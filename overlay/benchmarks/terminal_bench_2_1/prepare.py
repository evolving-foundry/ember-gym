# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare source rows for the terminal_bench_2_1 benchmark."""

import json
import hashlib
import tomllib
from pathlib import Path
from subprocess import run


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark.jsonl"
TASKS_REPOSITORY = "https://github.com/harbor-framework/terminal-bench-2-1.git"
TASKS_COMMIT = "7131e4375048a0e408a8fb404b5f499d726b695b"
TASKS_PATH = BENCHMARK_DIR / "terminal-bench-2-1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_task_checkout() -> None:
    if not TASKS_PATH.exists():
        run(["git", "clone", TASKS_REPOSITORY, str(TASKS_PATH)], check=True)
    if not (TASKS_PATH / ".git").exists():
        raise RuntimeError(f"task source path exists but is not a git checkout: {TASKS_PATH}")
    current_commit = run(
        ["git", "rev-parse", "HEAD"], cwd=TASKS_PATH, check=True, capture_output=True, text=True
    ).stdout.strip()
    if current_commit != TASKS_COMMIT:
        if run(["git", "status", "--porcelain"], cwd=TASKS_PATH, check=True, capture_output=True, text=True).stdout:
            raise RuntimeError(f"refusing to update dirty Terminal-Bench checkout at {TASKS_PATH}")
        run(["git", "fetch", "origin", TASKS_COMMIT], cwd=TASKS_PATH, check=True)
        run(["git", "checkout", "--detach", TASKS_COMMIT], cwd=TASKS_PATH, check=True)


def prepare() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ensure_task_checkout()

    num_samples = 0
    digest = hashlib.sha256()
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as f_out:
        for task_dir in sorted((TASKS_PATH / "tasks").iterdir()):
            if not task_dir.is_dir() or not (task_dir / "task.toml").is_file():
                continue
            with (task_dir / "task.toml").open("rb") as file:
                task_toml = tomllib.load(file)

            task_name = task_toml["task"]["name"]
            # Only pass agent-facing task fields into the Gym dataset. Reference
            # solutions and solution explanations remain inside the task source.
            sample = {
                "task_id": task_name,
                "responses_create_params": {
                    "input": [
                        {"role": "user", "content": (task_dir / "instruction.md").read_text(encoding="utf-8")}
                    ]
                },
                "task_name": task_name,
                "docker_image": task_toml["environment"]["docker_image"],
                "task_folder": task_dir.relative_to(BENCHMARK_DIR.parent.parent).as_posix(),
                "source_commit": TASKS_COMMIT,
            }
            encoded = (json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            f_out.write(encoded.decode("utf-8"))
            digest.update(encoded)
            num_samples += 1

    if num_samples != 89:
        raise ValueError(f"pinned Terminal-Bench 2.1 source has {num_samples} tasks; expected 89")
    manifest = {
        "schema_version": 1,
        "upstream_repository": TASKS_REPOSITORY,
        "upstream_commit": TASKS_COMMIT,
        "task_count": num_samples,
        "materialized_jsonl_sha256": digest.hexdigest(),
        "file": OUTPUT_PATH.name,
    }
    OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
