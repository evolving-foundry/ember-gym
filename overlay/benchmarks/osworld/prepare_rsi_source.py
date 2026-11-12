#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch pinned public OSWorld 1.x task definitions into Gym-compatible JSONL.

This materializes task metadata only. The VM image, file assets, and live
services stay in their upstream locations and are handled by the regular
OSWorld preparation workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_REPOSITORY = "xlang-ai/OSWorld"
SOURCE_COMMIT = "b138d348256078fa634fc3b73567a7337c793e6b"
CATALOG_URL = f"https://github.com/{SOURCE_REPOSITORY}/blob/{SOURCE_COMMIT}/evaluation_examples/test_all.json"
OUTPUT_FPATH = DATA_DIR / "osworld_1x_verified_source.jsonl"


def prepare(output_fpath: Path | None = None) -> Path:
    with TemporaryDirectory(prefix="osworld-source-") as temporary:
        checkout = Path(temporary) / "OSWorld"
        subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", str(checkout), "remote", "add", "origin", f"https://github.com/{SOURCE_REPOSITORY}.git"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "sparse-checkout", "init", "--cone"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "sparse-checkout", "set", "evaluation_examples"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", SOURCE_COMMIT],
            check=True,
            capture_output=True,
            text=True,
        )

        catalog_path = checkout / "evaluation_examples" / "test_all.json"
        catalog_bytes = catalog_path.read_bytes()
        catalog = json.loads(catalog_bytes)
        payloads = {}
        missing: list[str] = []
        for domain, task_ids in catalog.items():
            for task_id in task_ids:
                path = checkout / "evaluation_examples" / "examples" / domain / f"{task_id}.json"
                if not path.is_file():
                    missing.append(f"{domain}/{task_id}")
                    continue
                payloads[(domain, str(task_id))] = path.read_bytes()

    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("Pinned OSWorld catalog must be a non-empty domain-to-task mapping")
    requests_to_make = [
        (domain, str(task_id))
        for domain, task_ids in sorted(catalog.items())
        for task_id in task_ids
    ]
    requests_to_make = [item for item in requests_to_make if item in payloads]

    output_fpath = (output_fpath or OUTPUT_FPATH).expanduser().resolve()
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    task_hash = hashlib.sha256()
    task_ids: list[str] = []
    with output_fpath.open("w", encoding="utf-8", newline="\n") as stream:
        for domain, task_id in requests_to_make:
            payload = payloads[(domain, task_id)]
            task_hash.update(domain.encode("utf-8") + b"\0" + task_id.encode("utf-8") + b"\0" + payload)
            task = json.loads(payload)
            if str(task.get("id")) != task_id:
                raise ValueError(f"OSWorld task ID mismatch for {domain}/{task_id}")
            instruction = task.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"OSWorld task {domain}/{task_id} has no instruction")

            row = {
                "task_id": task_id,
                "domain": domain,
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": f"Please complete this OSWorld task:\n\n{instruction}",
                        }
                    ]
                },
                "verifier_metadata": {
                    "task_id": task_id,
                    "domain": domain,
                    "osworld_task": task,
                    "source_repository": SOURCE_REPOSITORY,
                    "source_commit": SOURCE_COMMIT,
                },
            }
            task_ids.append(task_id)
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": 1,
        "benchmark": "OSWorld-Verified-1.x",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "catalog_url": CATALOG_URL,
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "task_definitions_sha256": task_hash.hexdigest(),
        "row_count": len(task_ids),
        "catalog_reference_count": sum(len(values) for values in catalog.values()),
        "missing_catalog_references": sorted(missing),
        "task_ids_sha256": hashlib.sha256("\n".join(sorted(task_ids)).encode()).hexdigest(),
        "file": output_fpath.name,
        "note": "Task definitions only; VM images and external task assets are not bundled.",
    }
    output_fpath.with_suffix(output_fpath.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(task_ids)} pinned OSWorld 1.x tasks: {output_fpath}")
    return output_fpath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
