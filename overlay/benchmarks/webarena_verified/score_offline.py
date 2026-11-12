#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score captured WebArena-Verified runs with the official deterministic evaluator.

Expected artifact layout (as written by the official evaluator examples):

    <artifact-root>/<task-id>/agent_response_<task-id>.json
    <artifact-root>/<task-id>/network_<task-id>.har

This runner intentionally delegates grading to the pinned ``webarena-verified``
CLI. It is useful for offline replay and prevents local scoring logic from
silently drifting from the official evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_task_ids(path: Path) -> list[str]:
    task_ids: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("task_id")
            if value is None:
                raise ValueError(f"{path}:{line_number}: missing task_id")
            task_id = str(value)
            if task_id in task_ids:
                raise ValueError(f"{path}:{line_number}: duplicate task_id {task_id!r}")
            task_ids.append(task_id)
    return task_ids


def _read_eval_result(path: Path) -> dict[str, Any]:
    result_path = path / "eval_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"official evaluator did not write {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result.get("score"), (int, float)):
        raise ValueError(f"official result has no numeric score: {result_path}")
    return result


def _find_latest_artifacts(artifacts: Path, task_id: str) -> tuple[Path, Path]:
    response_name = f"agent_response_{task_id}.json"
    har_name = f"network_{task_id}.har"
    direct = artifacts / task_id
    if (direct / response_name).is_file() and (direct / har_name).is_file():
        return direct / response_name, direct / har_name

    candidates = [
        response_path
        for response_path in artifacts.rglob(response_name)
        if (response_path.parent / har_name).is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"task {task_id}: no response/HAR pair found under {artifacts}")
    # BrowserGym saves each rollout under a unique run ID. For RSI repeats,
    # select the latest complete attempt per task and record its exact hashes.
    selected = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    return selected, selected.parent / har_name


def score(
    *,
    dataset: Path,
    artifacts: Path,
    config: Path,
    output: Path,
    cli: str = "webarena-verified",
) -> Path:
    dataset = dataset.resolve()
    artifacts = artifacts.resolve()
    config = config.resolve()
    output = output.resolve()
    executable = shutil.which(cli)
    if executable is None:
        raise FileNotFoundError(
            f"cannot find {cli!r}; install webarena-verified or pass --cli with its executable path"
        )
    if not dataset.is_file() or not config.is_file():
        raise FileNotFoundError("dataset and evaluator config must both be existing files")

    task_ids = _load_task_ids(dataset)
    if not task_ids:
        raise ValueError(f"no tasks found in {dataset}")

    input_hashes: dict[str, dict[str, str]] = {}
    resolved_artifacts: dict[str, tuple[Path, Path]] = {}
    for task_id in task_ids:
        response_path, har_path = _find_latest_artifacts(artifacts, task_id)
        for path in (response_path, har_path):
            if not path.is_file():
                raise FileNotFoundError(f"task {task_id}: required run artifact missing: {path}")
        # Fail early on malformed agent output / HAR rather than invoking a partial batch.
        json.loads(response_path.read_text(encoding="utf-8"))
        har_doc = json.loads(har_path.read_text(encoding="utf-8"))
        if not isinstance(har_doc, dict) or not isinstance(har_doc.get("log"), dict):
            raise ValueError(f"task {task_id}: HAR must contain a top-level log object")
        input_hashes[task_id] = {
            "agent_response_path": str(response_path),
            "network_har_path": str(har_path),
            "agent_response_sha256": _sha256(response_path),
            "network_har_sha256": _sha256(har_path),
        }
        resolved_artifacts[task_id] = (response_path, har_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts_digest = hashlib.sha256(
        json.dumps(input_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    official_output = output.with_name(f"{output.stem}_official_{artifacts_digest[:12]}")
    official_output.mkdir(parents=True, exist_ok=True)
    for task_id, (response_path, har_path) in resolved_artifacts.items():
        task_dir = official_output / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(response_path, task_dir / response_path.name)
        shutil.copy2(har_path, task_dir / har_path.name)

    for task_id in task_ids:
        command = [
            executable,
            "eval-tasks",
            "--config",
            str(config),
            "--task-ids",
            task_id,
            "--output-dir",
            str(official_output),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode:
            tail = (completed.stdout + "\n" + completed.stderr)[-4000:]
            raise RuntimeError(f"official evaluator failed for task {task_id}:\n{tail}")

    results = {task_id: _read_eval_result(official_output / task_id) for task_id in task_ids}
    scores = [float(results[task_id]["score"]) for task_id in task_ids]
    summary = {
        "schema_version": 1,
        "evaluator": "webarena-verified CLI eval-tasks",
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "official_evaluator_config": str(config),
        "official_evaluator_config_sha256": _sha256(config),
        "artifact_root": str(artifacts),
        "official_evaluation_artifact_root": str(official_output),
        "task_count": len(task_ids),
        "score_mean": sum(scores) / len(scores),
        "score_sum": sum(scores),
        "task_ids": task_ids,
        "input_artifacts": input_hashes,
        "results": results,
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Scored {len(task_ids)} tasks; mean={summary['score_mean']:.4f}; wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="pinned WebArena-Verified JSONL")
    parser.add_argument("--artifacts", type=Path, required=True, help="root containing per-task response JSON and HAR")
    parser.add_argument("--config", type=Path, required=True, help="official evaluator environment config JSON")
    parser.add_argument("--output", type=Path, required=True, help="summary JSON output")
    parser.add_argument("--cli", default="webarena-verified", help="official CLI executable")
    args = parser.parse_args()
    score(dataset=args.dataset, artifacts=args.artifacts, config=args.config, output=args.output, cli=args.cli)


if __name__ == "__main__":
    main()
