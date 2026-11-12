#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize a pinned WebArena-Verified task split as UTF-8 JSONL.

This only prepares task data. Browser execution and HAR-based official scoring are
provided by the upstream ``webarena-verified`` package and need a BrowserGym/Gym
agent adapter before these rows can be run through eval_service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
UPSTREAM_REPOSITORY = "ServiceNow/webarena-verified"
UPSTREAM_COMMIT = "6473f72db5dcefc97b5725b59e734504edc28a21"
FULL_DATASET_URL = (
    f"https://raw.githubusercontent.com/{UPSTREAM_REPOSITORY}/{UPSTREAM_COMMIT}/"
    "assets/dataset/webarena-verified.json"
)
HARD_SUBSET_URL = (
    f"https://raw.githubusercontent.com/{UPSTREAM_REPOSITORY}/{UPSTREAM_COMMIT}/"
    "assets/dataset/subsets/webarena-verified-hard.json"
)
EXPECTED_ROWS = {"full": 812, "hard": 258}


def prepare(split: str = "hard", output_fpath: Path | None = None) -> Path:
    """Fetch one upstream split, validate its identity and save all task fields."""
    if split not in EXPECTED_ROWS:
        raise ValueError(f"unsupported split {split!r}; expected one of {sorted(EXPECTED_ROWS)}")

    response = requests.get(FULL_DATASET_URL, timeout=60)
    response.raise_for_status()
    full_bytes = response.content
    full_rows = json.loads(full_bytes)
    subset_sha256 = None
    if split == "hard":
        subset_response = requests.get(HARD_SUBSET_URL, timeout=30)
        subset_response.raise_for_status()
        subset_bytes = subset_response.content
        subset_sha256 = hashlib.sha256(subset_bytes).hexdigest()
        subset_data = json.loads(subset_bytes)
        task_ids = subset_data.get("task_ids") if isinstance(subset_data, dict) else None
        if not isinstance(task_ids, list):
            raise ValueError("pinned Hard-subset artifact has no task_ids list")
        task_ids = {str(task_id) for task_id in task_ids}
        rows_to_write = [row for row in full_rows if str(row.get("task_id")) in task_ids]
        if len(rows_to_write) != len(task_ids):
            raise ValueError("pinned Hard-subset task IDs do not match the pinned full task data")
    else:
        rows_to_write = full_rows
    expected_count = EXPECTED_ROWS[split]
    if len(rows_to_write) != expected_count:
        raise ValueError(
            f"upstream {split} row count changed: expected {expected_count}, received {len(rows_to_write)}"
        )

    output_fpath = (output_fpath or DATA_DIR / f"webarena_verified_{split}.jsonl").expanduser().resolve()
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    seen_ids: set[str] = set()
    with output_fpath.open("w", encoding="utf-8", newline="\n") as stream:
        for source_row in rows_to_write:
            row = dict(source_row)
            task_id = row.get("task_id")
            if task_id is None:
                raise ValueError("WebArena-Verified row is missing task_id")
            normalized_id = str(task_id)
            if normalized_id in seen_ids:
                raise ValueError(f"duplicate WebArena-Verified task_id: {normalized_id}")
            seen_ids.add(normalized_id)
            if not isinstance(row.get("intent"), str) or not row["intent"].strip():
                raise ValueError(f"task {normalized_id} has no intent")
            for key in ("intent_template_id", "revision"):
                if not isinstance(row.get(key), int):
                    raise ValueError(f"task {normalized_id} has no integer {key}")
            row["task_name"] = (
                f"webarena_verified.{row['intent_template_id']}.{normalized_id}.{row['revision']}"
            )
            # The BrowserGym task creates its own goal, but Gym's BaseRunRequest
            # still requires this envelope before dispatching the /run endpoint.
            row["responses_create_params"] = {"input": []}
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            stream.write(encoded.decode("utf-8"))
            digest.update(encoded)

    manifest = {
        "schema_version": 1,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "full_dataset_sha256": hashlib.sha256(full_bytes).hexdigest(),
        "hard_subset_sha256": subset_sha256,
        "split": split,
        "row_count": len(seen_ids),
        "task_ids_sha256": hashlib.sha256("\n".join(sorted(seen_ids)).encode("utf-8")).hexdigest(),
        "materialized_jsonl_sha256": digest.hexdigest(),
        "file": output_fpath.name,
    }
    output_fpath.with_suffix(output_fpath.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(seen_ids)} WebArena-Verified {split} tasks: {output_fpath}")
    return output_fpath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(EXPECTED_ROWS), default="hard")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    prepare(args.split, args.output)


if __name__ == "__main__":
    main()
