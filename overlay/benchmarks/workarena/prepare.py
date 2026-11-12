#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create deterministic BrowserGym WorkArena-L1 task/seed rows from its official catalog.

The catalog identifies task classes and official train/test assignment. Actual
WorkArena instances are generated against a gated ServiceNow instance at
rollout time; this script does not claim to materialize the 19,912 upstream
ServiceNow state instances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import requests


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
BROWSERGYM_COMMIT = "c05d7f38f4f788d6e4ee960512c3d5c8aabc6e42"
CATALOG_URL = (
    "https://raw.githubusercontent.com/ServiceNow/BrowserGym/"
    f"{BROWSERGYM_COMMIT}/browsergym/experiments/src/browsergym/experiments/benchmark/metadata/workarena.csv"
)


def prepare(
    *,
    output_fpath: Path | None = None,
    seeds: tuple[int, ...] = tuple(range(10)),
) -> Path:
    if not seeds:
        raise ValueError("at least one task seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("task seeds must be unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("task seeds must be non-negative")

    response = requests.get(CATALOG_URL, timeout=30)
    response.raise_for_status()
    catalog_bytes = response.content
    catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
    catalog_text = catalog_bytes.decode("utf-8-sig")
    tasks = [
        row
        for row in csv.DictReader(io.StringIO(catalog_text))
        if row.get("level", "").strip().lower() == "l1"
        and row.get("browsergym_split", "").strip().lower() == "train"
    ]
    if not tasks:
        raise ValueError("pinned BrowserGym catalog contains no WorkArena-L1 train tasks")

    output_fpath = (output_fpath or DATA_DIR / "workarena_l1_train.jsonl").expanduser().resolve()
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    ids: set[str] = set()
    rows_written = 0
    with output_fpath.open("w", encoding="utf-8", newline="\n") as stream:
        for task in tasks:
            task_name = task.get("task_name", "").strip()
            category = task.get("category", "").strip()
            if not task_name:
                raise ValueError("pinned WorkArena catalog contains an empty task_name")
            for seed in seeds:
                task_id = f"{task_name}::seed-{seed}"
                if task_id in ids:
                    raise ValueError(f"duplicate generated WorkArena task ID: {task_id}")
                ids.add(task_id)
                row = {
                    "task_id": task_id,
                    # BrowserGym supplies the task goal and observation at runtime;
                    # Gym still requires this request envelope in the input row.
                    "responses_create_params": {"input": []},
                    "task_name": task_name,
                    "seed": seed,
                    "level": "l1",
                    "category": category,
                    "catalog_split": "train",
                    "browsergym_commit": BROWSERGYM_COMMIT,
                }
                encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                stream.write(encoded.decode("utf-8"))
                digest.update(encoded)
                rows_written += 1

    manifest = {
        "schema_version": 1,
        "benchmark": "WorkArena-L1",
        "catalog_url": CATALOG_URL,
        "browsergym_commit": BROWSERGYM_COMMIT,
        "catalog_sha256": catalog_sha256,
        "catalog_train_task_count": len(tasks),
        "seeds": list(seeds),
        "row_count": rows_written,
        "task_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest(),
        "materialized_jsonl_sha256": digest.hexdigest(),
        "note": "task-seed catalog only; ServiceNow instances are provisioned at rollout time",
        "file": output_fpath.name,
    }
    output_fpath.with_suffix(output_fpath.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {rows_written} WorkArena-L1 task/seed rows from {len(tasks)} official train task classes")
    return output_fpath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=10)
    args = parser.parse_args()
    if args.seed_count < 1:
        parser.error("--seed-count must be >= 1")
    prepare(seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)), output_fpath=args.output)


if __name__ == "__main__":
    main()
