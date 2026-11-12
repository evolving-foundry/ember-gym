#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create an immutable, deterministic RSI train/holdout split from prepared JSONL rows.

This utility is for benchmark owners who provide a public or locally prepared task set but no
RSI-specific development split. The resulting holdout is a project-local promotion set; it must
not be reported as an official benchmark split or compared to an official leaderboard score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field(row: dict[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def split_jsonl(
    source: Path,
    train_output: Path,
    holdout_output: Path,
    *,
    id_field: str = "task_id",
    group_field: str | None = None,
    holdout_fraction: float = 0.2,
    seed: str = "agentgym-rsi-v1",
) -> Path:
    """Write disjoint stable splits and a content manifest; return the manifest path."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")
    source = source.expanduser().resolve()
    train_output = train_output.expanduser().resolve()
    holdout_output = holdout_output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if train_output == holdout_output or source in (train_output, holdout_output):
        raise ValueError("source, train, and holdout paths must be distinct")

    effective_group_field = group_field or id_field
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} must be a JSON object")
            task_id = _field(row, id_field)
            if task_id is None:
                raise ValueError(f"row {line_number} is missing id field {id_field!r}")
            task_id = str(task_id)
            if task_id in seen:
                raise ValueError(f"duplicate task id {task_id!r} at row {line_number}")
            seen.add(task_id)
            group_id = _field(row, effective_group_field)
            if group_id is None:
                raise ValueError(f"row {line_number} is missing group field {effective_group_field!r}")
            rows.append((task_id, str(group_id), line if line.endswith("\n") else line + "\n"))

    group_ids = {group_id for _, group_id, _ in rows}
    if len(group_ids) < 2:
        raise ValueError("at least two unique split groups are required")
    holdout_group_count = min(len(group_ids) - 1, max(1, math.ceil(len(group_ids) * holdout_fraction)))
    ranked_ids = sorted(
        (hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).hexdigest(), group_id)
        for group_id in group_ids
    )
    holdout_groups = {group_id for _, group_id in ranked_ids[:holdout_group_count]}
    train_rows = [(task_id, raw) for task_id, group_id, raw in rows if group_id not in holdout_groups]
    holdout_rows = [(task_id, raw) for task_id, group_id, raw in rows if group_id in holdout_groups]

    for output, selected in ((train_output, train_rows), (holdout_output, holdout_rows)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(raw for _, raw in selected), encoding="utf-8")

    manifest_path = train_output.with_suffix(train_output.suffix + ".manifest.json")
    manifest = {
        "schema_version": 1,
        "kind": "rsi-dataset-split",
        "source_sha256": _sha256_file(source),
        "id_field": id_field,
        "group_field": effective_group_field,
        "selection": "sha256(seed\\0group_id), lowest group hashes reserved for holdout",
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "holdout_count": len(holdout_rows),
        "group_count": len(group_ids),
        "train_group_count": len(group_ids - holdout_groups),
        "holdout_group_count": len(holdout_groups),
        "train_ids_sha256": hashlib.sha256("\n".join(sorted(i for i, _ in train_rows)).encode()).hexdigest(),
        "holdout_ids_sha256": hashlib.sha256("\n".join(sorted(i for i, _ in holdout_rows)).encode()).hexdigest(),
        "train_groups_sha256": hashlib.sha256("\n".join(sorted(group_ids - holdout_groups)).encode()).hexdigest(),
        "holdout_groups_sha256": hashlib.sha256("\n".join(sorted(holdout_groups)).encode()).hexdigest(),
        "train_sha256": _sha256_file(train_output),
        "holdout_sha256": _sha256_file(holdout_output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="prepared benchmark JSONL")
    parser.add_argument("--train", type=Path, required=True, help="output JSONL used for RSI search")
    parser.add_argument("--holdout", type=Path, required=True, help="output JSONL reserved for promotion")
    parser.add_argument("--id-field", default="task_id", help="task ID field; dotted paths are supported")
    parser.add_argument(
        "--group-field",
        default=None,
        help="optional grouping field; every row in a group stays in the same split",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="agentgym-rsi-v1")
    args = parser.parse_args()
    manifest = split_jsonl(
        args.source,
        args.train,
        args.holdout,
        id_field=args.id_field,
        group_field=args.group_field,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    print(manifest)


if __name__ == "__main__":
    main()
