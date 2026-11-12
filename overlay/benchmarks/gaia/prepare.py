#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare text-only tasks from GAIA's public validation split for NeMo Gym.

The script intentionally excludes rows with attachments: those require a separate file-aware
agent/evaluator contract. It never downloads or copies the private-answer test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "gaia_text_validation.jsonl"


def prepare(levels: tuple[int, ...] = (1, 2, 3)) -> Path:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    rows: list[dict[str, Any]] = []
    skipped_attachments = 0
    for level in levels:
        if level not in (1, 2, 3):
            raise ValueError(f"unsupported GAIA level: {level}")
        dataset = load_dataset(
            "gaia-benchmark/GAIA",
            f"2023_level{level}",
            split="validation",
            token=token,
        )
        for entry in dataset:
            if entry.get("file_name") or entry.get("file_path"):
                skipped_attachments += 1
                continue
            task_id = entry.get("task_id")
            question = entry.get("Question")
            answer = entry.get("Final answer")
            if not task_id or not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(f"invalid GAIA validation row at level {level}: {task_id!r}")
            prompt = (
                "Answer the user's question as accurately as possible. Use available tools when needed.\n"
                'Put only the final answer in the last line, prefixed exactly with "Final Answer:".'
            )
            rows.append(
                {
                    "task_id": str(task_id),
                    "question": question,
                    "expected_answer": answer,
                    "level": level,
                    "responses_create_params": {
                        "input": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": question},
                        ]
                    },
                }
            )
    if not rows:
        raise ValueError("GAIA preparation produced no text-only validation rows")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    encoded_rows = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    temporary_path = OUTPUT_FPATH.with_suffix(OUTPUT_FPATH.suffix + ".tmp")
    temporary_path.write_bytes(encoded_rows)
    temporary_path.replace(OUTPUT_FPATH)
    manifest = {
        "schema_version": 1,
        "benchmark": "GAIA",
        "source": "gaia-benchmark/GAIA",
        "split": "validation",
        "levels": list(levels),
        "task_count": len(rows),
        "attachment_rows_skipped": skipped_attachments,
        "materialized_jsonl_sha256": hashlib.sha256(encoded_rows).hexdigest(),
        "file": OUTPUT_FPATH.name,
        "note": "Public validation text-only tasks; private-answer test split excluded.",
    }
    OUTPUT_FPATH.with_suffix(OUTPUT_FPATH.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} text-only validation tasks; skipped {skipped_attachments} attachment tasks")
    return OUTPUT_FPATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3))
    args = parser.parse_args()
    prepare(tuple(args.levels))


if __name__ == "__main__":
    main()
