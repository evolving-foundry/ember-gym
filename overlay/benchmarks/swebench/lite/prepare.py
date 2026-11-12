# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the public SWE-bench Lite split for NeMo Gym."""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FPATH = REPO_ROOT / "benchmarks" / "swebench" / "data" / "swebench_lite_benchmark.jsonl"
PROMPT_TEMPLATE = REPO_ROOT / "benchmarks" / "swebench" / "minimax_prompt.txt"


def prepare() -> Path:
    dataset = load_dataset("SWE-bench/SWE-bench_Lite", split="test", token=os.environ.get("HF_TOKEN"))
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w", encoding="utf-8", newline="\n") as output:
        for source_row in dataset:
            prompt = (
                template.replace("{{ workspace_path }}", "/testbed")
                .replace("{{ instance.problem_statement }}", source_row["problem_statement"])
                .replace("{{ instance.repo_language ~ ' ' if instance.repo_language else '' }}", "")
            )
            row = {
                **source_row,
                "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                "subset": "lite",
                "split": "test",
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset)} SWE-bench Lite problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
