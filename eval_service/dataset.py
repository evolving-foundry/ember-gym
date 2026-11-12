# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RSI-level dataset splits (OF-1): train_probe (fixed small subset) and train_full (whole train split).

Per spec §8 open question 2, the train_probe subset is sampled deterministically per
(benchmark, dataset_version). The request seed is a run-identity/replay anchor: it neither selects
split rows nor guarantees a decoder seed, so probe scores stay horizontally comparable across
candidates.
"""

import json
import random
from pathlib import Path
from typing import Any, Callable, Optional

from agent_gym.layout import GymLayout
from eval_service.profile import BenchmarkSpec


def read_jsonl(fpath: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(fpath, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], fpath: str | Path) -> None:
    with open(fpath, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def probe_subset_seed(benchmark: BenchmarkSpec) -> str:
    return f"{benchmark.dataset_fpath}:{benchmark.dataset_version}"


def task_id(row: dict[str, Any], benchmark: BenchmarkSpec) -> str:
    """Read a benchmark's configured stable task ID, supporting dotted paths."""
    value: Any = row
    for part in benchmark.task_id_field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"row is missing configured task id field {benchmark.task_id_field!r}")
        value = value[part]
    if value is None:
        raise ValueError(f"row has null task id field {benchmark.task_id_field!r}")
    return str(value)


def _dataset_path(fpath: str, resolver: Optional[Callable[[str], Path]]) -> Path:
    path = Path(fpath)
    if path.is_absolute():
        return path
    if resolver is not None:
        return resolver(fpath)
    return GymLayout.discover().resolve_data(fpath)


def build_split_rows(
    benchmark: BenchmarkSpec,
    split: str,
    *,
    resolver: Optional[Callable[[str], Path]] = None,
) -> list[dict[str, Any]]:
    """Materialize the split's task rows from the benchmark dataset."""
    if split == "holdout":
        if not benchmark.holdout_dataset_fpath:
            raise ValueError("benchmark spec has no holdout dataset registered")
        return read_jsonl(_dataset_path(benchmark.holdout_dataset_fpath, resolver))
    rows = read_jsonl(_dataset_path(benchmark.dataset_fpath, resolver))
    if split == "train_full":
        return rows
    if split == "train_probe":
        rng = random.Random(probe_subset_seed(benchmark))
        return rng.sample(rows, k=min(benchmark.probe_split_size, len(rows)))
    raise ValueError(f"unsupported split: {split}")


def filter_rows(rows: list[dict[str, Any]], task_ids: list[str], benchmark: BenchmarkSpec) -> list[dict[str, Any]]:
    """Restrict split rows to the harness-pinned task set (RSI split contract), preserving the
    requested order. Raises on unknown ids — service._validate checks membership fail-fast, so
    this is a defensive re-check on the execution path."""
    index = {task_id(row, benchmark): row for row in rows}
    missing = [task_id for task_id in task_ids if task_id not in index]
    if missing:
        raise ValueError(f"task_ids not present in split rows: {missing[:5]}")
    return [index[task_id] for task_id in task_ids]
