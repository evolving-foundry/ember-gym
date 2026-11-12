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
"""Prepare the ARC-AGI benchmark (public evaluation split, 400 tasks).

Downloads the fchollet/ARC repo tarball (Apache-2.0) and flattens the
per-task JSON files under data/evaluation/ into one Gym JSONL row per task.
Each row keeps the structured grids for the verifier plus a pre-rendered
``question`` text consumed by the prompt template.
"""

import io
import json
import tarfile
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "arc_agi_benchmark.jsonl"

# Pinned to the fchollet/ARC master commit the 400-task split was prepared from
# (repo frozen upstream; re-pin deliberately, never track a moving branch).
TARBALL_URL = "https://github.com/fchollet/ARC/archive/399030444e0ab0cc8b4e199870fb20b863846f34.tar.gz"
UPSTREAM_SPLIT_DIR = "data/evaluation"


def render_grid(grid: list[list[int]]) -> str:
    return json.dumps(grid, separators=(", ", ": "))


def render_question(train_pairs: list[dict], test_input: list[list[int]]) -> str:
    """Render the task as text: example pairs, then the test input."""
    sections = []
    for index, pair in enumerate(train_pairs, start=1):
        sections.append(
            f"Example {index}:\n"
            f"Input: {render_grid(pair['input'])}\n"
            f"Output: {render_grid(pair['output'])}"
        )
    sections.append(f"Test input: {render_grid(test_input)}")
    return "\n\n".join(sections)


def prepare() -> Path:
    """Download fchollet/ARC, flatten the evaluation split, write Gym JSONL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    with urllib.request.urlopen(TARBALL_URL) as response, tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz") as tar:
        members = sorted(
            (
                member
                for member in tar.getmembers()
                if member.isfile()
                and f"/{UPSTREAM_SPLIT_DIR}/" in member.name
                and member.name.endswith(".json")
            ),
            key=lambda member: member.name,
        )

        with open(OUTPUT_FPATH, "w", encoding="utf-8") as fout:
            for member in members:
                task = json.loads(tar.extractfile(member).read().decode("utf-8"))
                task_id = Path(member.name).stem

                # Every ARC task has exactly one test pair. The public
                # evaluation split ships its ground-truth output; the hidden
                # test set (not distributed here) does not.
                test_pair = task["test"][0]
                new_entry = {
                    "task_id": task_id,
                    "train_pairs": task["train"],
                    "test_input": test_pair["input"],
                    "expected_output_grid": test_pair["output"],
                    "question": render_question(task["train"], test_pair["input"]),
                }
                fout.write(json.dumps(new_entry) + "\n")
                count += 1

    print(f"Wrote {count} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
