# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the arc_agi server.

Rows carry the full task: rendered prompt text plus the structured grids the
verifier compares against (``ARCAgiRunRequest`` at app.py). Verification is a
pure exact-grid match, so only ``expected_output_grid`` is consumed by
verify(); the other fields exist so prompts and RL pipelines can access the
task without re-reading the upstream dataset.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(
        description="Pre-rendered task text: example input/output pairs followed by the test input.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    train_pairs: List[dict] = Field(
        description="Example input/output grid pairs, [{input: [[int]], output: [[int]]}].",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    test_input: List[List[int]] = Field(
        description="The test input grid the model must transform.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_output_grid: List[List[int]] = Field(
        description="Ground-truth output grid; verify() rewards exact cell-for-cell match.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
