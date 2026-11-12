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
"""RSI × Gym evaluation service.

A standalone component (spec: docs/RSI_Gym_集成需求规划_v0.1.md) that wraps Gym's native
`gym eval run` entrypoint behind an agent-agnostic EvalRequest/EvalResult contract and enforces
the three hard constraints — overfit guard (OF), cost control (EC), harness freeze (HF) — at a
single entry point.
"""

from eval_service.contract import (
    EVAL_TOOL_PARAMETERS,
    GYM_EVALUATE_TOOL_DESCRIPTION,
    GYM_EVALUATE_TOOL_NAME,
    AdapterName,
    BudgetSpec,
    ErrorCode,
    EvalCost,
    EvalError,
    EvalMetrics,
    EvalRequest,
    EvalResult,
    EvalStatus,
    LineageCost,
    OverfitGuard,
    TierName,
    eval_id_for,
)
from eval_service.guards import Ledger
from eval_service.profile import (
    AdapterSpec,
    BenchmarkSpec,
    ConstraintBudgets,
    ExperimentProfile,
    ModelPin,
    TierPreset,
    profile_from_yaml,
)
from eval_service.promotion import PromotionPipeline
from eval_service.queue import BackgroundEvaluationRunner
from eval_service.runner import GymEvalRunner
from eval_service.service import EvaluationService


__all__ = [
    "EVAL_TOOL_PARAMETERS",
    "GYM_EVALUATE_TOOL_DESCRIPTION",
    "GYM_EVALUATE_TOOL_NAME",
    "AdapterName",
    "AdapterSpec",
    "BackgroundEvaluationRunner",
    "BenchmarkSpec",
    "BudgetSpec",
    "ConstraintBudgets",
    "ErrorCode",
    "EvalCost",
    "EvalError",
    "EvalMetrics",
    "EvalRequest",
    "EvalResult",
    "EvalStatus",
    "EvaluationService",
    "ExperimentProfile",
    "GymEvalRunner",
    "Ledger",
    "LineageCost",
    "ModelPin",
    "OverfitGuard",
    "PromotionPipeline",
    "TierName",
    "TierPreset",
    "eval_id_for",
    "profile_from_yaml",
]
