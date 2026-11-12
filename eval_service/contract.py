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
"""Agent-agnostic contract between the RSI outer search loop and the Gym evaluation service.

Spec: docs/RSI_Gym_集成需求规划_v0.1.md (§4 constraints, §5 contract).
Parameters that the spec deliberately omits (limit, max_concurrency, sampling params, harness
overrides) must stay absent from these models — `extra="forbid"` turns any attempt to pass them
into a validation error (EC-3 / HF-2: no entry point for the outer agent to make that mistake).
"""

import json
from enum import StrEnum
from hashlib import sha256
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


GYM_EVALUATE_TOOL_NAME = "gym_evaluate"
EVALUATION_MANIFEST_SCHEMA_VERSION = 2

AVAILABLE_SPLITS = ("train_probe", "train_full")
HOLDOUT_SPLIT = "holdout"

OVERFIT_RISK_STATEMENT = (
    "你正在对一个固定数据集反复优化。重复评估同一 split 会造成对评估集的过拟合，分数提升不再泛化。"
    "holdout split 不可由你选择或访问。"
)
COST_RISK_STATEMENT = (
    "每次评估是真实 rollout，耗时且消耗 token。先用 probe 级别粗筛，只对精英候选升级到更贵的评估层级。"
    "每次评估都会返回实际成本，请在搜索决策中显式消费它。"
)
HARNESS_FREEZE_RISK_STATEMENT = (
    "候选间的分数差异只有在 harness（Agent harness + model server + 环境配置）完全一致时才归因于候选本身。"
    "你不得修改 harness；若你的搜索变量就是 harness 配置，必须走 experiment_profile 变更流程，"
    "且结果与普通评估隔离比较。"
)

GYM_EVALUATE_TOOL_DESCRIPTION = f"""\
在固定 benchmark 上评估一个候选 Agent，返回 reward 与指标。

[过拟合警示] {OVERFIT_RISK_STATEMENT}
split 取值限定 train_probe（小子集粗筛）与 train_full（全量训练 split）；请求 holdout 会被直接拒绝
（HOLDOUT_LOCKED）。同一 split 的评估次数有软预算，超限后需显式传 acknowledge_exhausted_split 才会继续，
且结果标记 trusted: false。

[成本警示] {COST_RISK_STATEMENT}
tier 参数（probe|screen|full）决定 rollout 数/重复次数/并发的预设，你无法单独放大单次评估的规模。
budget 可为单次评估设置 timeout_s 与 max_rollouts 上限（与 tier 预设取交集）；超限的评估返回
budget_exhausted 状态并附已完成 rollout 的部分指标。每次评估返回实际成本与谱系累计成本
（lineage_cost_to_date）；谱系预算超限后需显式传 acknowledge_lineage_budget 才会继续。

[harness 冻结警示] {HARNESS_FREEZE_RISK_STATEMENT}
harness 配置（agent harness + 环境配置 + model server 参数）固化为内容寻址的 manifest（sha256），
运行态（framework/overlay/repo/evaluator）另行固化为 environment_digest，二者复合为 evaluation_digest；
结果同时回带 manifest_schema_version、harness_manifest_digest、environment_digest 与 evaluation_digest。
你不能逐次覆盖 harness 配置；显式传入与当前 pin 不一致的 harness_manifest 会被拒绝（MANIFEST_MISMATCH）。

结果中的 overfit_guard（eval_count / score_trend / trusted）与 warnings 必须纳入你的搜索决策。
"""

EVAL_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string", "description": "候选谱系标识，由外层搜索循环管理"},
        "adapter": {
            "type": "string",
            "pattern": "^[A-Za-z][A-Za-z0-9_.-]*$",
            "description": "profile 中注册的 Agent adapter 名称",
        },
        "benchmark": {"type": "string", "description": "已注册 benchmark 名"},
        "split": {"type": "string", "enum": list(AVAILABLE_SPLITS)},
        "tier": {"type": "string", "enum": ["probe", "screen", "full"]},
        "budget": {
            "type": "object",
            "properties": {
                "timeout_s": {"type": "integer", "minimum": 1},
                "max_rollouts": {"type": "integer", "minimum": 1},
            },
        },
        "harness_manifest": {"type": "string", "description": "缺省取当前 experiment_profile 的 pin"},
        "seed": {
            "type": "integer",
            "description": "评估运行幂等/复现锚点；不改变数据集切分，也不保证作为模型解码 seed",
        },
        "dataset_version": {
            "type": "string",
            "default": "auto",
            "description": "auto 或当前 benchmark 在 profile 中注册的版本",
        },
        "acknowledge_exhausted_split": {"type": "boolean"},
        "acknowledge_lineage_budget": {"type": "boolean"},
    },
    "required": ["candidate_id", "adapter", "benchmark", "split", "tier", "seed"],
}


class AdapterName(StrEnum):
    """Common adapter names retained for callers; profiles may register others."""

    DSH = "dsh"
    CODEX = "codex"
    CLAUDE_CODE = "claude_code"
    OPENCODE = "opencode"
    HERMES = "hermes"
    REMOTE = "remote"
    BROWSERGYM = "browsergym"
    TERMINUS_2 = "terminus_2"


class TierName(StrEnum):
    PROBE = "probe"
    SCREEN = "screen"
    FULL = "full"


class EvalStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    QUEUED = "queued"
    REJECTED = "rejected"


class ErrorCode(StrEnum):
    HOLDOUT_LOCKED = "HOLDOUT_LOCKED"
    ADAPTER_NOT_REGISTERED = "ADAPTER_NOT_REGISTERED"
    BENCHMARK_UNKNOWN = "BENCHMARK_UNKNOWN"
    INVALID_SPLIT = "INVALID_SPLIT"
    TIER_SPLIT_MISMATCH = "TIER_SPLIT_MISMATCH"
    SPLIT_EXHAUSTED = "SPLIT_EXHAUSTED"
    LINEAGE_BUDGET_EXCEEDED = "LINEAGE_BUDGET_EXCEEDED"
    MANIFEST_MISMATCH = "MANIFEST_MISMATCH"
    # §7.2: a rubric-scored benchmark refuses to start when its frozen library snapshot is
    # missing, tampered with, or does not match the sha256 pinned in the profile.
    RUBRIC_LIBRARY_MISMATCH = "RUBRIC_LIBRARY_MISMATCH"
    # §7.3(5): a judge endpoint outage is an explicit failure — never an all-zero degradation.
    RUBRIC_JUDGE_FAILED = "RUBRIC_JUDGE_FAILED"
    EVAL_FAILED = "EVAL_FAILED"
    TASK_SET_INVALID = "TASK_SET_INVALID"
    INVALID_DATASET_VERSION = "INVALID_DATASET_VERSION"
    # P1: the host activation surface does not identify the bundle named by the holdout request.
    ACTIVATION_MISMATCH = "ACTIVATION_MISMATCH"
    # The promotion control plane runs the complete holdout split, so only the full tier
    # (the one with no rollout cap) is admissible.
    PROMOTION_TIER_INVALID = "PROMOTION_TIER_INVALID"


class BudgetSpec(BaseModel):
    """EC-2: per-eval budget caps. Caps intersect with tier presets (EC-1)."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: Optional[int] = Field(default=None, gt=0)
    max_rollouts: Optional[int] = Field(default=None, gt=0)


class EvalRequest(BaseModel):
    """Outer loop → evaluation service. The trusted profile resolves `adapter`."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    # Content identity of the candidate bundle (for example sha256 of its file tree). This is
    # supplied by the host/controller that materializes the bundle; MCP callers cannot set it.
    # Optional for backwards compatibility: when present it participates in eval_id, so changing
    # the bundle under the same candidate_id can never replay an old result.
    candidate_digest: Optional[str] = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    adapter: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    benchmark: str = Field(min_length=1)
    # Validated by the service rather than an enum so that "holdout" yields HOLDOUT_LOCKED
    # (a targeted error with a fix hint) instead of a generic pydantic validation message (OF-1).
    split: str = Field(min_length=1)
    tier: TierName
    budget: Optional[BudgetSpec] = None
    harness_manifest: Optional[str] = None
    seed: int = Field(description="评估运行幂等/复现锚点；不改变数据集切分，也不保证作为模型解码 seed")
    dataset_version: str = "auto"
    # RSI split contract (EvaluationRequest.taskIds): the harness pins the exact task subset
    # (held-in / held-out halves of the split). None = whole split; the MCP agent path never
    # sets it (deliberately absent from EVAL_TOOL_PARAMETERS, like the other harness-only knobs).
    task_ids: Optional[list[str]] = None
    acknowledge_exhausted_split: Optional[bool] = None
    acknowledge_lineage_budget: Optional[bool] = None


class PromotionRequest(BaseModel):
    """Trusted promotion-control-plane request for one RSI candidate.

    This is deliberately a different model from EvalRequest. The holdout split is not a
    request field and task pinning is not expressible: the control plane always evaluates the
    complete, locked holdout split. ``extra="forbid"`` consequently rejects attempts to smuggle
    in ``split`` or ``task_ids`` at the HTTP boundary.

    F-01: only the ``full`` tier is admissible (it is the only preset with no rollout cap), so a
    promotion can never evaluate a truncated slice of the holdout. ``budget.max_rollouts`` is
    likewise not a holdout truncation knob — it is dropped when the request is converted, and a
    ``BudgetSpec.max_rollouts`` still cannot shorten a holdout run. ``timeout_s`` remains usable
    because it bounds wall-clock without shrinking the evaluated task set.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    candidate_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    adapter: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    benchmark: str = Field(min_length=1)
    tier: TierName = TierName.FULL
    budget: Optional[BudgetSpec] = None
    seed: int = Field(
        description=(
            "Rollout sampling seed. Recorded in the report, but deliberately excluded from the "
            "holdout one-shot identity: changing it cannot buy a second holdout."
        )
    )
    dataset_version: str = "auto"

    @model_validator(mode="after")
    def _require_full_tier(self) -> "PromotionRequest":
        if self.tier is not TierName.FULL:
            raise ValueError(
                f"promotion always evaluates the complete holdout split; tier must be 'full', got '{self.tier.value}'"
            )
        return self

    def to_eval_request(self) -> EvalRequest:
        # max_rollouts would cap the number of holdout tasks Gym materializes, which contradicts
        # the "complete holdout" contract; drop it rather than letting it reach the runner.
        budget = (
            BudgetSpec(timeout_s=self.budget.timeout_s)
            if self.budget is not None and self.budget.timeout_s is not None
            else None
        )
        return EvalRequest(
            candidate_id=self.candidate_id,
            candidate_digest=self.candidate_digest,
            adapter=self.adapter,
            benchmark=self.benchmark,
            split=HOLDOUT_SPLIT,
            tier=self.tier,
            budget=budget,
            seed=self.seed,
            dataset_version=self.dataset_version,
        )


class PromotionState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"
    REJECTED = "rejected"


class PromotionSubmitResponse(BaseModel):
    promotion_id: str
    state: PromotionState
    candidate_id: str
    candidate_digest: str
    benchmark: str
    reused: bool = False
    report: Optional[dict[str, Any]] = None


class EvalError(BaseModel):
    code: ErrorCode
    message: str
    fix_hint: str


class PromotionStatusResponse(BaseModel):
    promotion_id: str
    state: PromotionState
    report: Optional[dict[str, Any]] = None
    error: Optional[EvalError] = None


class EvalMetrics(BaseModel):
    """Aggregate metrics for one evaluation.

    F-03: failures are counted as zero. ``reward_mean`` and ``pass_rate`` use the attempted set
    as the denominator — completed plus failed rollouts, raised to Gym's materialized rollout
    count when that inventory exists (which also covers rollouts lost before they could report)
    — so a candidate cannot raise its score by crashing. ``rollouts_attempted`` records that
    denominator explicitly; it is None for legacy records written before this field existed,
    which lets the promotion comparator tell an already-effective pass_rate from the old raw one
    without double-penalizing.
    """

    reward_mean: Optional[float] = None
    pass_rate: Optional[float] = None
    rollouts: int = 0
    rollouts_failed: int = 0
    rollouts_attempted: Optional[int] = None


class EvalItemCheck(BaseModel):
    """One bounded task-level check exposed to the outer search loop."""

    id: str
    status: str
    score: Optional[float] = None
    reason: Optional[str] = None


class EvalItemDiagnostic(BaseModel):
    """A bounded, non-ground-truth diagnostic for a task-level rollout."""

    kind: str
    code: Optional[str] = None
    message: str
    truncated: bool = False


class EvalItemEvidence(BaseModel):
    """Per-task training evidence.

    This is intentionally absent from holdout results: only the locked promotion control plane
    may see holdout aggregates. Train results expose enough evidence for the proposer to diagnose
    a failure without leaking benchmark ground truth.
    """

    task_id: str
    rollouts: int = 0
    reward_mean: Optional[float] = None
    pass_rate: Optional[float] = None
    rollouts_failed: int = 0
    checks: list[EvalItemCheck] = Field(default_factory=list)
    diagnostics: list[EvalItemDiagnostic] = Field(default_factory=list)
    truncated: bool = False


class LineageCost(BaseModel):
    evals: int = 0
    total_tokens: int = 0


class EvalCost(BaseModel):
    wall_time_s: float = 0.0
    total_tokens: int = 0
    rollouts_executed: int = 0
    lineage_cost_to_date: LineageCost = LineageCost()


class CriterionSignal(BaseModel):
    """Per-criterion overfit signal merged into OverfitGuard (structure-compatible extension)."""

    hit_count: int = 0
    scored_rollouts: int = 0
    trusted: bool = True
    reason: str = ""


class OverfitGuard(BaseModel):
    split_eval_count: int = 0
    score_trend: str = "insufficient_history"
    trusted: bool = True
    # P7 (design doc §3.3/§5.3): rubric-scored benchmarks attach per-criterion trusted/overfit
    # signals — a criterion that hits on every rollout or none has no discriminating power for
    # promotion. None for benchmarks with programmatic verifiers.
    criterion_signals: Optional[dict[str, CriterionSignal]] = None


class RubricEvidence(BaseModel):
    """§6.1 rubric_evidence verdict detail, reused Gym-side as the judgment adapter's summary.

    Per-rollout scoring matrices land next to the rollouts in ``rubric_evidence.jsonl``
    (``evidence_ref``); this summary carries the aggregate verdict plus the frozen library
    identity that produced it (audit lineage, §6.1 ``rubric_library_sha256``).
    """

    rubric_library_sha256: str
    rubric_ids: list[str]
    rubric_weights: list[float]
    group_pass_counts: list[int]
    judge_model: str
    scored_rollouts: int = 0
    ungrouped_rollouts: int = 0
    evidence_ref: Optional[str] = None


class EvalResult(BaseModel):
    eval_id: str
    status: EvalStatus
    error: Optional[EvalError] = None
    metrics: Optional[EvalMetrics] = None
    items: list[EvalItemEvidence] = Field(default_factory=list)
    per_rollout_ref: Optional[str] = None
    cost: Optional[EvalCost] = None
    manifest_schema_version: int = 1
    candidate_digest: Optional[str] = None
    harness_manifest_digest: Optional[str] = None
    environment_digest: Optional[str] = None
    evaluation_digest: Optional[str] = None
    experiment_profile_id: Optional[str] = None
    dataset_version: Optional[str] = None
    overfit_guard: Optional[OverfitGuard] = None
    rubric_evidence: Optional[RubricEvidence] = None
    warnings: list[str] = []


def evaluation_digest_for(
    harness_manifest_digest: str,
    environment_digest: str,
    *,
    schema_version: int = EVALUATION_MANIFEST_SCHEMA_VERSION,
) -> str:
    """Combine harness and runtime-environment identities into the cache/ledger key."""
    canonical = json.dumps(
        {
            "schema_version": schema_version,
            "harness_manifest_digest": harness_manifest_digest,
            "environment_digest": environment_digest,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + sha256(canonical).hexdigest()


def eval_id_for(
    candidate_id: str,
    benchmark: str,
    split: str,
    tier: str,
    seed: int,
    manifest_digest: str,
    task_ids: Optional[list[str]] = None,
    *,
    candidate_digest: Optional[str] = None,
    environment_digest: Optional[str] = None,
    schema_version: int = EVALUATION_MANIFEST_SCHEMA_VERSION,
) -> str:
    """§5.3: idempotency key over (candidate_id, benchmark, split, tier, seed, evaluation digest).

    When environment_digest is supplied, the evaluation digest is the composite of the
    harness and environment identities. The legacy one-digest form is retained for reading
    old records and for callers that do not yet have a runtime identity.

    A pinned task_ids subset participates in the key so held-in/held-out evaluations of the same
    split never collide on the result cache (a collision would replay results from a different
    task set under the same eval_id). A candidate content digest, when supplied, does the same for
    bundle contents: the same candidate_id under a new bundle must not reuse an old score."""
    evaluation_digest = (
        evaluation_digest_for(manifest_digest, environment_digest, schema_version=schema_version)
        if environment_digest is not None
        else manifest_digest
    )
    key = [candidate_id, benchmark, split, tier, str(seed), evaluation_digest]
    if candidate_digest is not None:
        key.append(candidate_digest)
    if task_ids is not None:
        key.append(sha256(json.dumps(sorted(task_ids)).encode("utf-8")).hexdigest()[:16])
    return "eval-" + sha256("|".join(key).encode("utf-8")).hexdigest()[:16]


UNRESOLVED_MANIFEST = "unresolved"


def unresolved_eval_id(
    request: "EvalRequest",
    *,
    environment_digest: Optional[str] = None,
    schema_version: int = EVALUATION_MANIFEST_SCHEMA_VERSION,
) -> str:
    """Deterministic id for rejections raised before a manifest digest could be computed."""
    return eval_id_for(
        request.candidate_id,
        request.benchmark,
        request.split,
        request.tier.value,
        request.seed,
        UNRESOLVED_MANIFEST,
        task_ids=request.task_ids,
        candidate_digest=request.candidate_digest,
        environment_digest=environment_digest,
        schema_version=schema_version,
    )


def rejection(eval_id: str, code: ErrorCode, message: str, fix_hint: str) -> EvalResult:
    return EvalResult(
        eval_id=eval_id, status=EvalStatus.REJECTED, error=EvalError(code=code, message=message, fix_hint=fix_hint)
    )
