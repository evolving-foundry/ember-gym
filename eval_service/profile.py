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
"""Experiment profile: the pinned evaluation configuration (adapters, benchmarks, tier presets, budgets).

Spec: docs/RSI_Gym_集成需求规划_v0.1.md §3 (adapter pluggability) and §4 (tier presets EC-1,
constraint budgets OF-4/EC-4). Registering a new Agent = adding one AdapterSpec entry to a profile
YAML — no contract or other-adapter changes.
"""

import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval_service.contract import TierName


class ModelPin(BaseModel):
    """HF-1: model server parameters are part of the frozen harness manifest."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None


class TierPreset(BaseModel):
    """EC-1: rollout count / repeats / concurrency are preset per tier, never free request params (EC-3)."""

    model_config = ConfigDict(extra="forbid")

    limit: Optional[int] = Field(default=None, gt=0, description="rollout cap; None = run the whole split")
    num_repeats: int = Field(default=1, ge=1)
    concurrency: int = Field(default=8, ge=1)
    allowed_splits: list[str] = Field(default_factory=lambda: ["train_probe", "train_full"])


class AdapterSpec(BaseModel):
    """One Agent harness registration. `harness_agent` is the value passed to `gym eval run --agent`."""

    model_config = ConfigDict(extra="forbid")

    harness_agent: str = Field(min_length=1)
    version: str = "0"
    config_paths: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    # Non-secret runtime settings that affect the selected agent. The manifest
    # hashes their values without writing them to result artifacts.
    manifest_env: list[str] = Field(default_factory=list)
    description: str = ""


class RubricJudgePin(BaseModel):
    """§3.3: judge endpoint pin for rubric-scored benchmarks.

    The binary rubric protocol anchors the judge at temperature 0; the model name and pin travel
    into the harness manifest (HF-1). Endpoint credentials flow through env vars (§6.4) — only
    their NAMES are declared here and hashed into the manifest, never their values.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    temperature: float = 0.0
    manifest_env: list[str] = Field(default_factory=list)


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_fpath: str = Field(min_length=1)
    # OF-2: the locked holdout dataset is registered here but only reachable via the promotion
    # pipeline, never through the outer loop's evaluate() entry point.
    holdout_dataset_fpath: Optional[str] = None
    dataset_version: str = "auto"
    config_paths: list[str] = Field(default_factory=list)
    # Python/config/launch sources that influence this benchmark's reward path.
    source_paths: list[str] = Field(default_factory=list)
    manifest_env: list[str] = Field(default_factory=list)
    # Upstream datasets use different stable ID columns (for example instance_id or id).
    task_id_field: str = "task_id"
    probe_split_size: int = Field(default=32, ge=1)
    # §3.3 rubric judgment path: when rubric_library_fpath is set the benchmark has no programmatic
    # verifier and is scored by the rubric judgment adapter against this frozen snapshot
    # (DataDesigner export_frozen_snapshot output, evidence_scope=frozen_seed_only, §4.5). The
    # pinned sha256 is checked fail-fast before any rollout runs (§7.2), and the snapshot file
    # content joins the harness manifest (HF-1).
    rubric_library_fpath: Optional[str] = None
    rubric_library_sha256: Optional[str] = None
    rubric_judge: Optional[RubricJudgePin] = None
    # RuACT protocol anchor: rollout responses to one query are judged as a complete group.
    rubric_group_size: int = Field(default=8, ge=2)

    @model_validator(mode="after")
    def _rubric_fields_are_all_or_nothing(self) -> "BenchmarkSpec":
        if self.rubric_library_fpath is None:
            if self.rubric_library_sha256 is not None or self.rubric_judge is not None:
                raise ValueError("rubric_library_sha256 / rubric_judge require rubric_library_fpath")
        elif self.rubric_library_sha256 is None or self.rubric_judge is None:
            raise ValueError("a rubric-scored benchmark requires rubric_library_sha256 and rubric_judge alongside rubric_library_fpath")
        if (
            self.rubric_library_sha256 is not None
            and not re.fullmatch(r"sha256:[0-9a-f]{64}", self.rubric_library_sha256)
        ):
            raise ValueError("rubric_library_sha256 must be a lowercase 'sha256:<64 hex>' digest")
        return self


class ConstraintBudgets(BaseModel):
    """Soft budgets whose exhaustion requires explicit acknowledgement (OF-4 / EC-4)."""

    model_config = ConfigDict(extra="forbid")

    split_eval_soft_limit: int = Field(default=10, ge=1)
    lineage_max_evals: int = Field(default=500, ge=1)
    lineage_max_tokens: int = Field(default=2_000_000_000, ge=1)


DEFAULT_TIER_PRESETS: dict[str, TierPreset] = {
    # probe ≈ 32 rollouts on the small train_probe subset (EC-1); screen = whole split once;
    # full = train_full with repeated rollouts. Repeats stay tier-owned; the request seed is not
    # silently reinterpreted as a decoder seed.
    TierName.PROBE: TierPreset(limit=32, num_repeats=1, concurrency=8, allowed_splits=["train_probe"]),
    TierName.SCREEN: TierPreset(
        limit=None, num_repeats=1, concurrency=8, allowed_splits=["train_probe", "train_full"]
    ),
    TierName.FULL: TierPreset(limit=None, num_repeats=4, concurrency=8, allowed_splits=["train_full"]),
}


class ExperimentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1)
    adapters: dict[str, AdapterSpec] = Field(default_factory=dict)
    benchmarks: dict[str, BenchmarkSpec] = Field(default_factory=dict)
    tiers: dict[str, TierPreset] = Field(default_factory=lambda: dict(DEFAULT_TIER_PRESETS))
    budgets: ConstraintBudgets = Field(default_factory=ConstraintBudgets)
    model: Optional[ModelPin] = None
    # HF-1: harness_manifest of the current pin; EvalRequest.harness_manifest defaults to this.
    harness_pin: Optional[str] = None
    # P1: optional trust anchor written by the host activation adapter. When set, a holdout
    # request is rejected unless the activated bundle identity matches candidate_id/digest.
    # This is operational wiring, not part of the scored harness manifest.
    activation_manifest_fpath: Optional[str] = None

    def tier_preset(self, tier: TierName) -> TierPreset:
        return self.tiers[tier.value]


def profile_from_dict(data: dict[str, Any]) -> ExperimentProfile:
    return ExperimentProfile.model_validate(data)


def profile_from_yaml(fpath: str | Path) -> ExperimentProfile:
    with open(fpath, encoding="utf-8") as fh:
        return profile_from_dict(yaml.safe_load(fh))
