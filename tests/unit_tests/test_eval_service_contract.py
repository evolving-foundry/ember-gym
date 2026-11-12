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
import pydantic
import pytest

from eval_service.contract import (
    AVAILABLE_SPLITS,
    COST_RISK_STATEMENT,
    EVAL_TOOL_PARAMETERS,
    GYM_EVALUATE_TOOL_DESCRIPTION,
    GYM_EVALUATE_TOOL_NAME,
    HARNESS_FREEZE_RISK_STATEMENT,
    OVERFIT_RISK_STATEMENT,
    AdapterName,
    EvalRequest,
    TierName,
    eval_id_for,
    evaluation_digest_for,
)


class TestToolDescription:
    def test_embeds_the_three_risk_statements_verbatim(self) -> None:
        # Spec §5.1: the outer agent must be able to read all three risk statements (§4.1-§4.3)
        # before every evaluation decision.
        assert OVERFIT_RISK_STATEMENT in GYM_EVALUATE_TOOL_DESCRIPTION
        assert COST_RISK_STATEMENT in GYM_EVALUATE_TOOL_DESCRIPTION
        assert HARNESS_FREEZE_RISK_STATEMENT in GYM_EVALUATE_TOOL_DESCRIPTION

    def test_tool_name_matches_spec(self) -> None:
        assert GYM_EVALUATE_TOOL_NAME == "gym_evaluate"

    def test_seed_is_a_run_identity_anchor_not_a_sampling_promise(self) -> None:
        description = EVAL_TOOL_PARAMETERS["properties"]["seed"]["description"]
        assert "幂等/复现锚点" in description
        assert "不改变数据集切分" in description
        assert "不保证" in description


class TestEvalRequest:
    def test_minimal_request(self) -> None:
        request = EvalRequest(
            candidate_id="cand-001", adapter="claude_code", benchmark="toy", split="train_probe", tier="probe", seed=1
        )
        assert request.adapter == AdapterName.CLAUDE_CODE
        assert request.tier == TierName.PROBE
        assert request.dataset_version == "auto"

    @pytest.mark.parametrize(
        "forbidden", ["limit", "max_concurrency", "temperature", "agent", "allow_harness_override"]
    )
    def test_spec_omitted_params_are_not_expressible(self, forbidden: str) -> None:
        # EC-3 / HF-2: limit/seed-override/max_concurrency/harness overrides must not exist in the
        # schema — passing them is a contract error, not a silently-ignored default.
        payload = {
            "candidate_id": "cand-001",
            "adapter": "codex",
            "benchmark": "toy",
            "split": "train_probe",
            "tier": "probe",
            "seed": 1,
            forbidden: 32,
        }
        with pytest.raises(pydantic.ValidationError):
            EvalRequest(**payload)

    def test_budget_rejects_unknown_fields(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            EvalRequest(
                candidate_id="cand-001",
                adapter="codex",
                benchmark="toy",
                split="train_probe",
                tier="probe",
                seed=1,
                budget={"timeout_s": 60, "seed": 7},
            )

    def test_candidate_digest_must_be_a_sha256_digest(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            EvalRequest(
                candidate_id="cand-001",
                candidate_digest="not-a-digest",
                adapter="codex",
                benchmark="toy",
                split="train_probe",
                tier="probe",
                seed=1,
            )


class TestEvalId:
    def test_deterministic_for_identical_inputs(self) -> None:
        kwargs = ("cand-001", "toy", "train_probe", "probe", 1, "sha256:a")
        assert eval_id_for(*kwargs) == eval_id_for(*kwargs)
        assert eval_id_for(*kwargs).startswith("eval-")

    def test_varies_on_every_idempotency_component(self) -> None:
        base = ("cand-001", "toy", "train_probe", "probe", 1, "sha256:a")
        for i in range(len(base)):
            variant = list(base)
            variant[i] = str(variant[i]) + "-x" if isinstance(variant[i], str) else variant[i] + 1
            assert eval_id_for(*base) != eval_id_for(*variant)

    def test_environment_digest_participates_in_eval_id(self) -> None:
        base = ("cand-001", "toy", "train_probe", "probe", 1, "sha256:a")
        first = eval_id_for(*base, environment_digest="sha256:env-a")
        second = eval_id_for(*base, environment_digest="sha256:env-b")
        assert first != second

    def test_candidate_digest_participates_in_eval_id(self) -> None:
        base = ("cand-001", "toy", "train_probe", "probe", 1, "sha256:a")
        first = eval_id_for(*base, candidate_digest="sha256:" + "a" * 64)
        second = eval_id_for(*base, candidate_digest="sha256:" + "b" * 64)
        assert first != second
        assert eval_id_for(*base) == eval_id_for(*base, candidate_digest=None)

    def test_legacy_form_stays_stable_without_environment_identity(self) -> None:
        base = ("cand-001", "toy", "train_probe", "probe", 1, "sha256:a")
        assert eval_id_for(*base) == eval_id_for(*base, environment_digest=None)

    def test_evaluation_digest_combines_harness_and_environment(self) -> None:
        assert evaluation_digest_for("sha256:harness", "sha256:env-a") != evaluation_digest_for(
            "sha256:harness", "sha256:env-b"
        )


def test_holdout_is_absent_from_requestable_splits() -> None:
    # OF-1: holdout must not be selectable from the outer loop.
    assert "holdout" not in AVAILABLE_SPLITS
    assert set(AVAILABLE_SPLITS) == {"train_probe", "train_full"}
