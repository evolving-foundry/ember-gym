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
import json
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_gym.identity import environment_digest, environment_snapshot
from agent_gym.layout import GymLayout
from eval_service import dataset
from eval_service.contract import EvalRequest, EvalResult, EvalStatus, PromotionRequest
from eval_service.guards import Ledger, score_trend_summary
from eval_service.manifest import build_harness_manifest
from eval_service.profile import AdapterSpec, BenchmarkSpec, ConstraintBudgets, ExperimentProfile, profile_from_yaml
from eval_service.promotion import PromotionPipeline, _candidate_key, compare_promotion_reports
from eval_service.queue import BackgroundEvaluationRunner
from eval_service.runner import (
    GymEvalRunner,
    RunOutcome,
    _resolve_gym_bin,
    parse_rollout_items,
    parse_rollout_metrics,
)
from eval_service.service import (
    WARNING_CACHED,
    WARNING_ENVIRONMENT_CHANGED,
    WARNING_SCHEMA_CHANGED,
    EvaluationService,
)


class FakeRunner(GymEvalRunner):
    """Writes Gym-shaped output JSONL instead of launching a real `gym eval run` subprocess."""

    def __init__(self, behavior: str = "ok", repo_root: Path | None = None) -> None:
        super().__init__(repo_root=repo_root)
        self.behavior = behavior
        self.built_argv: list[list[str]] = []

    def execute(self, argv: list[str], timeout_s: float | None) -> RunOutcome:
        self.built_argv.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        materialized = output.with_name(output.stem + "_materialized_inputs.jsonl")
        if self.behavior == "limited":
            # Gym applied --limit: only the first two input rows were materialized/executed.
            dataset.write_jsonl([{"_ng_task_index": 0}, {"_ng_task_index": 1}], materialized)
            dataset.write_jsonl([{"_ng_task_index": 0, "reward": 1.0}, {"_ng_task_index": 1, "reward": 0.0}], output)
            return RunOutcome(EvalStatus.COMPLETED, [], 0, 0, 12.5, 0, "")
        if self.behavior == "with_missing":
            # All four rows were materialized; one produced a reward, one a failure, two nothing.
            dataset.write_jsonl([{"_ng_task_index": index} for index in range(4)], materialized)
            dataset.write_jsonl([{"_ng_task_index": 0, "reward": 1.0}], output)
            sidecar = output.with_name(output.stem + "_failures.jsonl")
            dataset.write_jsonl([{"_ng_task_index": 1, "_ng_failure_class": "agent_run_error"}], sidecar)
            return RunOutcome(EvalStatus.COMPLETED, [], 0, 0, 12.5, 0, "")
        if self.behavior in ("ok", "with_failures", "with_items"):
            rows = (
                [
                    {
                        "_ng_task_index": 0,
                        "reward": 1.0,
                        "ng_perf": {"total_tokens": 100},
                        "extracted_answer": "17",
                    },
                    {
                        "_ng_task_index": 1,
                        "reward": 0.0,
                        "extracted_answer": "17000",
                        "failure_kind": None,
                        "failure_reason": None,
                    },
                    {
                        "_ng_task_index": 2,
                        "reward": 0.0,
                        "extracted_answer": "100",
                        "failure_kind": None,
                        "failure_reason": None,
                    },
                    {"_ng_task_index": 3, "reward": 1.0, "extracted_answer": "3"},
                ]
                if self.behavior == "with_items"
                else [
                    {"_ng_task_index": 0, "reward": 1.0, "ng_perf": {"total_tokens": 100}},
                    {"_ng_task_index": 1, "reward": 1.0},
                    {"_ng_task_index": 2, "reward": 0.5, "ng_perf": {"total_tokens": 50}},
                    {"_ng_task_index": 3, "reward": 0.0},
                ]
            )
            dataset.write_jsonl(rows, output)
            if self.behavior == "with_failures":
                sidecar = output.with_name(output.stem + "_failures.jsonl")
                dataset.write_jsonl(
                    [
                        {"_ng_task_index": 0, "_ng_failure_class": "agent_run_error"},
                        {"_ng_task_index": 1, "_ng_failure_class": "agent_run_error"},
                    ],
                    sidecar,
                )
            return RunOutcome(EvalStatus.COMPLETED, [], 0, 0, 12.5, 0, "")
        if self.behavior == "partial":
            dataset.write_jsonl([{"reward": 1.0}], output)
            return RunOutcome(EvalStatus.BUDGET_EXHAUSTED, [], 0, 0, 30.0, None, "timed out")
        return RunOutcome(EvalStatus.FAILED, [], 0, 0, 1.0, 1, "boom: harness exploded")


def test_custom_gym_bin_path_is_not_rewritten(tmp_path: Path) -> None:
    custom = tmp_path / "bin" / "gym"

    assert _resolve_gym_bin(str(custom)) == str(custom)


def test_bare_gym_alias_resolves_to_agent_gym_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("eval_service.runner.shutil.which", lambda _: None)

    assert Path(_resolve_gym_bin("gym")).name == "agent-gym"


def make_profile(tmp_path: Path, with_holdout: bool = False, **budget_overrides) -> ExperimentProfile:
    dataset_fpath = tmp_path / "toy_train.jsonl"
    rows = [
        {
            "task_id": f"toy-{i}",
            "responses_create_params": {"input": [{"role": "user", "content": f"q{i}"}]},
            "verifier_metadata": {},
        }
        for i in range(10)
    ]
    dataset.write_jsonl(rows, dataset_fpath)
    holdout_fpath = None
    if with_holdout:
        holdout_fpath = tmp_path / "toy_holdout.jsonl"
        holdout_rows = [
            {
                "task_id": f"hold-{i}",
                "responses_create_params": {"input": [{"role": "user", "content": f"h{i}"}]},
                "verifier_metadata": {},
            }
            for i in range(3)
        ]
        dataset.write_jsonl(holdout_rows, holdout_fpath)
    return ExperimentProfile(
        profile_id="profile-test",
        adapters={
            "claude_code": AdapterSpec(harness_agent="claude_code_agent", version="1"),
            "codex": AdapterSpec(harness_agent="codex_agent", version="1", config_paths=["nonexistent/codex.yaml"]),
        },
        benchmarks={
            "toy": BenchmarkSpec(
                dataset_fpath=str(dataset_fpath),
                holdout_dataset_fpath=str(holdout_fpath) if holdout_fpath else None,
                dataset_version="v1",
                probe_split_size=4,
            )
        },
        budgets=ConstraintBudgets(**budget_overrides),
    )


def make_service(
    profile: ExperimentProfile, tmp_path: Path, behavior: str = "ok", max_in_flight: int = 1
) -> tuple[EvaluationService, FakeRunner]:
    runner = FakeRunner(behavior=behavior, repo_root=tmp_path)
    service = EvaluationService(
        profile=profile,
        runner=runner,
        ledger=Ledger(tmp_path / "ledger.json"),
        results_root=tmp_path / "results",
        gym_commit="fixed-commit",
        max_in_flight=max_in_flight,
    )
    return service, runner


def req(**overrides) -> EvalRequest:
    payload = {
        "candidate_id": "cand-1",
        "adapter": "claude_code",
        "benchmark": "toy",
        "split": "train_probe",
        "tier": "probe",
        "seed": 1,
    }
    payload.update(overrides)
    return EvalRequest(**payload)


def holdout_req(**overrides) -> EvalRequest:
    """Promotion-path request: holdout is full-tier only (F-01)."""
    payload = {"split": "holdout", "tier": "full"}
    payload.update(overrides)
    return req(**payload)


def make_identity_layout(
    root: Path,
    *,
    framework_version: str = "0.7.0",
    wheel_sha256: str = "a" * 64,
    component_tree_sha256: str = "b" * 64,
) -> GymLayout:
    overlay = root / "overlay"
    overlay.mkdir(parents=True, exist_ok=True)
    (root / "framework").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "overlay_version": "2026.11.09-r1",
        "framework": {
            "distribution": "nemo-gym",
            "version": framework_version,
            "wheel_sha256": wheel_sha256,
            "package_tree_sha256": "c" * 64,
        },
        "components": {
            "benchmarks/toy": {
                "kind": "component",
                "complete": True,
                "tree_sha256": component_tree_sha256,
            }
        },
    }
    (overlay / "overlay-manifest.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    return GymLayout(
        repo_root=root,
        overlay_root=overlay,
        workspace_root=root,
        data_root=root / "data",
        extra_roots=(),
        framework_root=root / "framework",
    )


class TestConcurrencyCap:
    def test_max_in_flight_sets_semaphore_capacity(self, tmp_path: Path) -> None:
        """§11.3: the deployment knob stays a constructor arg (never a profile field)."""
        service, _ = make_service(make_profile(tmp_path), tmp_path, max_in_flight=2)
        assert service._slots._initial_value == 2


class TestValidation:
    def test_unknown_benchmark_rejected(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        result = service.evaluate(req(benchmark="nope"))
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "BENCHMARK_UNKNOWN"

    def test_unregistered_adapter_rejected(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        result = service.evaluate(req(adapter="dsh"))
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "ADAPTER_NOT_REGISTERED"
        assert result.error.fix_hint

    def test_holdout_locked(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        result = service.evaluate(holdout_req())
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "HOLDOUT_LOCKED"

    def test_invalid_split_rejected(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        assert service.evaluate(req(split="validation")).error.code == "INVALID_SPLIT"

    def test_tier_split_mismatch_rejected(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        assert service.evaluate(req(split="train_full")).error.code == "TIER_SPLIT_MISMATCH"

    def test_manifest_mismatch_rejected(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        result = service.evaluate(req(harness_manifest="sha256:" + "0" * 64))
        assert result.error.code == "MANIFEST_MISMATCH"

    def test_unregistered_dataset_version_is_rejected(self, tmp_path: Path) -> None:
        service, fake = make_service(make_profile(tmp_path), tmp_path)
        result = service.evaluate(req(dataset_version="attacker-choice"))
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "INVALID_DATASET_VERSION"
        assert "v1" in result.error.fix_hint
        assert fake.built_argv == []

    def test_registered_dataset_version_is_accepted(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        result = service.evaluate(req(dataset_version="v1"))
        assert result.status == EvalStatus.COMPLETED
        assert result.dataset_version == "v1"

    def test_holdout_rejects_unregistered_dataset_version(self, tmp_path: Path) -> None:
        service, fake = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        result = service.evaluate_holdout(holdout_req(dataset_version="v2"))
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "INVALID_DATASET_VERSION"
        assert fake.built_argv == []

    def test_split_exhaustion_requires_acknowledgement(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path, split_eval_soft_limit=2), tmp_path)
        assert service.evaluate(req(seed=1)).status == EvalStatus.COMPLETED
        assert service.evaluate(req(seed=2)).status == EvalStatus.COMPLETED
        third = service.evaluate(req(seed=3))
        assert third.status == EvalStatus.REJECTED
        assert third.error.code == "SPLIT_EXHAUSTED"
        forced = service.evaluate(req(seed=3, acknowledge_exhausted_split=True))
        assert forced.status == EvalStatus.COMPLETED
        assert forced.overfit_guard.trusted is False
        assert forced.overfit_guard.split_eval_count == 3

    def test_lineage_budget_requires_acknowledgement(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path, lineage_max_evals=1), tmp_path)
        assert service.evaluate(req()).status == EvalStatus.COMPLETED
        blocked = service.evaluate(req(seed=2))
        assert blocked.status == EvalStatus.REJECTED
        assert blocked.error.code == "LINEAGE_BUDGET_EXCEEDED"
        assert service.evaluate(req(seed=2, acknowledge_lineage_budget=True)).status == EvalStatus.COMPLETED


class TestExecution:
    def test_successful_eval_metrics_and_persistence(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        result = service.evaluate(req())

        assert result.status == EvalStatus.COMPLETED
        assert result.metrics.reward_mean == pytest.approx(0.625)
        assert result.metrics.pass_rate == pytest.approx(0.5)
        assert result.metrics.rollouts == 4
        assert result.metrics.rollouts_failed == 0
        assert result.cost.total_tokens == 150
        assert result.cost.lineage_cost_to_date.evals == 1
        assert result.harness_manifest_digest.startswith("sha256:")
        assert result.environment_digest.startswith("sha256:")
        assert result.evaluation_digest.startswith("sha256:")
        assert result.manifest_schema_version == 2
        assert result.experiment_profile_id == "profile-test"
        assert result.dataset_version == "v1"
        assert result.overfit_guard.split_eval_count == 1
        assert result.overfit_guard.trusted is True
        assert (tmp_path / "results" / result.eval_id / "rollouts.jsonl").is_file()
        assert (tmp_path / "results" / result.eval_id / "repro.json").is_file()
        repro = json.loads((tmp_path / "results" / result.eval_id / "repro.json").read_text(encoding="utf-8"))
        assert repro["seed"] == 1 and repro["gym_commit"] == "fixed-commit"
        assert repro["manifest_schema_version"] == 2
        assert repro["environment_digest"] == result.environment_digest
        assert repro["evaluation_digest"] == result.evaluation_digest
        assert repro["environment_identity"]["schema_version"] == 2

    def test_idempotent_replay_hits_cache(self, tmp_path: Path) -> None:
        service, runner = make_service(make_profile(tmp_path), tmp_path)
        first = service.evaluate(req())
        replay = service.evaluate(req())
        assert replay.eval_id == first.eval_id
        assert WARNING_CACHED in replay.warnings
        assert len(runner.built_argv) == 1  # second request never spawned a rollout run

    def test_candidate_digest_separates_cache_and_is_persisted(self, tmp_path: Path) -> None:
        service, runner = make_service(make_profile(tmp_path), tmp_path)
        first_digest = "sha256:" + "a" * 64
        second_digest = "sha256:" + "b" * 64

        first = service.evaluate(req(candidate_digest=first_digest))
        replay = service.evaluate(req(candidate_digest=first_digest))
        changed = service.evaluate(req(candidate_digest=second_digest))

        assert first.candidate_digest == first_digest
        assert replay.eval_id == first.eval_id
        assert WARNING_CACHED in replay.warnings
        assert changed.eval_id != first.eval_id
        assert changed.candidate_digest == second_digest
        assert len(runner.built_argv) == 2
        persisted = json.loads((tmp_path / "results" / first.eval_id / "result.json").read_text(encoding="utf-8"))
        assert persisted["candidate_digest"] == first_digest
        repro = json.loads((tmp_path / "results" / changed.eval_id / "repro.json").read_text(encoding="utf-8"))
        assert repro["candidate_digest"] == second_digest

    def test_completed_eval_replays_after_split_budget_is_exhausted(self, tmp_path: Path) -> None:
        service, runner = make_service(make_profile(tmp_path, split_eval_soft_limit=1), tmp_path)
        first = service.evaluate(req(seed=1))
        assert first.status == EvalStatus.COMPLETED

        replay = service.evaluate(req(seed=1))
        assert replay.status == EvalStatus.COMPLETED
        assert WARNING_CACHED in replay.warnings
        assert len(runner.built_argv) == 1

    def test_completed_eval_replays_after_lineage_budget_is_exhausted(self, tmp_path: Path) -> None:
        service, runner = make_service(make_profile(tmp_path, lineage_max_evals=1), tmp_path)
        first = service.evaluate(req(seed=1))
        assert first.status == EvalStatus.COMPLETED

        replay = service.evaluate(req(seed=1))
        assert replay.status == EvalStatus.COMPLETED
        assert WARNING_CACHED in replay.warnings
        assert len(runner.built_argv) == 1

    def test_failed_run_reports_error_not_fake_success(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path, behavior="fail")
        result = service.evaluate(req())
        assert result.status == EvalStatus.FAILED
        assert result.error.code == "EVAL_FAILED"
        assert "boom" in result.error.fix_hint
        assert result.metrics is None

    def test_budget_exhausted_returns_partial_metrics(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path, behavior="partial")
        result = service.evaluate(req(budget={"timeout_s": 30}))
        assert result.status == EvalStatus.BUDGET_EXHAUSTED
        assert result.metrics.rollouts == 1
        assert result.metrics.reward_mean == 1.0

    def test_failure_sidecar_counted_as_zero_in_denominator(self, tmp_path: Path) -> None:
        """F-03: two infrastructure failures out of six attempts score zero, not excluded."""
        service, _ = make_service(make_profile(tmp_path), tmp_path, behavior="with_failures")
        result = service.evaluate(req())
        assert result.metrics.rollouts == 4
        assert result.metrics.rollouts_failed == 2
        assert result.metrics.rollouts_attempted == 6
        assert result.metrics.reward_mean == pytest.approx(2.5 / 6)
        assert result.metrics.pass_rate == pytest.approx(2 / 6)

    def test_train_items_expose_bounded_failure_evidence(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path, behavior="with_items")
        result = service.evaluate(req())

        assert result.status == EvalStatus.COMPLETED
        assert result.per_rollout_ref is not None
        assert len(result.items) == 4
        failed = next(item for item in result.items if item.reward_mean == 0.0)
        assert failed.rollouts == 1
        assert failed.reward_mean == 0.0
        assert failed.pass_rate == 0.0
        assert any(check.status == "failed" and check.id.startswith("answer-match/") for check in failed.checks)
        assert any("expected normalized answer" in diagnostic.message for diagnostic in failed.diagnostics)
        assert any("17000" in diagnostic.message for diagnostic in failed.diagnostics)

    def test_holdout_never_exposes_task_level_evidence(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path, with_holdout=True), tmp_path, behavior="with_items")
        result = service.evaluate_holdout(holdout_req())

        assert result.status == EvalStatus.COMPLETED
        assert result.metrics.rollouts == 4
        assert result.items == []

    def test_unattempted_rows_do_not_produce_failure_items(self, tmp_path: Path) -> None:
        """F-02: with --limit 2, the two unmaterialized probe rows must not appear as failures."""
        service, _ = make_service(make_profile(tmp_path), tmp_path, behavior="limited")
        result = service.evaluate(req(budget={"max_rollouts": 2}))

        assert result.status == EvalStatus.COMPLETED
        assert result.metrics.rollouts == 2
        assert len(result.items) == 2
        assert all(check.id != "rollout" for item in result.items for check in item.checks), (
            "an unattempted row must not be reported as a missing rollout"
        )

    def test_attempted_rows_distinguish_missing_from_failed(self, tmp_path: Path) -> None:
        """F-02/F-03: silent attempted rows get a missing check and a zero-score denominator."""
        service, _ = make_service(make_profile(tmp_path), tmp_path, behavior="with_missing")
        result = service.evaluate(req())

        assert result.metrics.rollouts == 1
        assert result.metrics.rollouts_failed == 1
        # Gym materialized four rollouts and count_missing_rollouts_as_zero scores the two
        # silent ones as zeros, so the run denominator is 4 -- not 2 (completed + failed).
        assert result.metrics.rollouts_attempted == 4
        assert result.metrics.reward_mean == pytest.approx(1.0 / 4)
        assert result.metrics.pass_rate == pytest.approx(1.0 / 4)
        items = result.items
        assert len(items) == 4  # every materialized task is attempted
        assert not any(check.status == "failed" for check in items[0].checks)
        assert any(check.id == "execution" for check in items[1].checks)
        # toy-2/toy-3 were materialized but produced no artifact: genuinely missing, scored zero.
        assert any(check.id == "rollout" for check in items[2].checks)
        assert items[2].reward_mean == 0.0 and items[2].pass_rate == 0.0
        assert any(check.id == "rollout" for check in items[3].checks)
        assert items[3].reward_mean == 0.0 and items[3].pass_rate == 0.0

    def test_materialized_repeats_shrink_task_denominator(self, tmp_path: Path) -> None:
        """F-03: a lost repeat scores zero instead of disappearing from the task average."""
        profile = make_profile(tmp_path)
        benchmark = profile.benchmarks["toy"]
        input_fpath = tmp_path / "repeat_input.jsonl"
        output = tmp_path / "repeat_rollouts.jsonl"
        dataset.write_jsonl([{"task_id": "toy-0"}], input_fpath)
        dataset.write_jsonl(
            [{"_ng_task_index": 0, "_ng_rollout_index": 0}, {"_ng_task_index": 0, "_ng_rollout_index": 1}],
            output.with_name(output.stem + "_materialized_inputs.jsonl"),
        )
        dataset.write_jsonl([{"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0}], output)

        items = parse_rollout_items(input_fpath, output, benchmark)

        assert len(items) == 1
        assert items[0].rollouts == 1
        assert items[0].reward_mean == pytest.approx(0.5)
        assert items[0].pass_rate == pytest.approx(0.5)

    def test_failed_task_item_metrics_count_failures_as_zero(self, tmp_path: Path) -> None:
        """F-03 at task level: one pass plus one failure averages to 0.5, not 1.0."""
        service, _ = make_service(make_profile(tmp_path), tmp_path, behavior="with_failures")
        result = service.evaluate(req())

        first = result.items[0]
        assert first.rollouts == 1 and first.rollouts_failed == 1
        assert first.reward_mean == pytest.approx(0.5)
        assert first.pass_rate == pytest.approx(0.5)

    def test_queue_when_in_flight_cap_reached(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path, max_in_flight=1)
        service._slots.acquire()
        try:
            result = service.evaluate(req())
            assert result.status == EvalStatus.QUEUED
            assert result.harness_manifest_digest.startswith("sha256:")
        finally:
            service._slots.release()


class TestArgvAssembly:
    def test_probe_tier_applies_preset_limit(self, tmp_path: Path) -> None:
        service, runner = make_service(make_profile(tmp_path), tmp_path)
        service.evaluate(req())
        argv = runner.built_argv[0]
        assert Path(argv[0]).stem == "agent-gym"
        assert argv[1:4] == ["eval", "run", "--no-serve"]
        assert "--agent" in argv and argv[argv.index("--agent") + 1] == "claude_code_agent"
        assert argv[argv.index("--limit") + 1] == "32"
        assert argv[argv.index("--concurrency") + 1] == "8"
        assert "--num-repeats" not in argv

    def test_budget_cap_intersects_with_tier_preset(self, tmp_path: Path) -> None:
        service, runner = make_service(make_profile(tmp_path), tmp_path)
        service.evaluate(req(budget={"max_rollouts": 8}))
        assert runner.built_argv[0][runner.built_argv[0].index("--limit") + 1] == "8"

    def test_full_tier_repeats_without_fake_seed_flag_and_config_paths(self, tmp_path: Path) -> None:
        service, runner = make_service(make_profile(tmp_path), tmp_path)
        service.evaluate(req(adapter="codex", split="train_full", tier="full"))
        argv = runner.built_argv[0]
        assert argv[argv.index("--num-repeats") + 1] == "4"
        assert "--seed" not in argv
        assert "+num_repeats_add_seed=true" not in argv
        resolved_config = Path(argv[argv.index("--config") + 1])
        assert resolved_config.is_absolute()
        assert resolved_config.name == "codex.yaml"

    def test_candidate_config_has_precedence_over_environment_config(self, tmp_path: Path) -> None:
        runner = GymEvalRunner(repo_root=tmp_path)
        argv = runner.build_argv(
            AdapterSpec(harness_agent="rsi_candidate", config_paths=["candidate.yaml"]),
            None,
            make_profile(tmp_path).tier_preset(req().tier),
            tmp_path / "input.jsonl",
            tmp_path / "output.jsonl",
            1,
            benchmark_config_paths=["environment.yaml"],
        )
        configs = [Path(argv[i + 1]) for i, value in enumerate(argv) if value == "--config"]
        assert [path.name for path in configs] == ["environment.yaml", "candidate.yaml"]
        assert all(path.is_absolute() for path in configs)


class TestDatasetSplits:
    def test_probe_split_is_deterministic_and_sized(self, tmp_path: Path) -> None:
        profile = make_profile(tmp_path)
        rows_a = dataset.build_split_rows(profile.benchmarks["toy"], "train_probe")
        rows_b = dataset.build_split_rows(profile.benchmarks["toy"], "train_probe")
        assert rows_a == rows_b and len(rows_a) == 4

    def test_full_split_returns_every_row(self, tmp_path: Path) -> None:
        profile = make_profile(tmp_path)
        assert len(dataset.build_split_rows(profile.benchmarks["toy"], "train_full")) == 10

    def test_request_seed_changes_run_identity_but_not_materialized_rows(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        first = service.evaluate(req(seed=11))
        second = service.evaluate(req(seed=22))
        first_rows = dataset.read_jsonl(tmp_path / "results" / first.eval_id / "input.jsonl")
        second_rows = dataset.read_jsonl(tmp_path / "results" / second.eval_id / "input.jsonl")
        assert first.eval_id != second.eval_id
        assert first_rows == second_rows


class TestGuards:
    def test_score_trend_summary(self) -> None:
        assert score_trend_summary([0.1, 0.2, 0.3]) == "monotonic_up_3"
        assert score_trend_summary([0.9, 0.5]) == "monotonic_down_2"
        assert score_trend_summary([0.5, 0.9, 0.5]) == "mixed"
        assert score_trend_summary([0.4]) == "insufficient_history"

    def test_ledger_roundtrip(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.json")
        ledger.record("eval-1", "cand-1", "toy", "train_probe", "v1", "sha256:a", 0.5, 100)
        reloaded = Ledger(tmp_path / "ledger.json")
        assert reloaded.lineage_cost("cand-1") == {"evals": 1, "total_tokens": 100}
        assert reloaded.overfit_stats("cand-1", "toy", "train_probe", "v1")["eval_count"] == 1


def test_parse_rollout_metrics_with_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    dataset.write_jsonl([{"reward": 1.0, "ng_perf": {"total_tokens": 10}}, {"reward": 0.25}], output)
    dataset.write_jsonl([{"_ng_failure_class": "x"}], tmp_path / "rollouts_failures.jsonl")
    rewards, failed, tokens = parse_rollout_metrics(output)
    assert rewards == [1.0, 0.25]
    assert failed == 1
    assert tokens == 10


class TestHoldoutPromotion:
    def test_outer_loop_entry_rejects_holdout_while_pipeline_accepts_it(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        assert service.evaluate(holdout_req()).error.code == "HOLDOUT_LOCKED"
        assert service.evaluate_holdout(holdout_req()).status == EvalStatus.COMPLETED

    def test_promotion_pipeline_requires_explicit_holdout_split(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        pipeline = PromotionPipeline(service)
        result, report = pipeline.run(req(split="train_probe"))
        assert result.error.code == "INVALID_SPLIT"
        assert report is None
        assert list(service.results_root.rglob("promotion")) == []

    def test_promotion_report_written_with_metrics(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        pipeline = PromotionPipeline(service)
        result, report_fpath = pipeline.run(holdout_req())
        assert result.status == EvalStatus.COMPLETED
        assert report_fpath is not None and report_fpath.is_file()
        report = json.loads(report_fpath.read_text(encoding="utf-8"))
        assert report["candidate_id"] == "cand-1"
        assert report["metrics"]["rollouts"] == 4
        assert report["manifest_schema_version"] == 2
        assert report["harness_manifest_digest"].startswith("sha256:")
        assert report["environment_digest"] == result.environment_digest
        assert report["evaluation_digest"] == result.evaluation_digest
        assert pipeline.reports("cand-1")[0]["eval_id"] == result.eval_id
        assert pipeline.reports("other-candidate") == []

    def test_holdout_without_registered_dataset_is_rejected(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path, with_holdout=False), tmp_path)
        result = service.evaluate_holdout(holdout_req())
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "INVALID_SPLIT"
        assert "no holdout dataset" in result.error.message

    def test_promotion_rejects_capped_tier(self, tmp_path: Path) -> None:
        """F-01: promotion is full-tier only, so a probe/screen limit can never truncate holdout."""
        service, _ = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        for tier in ("probe", "screen"):
            result = service.validate_holdout(req(split="holdout", tier=tier))
            assert result is not None
            assert result.error.code == "PROMOTION_TIER_INVALID"

    def test_promotion_request_model_rejects_non_full_tier(self) -> None:
        with pytest.raises(ValidationError):
            PromotionRequest(
                candidate_id="cand-1",
                candidate_digest="sha256:" + "a" * 64,
                adapter="claude_code",
                benchmark="toy",
                tier="probe",
                seed=1,
            )

    def test_holdout_argv_never_contains_a_limit(self, tmp_path: Path) -> None:
        """F-01 acceptance: the complete holdout runs, even when a budget cap is supplied."""
        service, runner = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        result = service.evaluate_holdout(holdout_req(budget={"timeout_s": 60, "max_rollouts": 1}))

        assert result.status == EvalStatus.COMPLETED
        assert "--limit" not in runner.built_argv[0]

    def test_empty_holdout_dataset_is_rejected(self, tmp_path: Path) -> None:
        profile = make_profile(tmp_path, with_holdout=True)
        (tmp_path / "toy_holdout.jsonl").write_text("", encoding="utf-8")
        service, runner = make_service(profile, tmp_path)
        result = service.evaluate_holdout(holdout_req())
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "INVALID_SPLIT"
        assert "empty" in result.error.message
        assert runner.built_argv == []

    def test_holdout_task_ids_are_rejected(self, tmp_path: Path) -> None:
        """F-01 defense in depth: a pinned subset is not a holdout verdict."""
        service, runner = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        result = service.evaluate_holdout(holdout_req(task_ids=["hold-0"]))
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "PROMOTION_TIER_INVALID"
        assert "task_ids" in result.error.message
        assert runner.built_argv == []


class TestBucketedAggregation:
    def test_buckets_never_merge_across_manifest_digests(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.json")
        ledger.record("eval-1", "cand-1", "toy", "train_probe", "v1", "sha256:a", 0.5, 10, experiment_profile_id="p1")
        ledger.record("eval-2", "cand-1", "toy", "train_probe", "v1", "sha256:b", 0.9, 10, experiment_profile_id="p2")
        summaries = ledger.bucket_summaries("cand-1")
        assert len(summaries) == 2  # two isolated buckets — no cross-digest mean exists
        assert {s["reward_mean_latest"] for s in summaries} == {0.5, 0.9}
        assert all("sha256:" in s["bucket"] or True for s in summaries)

    def test_bucket_key_includes_split_and_profile(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.json")
        ledger.record("eval-1", "cand-1", "toy", "train_probe", "v1", "sha256:a", 0.5, 10, experiment_profile_id="p1")
        ledger.record("eval-2", "cand-1", "toy", "train_full", "v1", "sha256:a", 0.5, 10, experiment_profile_id="p1")
        ledger.record("eval-3", "cand-1", "toy", "train_probe", "v1", "sha256:a", 0.5, 10, experiment_profile_id="p2")
        assert len(ledger.bucket_summaries("cand-1")) == 3

    def test_bucket_key_includes_candidate_digest(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.json")
        ledger.record(
            "eval-1",
            "cand-1",
            "toy",
            "train_probe",
            "v1",
            "sha256:a",
            0.5,
            10,
            candidate_digest="sha256:" + "a" * 64,
        )
        ledger.record(
            "eval-2",
            "cand-1",
            "toy",
            "train_probe",
            "v1",
            "sha256:a",
            0.9,
            10,
            candidate_digest="sha256:" + "b" * 64,
        )
        summaries = ledger.bucket_summaries("cand-1")
        assert len(summaries) == 2
        assert {summary["candidate_digest"] for summary in summaries} == {
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
        }


class TestEnvironmentIdentity:
    def test_framework_version_changes_environment_digest(self, tmp_path: Path) -> None:
        layout = make_identity_layout(tmp_path / "env")
        before = environment_digest(environment_snapshot(layout, repo_commit="fixed-commit"))

        make_identity_layout(tmp_path / "env", framework_version="0.7.1")
        after = environment_digest(environment_snapshot(layout, repo_commit="fixed-commit"))

        assert before != after

    def test_installed_framework_version_changes_digest_without_manifest_change(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        layout = make_identity_layout(tmp_path / "env")
        before = environment_digest(environment_snapshot(layout, repo_commit="fixed-commit"))

        monkeypatch.setattr(
            "agent_gym.identity.framework_identity",
            lambda _layout: {"distribution": "nemo-gym", "version": "0.7.1"},
        )
        after = environment_digest(environment_snapshot(layout, repo_commit="fixed-commit"))

        assert before != after

    def test_overlay_manifest_changes_digest_without_version_bump(self, tmp_path: Path) -> None:
        layout = make_identity_layout(tmp_path / "env")
        before = environment_digest(environment_snapshot(layout, repo_commit="fixed-commit"))

        make_identity_layout(tmp_path / "env", component_tree_sha256="d" * 64)
        after = environment_digest(environment_snapshot(layout, repo_commit="fixed-commit"))

        assert before != after

    def test_environment_digest_is_portable_across_layout_paths(self, tmp_path: Path) -> None:
        first_layout = make_identity_layout(tmp_path / "host-a")
        second_layout = make_identity_layout(tmp_path / "host-b")

        first = environment_snapshot(first_layout, repo_commit="fixed-commit")
        second = environment_snapshot(second_layout, repo_commit="fixed-commit")

        assert environment_digest(first) == environment_digest(second)
        assert environment_digest(first) == environment_digest({**first, "verified": False})

    def test_harness_change_does_not_change_environment_digest(self, tmp_path: Path) -> None:
        profile = make_profile(tmp_path)
        layout = make_identity_layout(tmp_path / "env")
        before = environment_digest(environment_snapshot(layout, repo_commit="fixed-commit"))

        changed_profile = profile.model_copy(deep=True)
        changed_profile.adapters["claude_code"].version = "2"
        first_harness = build_harness_manifest(profile, req(), tmp_path, gym_commit="fixed-commit", layout=layout)
        second_harness = build_harness_manifest(
            changed_profile, req(), tmp_path, gym_commit="fixed-commit", layout=layout
        )
        after = environment_digest(environment_snapshot(layout, repo_commit="fixed-commit"))

        assert first_harness != second_harness
        assert before == after


class TestEvaluationIdentityIsolation:
    def test_environment_change_does_not_replay_cached_result(self, tmp_path: Path) -> None:
        service, runner = make_service(make_profile(tmp_path), tmp_path)
        first = service.evaluate(req())

        service.environment_digest = "sha256:" + "d" * 64
        second = service.evaluate(req())

        assert second.eval_id != first.eval_id
        assert WARNING_CACHED not in second.warnings
        assert len(runner.built_argv) == 2

    def test_environment_and_schema_changes_are_warned(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        service.evaluate(req())

        service.environment_digest = "sha256:" + "e" * 64
        changed_environment = service.evaluate(req(seed=2))
        assert WARNING_ENVIRONMENT_CHANGED in changed_environment.warnings

        service.manifest_schema_version = 1
        changed_schema = service.evaluate(req(seed=3))
        assert WARNING_SCHEMA_CHANGED in changed_schema.warnings

    def test_overfit_stats_are_isolated_by_evaluation_digest(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.json")
        ledger.record(
            "eval-1",
            "cand-1",
            "toy",
            "train_probe",
            "v1",
            "sha256:harness",
            0.1,
            10,
            environment_digest="sha256:env-a",
            evaluation_digest="sha256:eval-a",
            manifest_schema_version=2,
        )
        ledger.record(
            "eval-2",
            "cand-1",
            "toy",
            "train_probe",
            "v1",
            "sha256:harness",
            0.9,
            10,
            environment_digest="sha256:env-b",
            evaluation_digest="sha256:eval-b",
            manifest_schema_version=2,
        )

        assert ledger.overfit_stats("cand-1", "toy", "train_probe", "v1", "sha256:eval-a") == {
            "eval_count": 1,
            "score_trend": "insufficient_history",
        }
        assert ledger.overfit_stats("cand-1", "toy", "train_probe", "v1")["eval_count"] == 2

    def test_schema_one_and_two_never_share_a_bucket(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "ledger.json")
        ledger.record("eval-legacy", "cand-1", "toy", "train_probe", "v1", "sha256:harness", 0.4, 10)
        ledger.record(
            "eval-v2",
            "cand-1",
            "toy",
            "train_probe",
            "v1",
            "sha256:harness",
            0.8,
            10,
            environment_digest="sha256:env",
            evaluation_digest="sha256:evaluation",
            manifest_schema_version=2,
        )

        summaries = ledger.bucket_summaries("cand-1")
        assert len(summaries) == 2
        assert {summary["manifest_schema_version"] for summary in summaries} == {1, 2}
        assert ledger.candidate_environment_digests("cand-1") == {"sha256:env"}
        assert ledger.candidate_schema_versions("cand-1") == {1, 2}

    def test_runtime_snapshot_is_persisted(self, tmp_path: Path) -> None:
        snapshot_path = tmp_path / "results" / "runtime_snapshot.json"
        service = EvaluationService(
            profile=make_profile(tmp_path),
            runner=FakeRunner(repo_root=tmp_path),
            ledger=Ledger(tmp_path / "ledger.json"),
            results_root=tmp_path / "results",
            gym_commit="fixed-commit",
            environment_snapshot_path=snapshot_path,
        )
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert snapshot["environment_digest"] == service.environment_digest
        assert snapshot["manifest_schema_version"] == 2
        assert snapshot["environment_identity"]["schema_version"] == 2


class TestBackgroundQueue:
    def test_submit_never_blocks_and_results_are_collectable(self, tmp_path: Path) -> None:
        service, fake = make_service(make_profile(tmp_path), tmp_path)
        bg = BackgroundEvaluationRunner(service)
        try:
            submissions = [bg.submit(req(seed=seed)) for seed in (1, 2, 3)]
            assert all(s.status == EvalStatus.QUEUED for s in submissions)
            results = {}
            for _ in range(200):
                results = {s.eval_id: bg.result(s.eval_id) for s in submissions}
                if all(r is not None for r in results.values()):
                    break
                time.sleep(0.02)
            assert all(r is not None and r.status == EvalStatus.COMPLETED for r in results.values())
            assert len(fake.built_argv) == 3  # each request executed exactly once
            assert bg.pending() == 0
            replay = bg.submit(req(seed=1))
            assert replay.eval_id == submissions[0].eval_id
            assert replay.status == EvalStatus.COMPLETED  # duplicate submit of finished eval
        finally:
            bg.shutdown()

    def test_pending_counts_unique_accepted_evals(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        started = threading.Event()
        release = threading.Event()
        original = service.evaluate

        def blocking(request: EvalRequest) -> EvalResult:
            started.set()
            assert release.wait(5)
            return original(request)

        service.evaluate = blocking
        bg = BackgroundEvaluationRunner(service)
        try:
            first = bg.submit(req(seed=1))
            second = bg.submit(req(seed=2))
            assert started.wait(5)
            # One running + one queued must count as two, not three.
            assert bg.pending() == 2
            release.set()
            for _ in range(200):
                results = {item.eval_id: bg.result(item.eval_id) for item in (first, second)}
                if all(result is not None for result in results.values()):
                    break
                time.sleep(0.02)
            assert all(result is not None for result in results.values())
        finally:
            release.set()
            bg.shutdown()

    def test_duplicate_submit_during_queued_retry_executes_once(self, tmp_path: Path) -> None:
        service, fake = make_service(make_profile(tmp_path), tmp_path)
        queued_once = threading.Event()
        second_call = threading.Event()
        release = threading.Event()
        original = service.evaluate
        calls = 0

        def controlled(request: EvalRequest) -> EvalResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                queued_once.set()
                eval_id = service.eval_id_of(request)
                assert eval_id is not None
                return EvalResult(eval_id=eval_id, status=EvalStatus.QUEUED)
            second_call.set()
            assert release.wait(5)
            return original(request)

        service.evaluate = controlled
        bg = BackgroundEvaluationRunner(service)
        try:
            first = bg.submit(req(seed=1))
            assert queued_once.wait(5)
            assert second_call.wait(5)
            duplicate = bg.submit(req(seed=1))
            assert duplicate.eval_id == first.eval_id
            assert bg.pending() == 1
            release.set()
            result = None
            for _ in range(200):
                result = bg.result(first.eval_id)
                if result is not None:
                    break
                time.sleep(0.02)
            assert result is not None and result.status == EvalStatus.COMPLETED
            assert len(fake.built_argv) == 1
        finally:
            release.set()
            bg.shutdown()

    def test_unresolvable_request_is_queued_then_rejected(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path), tmp_path)
        bg = BackgroundEvaluationRunner(service)
        try:
            submission = bg.submit(req(benchmark="nope"))
            assert submission.status == EvalStatus.QUEUED
            result = None
            for _ in range(200):
                result = bg.result(submission.eval_id)
                if result is not None:
                    break
                time.sleep(0.02)
            assert result is not None and result.status == EvalStatus.REJECTED
        finally:
            bg.shutdown()


class TestQueueJournal:
    def test_pending_queue_survives_runner_restart_without_reexecution(self, tmp_path: Path) -> None:
        """§11.2: a crashed runner's pending queue is restored from the journal; an eval whose
        result.json already exists settles by replay, never by re-running the harness."""
        first_service, _ = make_service(make_profile(tmp_path), tmp_path)
        first = first_service.evaluate(req(seed=1))
        assert first.status == EvalStatus.COMPLETED

        journal = tmp_path / "queue.json"
        restart_service, fake = make_service(make_profile(tmp_path), tmp_path)
        second_id = restart_service.eval_id_of(req(seed=2))
        journal.write_text(
            json.dumps(
                {
                    "pending": [
                        {"eval_id": first.eval_id, "request": req(seed=1).model_dump(mode="json")},
                        {"eval_id": second_id, "request": req(seed=2).model_dump(mode="json")},
                    ]
                }
            )
        )
        bg = BackgroundEvaluationRunner(restart_service, journal_path=journal)
        try:
            for _ in range(200):
                if json.loads(journal.read_text())["pending"] == []:
                    break
                time.sleep(0.02)
            outcomes = {}
            for _ in range(200):
                outcomes = {eval_id: bg.result(eval_id) for eval_id in (first.eval_id, second_id)}
                if all(r is not None for r in outcomes.values()):
                    break
                time.sleep(0.02)
            assert all(r is not None and r.status == EvalStatus.COMPLETED for r in outcomes.values())
            # seed=1 replayed from its result.json; only seed=2 actually executed.
            assert len(fake.built_argv) == 1
            assert bg.pending() == 0
            assert json.loads(journal.read_text())["pending"] == []
        finally:
            bg.shutdown()

    def test_corrupt_journal_starts_empty_but_service_still_works(self, tmp_path: Path) -> None:
        service, fake = make_service(make_profile(tmp_path), tmp_path)
        journal = tmp_path / "queue.json"
        journal.write_text("{not json")
        bg = BackgroundEvaluationRunner(service, journal_path=journal)
        try:
            submission = bg.submit(req(seed=1))
            result = None
            for _ in range(200):
                result = bg.result(submission.eval_id)
                if result is not None:
                    break
                time.sleep(0.02)
            assert result is not None and result.status == EvalStatus.COMPLETED
            assert len(fake.built_argv) == 1
        finally:
            bg.shutdown()

    def test_journal_rekeys_entries_when_environment_identity_changes(self, tmp_path: Path) -> None:
        old_service, _ = make_service(make_profile(tmp_path), tmp_path)
        old_eval_id = old_service.eval_id_of(req())

        restart_service, fake = make_service(make_profile(tmp_path), tmp_path)
        restart_service.environment_digest = "sha256:" + "9" * 64
        new_eval_id = restart_service.eval_id_of(req())
        assert new_eval_id != old_eval_id

        journal = tmp_path / "queue.json"
        journal.write_text(
            json.dumps({"pending": [{"eval_id": old_eval_id, "request": req().model_dump(mode="json")}]}),
            encoding="utf-8",
        )
        bg = BackgroundEvaluationRunner(restart_service, journal_path=journal)
        try:
            result = None
            for _ in range(200):
                result = bg.result(new_eval_id)
                if result is not None and json.loads(journal.read_text(encoding="utf-8"))["pending"] == []:
                    break
                time.sleep(0.02)
            assert result is not None and result.status == EvalStatus.COMPLETED
            assert result.environment_digest == restart_service.environment_digest
            assert bg.result(old_eval_id) is None
            assert bg.pending() == 0
            assert json.loads(journal.read_text(encoding="utf-8"))["pending"] == []
            assert len(fake.built_argv) == 1
        finally:
            bg.shutdown()


class TestRestartRecovery:
    def test_service_restart_recovers_completed_eval_by_eval_id(self, tmp_path: Path) -> None:
        """§7 鲁棒: a restarted service must resume from disk state, never re-run or fake-succeed."""
        profile = make_profile(tmp_path)
        first_service, runner = make_service(profile, tmp_path)
        first = first_service.evaluate(req())
        assert first.status == EvalStatus.COMPLETED

        restarted_service, restarted_runner = make_service(profile, tmp_path)
        replay = restarted_service.evaluate(req())
        assert replay.eval_id == first.eval_id
        assert replay.status == EvalStatus.COMPLETED
        assert replay.metrics.reward_mean == pytest.approx(first.metrics.reward_mean)
        assert WARNING_CACHED in replay.warnings
        assert restarted_runner.built_argv == []  # no re-execution after restart
        # Lineage accounting survived the restart too.
        assert replay.cost.lineage_cost_to_date.evals == 1


class TestPromotionFeedback:
    def test_latest_verdict_and_feedback_summary(self, tmp_path: Path) -> None:
        """§11.6: the holdout report flows back to the outer loop as a verdict + ledger buckets."""
        service, _ = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        pipeline = PromotionPipeline(service, reports_root=tmp_path / "reports")
        assert pipeline.latest_verdict("cand-1") is None  # nothing reported yet

        candidate_digest = "sha256:" + "c" * 64
        result, report_fpath = pipeline.run(holdout_req(candidate_digest=candidate_digest))
        assert result.status == EvalStatus.COMPLETED and report_fpath is not None
        assert result.candidate_digest == candidate_digest
        assert json.loads(report_fpath.read_text(encoding="utf-8"))["candidate_digest"] == candidate_digest

        verdict = pipeline.latest_verdict("cand-1")
        assert verdict["holdout_status"] == "completed"
        assert verdict["eligible_for_release"] is True
        assert verdict["decision_scope"] == "holdout_only"
        assert verdict["reward_mean"] == pytest.approx(result.metrics.reward_mean)

        summary = pipeline.feedback_summary("cand-1")
        assert summary["candidate_id"] == "cand-1"
        assert summary["holdout"] == verdict
        assert isinstance(summary["buckets"], list)
        # read-only: the feedback pass created no new report files
        assert len(pipeline.reports("cand-1")) == 1

    @pytest.mark.parametrize("candidate_id", ["../escaped", "nested/candidate"])
    def test_candidate_id_never_escapes_reports_root(self, tmp_path: Path, candidate_id: str) -> None:
        service, _ = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        pipeline = PromotionPipeline(service, reports_root=tmp_path / "reports")
        result, report_fpath = pipeline.run(holdout_req(candidate_id=candidate_id))

        assert result.status == EvalStatus.COMPLETED
        assert report_fpath is not None and report_fpath.is_file()
        assert report_fpath.parent == pipeline.reports_root
        assert [report["candidate_id"] for report in pipeline.reports(candidate_id)] == [candidate_id]

    def test_reports_treat_glob_characters_as_literal_candidate_ids(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        pipeline = PromotionPipeline(service, reports_root=tmp_path / "reports")
        pipeline.run(holdout_req(candidate_id="cand-1"))
        pipeline.run(holdout_req(candidate_id="*"))

        assert [report["candidate_id"] for report in pipeline.reports("*")] == ["*"]

    def test_budget_exhausted_report_yields_exhausted_verdict(self, tmp_path: Path) -> None:
        service, _ = make_service(make_profile(tmp_path, with_holdout=True), tmp_path, behavior="partial")
        pipeline = PromotionPipeline(service, reports_root=tmp_path / "reports")
        result, report_fpath = pipeline.run(holdout_req())
        assert result.status == EvalStatus.BUDGET_EXHAUSTED
        assert report_fpath is not None
        verdict = pipeline.latest_verdict("cand-1")
        assert verdict["holdout_status"] == "budget_exhausted"
        assert verdict["eligible_for_release"] is False
        assert verdict["reward_mean"] == pytest.approx(1.0)

    def test_legacy_report_naming_stays_visible(self, tmp_path: Path) -> None:
        """F-08: reports written under the pre-migration name are still discovered/readable."""
        service, _ = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        pipeline = PromotionPipeline(service, reports_root=tmp_path / "reports")
        pipeline.reports_root.mkdir(parents=True)
        legacy = pipeline.reports_root / f"{_candidate_key('cand-1')}.json"
        legacy.write_text(
            json.dumps(
                {
                    "promotion_id": "promo-" + "0" * 16,
                    "candidate_id": "cand-1",
                    "candidate_digest": "sha256:" + "d" * 64,
                    "benchmark": "toy",
                    "split": "holdout",
                    "status": "completed",
                    "metrics": {"reward_mean": 0.75, "pass_rate": 0.75, "rollouts": 4, "rollouts_failed": 0},
                }
            ),
            encoding="utf-8",
        )

        assert [report["candidate_id"] for report in pipeline.reports("cand-1")] == ["cand-1"]
        verdict = pipeline.latest_verdict("cand-1")
        assert verdict is not None and verdict["reward_mean"] == pytest.approx(0.75)


class TestPromotionComparison:
    @staticmethod
    def report(
        digest_char: str,
        *,
        pass_rate: float,
        rollouts: int,
        rollouts_failed: int = 0,
        reward_mean: float | None = None,
        rollouts_attempted: int | None = None,
        **overrides,
    ) -> dict:
        metrics = {
            "reward_mean": pass_rate if reward_mean is None else reward_mean,
            "pass_rate": pass_rate,
            "rollouts": rollouts,
            "rollouts_failed": rollouts_failed,
        }
        if rollouts_attempted is not None:
            metrics["rollouts_attempted"] = rollouts_attempted
        report = {
            "status": "completed",
            "split": "holdout",
            "candidate_digest": "sha256:" + digest_char * 64,
            "benchmark": "toy",
            "adapter": "claude_code",
            "tier": "full",
            "dataset_version": "v1",
            "manifest_schema_version": 2,
            "harness_manifest_digest": "sha256:" + "h" * 64,
            "environment_digest": "sha256:" + "e" * 64,
            "evaluation_digest": "sha256:" + "v" * 64,
            "metrics": metrics,
        }
        report.update(overrides)
        return report

    def test_compare_uses_effective_pass_rate_not_reward_mean(self) -> None:
        # Legacy reports (no rollouts_attempted): raw pass_rate over successful rollouts needs the
        # failure scaling, so 0.8 over 4 successes with 1 failure becomes 0.64.
        candidate = self.report("a", pass_rate=0.8, reward_mean=0.8, rollouts=4, rollouts_failed=1)
        incumbent = self.report("b", pass_rate=0.5, reward_mean=0.5, rollouts=4)
        decision = compare_promotion_reports(candidate, incumbent)
        assert decision["decision"] == "promote"
        assert decision["candidate_pass_rate"] == pytest.approx(0.64)
        assert decision["incumbent_pass_rate"] == pytest.approx(0.5)
        assert decision["delta"] == pytest.approx(0.14)

    def test_effective_pass_rate_is_not_scaled_twice(self) -> None:
        # New reports already divide by attempts; scaling again would wrongly cut 0.8 to 0.64.
        candidate = self.report("a", pass_rate=0.8, rollouts=4, rollouts_failed=1, rollouts_attempted=5)
        incumbent = self.report("b", pass_rate=0.5, rollouts=4, rollouts_attempted=4)
        decision = compare_promotion_reports(candidate, incumbent)
        assert decision["decision"] == "promote"
        assert decision["candidate_pass_rate"] == pytest.approx(0.8)

    def test_all_failed_candidate_is_not_a_promotion(self) -> None:
        candidate = self.report("a", pass_rate=0.0, reward_mean=0.0, rollouts=0, rollouts_failed=4)
        incumbent = self.report("b", pass_rate=0.5, rollouts=4)
        assert compare_promotion_reports(candidate, incumbent)["decision"] == "reject"

    def test_tie_keeps_incumbent(self) -> None:
        candidate = self.report("a", pass_rate=0.5, rollouts=4)
        incumbent = self.report("b", pass_rate=0.5, rollouts=4)
        decision = compare_promotion_reports(candidate, incumbent)
        assert decision["decision"] == "reject"
        assert decision["delta"] == pytest.approx(0.0)

    def test_strict_improvement_promotes_only_with_min_delta(self) -> None:
        candidate = self.report("a", pass_rate=0.55, rollouts=4)
        incumbent = self.report("b", pass_rate=0.5, rollouts=4)
        assert compare_promotion_reports(candidate, incumbent, min_delta=0.01)["decision"] == "promote"
        relaxed = compare_promotion_reports(candidate, incumbent, min_delta=0.1)
        assert relaxed["decision"] == "reject"

    def test_cross_environment_reports_are_incomparable(self) -> None:
        candidate = self.report("a", pass_rate=0.9, rollouts=4)
        incumbent = self.report("b", pass_rate=0.5, rollouts=4, environment_digest="sha256:" + "f" * 64)
        decision = compare_promotion_reports(candidate, incumbent)
        assert decision["decision"] == "reject"
        assert any("environment_digest" in reason for reason in decision["reasons"])

    def test_missing_identity_field_is_incomparable(self) -> None:
        candidate = self.report("a", pass_rate=0.9, rollouts=4, dataset_version=None)
        incumbent = self.report("b", pass_rate=0.5, rollouts=4)
        decision = compare_promotion_reports(candidate, incumbent)
        assert decision["decision"] == "reject"
        assert any("dataset_version" in reason for reason in decision["reasons"])


def test_holdout_requires_activated_bundle_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "activation.json"
    profile = make_profile(tmp_path, with_holdout=True).model_copy(
        update={"activation_manifest_fpath": str(manifest)}
    )
    service, _ = make_service(profile, tmp_path)
    digest = "sha256:" + "a" * 64

    missing = service.evaluate_holdout(holdout_req(candidate_digest=digest))
    assert missing.status == EvalStatus.REJECTED
    assert missing.error is not None and missing.error.code == "ACTIVATION_MISMATCH"

    manifest.write_text(
        json.dumps({"candidate_id": "cand-1", "candidate_digest": digest}),
        encoding="utf-8",
    )
    matching = service.evaluate_holdout(holdout_req(candidate_digest=digest))
    assert matching.status == EvalStatus.COMPLETED

    manifest.write_text(
        json.dumps({"candidate_id": "cand-2", "candidate_digest": digest}),
        encoding="utf-8",
    )
    wrong_id = service.evaluate_holdout(holdout_req(candidate_digest=digest))
    assert wrong_id.status == EvalStatus.REJECTED
    assert wrong_id.error is not None and wrong_id.error.code == "ACTIVATION_MISMATCH"

    manifest.write_text(
        json.dumps({"candidate_id": "cand-1", "candidate_digest": digest}),
        encoding="utf-8",
    )
    wrong = "sha256:" + "b" * 64
    mismatch = service.evaluate_holdout(holdout_req(candidate_digest=wrong))
    assert mismatch.status == EvalStatus.REJECTED
    assert mismatch.error is not None and mismatch.error.code == "ACTIVATION_MISMATCH"


class TestRsiIterationE2e:
    def test_probe_screen_holdout_iteration(self, tmp_path: Path) -> None:
        """§6 Phase 3 acceptance, approximated end-to-end: coarse probe screen → full-tier
        escalation → locked holdout acceptance report, with the search signal never seeing holdout."""
        service, _ = make_service(make_profile(tmp_path, with_holdout=True), tmp_path)
        pipeline = PromotionPipeline(service)

        # Coarse screen on the probe split, then escalate.
        probe = service.evaluate(req(split="train_probe", tier="probe", seed=1))
        assert probe.status == EvalStatus.COMPLETED
        assert probe.overfit_guard.trusted is True
        full = service.evaluate(req(split="train_full", tier="full", seed=1))
        assert full.status == EvalStatus.COMPLETED
        assert full.experiment_profile_id == "profile-test"

        # Holdout acceptance: locked to the outer entry, reported by the pipeline only.
        assert service.evaluate(holdout_req()).error.code == "HOLDOUT_LOCKED"
        holdout, report_fpath = pipeline.run(holdout_req())
        assert holdout.status == EvalStatus.COMPLETED
        assert holdout.overfit_guard.trusted is True
        report = json.loads(report_fpath.read_text(encoding="utf-8"))
        assert report["candidate_id"] == "cand-1"
        assert report["harness_manifest_digest"].startswith("sha256:")
        # HF-1: the manifest pins the tier preset, so probe and full evaluate under different
        # digests. Holdout is full-tier by construction (F-01), so it shares the full-tier
        # harness digest; the split still separates them in eval_id.
        assert probe.harness_manifest_digest != full.harness_manifest_digest
        assert holdout.harness_manifest_digest == full.harness_manifest_digest
        assert holdout.eval_id != full.eval_id
        assert report["dataset_version"] == "v1"


def test_example_profile_yaml_loads() -> None:
    example = Path(__file__).parents[2] / "eval_service" / "profiles" / "rsi_v0_example.yaml"
    profile = profile_from_yaml(example)
    assert set(profile.adapters) == {"claude_code", "codex", "hermes", "opencode", "dsh"}
    assert profile.adapters["claude_code"].harness_agent == "claude_code_agent"
    assert profile.adapters["dsh"].harness_agent == "dsh_agent"
    assert profile.budgets.split_eval_soft_limit == 10


def test_benchmark_cannot_override_rsi_candidate_agent() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="harness_agent"):
        BenchmarkSpec.model_validate({"dataset_fpath": "tasks.jsonl", "harness_agent": "reference_agent"})


def test_profile_can_register_new_agent_adapter(tmp_path: Path) -> None:
    profile = make_profile(tmp_path)
    profile.adapters["custom_agent"] = AdapterSpec(harness_agent="custom_gym_agent", version="1")
    service, runner = make_service(profile, tmp_path)
    result = service.evaluate(req(adapter="custom_agent"))
    assert result.status == EvalStatus.COMPLETED
    assert runner.built_argv[0][runner.built_argv[0].index("--agent") + 1] == "custom_gym_agent"


def test_tau_rsi_profile_routes_both_environments_to_candidate_endpoint(tmp_path: Path) -> None:
    import yaml

    root = Path(__file__).parents[2]
    profile = profile_from_yaml(root / "eval_service" / "profiles" / "rsi_v1_tau2.yaml")
    adapter = profile.adapters["remote"]
    runner = GymEvalRunner(repo_root=root)
    candidate_config = "responses_api_models/openai_model/configs/rsi_agent_chat_endpoint.yaml"
    candidate_path = str(runner.layout.resolve(candidate_config, must_exist=False))
    for name in ("tau2", "tau3_banking_bm25_grep"):
        benchmark = profile.benchmarks[name]
        argv = runner.build_argv(
            adapter,
            profile.model,
            profile.tier_preset(req().tier),
            tmp_path / "input.jsonl",
            tmp_path / "output.jsonl",
            1,
            benchmark_config_paths=benchmark.config_paths,
        )
        assert argv[argv.index("--agent") + 1] == "rsi_tau_agent"
        assert candidate_path in argv
        config_path = runner.layout.resolve(benchmark.config_paths[0])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        tau = config["rsi_tau_agent"]["responses_api_agents"]["tau2"]
        assert tau["model_server"]["name"] == "rsi_agent_endpoint"
        assert tau["user_model_server"]["name"] == "rsi_tau_user_model"
        simulator = config["rsi_tau_user_model"]["responses_api_models"]["openai_model"]
        assert simulator["openai_model"] == "${oc.env:TAU_USER_MODEL_NAME}"
        assert simulator["openai_base_url"] == "${oc.env:TAU_USER_BASE_URL}"
        assert "TAU_USER_THINKING_TYPE" in simulator["extra_body"]["thinking"]["type"]
        assert "OPENAI_API_KEY" not in str(config)


def test_native_agent_profiles_route_policy_to_candidate_endpoint(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    expected = {
        "rsi_v1_assistantbench.yaml",
        "rsi_v1_osworld.yaml",
        "rsi_v1_swebench_lite.yaml",
        "rsi_v1_swebench_verified.yaml",
        "rsi_v1_terminal_bench_2_1.yaml",
        "rsi_v1_webarena_verified.yaml",
        "rsi_v1_workarena.yaml",
    }
    candidate_config = "responses_api_models/openai_model/configs/rsi_policy_endpoint.yaml"
    policy_env = ["RSI_AGENT_BASE_URL", "RSI_AGENT_MODEL"]
    for name in expected:
        profile = profile_from_yaml(root / "eval_service" / "profiles" / name)
        for adapter in profile.adapters.values():
            if candidate_config not in adapter.config_paths:
                # DSH-only adapters invoke the materialized candidate directly and
                # therefore do not compose the native policy endpoint.
                assert "RSI_DSH_COMMAND" in adapter.manifest_env
                continue
            assert set(policy_env) <= set(adapter.manifest_env)
            assert adapter.config_paths[-1] in {
                candidate_config,
                "agent_gym/configs/sandbox/docker.yaml",
            }
            for benchmark in profile.benchmarks.values():
                runner = GymEvalRunner(repo_root=root)
                argv = runner.build_argv(
                    adapter,
                    profile.model,
                    profile.tier_preset(req().tier),
                    tmp_path / "input.jsonl",
                    tmp_path / "output.jsonl",
                    1,
                    benchmark_config_paths=benchmark.config_paths,
                )
                assert argv[argv.index("--agent") + 1] == adapter.harness_agent
                candidate_path = str(runner.layout.resolve(candidate_config, must_exist=False))
                benchmark_path = str(runner.layout.resolve(benchmark.config_paths[-1], must_exist=False))
                assert argv.index(candidate_path) > argv.index(benchmark_path)


def test_manifest_tracks_simulator_runtime_pin_without_storing_value(tmp_path: Path, monkeypatch) -> None:
    profile = make_profile(tmp_path)
    profile.benchmarks["toy"].manifest_env = ["TAU_USER_MODEL_NAME"]
    monkeypatch.setenv("TAU_USER_MODEL_NAME", "simulator-a")
    first = build_harness_manifest(profile, req(), tmp_path, gym_commit="fixed")
    monkeypatch.setenv("TAU_USER_MODEL_NAME", "simulator-b")
    second = build_harness_manifest(profile, req(), tmp_path, gym_commit="fixed")
    assert first != second
    assert "simulator-a" not in first
    assert "simulator-b" not in second
