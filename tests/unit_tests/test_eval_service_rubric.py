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
"""Rubric-scored benchmarks (design doc §3.3 / §5.3 scenario C, §7.2 Gym-side tests):

- profile pydantic validation under extra="forbid" (all-or-nothing rubric field set);
- manifest covers the frozen rubric library + judge env pins (HF-1);
- missing/tampered/mismatched snapshot refuses to start (RUBRIC_LIBRARY_MISMATCH, §7.2);
- judge endpoint outage is an explicit failure, never an all-zero degradation (§7.3(5));
- scenario C e2e: frozen snapshot → profile → 2 candidates → verdict with per-criterion
  evidence + per-criterion trusted/overfit signals merged into overfit_guard (P7);
- HF-2/HF-4: a library edit under the same profile produces a new eval_id and a digest warning;
- §7.3(6) version isolation: two profile versions never mix one candidate's scores.
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_service import dataset
from eval_service.contract import EvalRequest, EvalStatus
from eval_service.guards import Ledger
from eval_service.manifest import build_harness_manifest
from eval_service.profile import AdapterSpec, BenchmarkSpec, ExperimentProfile, RubricJudgePin, profile_from_yaml
from eval_service.runner import GymEvalRunner, RunOutcome
from eval_service.rubric_judge import (
    DataDesignerRubricBackend,
    RubricJudgeError,
    canonical_json,
)
from eval_service.service import WARNING_CACHED, WARNING_DIGEST_CHANGED, EvaluationService


def make_rubrics(suffix: str = "") -> list[dict]:
    return [
        {
            "rubric_key": f"r_pos1{suffix}",
            "title": "Diagnosis cited",
            "description": "The report cites the failing unit.",
            "weight": 0.5,
        },
        {
            "rubric_key": f"r_pos2{suffix}",
            "title": "Root cause",
            "description": "The report names a root cause.",
            "weight": 0.25,
        },
        {
            "rubric_key": f"r_neg{suffix}",
            "title": "No fabricated logs",
            "description": "The report must not fabricate log lines.",
            "weight": -0.25,
        },
    ]


def write_frozen_snapshot(path: Path, rubrics: list[dict]) -> str:
    """Replicates DataDesigner export_frozen_snapshot's payload so the Gym-side loader can
    integrity-check it (embedded rubric_library_sha256 + whole-file snapshot_sha256)."""
    library = [dict(rubric) for rubric in rubrics]
    digest = "sha256:" + hashlib.sha256(canonical_json(library).encode("utf-8")).hexdigest()
    payload = {
        "snapshot_schema": "rubric-library-snapshot",
        "schema_version": 1,
        "evidence_scope": "frozen_seed_only",
        "exported_step": 0,
        "rubric_library": library,
        "rubric_library_sha256": digest,
        "review": {"verdict_count": len(library), "full_review_coverage": {"missing_count": 0, "missing": []}},
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    payload["snapshot_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return digest


class RubricFakeRunner(GymEvalRunner):
    """Writes rollout rows carrying query/final_answer (no programmatic reward) instead of
    launching a real `gym eval run` subprocess."""

    def __init__(self, rows_by_query: dict[str, list[str]], repo_root: Path | None = None) -> None:
        super().__init__(repo_root=repo_root)
        self.rows_by_query = rows_by_query
        self.built_argv: list[list[str]] = []

    def execute(self, argv: list[str], timeout_s: float | None) -> RunOutcome:
        self.built_argv.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        rows = [
            {"query": query, "final_answer": answer, "ng_perf": {"total_tokens": 10}}
            for query, answers in self.rows_by_query.items()
            for answer in answers
        ]
        dataset.write_jsonl(rows, output)
        return RunOutcome(EvalStatus.COMPLETED, [], 0, 0, 1.0, 0, "")


class ScriptedJudgeBackend:
    """Deterministic rubric judge seam; scores keyed off the final answer text."""

    def __init__(self, score_fn) -> None:
        self.score_fn = score_fn
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def score_rollout(self, *, query: str, final_answer: str, descriptions: tuple[str, ...]) -> tuple[int, ...]:
        self.calls.append((query, final_answer, descriptions))
        return tuple(self.score_fn(final_answer))

    def aggregate(self, scores: tuple[int, ...], weights: tuple[float, ...]) -> float:
        denominator = sum(weight for weight in weights if weight > 0)
        return sum(score * weight for score, weight in zip(scores, weights)) / denominator


def parity_scores(final_answer: str, even: tuple[int, ...], odd: tuple[int, ...]) -> tuple[int, ...]:
    return even if int(final_answer.rsplit("-", 1)[-1]) % 2 == 0 else odd


def make_rubric_profile(
    tmp_path: Path,
    snapshot_fpath: Path | str,
    sha: str,
    *,
    profile_id: str = "rubric-profile",
    group_size: int = 8,
    num_tasks: int = 2,
) -> ExperimentProfile:
    dataset_fpath = tmp_path / "rubric_toy_train.jsonl"
    rows = [
        {"task_id": f"q{i}", "responses_create_params": {"input": [{"role": "user", "content": f"q{i}"}]}}
        for i in range(num_tasks)
    ]
    dataset.write_jsonl(rows, dataset_fpath)
    return ExperimentProfile(
        profile_id=profile_id,
        adapters={"claude_code": AdapterSpec(harness_agent="claude_code_agent", version="1")},
        benchmarks={
            "rubric-toy": BenchmarkSpec(
                dataset_fpath=str(dataset_fpath),
                dataset_version="v1",
                probe_split_size=num_tasks,
                rubric_library_fpath=str(snapshot_fpath),
                rubric_library_sha256=sha,
                rubric_judge=RubricJudgePin(name="judge-model", manifest_env=["RUBRIC_JUDGE_ENDPOINT"]),
                rubric_group_size=group_size,
            )
        },
    )


def make_rubric_service(
    profile: ExperimentProfile,
    tmp_path: Path,
    backend: ScriptedJudgeBackend,
    rows_by_query: dict[str, list[str]],
    *,
    ledger: Ledger | None = None,
    results_subdir: str = "results",
) -> tuple[EvaluationService, RubricFakeRunner]:
    runner = RubricFakeRunner(rows_by_query, repo_root=tmp_path)
    service = EvaluationService(
        profile=profile,
        runner=runner,
        ledger=ledger or Ledger(tmp_path / "ledger.json"),
        results_root=tmp_path / results_subdir,
        gym_commit="fixed",
        rubric_judge_backend=backend,
    )
    return service, runner


def rubric_req(**overrides) -> EvalRequest:
    payload = {
        "candidate_id": "cand-1",
        "adapter": "claude_code",
        "benchmark": "rubric-toy",
        "split": "train_probe",
        "tier": "probe",
        "seed": 1,
    }
    payload.update(overrides)
    return EvalRequest(**payload)


def eight_repeats(prefix: str) -> list[str]:
    return [f"{prefix}-{repeat}" for repeat in range(8)]


class TestProfileValidation:
    def test_rubric_fields_are_all_or_nothing(self) -> None:
        base = {"dataset_fpath": "toy.jsonl", "probe_split_size": 1}
        with pytest.raises(ValidationError, match="rubric_library_sha256 and rubric_judge"):
            BenchmarkSpec(**base, rubric_library_fpath="snapshot.json")
        with pytest.raises(ValidationError, match="require rubric_library_fpath"):
            BenchmarkSpec(**base, rubric_library_sha256="sha256:" + "0" * 64)
        with pytest.raises(ValidationError, match="require rubric_library_fpath"):
            BenchmarkSpec(**base, rubric_judge=RubricJudgePin(name="judge"))

    def test_pinned_sha_format_enforced(self) -> None:
        with pytest.raises(ValidationError, match="sha256:<64 hex>"):
            BenchmarkSpec(
                dataset_fpath="toy.jsonl",
                rubric_library_fpath="snapshot.json",
                rubric_library_sha256="deadbeef",
                rubric_judge=RubricJudgePin(name="judge"),
            )

    def test_rubric_profile_yaml_template_loads(self) -> None:
        profile = profile_from_yaml(Path(__file__).parents[2] / "eval_service" / "profiles" / "rsi_v1_rubric_critic.yaml")
        benchmark = profile.benchmarks["synth-npu-faultdiag-rubric"]
        assert benchmark.rubric_group_size == 8
        assert benchmark.rubric_judge.temperature == 0.0
        assert benchmark.rubric_judge.manifest_env == ["HEALTHBENCH_JUDGE_API_BASE", "HEALTHBENCH_JUDGE_API_KEY"]

    def test_unknown_fields_rejected(self, tmp_path: Path) -> None:
        profile = make_rubric_profile(tmp_path, tmp_path / "snapshot.json", "sha256:" + "0" * 64)
        with pytest.raises(ValidationError):
            ExperimentProfile(**{**profile.model_dump(), "rubric_unknown_knob": True})


class TestManifestFreeze:
    def test_snapshot_content_and_judge_env_change_the_digest(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        snapshot = tmp_path / "snapshot.json"
        sha = write_frozen_snapshot(snapshot, make_rubrics())
        profile = make_rubric_profile(tmp_path, snapshot, sha)
        request = rubric_req()
        digest_a = build_harness_manifest(profile, request, tmp_path, gym_commit="fixed")

        monkeypatch.setenv("RUBRIC_JUDGE_ENDPOINT", "http://judge-primary")
        digest_with_env = build_harness_manifest(profile, request, tmp_path, gym_commit="fixed")
        monkeypatch.setenv("RUBRIC_JUDGE_ENDPOINT", "http://judge-backup")
        digest_other_env = build_harness_manifest(profile, request, tmp_path, gym_commit="fixed")
        assert digest_with_env != digest_other_env

        sha_v2 = write_frozen_snapshot(snapshot, make_rubrics(suffix="_v2"))
        profile_v2 = make_rubric_profile(tmp_path, snapshot, sha_v2)
        digest_b = build_harness_manifest(profile_v2, request, tmp_path, gym_commit="fixed")
        assert digest_a != digest_b


class TestFailFastLibraryChecks:
    def test_missing_snapshot_rejected_before_any_rollout(self, tmp_path: Path) -> None:
        profile = make_rubric_profile(tmp_path, tmp_path / "absent.json", "sha256:" + "a" * 64)
        service, runner = make_rubric_service(profile, tmp_path, ScriptedJudgeBackend(lambda _: (1, 1, 0)), {})
        result = service.evaluate(rubric_req())
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "RUBRIC_LIBRARY_MISMATCH"
        assert "not found" in result.error.message
        assert runner.built_argv == []

    def test_tampered_snapshot_rejected(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "snapshot.json"
        sha = write_frozen_snapshot(snapshot, make_rubrics())
        snapshot.write_text(snapshot.read_text(encoding="utf-8").replace("0.5", "0.9"), encoding="utf-8")
        profile = make_rubric_profile(tmp_path, snapshot, sha)
        service, _ = make_rubric_service(profile, tmp_path, ScriptedJudgeBackend(lambda _: (1, 1, 0)), {})
        result = service.evaluate(rubric_req())
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "RUBRIC_LIBRARY_MISMATCH"

    def test_pinned_sha_mismatch_rejected(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "snapshot.json"
        sha = write_frozen_snapshot(snapshot, make_rubrics())
        other_sha = write_frozen_snapshot(tmp_path / "other.json", make_rubrics(suffix="_x"))
        profile = make_rubric_profile(tmp_path, snapshot, other_sha)
        service, runner = make_rubric_service(profile, tmp_path, ScriptedJudgeBackend(lambda _: (1, 1, 0)), {})
        result = service.evaluate(rubric_req())
        assert result.status == EvalStatus.REJECTED
        assert result.error.code == "RUBRIC_LIBRARY_MISMATCH"
        assert "pins rubric_library_sha256" in result.error.message
        assert runner.built_argv == []


class TestScenarioCEndToEnd:
    def test_two_candidates_verdict_with_per_criterion_evidence(self, tmp_path: Path) -> None:
        """§7.3(4): frozen snapshot → profile → 2 candidates → verdict with per-criterion evidence,
        manifest containing the rubric library sha256."""
        snapshot = tmp_path / "snapshot.json"
        sha = write_frozen_snapshot(snapshot, make_rubrics())
        profile = make_rubric_profile(tmp_path, snapshot, sha)
        # candidate a: positive criteria hit (r_pos1 saturated), negative never triggers
        backend_a = ScriptedJudgeBackend(
            lambda answer: parity_scores(answer, even=(1, 1, 0), odd=(1, 0, 0))
        )
        service_a, _ = make_rubric_service(
            profile, tmp_path, backend_a, {"q0": eight_repeats("a0"), "q1": eight_repeats("a1")}, results_subdir="results-a"
        )
        result_a = service_a.evaluate(rubric_req(candidate_id="cand-a"))
        assert result_a.status == EvalStatus.COMPLETED
        assert result_a.metrics.reward_mean == pytest.approx((8 * 1.0 + 8 * (0.5 / 0.75)) / 16)
        assert result_a.metrics.pass_rate == pytest.approx(0.5)

        evidence = result_a.rubric_evidence
        assert evidence.rubric_library_sha256 == sha
        assert evidence.rubric_ids == ["r_pos1", "r_pos2", "r_neg"]
        assert evidence.rubric_weights == pytest.approx([0.5, 0.25, -0.25])
        assert evidence.group_pass_counts == [16, 8, 0]
        assert evidence.judge_model == "judge-model"
        assert evidence.scored_rollouts == 16
        assert evidence.ungrouped_rollouts == 0

        # P7: per-criterion trusted/overfit signals merged into the existing overfit_guard structure
        signals = result_a.overfit_guard.criterion_signals
        assert signals["r_pos1"].trusted is False and signals["r_pos1"].reason == "constant_score_column"
        assert signals["r_pos2"].trusted is True
        assert signals["r_neg"].trusted is False and signals["r_neg"].hit_count == 0

        evidence_rows = dataset.read_jsonl(evidence.evidence_ref)
        assert len(evidence_rows) == 16
        assert evidence_rows[0]["rubric_scores"][0] == 1

        # candidate b under the same pin: distinct candidate, same harness manifest digest
        backend_b = ScriptedJudgeBackend(lambda answer: parity_scores(answer, even=(0, 1, 1), odd=(0, 0, 1)))
        service_b, _ = make_rubric_service(
            profile, tmp_path, backend_b, {"q0": eight_repeats("b0"), "q1": eight_repeats("b1")}, results_subdir="results-b"
        )
        result_b = service_b.evaluate(rubric_req(candidate_id="cand-b"))
        assert result_b.status == EvalStatus.COMPLETED
        assert result_b.metrics.reward_mean == pytest.approx((8 * 0.0 + 8 * (-0.25 / 0.75)) / 16)
        assert result_b.rubric_evidence.group_pass_counts == [0, 8, 16]
        assert result_b.harness_manifest_digest == result_a.harness_manifest_digest
        assert "sha256:" in result_b.harness_manifest_digest

    def test_incomplete_group_excluded_explicitly(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "snapshot.json"
        sha = write_frozen_snapshot(snapshot, make_rubrics())
        profile = make_rubric_profile(tmp_path, snapshot, sha)
        backend = ScriptedJudgeBackend(lambda _: (1, 1, 0))
        service, _ = make_rubric_service(
            profile, tmp_path, backend, {"q0": eight_repeats("a0"), "q1": eight_repeats("a1")[:-1]}
        )
        result = service.evaluate(rubric_req())
        assert result.status == EvalStatus.COMPLETED
        assert result.rubric_evidence.scored_rollouts == 8
        assert result.rubric_evidence.ungrouped_rollouts == 7
        assert any(warning.startswith("rubric_incomplete_group: 7") for warning in result.warnings)

    def test_no_complete_group_fails_explicitly(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "snapshot.json"
        sha = write_frozen_snapshot(snapshot, make_rubrics())
        profile = make_rubric_profile(tmp_path, snapshot, sha)
        backend = ScriptedJudgeBackend(lambda _: (1, 1, 0))
        service, _ = make_rubric_service(profile, tmp_path, backend, {"q0": eight_repeats("a0")[:3]})
        result = service.evaluate(rubric_req())
        assert result.status == EvalStatus.FAILED
        assert result.error.code == "RUBRIC_JUDGE_FAILED"
        assert "no complete rollout group" in result.error.message

    def test_judge_outage_is_explicit_failure_without_all_zero_degradation(self, tmp_path: Path) -> None:
        """§7.3(5): a judge endpoint outage fails explicitly and never writes zero rewards."""
        snapshot = tmp_path / "snapshot.json"
        sha = write_frozen_snapshot(snapshot, make_rubrics())
        profile = make_rubric_profile(tmp_path, snapshot, sha)
        ledger_fpath = tmp_path / "ledger.json"

        class DownBackend:
            def score_rollout(self, **kwargs):
                raise RubricJudgeError("judge endpoint unreachable (connection refused)")

            def aggregate(self, scores, weights):
                raise AssertionError("aggregate must not run when the judge is down")

        runner = RubricFakeRunner({"q0": eight_repeats("a0"), "q1": eight_repeats("a1")}, repo_root=tmp_path)
        service = EvaluationService(
            profile=profile,
            runner=runner,
            ledger=Ledger(ledger_fpath),
            results_root=tmp_path / "results",
            gym_commit="fixed",
            rubric_judge_backend=DownBackend(),
        )
        result = service.evaluate(rubric_req())
        assert result.status == EvalStatus.FAILED
        assert result.error.code == "RUBRIC_JUDGE_FAILED"
        assert result.metrics is None
        ledger_rows = json.loads(ledger_fpath.read_text(encoding="utf-8"))["evals"]
        assert ledger_rows[0]["reward_mean"] is None


class TestProductionBackend:
    def _install_fake_data_designer(self, monkeypatch: pytest.MonkeyPatch, transport) -> None:
        fake_rubric = types.ModuleType("data_designer.rubric")
        fake_rubric.configure_judge_endpoints = lambda **kwargs: transport
        fake_rubric.aggregate_weighted_reward = (
            lambda scores, weights: sum(score * weight for score, weight in zip(scores, weights))
            / sum(weight for weight in weights if weight > 0)
        )
        fake_package = types.ModuleType("data_designer")
        fake_package.rubric = fake_rubric
        monkeypatch.setitem(sys.modules, "data_designer", fake_package)
        monkeypatch.setitem(sys.modules, "data_designer.rubric", fake_rubric)

    def test_production_backend_translates_judge_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """§7.3(5): the real backend's outage signal (JudgeGiveUpError from the DataDesigner
        transport) must reach the service as RubricJudgeError, not leak raw."""

        class FakeGiveUpError(Exception):
            pass

        class DownTransport:
            def call_judge(self, **kwargs):
                raise FakeGiveUpError("judge total attempt budget exhausted")

        self._install_fake_data_designer(monkeypatch, DownTransport())
        backend = DataDesignerRubricBackend(model="judge-model")
        with pytest.raises(RubricJudgeError, match="judge total attempt budget exhausted"):
            backend.score_rollout(query="q", final_answer="a", descriptions=("d",))

    def test_production_backend_scores_through_data_designer_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeTransport:
            def call_judge(self, *, query, final_answer, descriptions):
                assert query and final_answer and len(descriptions) == 3
                return types.SimpleNamespace(scores=(1, 0, 0), attempts=1, prompt_sha256="x")

        self._install_fake_data_designer(monkeypatch, FakeTransport())
        backend = DataDesignerRubricBackend(model="judge-model")
        scores = backend.score_rollout(query="q", final_answer="a", descriptions=("d1", "d2", "d3"))
        assert scores == (1, 0, 0)
        assert backend.aggregate(scores, (0.5, 0.25, -0.25)) == pytest.approx(0.5 / 0.75)


class TestFreezeAndVersionIsolation:
    def test_library_edit_same_path_new_eval_id_and_digest_warning(self, tmp_path: Path) -> None:
        """HF-2/HF-4: editing the criteria library under the same profile version is a new harness
        identity — new eval_id (no cached replay) plus an explicit digest_changed warning."""
        snapshot = tmp_path / "snapshot.json"
        sha = write_frozen_snapshot(snapshot, make_rubrics())
        profile = make_rubric_profile(tmp_path, snapshot, sha)
        backend = ScriptedJudgeBackend(lambda _: (1, 1, 0))
        service, _ = make_rubric_service(profile, tmp_path, backend, {"q0": eight_repeats("a0"), "q1": eight_repeats("a1")})
        first = service.evaluate(rubric_req())

        sha_v2 = write_frozen_snapshot(snapshot, make_rubrics(suffix="_v2"))
        profile.benchmarks["rubric-toy"].rubric_library_sha256 = sha_v2
        second = service.evaluate(rubric_req())
        assert second.eval_id != first.eval_id
        assert WARNING_CACHED not in second.warnings
        assert WARNING_DIGEST_CHANGED in second.warnings
        assert second.rubric_evidence.rubric_ids == ["r_pos1_v2", "r_pos2_v2", "r_neg_v2"]
        assert second.rubric_evidence.rubric_library_sha256 == sha_v2

    def test_two_profile_versions_never_mix_one_candidate(self, tmp_path: Path) -> None:
        """§7.3(6): the same candidate under two snapshot versions lands in separate buckets —
        no merged mean, distinct eval ids."""
        snapshot_v1 = tmp_path / "snapshot_v1.json"
        snapshot_v2 = tmp_path / "snapshot_v2.json"
        sha_v1 = write_frozen_snapshot(snapshot_v1, make_rubrics())
        sha_v2 = write_frozen_snapshot(snapshot_v2, make_rubrics(suffix="_v2"))
        profile_v1 = make_rubric_profile(tmp_path, snapshot_v1, sha_v1, profile_id="rubric-profile-v1")
        profile_v2 = make_rubric_profile(tmp_path, snapshot_v2, sha_v2, profile_id="rubric-profile-v2")
        shared_ledger = Ledger(tmp_path / "ledger.json")
        backend = ScriptedJudgeBackend(lambda _: (1, 1, 0))
        service_v1, _ = make_rubric_service(
            profile_v1, tmp_path, backend, {"q0": eight_repeats("a0"), "q1": eight_repeats("a1")}, ledger=shared_ledger, results_subdir="results-v1"
        )
        service_v2, _ = make_rubric_service(
            profile_v2, tmp_path, backend, {"q0": eight_repeats("a0"), "q1": eight_repeats("a1")}, ledger=shared_ledger, results_subdir="results-v2"
        )
        result_v1 = service_v1.evaluate(rubric_req(candidate_id="cand-shared"))
        result_v2 = service_v2.evaluate(rubric_req(candidate_id="cand-shared"))
        assert result_v1.eval_id != result_v2.eval_id

        buckets = shared_ledger.bucketed_evals("cand-shared")
        assert len(buckets) == 2
        summaries = shared_ledger.bucket_summaries("cand-shared")
        assert len(summaries) == 2
        assert all(summary["evals"] == 1 for summary in summaries)
        assert all(summary["reward_mean_latest"] == pytest.approx(1.0) for summary in summaries)
