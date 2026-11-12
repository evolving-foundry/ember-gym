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
"""Evaluation service orchestrator: the single entry point the RSI outer loop calls.

Constraint pipeline order is fixed by spec §4.4: harness manifest (HF) → split legality (OF) →
budgets (EC) → execute. Every rejection is fail-fast with a structured error (code, message,
fix_hint); no partial execution is ever produced.
"""

import json
import os
import threading
from pathlib import Path
from typing import Optional

from agent_gym.identity import environment_digest, environment_snapshot
from agent_gym.layout import GymLayout
from eval_service import dataset
from eval_service.contract import (
    AVAILABLE_SPLITS,
    EVALUATION_MANIFEST_SCHEMA_VERSION,
    HOLDOUT_SPLIT,
    BudgetSpec,
    CriterionSignal,
    ErrorCode,
    EvalCost,
    EvalError,
    EvalMetrics,
    EvalRequest,
    EvalResult,
    EvalStatus,
    LineageCost,
    OverfitGuard,
    RubricEvidence,
    TierName,
    eval_id_for,
    evaluation_digest_for,
    rejection,
    unresolved_eval_id,
)
from eval_service.guards import Ledger
from eval_service.manifest import build_harness_manifest
from eval_service.profile import BenchmarkSpec, ExperimentProfile
from eval_service.runner import GymEvalRunner, materialized_rollout_counts, parse_rollout_items, parse_rollout_metrics
from eval_service.rubric_judge import (
    DataDesignerRubricBackend,
    RubricJudgeBackend,
    RubricJudgeError,
    RubricLibraryError,
    RubricVerdict,
    judge_rubric_rollouts,
    load_rubric_library,
    rubric_library_path,
)


WARNING_SPLIT_EXHAUSTED = "split_eval_soft_limit_exceeded; results marked trusted=false (OF-4)"
WARNING_DIGEST_CHANGED = "digest_changed: candidate has prior evals under a different harness manifest (HF-4)"
WARNING_ENVIRONMENT_CHANGED = "environment_changed: candidate has prior evals under a different runtime environment"
WARNING_SCHEMA_CHANGED = "manifest_schema_changed: candidate has prior evals under a different identity schema"
WARNING_CACHED = "cached_result: idempotent replay of a previously completed evaluation"
WARNING_RUBRIC_INCOMPLETE_GROUP = (
    "rubric_incomplete_group: {count} rollout(s) excluded from rubric scoring — incomplete n-group (§5.3)"
)


class EvaluationService:
    """Single entry point for rollout evaluations (spec §5).

    `max_in_flight` caps concurrent harness executions (a BoundedSemaphore; requests that
    arrive when the cap is reached get QUEUED and retried by the background runner). It is a
    deployment-level knob — an EvaluationService constructor argument, deliberately NOT part
    of the experiment profile, which describes what to measure rather than how the service
    scales. Tune it per host (default 1 = serial, the safest setting on shared machines).
    """

    def __init__(
        self,
        profile: ExperimentProfile,
        runner: GymEvalRunner,
        ledger: Ledger,
        results_root: Path,
        repo_root: Optional[Path] = None,
        gym_commit: str = "unknown",
        max_in_flight: int = 1,
        *,
        layout: Optional[GymLayout] = None,
        rubric_judge_backend: Optional[RubricJudgeBackend] = None,
        environment_snapshot_path: Optional[Path] = None,
    ) -> None:
        self.profile = profile
        self.runner = runner
        self.ledger = ledger
        self.results_root = Path(results_root)
        self.layout = layout or getattr(runner, "layout", None) or GymLayout.discover(repo_root=repo_root)
        self.repo_root = repo_root or getattr(runner, "repo_root", None) or self.layout.repo_root
        self.gym_commit = gym_commit
        snapshot_repo_commit = gym_commit if gym_commit not in ("", "unknown") else None
        self.activation_manifest_fpath = self._resolve_activation_manifest(profile)
        self.environment_identity = environment_snapshot(self.layout, repo_commit=snapshot_repo_commit)
        self.environment_digest = environment_digest(self.environment_identity)
        self.manifest_schema_version = EVALUATION_MANIFEST_SCHEMA_VERSION
        if environment_snapshot_path is not None:
            self._write_environment_snapshot(Path(environment_snapshot_path))
        self._slots = threading.BoundedSemaphore(max_in_flight)
        # §3.3: judge seam for rubric-scored benchmarks. None = build the production
        # DataDesigner backend lazily from the profile's rubric_judge pin on first use.
        self.rubric_judge_backend = rubric_judge_backend

    def _resolve_activation_manifest(self, profile: ExperimentProfile) -> Optional[Path]:
        configured = profile.activation_manifest_fpath
        if not configured:
            return None
        expanded = os.path.expandvars(os.path.expanduser(configured))
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = self.layout.resolve(candidate, must_exist=False)
        return candidate.resolve(strict=False)

    def _activation_identity_error(self, request: EvalRequest) -> Optional[str]:
        """P1: fail closed when the activated bundle is not the requested candidate."""
        path = self.activation_manifest_fpath
        if path is None:
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return f"activation manifest is missing: {path}"
        except (OSError, json.JSONDecodeError) as exc:
            return f"activation manifest is unreadable: {path}: {exc}"
        if not isinstance(record, dict):
            return f"activation manifest is not a JSON object: {path}"
        if record.get("candidate_digest") != request.candidate_digest:
            return (
                "activated bundle digest does not match candidate_digest: "
                f"manifest={record.get('candidate_digest')!r}, request={request.candidate_digest!r}"
            )
        if record.get("candidate_id") != request.candidate_id:
            return (
                "activated bundle candidate_id does not match request: "
                f"manifest={record.get('candidate_id')!r}, request={request.candidate_id!r}"
            )
        return None

    def _write_environment_snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "manifest_schema_version": self.manifest_schema_version,
            "environment_digest": self.environment_digest,
            "environment_identity": self.environment_identity,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- validation (fail-fast, ordered per §4.4) ----

    def _resolve_dataset_version(self, request: EvalRequest) -> str:
        benchmark = self.profile.benchmarks[request.benchmark]
        return request.dataset_version if request.dataset_version != "auto" else benchmark.dataset_version

    def _dataset_version_rejection(self, request: EvalRequest, digest: str) -> Optional[EvalResult]:
        benchmark = self.profile.benchmarks[request.benchmark]
        if request.dataset_version in ("auto", benchmark.dataset_version):
            return None
        return rejection(
            self._eval_id(request, digest),
            ErrorCode.INVALID_DATASET_VERSION,
            f"dataset_version '{request.dataset_version}' is not registered for benchmark '{request.benchmark}'",
            f"use 'auto' or the profile-pinned version '{benchmark.dataset_version}'",
        )

    def _manifest_digest(self, request: EvalRequest) -> str:
        return build_harness_manifest(
            self.profile,
            request,
            self.repo_root,
            gym_commit=self.gym_commit,
            layout=self.layout,
        )

    def _evaluation_digest(self, harness_manifest_digest: str) -> str:
        return evaluation_digest_for(
            harness_manifest_digest,
            self.environment_digest,
            schema_version=self.manifest_schema_version,
        )

    def evaluation_digest_for_request(self, request: EvalRequest) -> str:
        """Return the harness+runtime identity for a request without executing it."""
        return self._evaluation_digest(self._manifest_digest(request))

    def _unresolved_eval_id(self, request: EvalRequest) -> str:
        return unresolved_eval_id(
            request,
            environment_digest=self.environment_digest,
            schema_version=self.manifest_schema_version,
        )

    def _eval_id(self, request: EvalRequest, digest: str) -> str:
        return eval_id_for(
            request.candidate_id,
            request.benchmark,
            request.split,
            request.tier.value,
            request.seed,
            digest,
            task_ids=request.task_ids,
            candidate_digest=request.candidate_digest,
            environment_digest=self.environment_digest,
            schema_version=self.manifest_schema_version,
        )

    def _validate(self, request: EvalRequest) -> Optional[EvalResult]:
        if request.benchmark not in self.profile.benchmarks:
            return rejection(
                self._unresolved_eval_id(request),
                ErrorCode.BENCHMARK_UNKNOWN,
                f"benchmark '{request.benchmark}' is not registered in experiment profile '{self.profile.profile_id}'",
                "pick a registered benchmark or register it in the profile",
            )
        if request.adapter not in self.profile.adapters:
            return rejection(
                self._unresolved_eval_id(request),
                ErrorCode.ADAPTER_NOT_REGISTERED,
                f"adapter '{request.adapter}' is not registered in experiment profile '{self.profile.profile_id}'",
                "register the adapter (one profile YAML entry) or choose a registered one",
            )

        # HF: manifest identity check — no per-request harness overrides exist by construction.
        digest = self._manifest_digest(request)
        invalid_dataset_version = self._dataset_version_rejection(request, digest)
        if invalid_dataset_version is not None:
            return invalid_dataset_version
        if request.harness_manifest is not None and request.harness_manifest != digest:
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.MANIFEST_MISMATCH,
                "requested harness manifest does not match the manifest computed from the current profile pin",
                "omit harness_manifest to evaluate under the current pin, or run under the profile that owns it",
            )

        # §7.2: a rubric-scored benchmark refuses to start when its frozen library snapshot is
        # missing, tampered with, or does not match the sha256 pinned in the profile (fail-fast,
        # before any rollout runs).
        benchmark = self.profile.benchmarks[request.benchmark]
        if benchmark.rubric_library_fpath is not None:
            try:
                _, library_sha256 = load_rubric_library(rubric_library_path(benchmark, self.layout))
            except RubricLibraryError as error:
                return rejection(
                    self._eval_id(request, digest),
                    ErrorCode.RUBRIC_LIBRARY_MISMATCH,
                    f"rubric library snapshot failed validation: {error}",
                    "export a fresh frozen snapshot (DataDesigner rubric snapshot export) and pin its rubric_library_sha256 in the profile",
                )
            if library_sha256 != benchmark.rubric_library_sha256:
                return rejection(
                    self._eval_id(request, digest),
                    ErrorCode.RUBRIC_LIBRARY_MISMATCH,
                    (
                        f"profile pins rubric_library_sha256 {benchmark.rubric_library_sha256} but the frozen "
                        f"snapshot carries {library_sha256}"
                    ),
                    "update the profile pin to the snapshot's rubric_library_sha256, or export a new snapshot version",
                )

        # OF: split legality — holdout is never requestable from the outer loop (OF-1/OF-2).
        if request.split == HOLDOUT_SPLIT:
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.HOLDOUT_LOCKED,
                "holdout split is locked and cannot be requested; it is reserved for the promotion pipeline",
                "evaluate on train_probe or train_full",
            )
        if request.split not in AVAILABLE_SPLITS:
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.INVALID_SPLIT,
                f"split '{request.split}' is not one of {list(AVAILABLE_SPLITS)}",
                "use train_probe for coarse screening or train_full for the full training split",
            )

        # EC: tier/split escalation ladder.
        tier = self.profile.tier_preset(request.tier)
        if request.split not in tier.allowed_splits:
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.TIER_SPLIT_MISMATCH,
                f"tier '{request.tier.value}' does not allow split '{request.split}' (allowed: {tier.allowed_splits})",
                "screen with probe/train_probe first, then escalate (EC-1)",
            )

        # RSI split contract: a harness-pinned task subset must be a non-empty, duplicate-free
        # subset of the requested split. Fail-fast here — the background drain loop has no
        # exception boundary, so a bad list reaching _execute would silently stall the worker.
        if request.task_ids is not None:
            if not request.task_ids:
                return rejection(
                    self._eval_id(request, digest),
                    ErrorCode.TASK_SET_INVALID,
                    "task_ids must be a non-empty list when provided",
                    "omit task_ids to evaluate the whole split, or list the pinned task ids",
                )
            if len(set(request.task_ids)) != len(request.task_ids):
                return rejection(
                    self._eval_id(request, digest),
                    ErrorCode.TASK_SET_INVALID,
                    "task_ids contains duplicate entries",
                    "each task id must appear exactly once",
                )
            known = {
                dataset.task_id(row, self.profile.benchmarks[request.benchmark])
                for row in dataset.build_split_rows(
                    self.profile.benchmarks[request.benchmark],
                    request.split,
                    resolver=self.layout.resolve_data,
                )
            }
            unknown = sorted(set(request.task_ids) - known)
            if unknown:
                return rejection(
                    self._eval_id(request, digest),
                    ErrorCode.TASK_SET_INVALID,
                    f"task_ids not present in split '{request.split}': {unknown[:5]}",
                    "list valid tasks via POST /tasks for the same benchmark+split",
                )

        # Idempotent replay must win over soft budgets: a completed evaluation is a durable fact,
        # so retrying it after OF-4/EC-4 fired must return the cached result instead of a rejection.
        # Holdout/manifest/split/tier/task validation already ran above; only budget checks remain.
        if (self.results_root / self._eval_id(request, digest) / "result.json").is_file():
            return None

        # OF-4: soft budget on split eval count.
        dataset_version = self._resolve_dataset_version(request)
        stats = self.ledger.overfit_stats(
            request.candidate_id,
            request.benchmark,
            request.split,
            dataset_version,
            self._evaluation_digest(digest),
        )
        if (
            stats["eval_count"] >= self.profile.budgets.split_eval_soft_limit
            and request.acknowledge_exhausted_split is not True
        ):
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.SPLIT_EXHAUSTED,
                (
                    f"split '{request.split}' already evaluated {stats['eval_count']} times for this lineage "
                    f"(soft limit {self.profile.budgets.split_eval_soft_limit}); continued evaluation risks overfitting"
                ),
                "set acknowledge_exhausted_split=true to force (results will be trusted=false)",
            )

        # EC-4: lineage cost budget.
        cost = self.ledger.lineage_cost(request.candidate_id)
        over_budget = (
            cost["evals"] >= self.profile.budgets.lineage_max_evals
            or cost["total_tokens"] >= self.profile.budgets.lineage_max_tokens
        )
        if over_budget and request.acknowledge_lineage_budget is not True:
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.LINEAGE_BUDGET_EXCEEDED,
                f"lineage cost {cost} exceeds budget for candidate '{request.candidate_id}'",
                "set acknowledge_lineage_budget=true to force (cost stays visible in results)",
            )
        return None

    # ---- execution ----

    def _effective_rollout_cap(self, tier_limit: Optional[int], budget: Optional[BudgetSpec]) -> Optional[int]:
        caps = [c for c in (tier_limit, budget.max_rollouts if budget else None) if c is not None]
        return min(caps) if caps else None

    def validate(self, request: EvalRequest) -> Optional[EvalResult]:
        """Public fail-fast pre-check (spec §4.4): returns the in-band rejection the service would
        produce, or None when the request is acceptable. Deployment boundaries (HTTP/MCP) call
        this before queueing so contract violations never round-trip through the worker."""
        return self._validate(request)

    def evaluate(self, request: EvalRequest) -> EvalResult:
        invalid = self._validate(request)
        if invalid is not None:
            return invalid

        digest = self._manifest_digest(request)
        return self._finish(request, digest)

    def eval_id_of(self, request: EvalRequest) -> Optional[str]:
        """Idempotency key for a request; None when the adapter/benchmark pair cannot resolve."""
        try:
            return self._eval_id(request, self._manifest_digest(request))
        except KeyError:
            return None

    def unresolved_eval_id_of(self, request: EvalRequest) -> str:
        """Idempotency key for requests rejected before a harness manifest can be built."""
        return self._unresolved_eval_id(request)

    def validate_holdout(self, request: EvalRequest) -> Optional[EvalResult]:
        """Preflight the locked holdout path without executing it.

        The promotion control plane uses this to reject unknown adapters/benchmarks or a
        benchmark without a registered holdout dataset before accepting a background job.
        """
        if request.split != HOLDOUT_SPLIT:
            return rejection(
                self._unresolved_eval_id(request),
                ErrorCode.INVALID_SPLIT,
                f"the promotion pipeline requires split='holdout', got '{request.split}'",
                "use evaluate() for train_probe/train_full; the holdout entry is pipeline-only",
            )
        # F-01: only the full tier has no rollout cap. Rejecting capped tiers here keeps the
        # "complete holdout" promise structural instead of relying on the caller's argv.
        if request.tier is not TierName.FULL:
            return rejection(
                self._unresolved_eval_id(request),
                ErrorCode.PROMOTION_TIER_INVALID,
                (
                    f"promotion always evaluates the complete holdout split; tier "
                    f"'{request.tier.value}' would truncate it"
                ),
                "submit the promotion request without a tier override (promotion is full-tier only)",
            )
        # Defense in depth: the control plane cannot express task pinning, but a hand-built
        # EvalRequest could still ask for a subset, which is not a holdout verdict (F-01).
        if request.task_ids is not None:
            return rejection(
                self._unresolved_eval_id(request),
                ErrorCode.PROMOTION_TIER_INVALID,
                "holdout evaluation always covers the complete locked split; task_ids would truncate it",
                "submit the promotion request without task_ids (promotion is whole-split only)",
            )
        if request.benchmark not in self.profile.benchmarks:
            return rejection(
                self._unresolved_eval_id(request),
                ErrorCode.BENCHMARK_UNKNOWN,
                f"benchmark '{request.benchmark}' is not registered in experiment profile '{self.profile.profile_id}'",
                "register the benchmark (with holdout_dataset_fpath) in the profile",
            )
        if request.adapter not in self.profile.adapters:
            return rejection(
                self._unresolved_eval_id(request),
                ErrorCode.ADAPTER_NOT_REGISTERED,
                f"adapter '{request.adapter}' is not registered in experiment profile '{self.profile.profile_id}'",
                "register the adapter in the profile",
            )
        digest = self._manifest_digest(request)
        invalid_dataset_version = self._dataset_version_rejection(request, digest)
        if invalid_dataset_version is not None:
            return invalid_dataset_version
        if request.harness_manifest is not None and request.harness_manifest != digest:
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.MANIFEST_MISMATCH,
                "requested harness manifest does not match the manifest computed from the current profile pin",
                "omit harness_manifest to evaluate under the current pin",
            )
        activation_error = self._activation_identity_error(request)
        if activation_error is not None:
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.ACTIVATION_MISMATCH,
                activation_error,
                "activate the exact candidate bundle before submitting /promotion",
            )
        benchmark = self.profile.benchmarks[request.benchmark]
        if not benchmark.holdout_dataset_fpath:
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.INVALID_SPLIT,
                f"benchmark '{request.benchmark}' has no holdout dataset registered in the profile",
                "register holdout_dataset_fpath for the benchmark before running promotion",
            )
        # F-01: an unreadable or empty holdout cannot produce a complete-split verdict. Fail here
        # instead of letting Gym materialize zero tasks and report a vacuous score.
        try:
            holdout_rows = dataset.build_split_rows(benchmark, HOLDOUT_SPLIT, resolver=self.layout.resolve_data)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.INVALID_SPLIT,
                f"holdout dataset for benchmark '{request.benchmark}' is unreadable: {error}",
                "fix holdout_dataset_fpath in the profile so it points at a readable JSONL file",
            )
        if not holdout_rows:
            return rejection(
                self._eval_id(request, digest),
                ErrorCode.INVALID_SPLIT,
                f"holdout dataset for benchmark '{request.benchmark}' is empty",
                "populate the locked holdout split before running promotion",
            )
        if benchmark.rubric_library_fpath is not None:
            try:
                load_rubric_library(rubric_library_path(benchmark, self.layout))
            except RubricLibraryError as error:
                return rejection(
                    self._eval_id(request, digest),
                    ErrorCode.RUBRIC_LIBRARY_MISMATCH,
                    f"rubric library snapshot failed validation: {error}",
                    "export a fresh frozen snapshot (DataDesigner rubric snapshot export) and pin its rubric_library_sha256 in the profile",
                )
        return None

    def evaluate_holdout(self, request: EvalRequest) -> EvalResult:
        """OF-2: the locked holdout entry point, reserved for the promotion control plane.

        Deliberately NOT reachable through evaluate()/the outer-loop tool: holdout results may
        only enter promotion reports, never the outer search loop's feedback signal.
        """
        invalid = self.validate_holdout(request)
        if invalid is not None:
            return invalid
        digest = self._manifest_digest(request)
        return self._finish(request, digest)

    def _finish(self, request: EvalRequest, digest: str) -> EvalResult:
        evaluation_digest = self._evaluation_digest(digest)
        eval_id = self._eval_id(request, digest)
        eval_dir = self.results_root / eval_id
        result_fpath = eval_dir / "result.json"

        if result_fpath.is_file():
            return self._load_cached(eval_id, result_fpath)

        if not self._slots.acquire(blocking=False):
            # EC-5: lineage-level concurrency cap — queue instead of blocking the outer loop.
            return EvalResult(
                eval_id=eval_id,
                status=EvalStatus.QUEUED,
                manifest_schema_version=self.manifest_schema_version,
                candidate_digest=request.candidate_digest,
                harness_manifest_digest=digest,
                environment_digest=self.environment_digest,
                evaluation_digest=evaluation_digest,
            )
        try:
            # Re-check after acquiring the slot: a concurrent identical request may have completed
            # the run between the check above and the slot acquisition.
            if result_fpath.is_file():
                return self._load_cached(eval_id, result_fpath)
            return self._execute(request, eval_id, eval_dir, digest, evaluation_digest)
        finally:
            self._slots.release()

    def _load_cached(self, eval_id: str, result_fpath: Path) -> EvalResult:
        cached = EvalResult.model_validate_json(result_fpath.read_text(encoding="utf-8"))
        cached.warnings = cached.warnings + [WARNING_CACHED]
        return cached

    def _execute(
        self,
        request: EvalRequest,
        eval_id: str,
        eval_dir: Path,
        digest: str,
        evaluation_digest: str,
    ) -> EvalResult:
        adapter = self.profile.adapters[request.adapter]
        benchmark = self.profile.benchmarks[request.benchmark]
        tier = self.profile.tier_preset(request.tier)
        dataset_version = self._resolve_dataset_version(request)
        result_fpath = eval_dir / "result.json"

        eval_dir.mkdir(parents=True, exist_ok=True)
        input_fpath = eval_dir / "input.jsonl"
        output_fpath = eval_dir / "rollouts.jsonl"
        rows = dataset.build_split_rows(benchmark, request.split, resolver=self.layout.resolve_data)
        if request.task_ids is not None:
            rows = dataset.filter_rows(rows, request.task_ids, benchmark)
        dataset.write_jsonl(rows, input_fpath)

        # F-01: holdout always runs the complete locked split. Neither the tier preset nor
        # budget.max_rollouts may turn it into a subset (the promotion tier is pinned to full, but
        # defense-in-depth keeps a hand-built holdout EvalRequest from sneaking a cap through).
        max_rollouts = (
            None if request.split == HOLDOUT_SPLIT else self._effective_rollout_cap(tier.limit, request.budget)
        )
        timeout_s = float(request.budget.timeout_s) if request.budget and request.budget.timeout_s else None
        argv = self.runner.build_argv(
            adapter,
            self.profile.model,
            tier,
            input_fpath,
            output_fpath,
            max_rollouts,
            benchmark_config_paths=benchmark.config_paths,
        )
        self.runner.write_repro_record(
            eval_dir,
            argv,
            request.seed,
            dataset_version,
            digest,
            self.gym_commit,
            adapter,
            benchmark_config_paths=benchmark.config_paths,
            candidate_digest=request.candidate_digest,
            environment_identity=self.environment_identity,
            environment_digest=self.environment_digest,
            evaluation_digest=evaluation_digest,
            manifest_schema_version=self.manifest_schema_version,
        )

        outcome = self.runner.execute(argv, timeout_s)
        rewards, rollouts_failed, total_tokens = parse_rollout_metrics(output_fpath)

        if outcome.status == EvalStatus.FAILED and not rewards:
            result = EvalResult(
                eval_id=eval_id,
                status=EvalStatus.FAILED,
                error=EvalError(
                    code=ErrorCode.EVAL_FAILED,
                    message=f"gym eval run exited with code {outcome.returncode}",
                    fix_hint=(outcome.stdout_tail or "inspect eval_dir logs")[:500],
                ),
                manifest_schema_version=self.manifest_schema_version,
                candidate_digest=request.candidate_digest,
                harness_manifest_digest=digest,
                environment_digest=self.environment_digest,
                evaluation_digest=evaluation_digest,
                experiment_profile_id=self.profile.profile_id,
                dataset_version=dataset_version,
                warnings=[],
            )
            self._finalize(
                result,
                request,
                dataset_version,
                digest,
                evaluation_digest,
                result_fpath,
                reward_mean=None,
                total_tokens=0,
            )
            return result

        # §5.3 scenario C: a rubric-scored benchmark replaces programmatic rewards with the frozen
        # library's judgment. Judge failures are explicit — no all-zero degradation (§7.3(5)).
        rubric_verdict: Optional[RubricVerdict] = None
        if benchmark.rubric_library_fpath is not None:
            try:
                rubric_verdict = self._judge_rubric(eval_dir, output_fpath, benchmark)
            except RubricLibraryError as error:
                result = EvalResult(
                    eval_id=eval_id,
                    status=EvalStatus.FAILED,
                    error=EvalError(
                        code=ErrorCode.RUBRIC_LIBRARY_MISMATCH,
                        message=f"rubric library snapshot failed validation during judgment: {error}",
                        fix_hint="export a fresh frozen snapshot and pin its rubric_library_sha256 in the profile",
                    ),
                    harness_manifest_digest=digest,
                    experiment_profile_id=self.profile.profile_id,
                    dataset_version=dataset_version,
                    warnings=[],
                )
                self._finalize(result, request, dataset_version, digest, evaluation_digest, result_fpath, reward_mean=None, total_tokens=total_tokens)
                return result
            except RubricJudgeError as error:
                result = EvalResult(
                    eval_id=eval_id,
                    status=EvalStatus.FAILED,
                    error=EvalError(
                        code=ErrorCode.RUBRIC_JUDGE_FAILED,
                        message=f"rubric judgment failed: {error}",
                        fix_hint="judge endpoint outages fail explicitly — retry after the endpoint recovers; no all-zero degradation is produced (§7.3(5))",
                    ),
                    harness_manifest_digest=digest,
                    experiment_profile_id=self.profile.profile_id,
                    dataset_version=dataset_version,
                    warnings=[],
                )
                self._finalize(result, request, dataset_version, digest, evaluation_digest, result_fpath, reward_mean=None, total_tokens=total_tokens)
                return result
            rewards = list(rubric_verdict.rewards)

        # F-03: failures and lost rollouts are zero-score attempts, not exclusions. The denominator
        # is the materialized rollout inventory when Gym left one (it already covers failures and
        # rollouts killed before they could report); otherwise completed + failed rows. Both
        # aggregates use it, so a candidate cannot lift its score by crashing or going missing.
        materialized = materialized_rollout_counts(output_fpath)
        attempted = len(rewards) + rollouts_failed
        if materialized is not None:
            attempted = max(attempted, materialized[0])
        reward_mean = sum(rewards) / attempted if attempted else None
        items = (
            []
            if request.split == HOLDOUT_SPLIT
            else parse_rollout_items(input_fpath, output_fpath, benchmark)
        )
        # FAILED with partial rollouts must still carry the exit detail — an error-less failed
        # result is undiagnosable for the outer loop (plugin surfaces "without an error payload").
        error = (
            EvalError(
                code=ErrorCode.EVAL_FAILED,
                message=f"gym eval run exited with code {outcome.returncode}",
                fix_hint=(outcome.stdout_tail or "inspect eval_dir logs")[:500],
            )
            if outcome.status == EvalStatus.FAILED
            else None
        )
        result = EvalResult(
            eval_id=eval_id,
            status=outcome.status,
            error=error,
            metrics=EvalMetrics(
                reward_mean=reward_mean,
                pass_rate=(sum(1 for r in rewards if r >= 1.0) / attempted) if attempted else None,
                rollouts=len(rewards),
                rollouts_failed=rollouts_failed,
                rollouts_attempted=attempted,
            ),
            items=items,
            per_rollout_ref=str(output_fpath),
            cost=EvalCost(
                wall_time_s=outcome.wall_time_s,
                total_tokens=total_tokens,
                rollouts_executed=len(rewards),
            ),
            manifest_schema_version=self.manifest_schema_version,
            candidate_digest=request.candidate_digest,
            harness_manifest_digest=digest,
            environment_digest=self.environment_digest,
            evaluation_digest=evaluation_digest,
            experiment_profile_id=self.profile.profile_id,
            dataset_version=dataset_version,
            rubric_evidence=(
                RubricEvidence(
                    rubric_library_sha256=rubric_verdict.rubric_library_sha256,
                    rubric_ids=list(rubric_verdict.rubric_ids),
                    rubric_weights=list(rubric_verdict.rubric_weights),
                    group_pass_counts=list(rubric_verdict.group_pass_counts),
                    judge_model=rubric_verdict.judge_model,
                    scored_rollouts=rubric_verdict.scored_rollouts,
                    ungrouped_rollouts=rubric_verdict.ungrouped_rollouts,
                    evidence_ref=str(eval_dir / "rubric_evidence.jsonl"),
                )
                if rubric_verdict is not None
                else None
            ),
            warnings=(
                [WARNING_RUBRIC_INCOMPLETE_GROUP.format(count=rubric_verdict.ungrouped_rollouts)]
                if rubric_verdict is not None and rubric_verdict.ungrouped_rollouts
                else []
            ),
        )
        self._finalize(
            result,
            request,
            dataset_version,
            digest,
            evaluation_digest,
            result_fpath,
            reward_mean=reward_mean,
            total_tokens=total_tokens,
            rubric_verdict=rubric_verdict,
        )
        return result

    def _judge_rubric(self, eval_dir: Path, output_fpath: Path, benchmark: BenchmarkSpec) -> RubricVerdict:
        """Run the rubric judgment adapter over the rollout rows (§5.3 steps 1–3) and persist the
        per-rollout evidence next to them."""
        criteria, library_sha256 = load_rubric_library(rubric_library_path(benchmark, self.layout))
        backend = self.rubric_judge_backend
        if backend is None:
            backend = DataDesignerRubricBackend(model=benchmark.rubric_judge.name)
        verdict = judge_rubric_rollouts(
            dataset.read_jsonl(output_fpath),
            criteria,
            backend=backend,
            group_size=benchmark.rubric_group_size,
            library_sha256=library_sha256,
            judge_model=benchmark.rubric_judge.name,
        )
        dataset.write_jsonl(list(verdict.evidence_rows), eval_dir / "rubric_evidence.jsonl")
        return verdict

    def _finalize(
        self,
        result: EvalResult,
        request: EvalRequest,
        dataset_version: str,
        digest: str,
        evaluation_digest: str,
        result_fpath: Path,
        reward_mean: Optional[float],
        total_tokens: int,
        rubric_verdict: Optional[RubricVerdict] = None,
    ) -> None:
        self.ledger.record(
            result.eval_id,
            request.candidate_id,
            request.benchmark,
            request.split,
            dataset_version,
            digest,
            reward_mean,
            total_tokens,
            experiment_profile_id=result.experiment_profile_id,
            candidate_digest=request.candidate_digest,
            environment_digest=self.environment_digest,
            evaluation_digest=evaluation_digest,
            manifest_schema_version=self.manifest_schema_version,
        )

        lineage = self.ledger.lineage_cost(request.candidate_id)
        if result.cost is not None:
            result.cost.lineage_cost_to_date = LineageCost(**lineage)

        stats = self.ledger.overfit_stats(
            request.candidate_id,
            request.benchmark,
            request.split,
            dataset_version,
            evaluation_digest,
        )
        trusted = stats["eval_count"] <= self.profile.budgets.split_eval_soft_limit
        result.overfit_guard = OverfitGuard(
            split_eval_count=stats["eval_count"], score_trend=stats["score_trend"], trusted=trusted
        )
        if rubric_verdict is not None:
            # P7: per-criterion trusted/overfit signals merge into the existing overfit_guard
            # structure — a criterion hitting on every rollout or none carries no promotion signal.
            result.overfit_guard.criterion_signals = {
                rubric_id: CriterionSignal(**signal) for rubric_id, signal in rubric_verdict.criterion_signals.items()
            }
        if not trusted:
            result.warnings.append(WARNING_SPLIT_EXHAUSTED)
        if self.ledger.candidate_digests(request.candidate_id) - {digest}:
            result.warnings.append(WARNING_DIGEST_CHANGED)
        if self.ledger.candidate_environment_digests(request.candidate_id) - {self.environment_digest}:
            result.warnings.append(WARNING_ENVIRONMENT_CHANGED)
        if self.ledger.candidate_schema_versions(request.candidate_id) - {self.manifest_schema_version}:
            result.warnings.append(WARNING_SCHEMA_CHANGED)

        result_fpath.write_text(result.model_dump_json(indent=2), encoding="utf-8")
