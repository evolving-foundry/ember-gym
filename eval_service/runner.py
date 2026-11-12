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
"""Subprocess wrapper around Gym's native evaluation entrypoint (`gym eval run --no-serve`).

Spec §6 Phase 1: EvalRequest → gym eval run subprocess → parse JSONL → metrics. The runner never
builds a second rollout loop; it only assembles argv from tier presets (EC-1), enforces the
wall-clock budget (EC-2), and folds Gym's failure sidecar rows into `rollouts_failed`.
"""

import dataclasses
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent_gym.identity import configuration_identity
from agent_gym.layout import GymLayout, LayoutError
from agent_gym.overlay import OverlayError
from eval_service import dataset
from eval_service.contract import EvalItemCheck, EvalItemDiagnostic, EvalItemEvidence, EvalStatus
from eval_service.profile import AdapterSpec, BenchmarkSpec, ModelPin, TierPreset


STDOUT_TAIL_CHARS = 2000
MAX_ITEM_CHECKS = 20
MAX_ITEM_DIAGNOSTICS = 8
MAX_ITEM_TEXT_CHARS = 1200


_CANONICAL_ALIASES = {"gym", "ng"}


def _resolve_gym_bin(gym_bin: str) -> str:
    """Return the canonical AgentGym launcher while preserving custom executable paths."""
    requested = Path(gym_bin)
    if requested.stem not in _CANONICAL_ALIASES or str(requested) != requested.name:
        return gym_bin
    suffix = requested.suffix if requested.suffix else ""
    candidate_name = f"agent-gym{suffix}"
    candidate = Path(candidate_name)
    if candidate.is_file():
        return str(candidate)
    sibling = Path(sys.executable).with_name(candidate_name)
    if sibling.is_file():
        return str(sibling)
    on_path = shutil.which(candidate.stem)
    if on_path:
        return on_path
    return str(candidate)


@dataclasses.dataclass
class RunOutcome:
    status: EvalStatus
    rewards: list[float]
    rollouts_failed: int
    total_tokens: int
    wall_time_s: float
    returncode: Optional[int]
    stdout_tail: str


def _extract_tokens(row: dict[str, Any]) -> int:
    response = row.get("response") or {}
    usage = response.get("usage") if isinstance(response, dict) else None
    for source in (row.get("ng_perf"), usage):
        if isinstance(source, dict):
            value = source.get("total_tokens")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return int(value)
    return 0


def parse_rollout_metrics(output_jsonl_fpath: str | Path) -> tuple[list[float], int, int]:
    """Returns (rewards, rollouts_failed, total_tokens).

    Success rows carry a numeric `reward`; infrastructure failures land in the
    `<stem>_failures.jsonl` sidecar (counted separately, never merged into reward stats).
    A missing output file (crash before any rollout completed) yields empty metrics.
    """
    output = Path(output_jsonl_fpath)
    rewards: list[float] = []
    tokens = 0
    if output.is_file():
        for row in dataset.read_jsonl(output):
            reward = row.get("reward")
            if isinstance(reward, (int, float)):
                rewards.append(float(reward))
            tokens += _extract_tokens(row)
    failures_fpath = Path(output_jsonl_fpath).with_name(Path(output_jsonl_fpath).stem + "_failures.jsonl")
    rollouts_failed = len(dataset.read_jsonl(failures_fpath)) if failures_fpath.is_file() else 0
    return rewards, rollouts_failed, tokens


def _task_index(row: dict[str, Any]) -> Optional[int]:
    value = row.get("_ng_task_index")
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _bounded(value: Any, limit: int = MAX_ITEM_TEXT_CHARS) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    return text[:limit], len(text) > limit


def _input_task_ids(input_jsonl_fpath: str | Path, benchmark: BenchmarkSpec) -> dict[int, str]:
    input_fpath = Path(input_jsonl_fpath)
    if not input_fpath.is_file():
        return {}
    task_ids: dict[int, str] = {}
    for index, row in enumerate(dataset.read_jsonl(input_fpath)):
        try:
            task_ids[index] = dataset.task_id(row, benchmark)
        except (TypeError, ValueError):
            fallback = row.get("task_id")
            if isinstance(fallback, (str, int)) and not isinstance(fallback, bool):
                task_ids[index] = str(fallback)
    return task_ids


def materialized_rollout_counts(output_jsonl_fpath: str | Path) -> Optional[tuple[int, dict[int, int]]]:
    """Rollouts Gym actually materialized (``<output_stem>_materialized_inputs.jsonl``).

    Gym writes this file after applying ``--limit``/repeat expansion, one line per planned
    rollout, so it is the authoritative "attempted" inventory: returns ``(total_rows,
    rows_per_task_index)``. A rollout that is materialized but leaves no row in ``rollouts.jsonl``
    and no row in the failures sidecar was killed or lost mid-flight; Gym's
    ``count_missing_rollouts_as_zero`` scores it zero, and eval_service must use the same
    denominator or a lost rollout silently leaves the score (F-03).

    None means the artifact is absent (old Gym build or crash before materialization); callers
    then fall back to completed + failed rows. Malformed lines are skipped defensively — a
    truncated tail must not abort evidence parsing — while still counting toward the total, as
    Gym counts raw lines.
    """
    output = Path(output_jsonl_fpath)
    materialized = output.with_name(output.stem + "_materialized_inputs.jsonl")
    if not materialized.is_file():
        return None
    total = 0
    per_task: dict[int, int] = {}
    with open(materialized, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = _task_index(row)
            if index is not None:
                per_task[index] = per_task.get(index, 0) + 1
    return total, per_task


def parse_rollout_items(
    input_jsonl_fpath: str | Path,
    output_jsonl_fpath: str | Path,
    benchmark: BenchmarkSpec,
) -> list[EvalItemEvidence]:
    """Build bounded per-task evidence for the train split.

    ``rollouts.jsonl`` has no task id column; Gym records ``_ng_task_index`` referring to the
    row order in ``input.jsonl``. Only attempted tasks get an item: Gym's
    ``_materialized_inputs.jsonl`` names the rows ``--limit`` actually dispatched, so a row that
    was never run is omitted rather than misreported as a failed rollout (F-02). Within an
    attempted task, a missing rollout check is emitted only when neither a reward row nor a
    failure-sidecar row exists — an explicit failure is reported as ``execution``.

    ``len(items)`` therefore counts attempted tasks, while ``metrics.rollouts`` counts scored
    rollouts (with repeats); a task can contribute several rollouts under ``num_repeats``. The
    item denominator is the materialized rollout count for the task, so a materialized-but-silent
    repeat scores zero instead of shrinking the denominator (F-03).

    The parser is deliberately defensive: malformed rows or missing sidecar files must not turn a
    completed evaluation into a failed one.
    """
    task_ids = _input_task_ids(input_jsonl_fpath, benchmark)
    materialized = materialized_rollout_counts(output_jsonl_fpath)
    if not task_ids:
        return []

    rewards: dict[int, list[float]] = {index: [] for index in task_ids}
    rollout_rows: dict[int, list[dict[str, Any]]] = {index: [] for index in task_ids}
    output_fpath = Path(output_jsonl_fpath)
    if output_fpath.is_file():
        for row in dataset.read_jsonl(output_fpath):
            index = _task_index(row)
            reward = row.get("reward")
            if index not in task_ids or isinstance(reward, bool):
                continue
            if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
                continue
            rewards[index].append(float(reward))
            rollout_rows[index].append(row)

    failed_by_task: dict[int, int] = {}
    failures_fpath = output_fpath.with_name(output_fpath.stem + "_failures.jsonl")
    if failures_fpath.is_file():
        for row in dataset.read_jsonl(failures_fpath):
            index = _task_index(row)
            if index in task_ids:
                failed_by_task[index] = failed_by_task.get(index, 0) + 1

    if materialized is None:
        attempted_indices = set(task_ids)
        per_task_attempts: dict[int, int] = {}
    else:
        _, per_task_attempts = materialized
        # Union the materialized rows with any task that nevertheless produced output or a
        # failure record, so an out-of-order artifact can never hide evidence.
        attempted_indices = {index for index in per_task_attempts if index in task_ids} | {
            index for index in task_ids if rewards[index] or failed_by_task.get(index)
        }

    items: list[EvalItemEvidence] = []
    for index, task_id in sorted(task_ids.items()):
        if index not in attempted_indices:
            continue
        task_rewards = rewards[index]
        rows = rollout_rows[index]
        checks: list[EvalItemCheck] = []
        diagnostics: list[EvalItemDiagnostic] = []
        truncated = False

        for row in rows:
            reward = float(row["reward"])
            passed = reward >= 1.0
            failure_kind = row.get("failure_kind")
            failure_reason = row.get("failure_reason")
            extracted_answer = row.get("extracted_answer")
            reason: Optional[str] = None
            if not passed:
                if isinstance(failure_reason, str) and failure_reason:
                    reason, cut = _bounded(failure_reason)
                    truncated = truncated or cut
                else:
                    reason = "The rollout did not match the expected normalized answer."
            if len(checks) < MAX_ITEM_CHECKS:
                checks.append(
                    EvalItemCheck(
                        id=f"answer-match/{len(checks)}",
                        status="passed" if passed else "failed",
                        score=reward,
                        reason=reason,
                    )
                )
            else:
                truncated = True

            if not passed and len(diagnostics) < MAX_ITEM_DIAGNOSTICS:
                if isinstance(extracted_answer, str) and extracted_answer:
                    message, cut = _bounded(f"Extracted answer: {extracted_answer}")
                    diagnostics.append(
                        EvalItemDiagnostic(kind="gym-rollout", code="extracted_answer", message=message, truncated=cut)
                    )
                    truncated = truncated or cut
                if isinstance(failure_reason, str) and failure_reason:
                    message, cut = _bounded(failure_reason)
                    diagnostics.append(
                        EvalItemDiagnostic(
                            kind="verifier",
                            code=str(failure_kind) if failure_kind is not None else "answer_mismatch",
                            message=message,
                            truncated=cut,
                        )
                    )
                    truncated = truncated or cut
                elif len(diagnostics) < MAX_ITEM_DIAGNOSTICS:
                    diagnostics.append(
                        EvalItemDiagnostic(
                            kind="verifier",
                            code=str(failure_kind) if failure_kind is not None else "answer_mismatch",
                            message="The rollout did not match the expected normalized answer.",
                        )
                    )
            elif not passed:
                truncated = True

        failed_count = failed_by_task.get(index, 0)
        if failed_count:
            if len(checks) < MAX_ITEM_CHECKS:
                checks.append(
                    EvalItemCheck(
                        id="execution",
                        status="failed",
                        score=0.0,
                        reason=f"Gym reported {failed_count} infrastructure rollout failure(s).",
                    )
                )
            else:
                truncated = True
            if len(diagnostics) < MAX_ITEM_DIAGNOSTICS:
                diagnostics.append(
                    EvalItemDiagnostic(
                        kind="gym-execution",
                        code="rollout_failure",
                        message=f"Gym reported {failed_count} infrastructure rollout failure(s) for this task.",
                    )
                )
            else:
                truncated = True

        if not task_rewards and not failed_count:
            if len(checks) < MAX_ITEM_CHECKS:
                checks.append(
                    EvalItemCheck(
                        id="rollout",
                        status="failed",
                        score=0.0,
                        reason="No completed rollout was recorded for this task.",
                    )
                )
            if len(diagnostics) < MAX_ITEM_DIAGNOSTICS:
                diagnostics.append(
                    EvalItemDiagnostic(
                        kind="gym-execution",
                        code="missing_rollout",
                        message="No completed rollout was recorded for this task.",
                    )
                )

        # F-03: task-level aggregates use the same zero-for-failure denominator as the run-level
        # metrics. The materialized count catches repeats that were dispatched but never reported,
        # so one task's crash or lost rollout cannot make its surviving rollouts look better.
        attempted = max(len(task_rewards) + failed_count, per_task_attempts.get(index, 0))
        items.append(
            EvalItemEvidence(
                task_id=task_id,
                rollouts=len(task_rewards),
                reward_mean=(sum(task_rewards) / attempted) if attempted else None,
                pass_rate=(sum(1 for reward in task_rewards if reward >= 1.0) / attempted) if attempted else None,
                rollouts_failed=failed_count,
                checks=checks,
                diagnostics=diagnostics,
                truncated=truncated,
            )
        )
    return items


class GymEvalRunner:
    def __init__(
        self,
        gym_bin: str = "gym",
        repo_root: Optional[Path] = None,
        *,
        layout: Optional[GymLayout] = None,
    ) -> None:
        self.gym_bin = _resolve_gym_bin(gym_bin)
        self.layout = layout or GymLayout.discover(repo_root=repo_root)
        self.repo_root = Path(repo_root) if repo_root is not None else self.layout.repo_root
        self.workspace_root = self.layout.workspace_root

    def build_argv(
        self,
        adapter: AdapterSpec,
        model: Optional[ModelPin],
        tier: TierPreset,
        input_fpath: Path,
        output_fpath: Path,
        max_rollouts: Optional[int],
        benchmark_config_paths: Optional[list[str]] = None,
    ) -> list[str]:
        """EC-3: limit/repeats/concurrency come only from tier presets (intersected with the request
        budget cap); they are not free request parameters."""
        argv = [
            self.gym_bin,
            "eval",
            "run",
            "--no-serve",
            "--agent",
            adapter.harness_agent,
            "--input",
            str(input_fpath),
            "--output",
            str(output_fpath),
        ]
        # Gym merges later root configs over earlier ones. The candidate adapter
        # must win if an environment composition also contains a reference agent.
        for config_path in (*(benchmark_config_paths or []), *adapter.config_paths):
            resolved = self.layout.resolve(config_path, must_exist=False)
            argv += ["--config", str(resolved)]
        if max_rollouts is not None:
            argv += ["--limit", str(max_rollouts)]
        if tier.num_repeats > 1:
            argv += ["--num-repeats", str(tier.num_repeats)]
        argv += ["--concurrency", str(tier.concurrency)]
        if model is not None:
            argv += ["--model", model.name]
            if model.temperature is not None:
                argv += ["--temperature", str(model.temperature)]
            if model.top_p is not None:
                argv += ["--top-p", str(model.top_p)]
            if model.max_output_tokens is not None:
                argv += ["--max-output-tokens", str(model.max_output_tokens)]
        return argv

    def execute(self, argv: list[str], timeout_s: Optional[float]) -> RunOutcome:
        started = datetime.now(timezone.utc)
        status = EvalStatus.COMPLETED
        returncode: Optional[int] = None
        stdout = ""
        try:
            proc = subprocess.run(
                argv,
                cwd=self.workspace_root,
                env=self.layout.gym_env(),
                capture_output=True,
                timeout=timeout_s,
            )
            returncode = proc.returncode
            stdout = proc.stdout.decode(errors="replace")
            if returncode != 0:
                status = EvalStatus.FAILED
        except subprocess.TimeoutExpired as exc:
            # EC-2: over-budget runs are budget_exhausted, not errors; partial rollouts still count.
            status = EvalStatus.BUDGET_EXHAUSTED
            stdout = (exc.stdout or b"").decode(errors="replace")
        wall_time_s = (datetime.now(timezone.utc) - started).total_seconds()
        return RunOutcome(
            status=status,
            rewards=[],
            rollouts_failed=0,
            total_tokens=0,
            wall_time_s=wall_time_s,
            returncode=returncode,
            stdout_tail=stdout[-STDOUT_TAIL_CHARS:],
        )

    def write_repro_record(
        self,
        eval_dir: Path,
        argv: list[str],
        seed: int,
        dataset_version: str,
        manifest_digest: str,
        gym_commit: str,
        adapter: AdapterSpec,
        benchmark_config_paths: Optional[list[str]] = None,
        *,
        candidate_digest: Optional[str] = None,
        environment_identity: Optional[dict[str, Any]] = None,
        environment_digest: Optional[str] = None,
        evaluation_digest: Optional[str] = None,
        manifest_schema_version: int = 1,
    ) -> None:
        """§7 reproducibility: everything needed to re-run lands next to the rollouts."""
        eval_dir.mkdir(parents=True, exist_ok=True)
        try:
            runtime_identity = configuration_identity(self.layout)
        except (LayoutError, OverlayError, OSError):
            runtime_identity = {
                "repo": {"root": str(self.repo_root)},
                "overlay": {"root": str(self.layout.overlay_root)},
            }
        record = {
            "argv": argv,
            "seed": seed,
            "dataset_version": dataset_version,
            "manifest_schema_version": manifest_schema_version,
            "harness_manifest_digest": manifest_digest,
            "environment_digest": environment_digest,
            "evaluation_digest": evaluation_digest or manifest_digest,
            "gym_commit": gym_commit,
            "candidate_digest": candidate_digest,
            "adapter": {
                "harness_agent": adapter.harness_agent,
                "version": adapter.version,
            },
            "benchmark_config_paths": benchmark_config_paths or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment_identity": environment_identity,
            "runtime_identity": runtime_identity,
            "resolved_paths": {
                "benchmark_configs": [
                    str(self.layout.resolve(path, must_exist=False)) for path in (benchmark_config_paths or [])
                ],
                "adapter_configs": [str(self.layout.resolve(path, must_exist=False)) for path in adapter.config_paths],
            },
        }
        with open(eval_dir / "repro.json", "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
