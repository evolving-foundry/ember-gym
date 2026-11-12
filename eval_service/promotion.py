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
"""Promotion pipeline (OF-2): the only path to the locked holdout split.

Outer-loop agents cannot reach this module — evaluate() rejects holdout outright. Holdout
results are written to promotion reports and never fed back into the search loop's signal.
"""

import json
import logging
import math
import re
import threading
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

from eval_service.atomic import atomic_write_json
from eval_service.contract import EvalRequest, EvalResult, EvalStatus
from eval_service.service import EvaluationService


REPORTED_STATUSES = (EvalStatus.COMPLETED, EvalStatus.BUDGET_EXHAUSTED)
PROMOTION_ID_PATTERN = re.compile(r"^promo-[0-9a-f]{16}$")
LOG = logging.getLogger(__name__)


def _candidate_key(candidate_id: str) -> str:
    """Stable filesystem-safe key; candidate_id itself stays in the report payload."""
    return sha256(candidate_id.encode("utf-8")).hexdigest()[:16]


def promotion_id_for(
    candidate_digest: Optional[str],
    benchmark: str,
    *,
    adapter: str,
    tier: str,
    evaluation_digest: str,
) -> str:
    """Stable id for one holdout attempt under a fixed harness/environment identity.

    Seed is deliberately excluded. The holdout split is one-shot per candidate content and
    environment: a caller cannot obtain a second independent draw by varying seed. A changed
    environment or harness still produces a new identity, because the old report no longer
    describes the current measurement setup.
    """
    material = "|".join(
        [
            candidate_digest or "unidentified",
            benchmark,
            adapter,
            tier,
            evaluation_digest,
        ]
    ).encode("utf-8")
    return "promo-" + sha256(material).hexdigest()[:16]


class PromotionPipeline:
    def __init__(self, service: EvaluationService, reports_root: Optional[Path] = None) -> None:
        self.service = service
        self.reports_root = Path(reports_root) if reports_root else service.results_root / "promotion"
        self._reports_lock = threading.RLock()
        self._reports_cache: Optional[list[dict]] = None

    def _invalidate_reports_cache(self) -> None:
        with self._reports_lock:
            self._reports_cache = None

    def run(
        self, request: EvalRequest, *, promotion_id: Optional[str] = None
    ) -> tuple[EvalResult, Optional[Path]]:
        """Run a holdout evaluation and, if it produced measurements, persist a promotion report."""
        result = self.service.evaluate_holdout(request)
        report_fpath = None
        if result.status in REPORTED_STATUSES:
            promotion_id = promotion_id or promotion_id_for(
                request.candidate_digest,
                request.benchmark,
                adapter=request.adapter,
                tier=request.tier.value,
                evaluation_digest=result.evaluation_digest or "unresolved",
            )
            report = {
                "promotion_id": promotion_id,
                "eval_id": result.eval_id,
                "candidate_id": request.candidate_id,
                "candidate_digest": result.candidate_digest,
                "adapter": request.adapter,
                "benchmark": request.benchmark,
                "split": request.split,
                "tier": request.tier.value,
                "seed": request.seed,
                "status": result.status.value,
                "metrics": result.metrics.model_dump() if result.metrics else None,
                "cost": result.cost.model_dump() if result.cost else None,
                "manifest_schema_version": result.manifest_schema_version,
                "harness_manifest_digest": result.harness_manifest_digest,
                "environment_digest": result.environment_digest,
                "evaluation_digest": result.evaluation_digest,
                "experiment_profile_id": result.experiment_profile_id,
                "dataset_version": result.dataset_version,
                # §3.3: per-criterion rubric evidence travels into the promotion feedback summary.
                "rubric_evidence": result.rubric_evidence.model_dump() if result.rubric_evidence else None,
                "warnings": result.warnings,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.reports_root.mkdir(parents=True, exist_ok=True)
            report_fpath = self.reports_root / f"{_candidate_key(request.candidate_id)}-{result.eval_id}.json"
            atomic_write_json(report_fpath, report)
            self._invalidate_reports_cache()
        return result, report_fpath

    def all_reports(self) -> list[dict]:
        """Read every promotion report in the reports root (top-level JSON files only).

        Reports are append-only within one service process. Cache the parsed set so status
        polling and repeated promotion submissions do not rescan every JSON file.
        """
        with self._reports_lock:
            if self._reports_cache is not None:
                return list(self._reports_cache)
            records = []
            if not self.reports_root.is_dir():
                self._reports_cache = records
                return list(records)
            for fpath in sorted(self.reports_root.glob("*.json")):
                try:
                    record = json.loads(fpath.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    LOG.warning("skipping unreadable promotion report %s: %s", fpath, exc)
                    continue
                if isinstance(record, dict):
                    records.append(record)
            self._reports_cache = records
            return list(records)

    def reports(self, candidate_id: str) -> list[dict]:
        return [record for record in self.all_reports() if record.get("candidate_id") == candidate_id]

    def find_report(
        self,
        candidate_digest: str,
        benchmark: str,
        *,
        adapter: str,
        tier: str,
        evaluation_digest: str,
    ) -> Optional[dict]:
        """Find the one-shot report for this exact experiment identity."""
        wanted = {
            "candidate_digest": candidate_digest,
            "benchmark": benchmark,
            "adapter": adapter,
            "tier": tier,
            "evaluation_digest": evaluation_digest,
        }
        for record in self.all_reports():
            if all(record.get(field) == value for field, value in wanted.items()):
                return record
        return None

    def report_by_promotion_id(self, promotion_id: str) -> Optional[dict]:
        for record in self.all_reports():
            if record.get("promotion_id") == promotion_id:
                return record
        return None

    def latest_verdict(self, candidate_id: str) -> Optional[dict]:
        """Outer-loop consumption entry (§11.6): latest holdout report plus a derived verdict.

        A completed holdout is only *eligible* for release. This method deliberately does not
        compare against an incumbent and must not be treated as the final release decision;
        ``compare_promotion_reports`` (or the host release controller) owns that policy.
        """
        reported = {status.value for status in REPORTED_STATUSES}
        reports = [r for r in self.reports(candidate_id) if r.get("status") in reported]
        if not reports:
            return None
        report = max(reports, key=lambda r: str(r.get("generated_at", "")))
        metrics = report.get("metrics") or {}
        completed = report["status"] == EvalStatus.COMPLETED.value
        successful = metrics.get("rollouts")
        return {
            "report": report,
            "holdout_status": report["status"],
            "eligible_for_release": completed and isinstance(successful, int) and successful > 0,
            "decision_scope": "holdout_only",
            "reward_mean": metrics.get("reward_mean"),
            "pass_rate": metrics.get("pass_rate"),
            "rollouts": successful,
            "rollouts_failed": metrics.get("rollouts_failed"),
        }

    def feedback_summary(self, candidate_id: str) -> Optional[dict]:
        """One-stop feedback for the next profile iteration (§11.6): the holdout verdict plus
        the candidate's search-side bucket summaries from the ledger. Read-only."""
        verdict = self.latest_verdict(candidate_id)
        if verdict is None:
            return None
        return {
            "candidate_id": candidate_id,
            "holdout": verdict,
            "buckets": self.service.ledger.bucket_summaries(candidate_id),
        }


def _completed_metric(
    record: dict[str, Any], label: str
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if record.get("status") != EvalStatus.COMPLETED.value:
        return None, None, f"{label} report status is {record.get('status')!r}, not completed"
    if record.get("split") != "holdout":
        return None, None, f"{label} report is not a holdout report"
    metrics = record.get("metrics") or {}
    reward = metrics.get("reward_mean")
    if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
        return None, None, f"{label} report has no finite reward_mean"
    successful = metrics.get("rollouts")
    failed = metrics.get("rollouts_failed", 0)
    pass_rate = metrics.get("pass_rate")
    if not isinstance(successful, int) or not isinstance(failed, int):
        return float(reward), None, f"{label} report has invalid rollout counts"
    attempted = successful + failed
    if attempted <= 0:
        return float(reward), None, f"{label} report has no attempted rollouts"
    if successful == 0:
        return float(reward), 0.0, None
    if not isinstance(pass_rate, (int, float)) or not math.isfinite(float(pass_rate)):
        return float(reward), None, f"{label} report has no finite pass_rate"
    # F-03: reports written after the failure-zero change carry rollouts_attempted and a
    # pass_rate whose denominator already includes failures. Legacy reports have a raw
    # pass_rate over successful rollouts only, which still needs the scaling.
    recorded_attempted = metrics.get("rollouts_attempted")
    if isinstance(recorded_attempted, int) and recorded_attempted > 0:
        return float(reward), float(pass_rate), None
    return float(reward), float(pass_rate) * successful / attempted, None


# Identity fields that make two holdout measurements comparable. A release decision across any
# of these is meaningless: the numbers no longer describe the same measurement setup (F-05).
_COMPARABILITY_FIELDS = (
    "benchmark",
    "adapter",
    "tier",
    "split",
    "dataset_version",
    "manifest_schema_version",
    "harness_manifest_digest",
    "environment_digest",
    "evaluation_digest",
)


def compare_promotion_reports(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    """Compare two holdout reports on effective pass rate; strict improvement replaces incumbent.

    The Gym promotion report says whether a holdout run completed and what it measured. It
    must not by itself mean "promote": the release controller compares the candidate with the
    current incumbent and treats ties as keeping the incumbent. Infrastructure failures count
    as zero in the denominator, matching the RSI release-gym policy; this helper exists for
    non-RSI hosts and must stay aligned with that policy.

    Reports are only comparable when they describe the same experiment identity (F-05): same
    benchmark/adapter/tier/dataset version, same manifest schema, and the same harness + runtime
    environment. A missing identity field is treated as incomparable rather than guessed.
    """
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    candidate_reward, candidate_rate, candidate_error = _completed_metric(candidate, "candidate")
    incumbent_reward, incumbent_rate, incumbent_error = _completed_metric(incumbent, "incumbent")
    reasons = [reason for reason in (candidate_error, incumbent_error) if reason]
    if not candidate.get("candidate_digest"):
        reasons.append("candidate report has no candidate_digest")
    if not incumbent.get("candidate_digest"):
        reasons.append("incumbent report has no candidate_digest")
    if not reasons:
        for field in _COMPARABILITY_FIELDS:
            candidate_value = candidate.get(field)
            incumbent_value = incumbent.get(field)
            if candidate_value is None or incumbent_value is None:
                reasons.append(f"both reports must carry {field} to be comparable")
            elif candidate_value != incumbent_value:
                reasons.append(
                    f"candidate and incumbent are not comparable: {field} {candidate_value!r} != {incumbent_value!r}"
                )
    delta = None
    if not reasons and candidate_rate is not None and incumbent_rate is not None:
        delta = candidate_rate - incumbent_rate
        if delta <= min_delta:
            reasons.append(
                f"candidate effective pass rate {candidate_rate:.6f} did not exceed incumbent "
                f"{incumbent_rate:.6f} by min_delta {min_delta:.6f}"
            )
    decision = "promote" if not reasons else "reject"
    return {
        "decision": decision,
        "reason": reasons[0] if reasons else "candidate strictly beats incumbent on holdout",
        "reasons": reasons,
        "candidate_reward_mean": candidate_reward,
        "incumbent_reward_mean": incumbent_reward,
        "candidate_pass_rate": candidate_rate,
        "incumbent_pass_rate": incumbent_rate,
        "delta": delta,
        "min_delta": min_delta,
    }
