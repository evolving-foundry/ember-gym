# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loopback-only promotion control plane for the locked holdout split.

The evaluation data plane (`/gym_evaluate`, MCP `gym_evaluate`) rejects holdout outright. This
module is the separate, trusted path: one bundle+benchmark promotion attempt, one background
holdout rollout, and an aggregate report that never contains holdout task rows or answers.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from eval_service.atomic import atomic_write_json
from eval_service.contract import (
    ErrorCode,
    EvalError,
    EvalRequest,
    EvalResult,
    EvalStatus,
    PromotionRequest,
    PromotionState,
    PromotionStatusResponse,
    PromotionSubmitResponse,
)
from eval_service.promotion import PROMOTION_ID_PATTERN, PromotionPipeline, promotion_id_for


LOG = logging.getLogger(__name__)

# F-07: each retry re-runs the preflight (manifest hashing), so back off instead of polling at a
# fixed short interval while an evaluation slot is busy.
MAX_POLL_INTERVAL_S = 5.0


class PromotionPreflightError(Exception):
    """The request is structurally valid but cannot run under the current profile."""

    def __init__(self, result: EvalResult) -> None:
        super().__init__(result.error.message if result.error else "promotion preflight failed")
        self.result = result


class PromotionConflict(Exception):
    """A promotion for the same bundle+benchmark is already running."""

    def __init__(self, promotion_id: str) -> None:
        super().__init__(f"promotion {promotion_id} is already running")
        self.promotion_id = promotion_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PromotionController:
    """Serialize trusted holdout runs and enforce the one-shot gate per experiment identity.

    No task ids or split are accepted from the caller. The identity is
    ``candidate_digest + benchmark + adapter + tier + evaluation_digest`` (the evaluation digest
    folds in the harness manifest and runtime environment); ``seed`` is deliberately excluded, so
    varying it cannot buy a second holdout. A second request for the same identity reuses the
    persisted report; a concurrent duplicate gets a conflict instead of starting a second holdout
    rollout.
    """

    def __init__(
        self,
        pipeline: PromotionPipeline,
        *,
        jobs_root: Optional[Path] = None,
        max_queue_wait_s: float = 3600.0,
        poll_interval_s: float = 0.25,
    ) -> None:
        self.pipeline = pipeline
        self.jobs_root = Path(jobs_root) if jobs_root else pipeline.reports_root / ".jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.max_queue_wait_s = max_queue_wait_s
        self.poll_interval_s = poll_interval_s
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active: dict[tuple[Any, ...], str] = {}
        self._stop = threading.Event()
        self._queue: queue.Queue[Optional[tuple[str, EvalRequest]]] = queue.Queue()
        self._worker = threading.Thread(target=self._drain, name="gym-promotion-worker", daemon=True)
        self._worker.start()

    def _job_path(self, promotion_id: str) -> Path:
        return self.jobs_root / f"{promotion_id}.json"

    def _persist(self, job: dict[str, Any]) -> None:
        atomic_write_json(self._job_path(job["promotion_id"]), job)

    def _load(self, promotion_id: str) -> Optional[dict[str, Any]]:
        try:
            raw = json.loads(self._job_path(promotion_id).read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict) or raw.get("promotion_id") != promotion_id:
            return None
        if raw.get("state") in (PromotionState.QUEUED.value, PromotionState.RUNNING.value):
            raw["state"] = PromotionState.FAILED.value
            raw["error"] = EvalError(
                code=ErrorCode.EVAL_FAILED,
                message="promotion job was interrupted before reaching a terminal state",
                fix_hint="submit the promotion again; any completed report is still reusable",
            ).model_dump(mode="json")
            raw["updated_at"] = _now()
            self._persist(raw)
        return raw

    def submit(self, request: PromotionRequest) -> PromotionSubmitResponse:
        eval_request = request.to_eval_request()
        invalid = self.pipeline.service.validate_holdout(eval_request)
        if invalid is not None:
            raise PromotionPreflightError(invalid)

        evaluation_digest = self.pipeline.service.evaluation_digest_for_request(eval_request)
        promotion_id = promotion_id_for(
            request.candidate_digest,
            request.benchmark,
            adapter=request.adapter,
            tier=request.tier.value,
            evaluation_digest=evaluation_digest,
        )
        key = (
            request.candidate_digest,
            request.benchmark,
            request.adapter,
            request.tier.value,
            evaluation_digest,
        )
        with self._lock:
            existing = self.pipeline.find_report(
                request.candidate_digest,
                request.benchmark,
                adapter=request.adapter,
                tier=request.tier.value,
                evaluation_digest=evaluation_digest,
            )
            if existing is not None:
                return PromotionSubmitResponse(
                    promotion_id=existing.get("promotion_id", promotion_id),
                    state=PromotionState(existing["status"]),
                    candidate_id=existing.get("candidate_id", request.candidate_id),
                    candidate_digest=request.candidate_digest,
                    benchmark=request.benchmark,
                    reused=True,
                    report=existing,
                )
            active = self._active.get(key)
            if active is not None:
                raise PromotionConflict(active)

            job = {
                "promotion_id": promotion_id,
                "candidate_id": request.candidate_id,
                "candidate_digest": request.candidate_digest,
                "benchmark": request.benchmark,
                "adapter": request.adapter,
                "tier": request.tier.value,
                "seed": request.seed,
                "dataset_version": request.dataset_version,
                "evaluation_digest": evaluation_digest,
                "state": PromotionState.QUEUED.value,
                "created_at": _now(),
                "updated_at": _now(),
                "report": None,
                "error": None,
            }
            self._jobs[promotion_id] = job
            self._active[key] = promotion_id
            self._persist(job)

        self._queue.put((promotion_id, eval_request))
        return PromotionSubmitResponse(
            promotion_id=promotion_id,
            state=PromotionState.QUEUED,
            candidate_id=request.candidate_id,
            candidate_digest=request.candidate_digest,
            benchmark=request.benchmark,
        )

    def _update(self, promotion_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(promotion_id)
            if job is None:
                return
            job.update(changes)
            job["updated_at"] = _now()
            self._persist(job)

    def _finish(
        self,
        promotion_id: str,
        state: PromotionState,
        *,
        report: Optional[dict[str, Any]] = None,
        error: Optional[EvalError] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(promotion_id)
            if job is None:
                return
            job["state"] = state.value
            job["report"] = report
            job["error"] = error.model_dump(mode="json") if error is not None else None
            job["updated_at"] = _now()
            self._active.pop(
                (
                    job["candidate_digest"],
                    job["benchmark"],
                    job["adapter"],
                    job["tier"],
                    job["evaluation_digest"],
                ),
                None,
            )
            self._persist(job)

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            promotion_id, request = item
            if self._stop.is_set():
                self._finish(
                    promotion_id,
                    PromotionState.FAILED,
                    error=EvalError(
                        code=ErrorCode.EVAL_FAILED,
                        message="promotion worker is shutting down",
                        fix_hint="submit the promotion again after the service restarts",
                    ),
                )
                continue
            self._update(promotion_id, state=PromotionState.RUNNING.value)
            deadline = time.monotonic() + self.max_queue_wait_s
            poll_interval = self.poll_interval_s
            try:
                while True:
                    result, report_path = self.pipeline.run(request, promotion_id=promotion_id)
                    if result.status != EvalStatus.QUEUED:
                        break
                    if time.monotonic() >= deadline:
                        self._finish(
                            promotion_id,
                            PromotionState.FAILED,
                            error=EvalError(
                                code=ErrorCode.EVAL_FAILED,
                                message="promotion waited too long for an evaluation slot",
                                fix_hint="retry after outstanding evaluations settle",
                            ),
                        )
                        break
                    if self._stop.wait(poll_interval):
                        self._finish(
                            promotion_id,
                            PromotionState.FAILED,
                            error=EvalError(
                                code=ErrorCode.EVAL_FAILED,
                                message="promotion worker is shutting down",
                                fix_hint="submit the promotion again after the service restarts",
                            ),
                        )
                        break
                    poll_interval = min(poll_interval * 2, MAX_POLL_INTERVAL_S)
                if result is not None and result.status != EvalStatus.QUEUED:
                    report = None
                    if report_path is not None:
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                    self._finish(
                        promotion_id,
                        PromotionState(result.status.value),
                        report=report,
                        error=result.error,
                    )
            except Exception:
                LOG.exception("promotion worker crashed for %s", promotion_id)
                self._finish(
                    promotion_id,
                    PromotionState.FAILED,
                    error=EvalError(
                        code=ErrorCode.EVAL_FAILED,
                        message="promotion worker crashed before producing a report",
                        fix_hint="inspect the service log and retry the promotion",
                    ),
                )

    def status(self, promotion_id: str) -> Optional[PromotionStatusResponse]:
        if PROMOTION_ID_PATTERN.fullmatch(promotion_id) is None:
            return None
        with self._lock:
            job = self._jobs.get(promotion_id) or self._load(promotion_id)
            if job is not None:
                error = job.get("error")
                return PromotionStatusResponse(
                    promotion_id=promotion_id,
                    state=PromotionState(job["state"]),
                    report=job.get("report"),
                    error=EvalError.model_validate(error) if error else None,
                )
            report = self.pipeline.report_by_promotion_id(promotion_id)
            if report is not None:
                return PromotionStatusResponse(
                    promotion_id=promotion_id,
                    state=PromotionState(report["status"]),
                    report=report,
                )
            return None

    def shutdown(self) -> None:
        self._stop.set()
        self._queue.put(None)
        self._worker.join(timeout=30)
