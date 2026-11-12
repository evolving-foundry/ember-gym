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
"""Background evaluation runner (§6 Phase 2): non-blocking submit with a draining worker.

submit() always returns immediately (queued result), identical requests dedupe to a single
execution by eval_id, and completed results are retrievable by eval_id — the outer loop never
blocks on a rollout. With a journal_path, the pending queue is mirrored to disk and restored
on restart (§11.2); already-finished entries settle instantly through the service's
result.json replay, so no work is re-executed after a crash.
"""

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Optional

from eval_service.contract import (
    ErrorCode,
    EvalError,
    EvalRequest,
    EvalResult,
    EvalStatus,
)
from eval_service.service import EvaluationService


LOG = logging.getLogger(__name__)

# F-07: a QUEUED result means every execution slot is busy elsewhere. Retry after yielding the
# CPU instead of immediately requeueing (which would spin this worker for the whole duration of a
# long rollout).
QUEUE_RETRY_INTERVAL_S = 0.25


class QueueJournal:
    """On-disk mirror of the runner's pending queue, following the Ledger pattern.

    File schema (synthetic example): {"pending": [{"eval_id": "eval-...",
    "request": {...EvalRequest dump...}}]}. Full rewrite on every mutation with a
    tmp-file replace, so a crash can never leave a truncated journal.
    """

    def __init__(self, fpath: Path) -> None:
        self.fpath = Path(fpath)
        self._entries: list[tuple[str, EvalRequest]] = []
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.fpath.read_text())
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("queue journal %s is unreadable (%s); starting from an empty queue", self.fpath, exc)
            return
        entries: list[tuple[str, EvalRequest]] = []
        for item in raw.get("pending") or []:
            try:
                entries.append((item["eval_id"], EvalRequest.model_validate(item["request"])))
            except (KeyError, TypeError, ValueError) as exc:
                LOG.warning("skipping unreadable queue journal entry in %s: %s", self.fpath, exc)
        self._entries = entries

    def _save(self) -> None:
        payload = {
            "pending": [
                {"eval_id": eval_id, "request": request.model_dump(mode="json")} for eval_id, request in self._entries
            ]
        }
        self.fpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.fpath.with_suffix(self.fpath.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.fpath)

    def pending_entries(self) -> list[tuple[str, EvalRequest]]:
        return list(self._entries)

    def add(self, eval_id: str, request: EvalRequest) -> None:
        self._entries.append((eval_id, request))
        self._save()

    def remove(self, eval_id: str) -> None:
        if all(entry_id != eval_id for entry_id, _ in self._entries):
            return
        self._entries = [(entry_id, req) for entry_id, req in self._entries if entry_id != eval_id]
        self._save()

    def replace(self, entries: list[tuple[str, EvalRequest]]) -> None:
        self._entries = list(entries)
        self._save()


class BackgroundEvaluationRunner:
    def __init__(self, service: EvaluationService, journal_path: Optional[Path] = None) -> None:
        self.service = service
        self._pending: deque[EvalRequest] = deque()
        # Accepted covers queued and running evals until they reach a terminal result.
        self._accepted: set[str] = set()
        self._results: dict[str, EvalResult] = {}
        self._condition = threading.Condition()
        self._shutdown = False
        self._journal = QueueJournal(journal_path) if journal_path is not None else None
        if self._journal is not None:
            with self._condition:
                restored: list[tuple[str, EvalRequest]] = []
                for _, request in self._journal.pending_entries():
                    # Re-key restored entries against the current runtime identity. A journal
                    # written before a framework/overlay upgrade carries the old eval_id; using
                    # it verbatim would leave a ghost pending entry after the worker settles the
                    # request under the new environment digest.
                    eval_id = self.service.eval_id_of(request) or self.service.unresolved_eval_id_of(request)
                    if eval_id in self._accepted or eval_id in self._results:
                        continue
                    self._accepted.add(eval_id)
                    self._pending.append(request)
                    restored.append((eval_id, request))
                self._journal.replace(restored)
        self._worker = threading.Thread(target=self._drain, name="eval-service-worker", daemon=True)
        self._worker.start()

    def submit(self, request: EvalRequest) -> EvalResult:
        """Queue an evaluation; returns the eval_id's queued result (or the completed one on
        duplicate submit of an already-finished evaluation)."""
        eval_id = self.service.eval_id_of(request)
        if eval_id is None:
            # Unresolvable request (unknown adapter/benchmark): the service rejects it under
            # this deterministic id when the worker processes it.
            eval_id = self.service.unresolved_eval_id_of(request)
        with self._condition:
            if eval_id in self._results:
                return self._results[eval_id]
            if eval_id in self._accepted:
                return self._queued_result(eval_id, request)
            self._accepted.add(eval_id)
            self._pending.append(request)
            if self._journal is not None:
                self._journal.add(eval_id, request)
            self._condition.notify()
            return self._queued_result(eval_id, request)

    def _queued_result(self, eval_id: str, request: EvalRequest) -> EvalResult:
        return EvalResult(
            eval_id=eval_id,
            status=EvalStatus.QUEUED,
            manifest_schema_version=self.service.manifest_schema_version,
            candidate_digest=request.candidate_digest,
            environment_digest=self.service.environment_digest,
        )

    def result(self, eval_id: str) -> Optional[EvalResult]:
        with self._condition:
            return self._results.get(eval_id)

    def pending(self) -> int:
        # Outstanding = queued + running; _accepted changes only at submit/terminal.
        with self._condition:
            return len(self._accepted)

    def shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify()
        self._worker.join(timeout=30)

    def _drain(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._shutdown:
                    self._condition.wait(timeout=1.0)
                if self._shutdown and not self._pending:
                    return
                request = self._pending.popleft()
            try:
                result = self.service.evaluate(request)
            except Exception as exc:
                # 服务进程内崩溃(如 spawn 失败)不能杀死 worker 线程,否则该评估永远停在
                # queued、外层循环无限轮询(设计 D4 的"结果未知"劣化形态)。落为 in-band
                # failed 结果,让提交方得到确定性失败。
                LOG.exception("evaluation worker crashed while evaluating request %s", request)
                result = EvalResult(
                    eval_id=self.service.eval_id_of(request) or self.service.unresolved_eval_id_of(request),
                    status=EvalStatus.FAILED,
                    manifest_schema_version=self.service.manifest_schema_version,
                    environment_digest=self.service.environment_digest,
                    error=EvalError(
                        code=ErrorCode.EVAL_FAILED,
                        message=f"evaluation worker crashed: {exc}",
                        fix_hint="inspect eval_service logs for the traceback",
                    ),
                )
            with self._condition:
                eval_id = result.eval_id
                if result.status == EvalStatus.QUEUED:
                    # Service slots were busy; retry later without losing duplicate-submit
                    # suppression. The eval_id stays accepted until it reaches a terminal state.
                    # wait() releases the condition so a submit/shutdown can wake us; the timeout
                    # covers the case where the slot frees up in another thread without notifying.
                    self._pending.appendleft(request)
                    self._condition.wait(timeout=QUEUE_RETRY_INTERVAL_S)
                    continue
                self._results[eval_id] = result
                self._accepted.discard(eval_id)
                if self._journal is not None:
                    self._journal.remove(eval_id)
