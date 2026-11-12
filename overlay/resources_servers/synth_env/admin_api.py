# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""C-SE-2 admin plane (design §3.13): the operating interface of a long-lived host.

Scope boundaries (§3.13.6): no multi-host orchestration (deployment layer), no
evaluation queueing (eval_service, deferred). The strategy side never sees these
routes: ``SynthEnvServer.mcp_tools`` filters every ``/admin/*`` tool out of MCP
auto-exposure (决议 9), and tool names containing ``/`` would fail MCP name
validation anyway — the filter is a hard startup requirement under
``expose_tools_over_mcp: true``.

Discipline enforced here:
- runtime load runs the full triple-pin validation and never replaces a loaded
  version (资产不可变);
- unload is refused while live sessions reference the version (拒绝，不强杀);
- every admin mutation (load/unload/terminate/reap/drain) plus budget
  exhaustion lands in the journal (动作/操作者/时间/原因, ledger audit culture).
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from resources_servers.synth_env.package_loader import (
    PackageBusyError,
    PackageNotFoundError,
    SynthEnvPackageError,
    load_package_zip,
)
from resources_servers.synth_env.task_data import EnvReleasePin


class Journal:
    """Append-only admin audit trail (M1: in-memory; optional JSONL sink per entry)."""

    def __init__(self, jsonl_fpath: Optional[str] = None):
        self.entries: list[dict[str, Any]] = []
        self._jsonl_fpath = jsonl_fpath

    def append(self, action: str, operator: str = "admin_api", detail: Optional[dict[str, Any]] = None) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "operator": operator,
            "detail": detail or {},
        }
        self.entries.append(entry)
        if self._jsonl_fpath:
            with Path(self._jsonl_fpath).open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---- admin wire models -------------------------------------------------------


class AdminLoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zip_base64: str
    expect: Optional[EnvReleasePin] = None


class AdminUnloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_id: str
    version: str


class AdminTerminateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str


class AdminReapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttl_s: Optional[float] = Field(default=None, gt=0)


def _active_pin_keys(server: Any) -> list[tuple[str, str]]:
    # A list, not a set: PackageStore.unload reports how many live sessions
    # reference a version, so per-session entries must not collapse.
    return [state.pin_key for state in server._sessions.values() if not state.done]


def register_admin_routes(app: FastAPI, server: Any) -> None:
    app.get("/admin/packages")(server.admin_list_packages)
    app.get("/admin/packages/{ref_id}")(server.admin_package_detail)
    app.post("/admin/packages/load")(server.admin_load_package)
    app.post("/admin/packages/unload")(server.admin_unload_package)
    app.get("/admin/sessions")(server.admin_list_sessions)
    app.get("/admin/sessions/{session_id}")(server.admin_session_detail)
    app.post("/admin/sessions/{session_id}/terminate")(server.admin_terminate_session)
    app.post("/admin/sessions/reap")(server.admin_reap_sessions)
    app.get("/admin/metrics")(server.admin_metrics)
    app.post("/admin/drain")(server.admin_drain)


# ---- route handlers (methods on SynthEnvServer, registered above) --------------


def admin_list_packages(self: Any) -> list[dict[str, Any]]:
    return self._store.list()


def admin_package_detail(self: Any, ref_id: str) -> dict[str, Any]:
    versions = self._store._packages.get(ref_id)
    if not versions:
        raise HTTPException(status_code=404, detail=f"no loaded package with ref_id {ref_id!r}")
    return {
        version: {
            "fingerprint": package.manifest.fingerprint,
            "profile": package.manifest.profile,
            "description": package.manifest.description,
            "action_schema": package.env_spec.action_schema,
            "command_max_length": package.env_spec.command_max_length,
            "danger_patterns": package.env_spec.danger_patterns,
            "task_count": len(package.tasks),
            "replay_entries": len(package.replay),
            "task_ids": sorted(package.tasks),
            # §3.13 包详情: 规则数 / grounding 规模 (office packages only).
            "rules_count": (
                len((package.office.rules or {}).get("fsms", {})) + len((package.office.rules or {}).get("assertions", []))
                if package.office is not None
                else 0
            ),
            "grounding_rows": len(package.office.grounding_rows) if package.office is not None else 0,
        }
        for version, package in sorted(versions.items())
    }


def admin_load_package(self: Any, body: AdminLoadRequest) -> dict[str, Any]:
    try:
        zip_bytes = base64.b64decode(body.zip_base64, validate=True)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=f"zip_base64 is not valid base64: {error}") from error
    try:
        package = load_package_zip(zip_bytes, expect=body.expect)
        self._store.load(package)
    except SynthEnvPackageError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    self._journal.append("load", detail={"ref_id": package.manifest.ref_id, "version": package.manifest.version})
    return {
        "loaded": {
            "ref_id": package.manifest.ref_id,
            "version": package.manifest.version,
            "fingerprint": package.manifest.fingerprint,
        }
    }


def admin_unload_package(self: Any, body: AdminUnloadRequest) -> dict[str, Any]:
    try:
        self._store.unload(body.ref_id, body.version, _active_pin_keys(self))
    except PackageBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except PackageNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    self._journal.append("unload", detail={"ref_id": body.ref_id, "version": body.version})
    return {"unloaded": {"ref_id": body.ref_id, "version": body.version}}


def admin_list_sessions(self: Any) -> list[dict[str, Any]]:
    sessions = []
    for state in self._sessions.values():
        sessions.append(
            {
                "session_id": state.session_id,
                "task_id": state.task_id,
                "env_release": state.pin.model_dump(),
                "step": state.step,
                "max_steps": state.max_steps,
                "llm_sim_calls": state.llm_sim_calls,
                "done": state.done,
                "budget_remaining": {
                    "llm_sim_calls": (
                        None
                        if state.budgets.llm_sim_calls is None
                        else max(0, state.budgets.llm_sim_calls - state.llm_sim_calls)
                    ),
                    "wall_clock_s": (
                        None
                        if state.budgets.wall_clock_s is None
                        else max(0.0, state.budgets.wall_clock_s - (monotonic() - state.started_at))
                    ),
                },
            }
        )
    return sessions


def admin_session_detail(self: Any, session_id: str) -> dict[str, Any]:
    state = self._sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"no active session {session_id!r}")
    return {
        "session_id": state.session_id,
        "task_id": state.task_id,
        "env_release": state.pin.model_dump(),
        "step": state.step,
        "max_steps": state.max_steps,
        "done": state.done,
        "budget_exceeded": state.budget_exceeded,
        "failure_class": state.failure_class,
        "violations": list(state.violations),
        "trace_tail": state.history[-10:],
        "budget_remaining": {
            "llm_sim_calls": (
                None
                if state.budgets.llm_sim_calls is None
                else max(0, state.budgets.llm_sim_calls - state.llm_sim_calls)
            )
        },
    }


def admin_terminate_session(self: Any, session_id: str, body: AdminTerminateRequest) -> dict[str, Any]:
    state = self._sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"no active session {session_id!r}")
    state.done = True
    state.failure_class = state.failure_class or "admin_terminated"
    self._journal.append(
        "terminate", detail={"session_id": session_id, "task_id": state.task_id, "reason": body.reason}
    )
    return {"terminated": session_id, "reason": body.reason}


def admin_reap_sessions(self: Any, body: AdminReapRequest) -> dict[str, Any]:
    ttl_s = body.ttl_s if body.ttl_s is not None else self.config.session_ttl_s
    now = monotonic()
    stale = [
        session_id
        for session_id, state in self._sessions.items()
        if (now - state.last_activity_at) > ttl_s
    ]
    for session_id in stale:
        del self._sessions[session_id]
    self._journal.append("reap", detail={"ttl_s": ttl_s, "reaped": stale})
    return {"reaped": stale, "ttl_s": ttl_s}


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Deterministic nearest-rank percentile over an ascending list."""
    if not sorted_values:
        return 0.0
    index = max(0, -(-int(pct * len(sorted_values)) // 100) - 1)
    return sorted_values[index]


def admin_metrics(self: Any) -> dict[str, Any]:
    states = list(self._sessions.values())
    total_steps = sum(state.step for state in states)
    total_violations = sum(len(state.violations) for state in states)
    # Latency percentiles and the drift proxy come from the per-session
    # simulator traces (§3.13 metrics line, §7.1 drift monitoring v1):
    # a generation with zero grounding hits produced unanchored content.
    traces = [trace for state in states if state.llm is not None for trace in state.llm.traces]
    durations = sorted(trace.duration_ms for trace in traces)
    ungrounded = sum(1 for trace in traces if not trace.grounded_ids)
    return {
        "active_sessions": len(states),
        "packages_loaded": len(self._store.list()),
        "llm_sim_calls_total": sum(state.llm_sim_calls for state in states),
        "llm_sim_latency_ms": {
            "p50": round(_percentile(durations, 50), 3),
            "p95": round(_percentile(durations, 95), 3),
            "count": len(durations),
        },
        "llm_sim_ungrounded_calls": ungrounded,
        "drift_rate": round(ungrounded / len(traces), 4) if traces else 0.0,
        "steps_total": total_steps,
        "violations_total": total_violations,
        "violation_rate": (total_violations / total_steps) if total_steps else 0.0,
        "draining": self._draining,
    }


def admin_drain(self: Any) -> dict[str, Any]:
    self._draining = True
    self._journal.append("drain", detail={"active_sessions": len(self._sessions)})
    return {"draining": True, "active_sessions": len(self._sessions)}
