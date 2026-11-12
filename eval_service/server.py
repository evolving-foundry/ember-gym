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
"""Dual-protocol deployment server (HTTP + MCP) for the evaluation service.

One service core, two thin protocol layers: plain JSON-over-HTTP POST routes and MCP tools
(Streamable HTTP at /mcp) converge on the same BackgroundEvaluationRunner / PromotionPipeline
calls. MCP tool names, descriptions, and input schemas are the contract.py constants verbatim —
the TS outer loop reads exactly what the in-process contract states (zero drift), which is why
tools are hand-registered instead of harvested from route signatures.

OF-2 holds at the deployment boundary: the evaluation data plane (`/gym_evaluate`, `/tasks`,
MCP `gym_evaluate`) cannot reach the holdout split. The separate `/promotion` control plane is
loopback-only, is not registered as an MCP tool, accepts no split/task ids, and is the only HTTP
path that can enqueue a holdout run. `/feedback` remains read-only. Validation rejections are
in-band EvalResults (HTTP 200, status=rejected); only a request body violating the contract
models yields HTTP 422.

Boot: `python -m eval_service.server --profile-path <yaml> --results-root <dir> --ledger-path <json>`.
No --reload ever: on the /mnt/c mount the file watcher is pathologically slow.
"""

import argparse
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import mcp.types as types
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from mcp.server.lowlevel import Server as LowLevelMCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.routing import Mount, Route

from agent_gym.bootstrap import prepare_runtime
from agent_gym.layout import GymLayout
from eval_service import dataset
from eval_service.contract import (
    AVAILABLE_SPLITS,
    EVAL_TOOL_PARAMETERS,
    GYM_EVALUATE_TOOL_DESCRIPTION,
    GYM_EVALUATE_TOOL_NAME,
    HOLDOUT_SPLIT,
    EvalRequest,
    EvalResult,
    PromotionRequest,
    PromotionStatusResponse,
    PromotionSubmitResponse,
)
from eval_service.guards import Ledger
from eval_service.profile import profile_from_yaml
from eval_service.promotion import PromotionPipeline
from eval_service.promotion_control import PromotionConflict, PromotionController, PromotionPreflightError
from eval_service.queue import BackgroundEvaluationRunner
from eval_service.runner import GymEvalRunner
from eval_service.service import EvaluationService


LOG = logging.getLogger(__name__)

MCP_URL_PATH = "/mcp"

_LOOPBACK_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient", "::ffff:127.0.0.1"}

RESULT_TOOL_NAME = "gym_eval_result"
RESULT_TOOL_DESCRIPTION = (
    "按 eval_id 查询评估结果。gym_evaluate 返回 queued 时用它轮询；服务重启后仍可取回（结果落盘回放）。"
)
RESULT_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"eval_id": {"type": "string", "description": "gym_evaluate 返回的评估 id（幂等键）"}},
    "required": ["eval_id"],
}

FEEDBACK_TOOL_NAME = "gym_feedback"
FEEDBACK_TOOL_DESCRIPTION = (
    "读取一个候选的晋升结论（holdout latest_verdict）与搜索侧分桶汇总，作为下一轮 profile 迭代的输入。只读。"
)
FEEDBACK_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"candidate_id": {"type": "string", "description": "候选谱系标识"}},
    "required": ["candidate_id"],
}

# eval ids are "eval-" + 16 hex (contract.eval_id_for); enforced before any filesystem access.
_EVAL_ID_PATTERN = re.compile(r"eval-[0-9a-f]{16}")


class ServerConfig(BaseModel):
    """Deployment-level wiring; deliberately separate from the experiment profile (what to
    measure) and from EvalRequest (per-eval semantics)."""

    model_config = ConfigDict(extra="forbid")

    profile_path: Path
    results_root: Path
    ledger_path: Path
    journal_path: Optional[Path] = None
    reports_root: Optional[Path] = None
    repo_root: Optional[Path] = None
    overlay_root: Optional[Path] = None
    workspace_root: Optional[Path] = None
    data_root: Optional[Path] = None
    gym_bin: str = "gym"
    gym_commit: str = "unknown"
    max_in_flight: int = Field(default=1, ge=1)
    host: str = "127.0.0.1"
    port: int = 8399
    expose_tools_over_mcp: bool = True
    promotion_enabled: bool = True
    promotion_allow_remote: bool = False
    promotion_jobs_root: Optional[Path] = None
    promotion_max_queue_wait_s: float = Field(default=3600.0, gt=0)
    promotion_poll_interval_s: float = Field(default=0.25, gt=0)


class EvalIdBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_id: str = Field(min_length=1)


class CandidateIdBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)


class PendingBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskListRequest(BaseModel):
    """RSI split contract: the harness lists a training split's task universe to pin the
    held-in/held-out subsets. Metadata only — no rollouts are executed here, and holdout
    stays unreachable at this boundary too (OF-2)."""

    model_config = ConfigDict(extra="forbid")

    benchmark: str = Field(min_length=1)
    split: str = Field(min_length=1)


class TaskListResponse(BaseModel):
    tasks: list[str]


class ResultResponse(BaseModel):
    found: bool
    result: Optional[EvalResult] = None


class FeedbackResponse(BaseModel):
    found: bool
    summary: Optional[dict] = None


def _replay_result(results_root: Path, eval_id: str) -> Optional[EvalResult]:
    """Cross-restart result lookup: BackgroundEvaluationRunner only knows in-memory results, but
    every settled eval wrote results_root/<eval_id>/result.json (the service's idempotency cache)."""
    if _EVAL_ID_PATTERN.fullmatch(eval_id) is None:
        return None
    fpath = results_root / eval_id / "result.json"
    if not fpath.is_file():
        return None
    try:
        return EvalResult.model_validate_json(fpath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pydantic ValidationError subclasses ValueError
        LOG.warning("result replay failed for %s: %s", eval_id, exc)
        return None


def _is_loopback_client(host: Optional[str]) -> bool:
    if host is None:
        return False
    normalized = host.strip("[]").lower()
    return normalized in _LOOPBACK_CLIENTS or normalized.startswith("127.")


def _tool_defs() -> list[types.Tool]:
    return [
        types.Tool(
            name=GYM_EVALUATE_TOOL_NAME, description=GYM_EVALUATE_TOOL_DESCRIPTION, inputSchema=EVAL_TOOL_PARAMETERS
        ),
        types.Tool(name=RESULT_TOOL_NAME, description=RESULT_TOOL_DESCRIPTION, inputSchema=RESULT_TOOL_PARAMETERS),
        types.Tool(
            name=FEEDBACK_TOOL_NAME, description=FEEDBACK_TOOL_DESCRIPTION, inputSchema=FEEDBACK_TOOL_PARAMETERS
        ),
    ]


def _tool_names() -> list[str]:
    return [GYM_EVALUATE_TOOL_NAME, RESULT_TOOL_NAME, FEEDBACK_TOOL_NAME]


def _tool_error(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=True)


def _tool_payload(payload: dict) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))])


def _mount_mcp(app: FastAPI, mcp_server: LowLevelMCPServer) -> None:
    """Mount the Streamable-HTTP /mcp endpoint, mirroring nemo_gym.mcp_auto_exposure's tail: the
    session manager runs inside the app lifespan, and Route+Mount are both registered so /mcp and
    /mcp/<subpath> reach the SDK without a 307 in between."""
    if any(isinstance(r, (Route, Mount)) and getattr(r, "path", None) == MCP_URL_PATH for r in app.router.routes):
        raise ValueError("an /mcp mount already exists; refusing to shadow it")

    manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=True,
        stateless=True,
        # Server-to-server on localhost; the SDK's Host/Origin allowlist defends browsers, which
        # never talk to this endpoint (same rationale as mcp_auto_exposure).
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    class _MCPEndpoint:
        async def __call__(self, scope, receive, send):
            await manager.handle_request(scope, receive, send)

    endpoint = _MCPEndpoint()
    app.router.routes.insert(0, Route(MCP_URL_PATH, endpoint, include_in_schema=False))
    app.router.routes.insert(1, Mount(MCP_URL_PATH, app=endpoint))

    main_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_wrapper(app_: FastAPI):
        async with manager.run():
            async with main_lifespan(app_) as state:
                yield state

    app.router.lifespan_context = lifespan_wrapper


def build_app(
    config: ServerConfig,
    runner: Optional[GymEvalRunner] = None,
    *,
    layout: Optional[GymLayout] = None,
) -> FastAPI:
    """Wire profile → runner → ledger → service → background runner → promotion, then expose
    them over HTTP and (opt-in) MCP. `runner` overrides the GymEvalRunner default (tests inject
    a fake; deployments leave it None)."""
    layout = (
        layout
        or (getattr(runner, "layout", None) if runner is not None else None)
        or GymLayout.discover(
            repo_root=config.repo_root,
            overlay_root=config.overlay_root,
            workspace_root=config.workspace_root,
            data_root=config.data_root,
        )
    )
    results_root = layout.resolve_write(config.results_root)
    ledger_path = layout.resolve_write(config.ledger_path)
    journal_path = layout.resolve_write(config.journal_path) if config.journal_path else None
    reports_root = layout.resolve_write(config.reports_root) if config.reports_root else None
    profile = profile_from_yaml(layout.resolve(config.profile_path))
    gym_runner = runner or GymEvalRunner(
        gym_bin=config.gym_bin,
        repo_root=layout.repo_root,
        layout=layout,
    )
    ledger = Ledger(ledger_path)
    service = EvaluationService(
        profile=profile,
        runner=gym_runner,
        ledger=ledger,
        results_root=results_root,
        repo_root=layout.repo_root,
        gym_commit=config.gym_commit,
        max_in_flight=config.max_in_flight,
        layout=layout,
        environment_snapshot_path=results_root / "runtime_snapshot.json",
    )
    background = BackgroundEvaluationRunner(service, journal_path=journal_path)
    promotion = PromotionPipeline(service, reports_root=reports_root)
    promotion_jobs_root = (
        layout.resolve_write(config.promotion_jobs_root) if config.promotion_jobs_root else None
    )
    promotion_control = PromotionController(
        promotion,
        jobs_root=promotion_jobs_root,
        max_queue_wait_s=config.promotion_max_queue_wait_s,
        poll_interval_s=config.promotion_poll_interval_s,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        promotion_control.shutdown()
        background.shutdown()

    app = FastAPI(title="gym-eval-service", lifespan=lifespan)

    def _check_promotion_access(request: Request) -> None:
        if not config.promotion_enabled:
            raise HTTPException(status_code=404, detail="promotion control plane is disabled")
        if not config.promotion_allow_remote and not _is_loopback_client(
            request.client.host if request.client else None
        ):
            raise HTTPException(
                status_code=403,
                detail="promotion control plane is loopback-only; use localhost or a Unix socket",
            )

    @app.post("/gym_evaluate", response_model=EvalResult)
    async def gym_evaluate(request: EvalRequest) -> EvalResult:
        """Fail-fast validation (§4.4) then non-blocking submit (QUEUED / completed on duplicate)."""
        invalid = service.validate(request)
        if invalid is not None:
            return invalid  # in-band rejection — never queued
        return background.submit(request)

    @app.post("/results", response_model=ResultResponse)
    async def results(body: EvalIdBody) -> ResultResponse:
        result = background.result(body.eval_id) or _replay_result(results_root, body.eval_id)
        return ResultResponse(found=result is not None, result=result)

    @app.post("/pending", response_model=dict)
    async def pending(_: PendingBody) -> dict:
        return {"pending": background.pending()}

    @app.post("/tasks", response_model=TaskListResponse)
    async def tasks(body: TaskListRequest) -> TaskListResponse:
        if body.benchmark not in profile.benchmarks:
            raise HTTPException(
                status_code=400,
                detail=f"benchmark '{body.benchmark}' is not registered in experiment profile '{profile.profile_id}'",
            )
        if body.split == HOLDOUT_SPLIT or body.split not in AVAILABLE_SPLITS:
            raise HTTPException(
                status_code=400,
                detail=f"split '{body.split}' is not listable (use one of {list(AVAILABLE_SPLITS)})",
            )
        rows = dataset.build_split_rows(
            profile.benchmarks[body.benchmark],
            body.split,
            resolver=layout.resolve_data,
        )
        benchmark = profile.benchmarks[body.benchmark]
        return TaskListResponse(tasks=[dataset.task_id(row, benchmark) for row in rows])

    @app.post("/feedback", response_model=FeedbackResponse)
    async def feedback(body: CandidateIdBody) -> FeedbackResponse:
        summary = promotion.feedback_summary(body.candidate_id)
        return FeedbackResponse(found=summary is not None, summary=summary)

    @app.post("/promotion", response_model=PromotionSubmitResponse)
    async def promotion_submit(
        body: PromotionRequest, request: Request, response: Response
    ) -> PromotionSubmitResponse:
        """Enqueue one locked holdout run; never exposed through MCP."""
        _check_promotion_access(request)
        try:
            submitted = promotion_control.submit(body)
        except PromotionPreflightError as exc:
            error = exc.result.error
            detail = (
                error.model_dump(mode="json")
                if error is not None
                else {
                    "code": "PROMOTION_PREFLIGHT_FAILED",
                    "message": "promotion request cannot run under the current profile",
                    "fix_hint": "inspect the experiment profile and holdout dataset registration",
                }
            )
            raise HTTPException(status_code=400, detail=detail) from exc
        except PromotionConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "PROMOTION_IN_PROGRESS",
                    "promotion_id": exc.promotion_id,
                    "message": "a promotion for this candidate digest and benchmark is already running",
                    "fix_hint": "poll GET /promotion/{promotion_id} instead of submitting again",
                },
            ) from exc
        response.status_code = 200 if submitted.reused else 202
        return submitted

    @app.get("/promotion/{promotion_id}", response_model=PromotionStatusResponse)
    async def promotion_status(promotion_id: str, request: Request) -> PromotionStatusResponse:
        _check_promotion_access(request)
        status = promotion_control.status(promotion_id)
        if status is None:
            raise HTTPException(status_code=404, detail="promotion not found")
        return status

    if not config.expose_tools_over_mcp:
        return app

    mcp_server = LowLevelMCPServer("gym-eval-service")

    @mcp_server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return _tool_defs()

    @mcp_server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict):
        # Contract enforcement is exact via the pydantic models below (extra="forbid" → precise
        # per-field errors surfaced as MCP isError results), so the SDK-side schema re-validation
        # that validate_input=True would perform is redundant.
        try:
            if name == GYM_EVALUATE_TOOL_NAME:
                # F-04: candidate_digest is host/controller-supplied (it ties a result to the
                # materialized bundle). Accepting it here would let a data-plane caller forge the
                # candidate identity and poison eval_id/ledger/promotion lineage, so reject it
                # explicitly instead of silently dropping it.
                if arguments.get("candidate_digest") is not None:
                    return _tool_error(
                        "candidate_digest is host-supplied and cannot be set through gym_evaluate; "
                        "call POST /gym_evaluate with the activated bundle digest instead"
                    )
                request = EvalRequest.model_validate(arguments)
                invalid = service.validate(request)
                if invalid is not None:
                    return _tool_payload(invalid.model_dump())  # in-band rejection — never queued
                return _tool_payload(background.submit(request).model_dump())
            if name == RESULT_TOOL_NAME:
                body = EvalIdBody.model_validate(arguments)
                result = background.result(body.eval_id) or _replay_result(results_root, body.eval_id)
                return _tool_payload({"found": result is not None, "result": result.model_dump() if result else None})
            if name == FEEDBACK_TOOL_NAME:
                body = CandidateIdBody.model_validate(arguments)
                summary = promotion.feedback_summary(body.candidate_id)
                return _tool_payload({"found": summary is not None, "summary": summary})
            return _tool_error(f"Unknown tool: {name!r}. Available tools: {sorted(_tool_names())}")
        except ValidationError as exc:
            return _tool_error(exc.json())

    _mount_mcp(app, mcp_server)
    return app


def _config_from_args(argv: Optional[list[str]] = None) -> ServerConfig:
    parser = argparse.ArgumentParser(
        prog="python -m eval_service.server", description="Evaluation service over HTTP + MCP"
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config file; CLI flags override its values")
    for name, field in ServerConfig.model_fields.items():
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None, help=field.description)
    args = parser.parse_args(argv)

    data: dict[str, Any] = {}
    if args.config is not None:
        data.update(yaml.safe_load(args.config.read_text(encoding="utf-8")) or {})
    for name in ServerConfig.model_fields:
        value = getattr(args, name)
        if value is not None:
            data[name] = value
    return ServerConfig.model_validate(data)


def main() -> None:
    config = _config_from_args()
    layout = prepare_runtime(
        repo_root=config.repo_root,
        overlay_root=config.overlay_root,
        workspace_root=config.workspace_root,
        data_root=config.data_root,
        require_overlay=True,
        verify_trees=True,
    )
    app = build_app(config, layout=layout)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
