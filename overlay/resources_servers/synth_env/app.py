# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SynthEnvServer — the synthetic environment host (design C-SE-1 / C-SE-2).

A ``SimpleResourcesServer`` subclass: the inherited ``/seed_session`` session
cookie, traced ``/verify``, ``/aggregate_metrics`` and MCP auto-exposure are
kept untouched; the environment routes (``/reset`` ``/observe`` ``/step``
``/bash``) are appended in ``setup_webserver``. The environment is modal/
dynamics pluggable (profiles/dynamics), with M1 shipping the terminal profile
over replay dynamics only — zero LLM anywhere on the hot path.

MCP (design §3.11/决议 9): with ``expose_tools_over_mcp: true`` the typed POST
routes become ``/mcp`` tools; ``/bash`` gives the action its profile-semantic
name. ``/admin/*`` routes are filtered out of exposure by ``mcp_tools`` — the
strategy side can never see them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, create_model

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.synth_env import admin_api, host_api
from resources_servers.synth_env.package_loader import PackageStore, load_package_zip


class SynthEnvServerConfig(BaseResourcesServerConfig):
    # Asset package zips loaded at startup (logical paths resolved via component
    # search roots: extra roots > cwd > install root); startup load failure
    # aborts boot — a broken package must never serve.
    package_paths: list[str] = []
    # Default TTL for /admin/sessions/reap (design §3.13.5).
    session_ttl_s: float = 3600.0
    # Optional JSONL sink for the admin journal (M1 default: in-memory only);
    # relative paths resolve under the writable workspace.
    journal_fpath: Optional[str] = None
    # Optional JSONL sink for llm_sim call traces (design §3.2 漂移可归因);
    # resolved like the journal — writable workspace only, never the overlay.
    llm_sim_trace_fpath: Optional[str] = None
    # Simulator model ref for the office profile's caged llm_sim (决议 3) —
    # strictly separate from the policy model; office packages may pin the same
    # name in their manifest, and seed-time validation rejects mismatches.
    llm_sim_model_server: Optional[ModelServerRef] = None

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED


def _resolve_workspace_sink(configured: Optional[str]) -> Optional[str]:
    """Resolve a JSONL sink (journal / llm_sim trace) under the writable workspace.

    Mirrors ``resolve_dsh_workspace_root`` (dsh_agent): relative paths anchor at
    ``AGENT_GYM_WORKSPACE`` (falling back to cwd outside the launcher), and any
    sink — relative or absolute — inside the read-only overlay is rejected.
    """
    if configured is None:
        return configured
    if Path(configured).is_absolute():
        resolved = Path(configured).resolve(strict=False)
    else:
        workspace = os.environ.get("AGENT_GYM_WORKSPACE")
        root = Path(workspace).expanduser() if workspace else Path.cwd()
        resolved = (root / configured).resolve(strict=False)
    overlay = os.environ.get("AGENT_GYM_OVERLAY")
    if overlay:
        overlay_root = Path(overlay).expanduser().resolve(strict=False)
        if resolved == overlay_root or overlay_root in resolved.parents:
            raise RuntimeError(f"journal must not be written inside the read-only overlay: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved)


class SynthEnvServer(SimpleResourcesServer):
    config: SynthEnvServerConfig

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        self._store = PackageStore()
        self._sessions = {}
        self._journal = admin_api.Journal(jsonl_fpath=_resolve_workspace_sink(self.config.journal_fpath))
        self._llm_sim_trace_fpath = _resolve_workspace_sink(self.config.llm_sim_trace_fpath)
        self._draining = False
        for raw_path in self.config.package_paths:
            path = Path(raw_path)
            if not path.is_absolute():
                path = _resolve_under_cwd_or_install(raw_path)
                if not path.is_file():
                    raise FileNotFoundError(f"package not found under component search roots: {raw_path}")
            self._store.load(load_package_zip(path.read_bytes()))

    def setup_webserver(self) -> FastAPI:
        # super() registers /seed_session (session cookie), traced /verify,
        # /aggregate_metrics and /reverify_mode — none of that is re-registered here.
        app = super().setup_webserver()
        app.post("/reset")(self.reset)
        app.get("/observe")(self.observe)
        app.post("/step")(self.step)
        # Profile-semantic action name: becomes the MCP tool "bash" (§3.11).
        app.post("/bash")(self.bash)
        # Office profile: one typed route per declared action (§3.11 — the tool
        # name is the profile semantics: submit_ticket, approve, …).
        for action_name, model in self._office_action_models().items():
            app.post(f"/{action_name}")(self._make_action_route(action_name, model))
        admin_api.register_admin_routes(app, self)
        return app

    _RESERVED_ROUTES = frozenset(
        {"seed_session", "verify", "aggregate_metrics", "reverify_mode", "reset", "observe", "step", "bash", "mcp"}
    )

    def _office_action_models(self) -> dict[str, type[BaseModel]]:
        """Typed request models for every action declared by loaded office packages."""
        models: dict[str, type[BaseModel]] = {}
        for package in self._store.packages():
            if package.manifest.profile != "office":
                continue
            for definition in package.env_spec.actions or []:
                name = definition.get("name")
                if not name or name in self._RESERVED_ROUTES:
                    continue
                parameters = definition.get("parameters") or {}
                required = set(parameters.get("required") or [])
                fields: dict[str, tuple[Any, Any]] = {}
                for arg_name, spec in (parameters.get("properties") or {}).items():
                    annotation: Any = {"string": str, "number": float, "boolean": bool}.get(spec.get("type"), str)
                    if spec.get("type") == "enum" and spec.get("enum"):
                        annotation = Literal[tuple(spec["enum"])]  # type: ignore[valid-type]
                    description = spec.get("description")
                    if arg_name in required:
                        fields[arg_name] = (annotation, Field(default=..., description=description))
                    else:
                        fields[arg_name] = (Optional[annotation], Field(default=None, description=description))
                models[name] = create_model(f"{name}_request", **fields)
        return models

    def _make_action_route(self, action_name: str, model: type[BaseModel]):
        server = self

        async def action_route(request, body):
            return await server.step(
                request, host_api.SynthStepRequest(action={"name": action_name, "args": body.model_dump()})
            )

        # __annotations__ set as live objects: the module uses postponed
        # annotations, and FastAPI must resolve the dynamic model by reference.
        action_route.__name__ = action_name
        action_route.__annotations__ = {
            "request": Request,
            "body": model,
            "return": host_api.SynthStepResponse,
        }
        return action_route

    # ---- MCP (决议 9: /admin/* never exposed to the strategy side) -------------

    def mcp_tools(self, harvested, catchall):
        return [tool for tool in harvested if not (tool.path or "").startswith("/admin")]

    # ---- /aggregate_metrics contract slot (§3.13: 填充既有契约槽) ----------------

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """synth_env observability aggregates over the scored verify responses.

        Masked samples are excluded, mirroring the wheel's scored-subset
        discipline for quality metrics.
        """
        rows = [row for group in tasks for row in group if not row.get("mask_sample")]
        if not rows:
            return {}

        def mean(key: str) -> float:
            return round(sum(row.get(key, 0) for row in rows) / len(rows), 4)

        def rate(key: str) -> float:
            return round(sum(1 for row in rows if row.get(key)) / len(rows), 4)

        return {
            "synth_env_violations_mean": mean("violations"),
            "synth_env_steps_mean": mean("steps"),
            "synth_env_truncated_rate": rate("truncated"),
            "synth_env_budget_exceeded_rate": rate("budget_exceeded"),
            "synth_env_llm_sim_calls_mean": mean("llm_sim_calls"),
        }

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in agent_metrics.items() if key.startswith("synth_env_")}

    # ---- C-SE-1 routes ---------------------------------------------------------

    async def seed_session(self, request: Request, body: host_api.SynthSeedRequest) -> host_api.SynthSeedResponse:
        return host_api.handle_seed(self, request, body)

    async def reset(self, request: Request) -> host_api.SynthInitialObs:
        return host_api.handle_reset(self, request)

    async def observe(self, request: Request) -> host_api.SynthObserveResponse:
        return host_api.handle_observe(self, request)

    async def step(self, request: Request, body: host_api.SynthStepRequest) -> host_api.SynthStepResponse:
        return await host_api.handle_step(self, request, body)

    async def bash(self, request: Request, body: host_api.BashActionRequest) -> host_api.SynthStepResponse:
        return await self.step(
            request, host_api.SynthStepRequest(action={"name": "bash", "args": {"command": body.command}})
        )

    async def verify(self, request: Request, body: BaseVerifyRequest) -> host_api.SynthVerifyResponse:
        return host_api.handle_verify(self, request, body)

    # ---- C-SE-2 admin routes (handlers live in admin_api.py) -------------------

    admin_list_packages = admin_api.admin_list_packages
    admin_package_detail = admin_api.admin_package_detail
    admin_load_package = admin_api.admin_load_package
    admin_unload_package = admin_api.admin_unload_package
    admin_list_sessions = admin_api.admin_list_sessions
    admin_session_detail = admin_api.admin_session_detail
    admin_terminate_session = admin_api.admin_terminate_session
    admin_reap_sessions = admin_api.admin_reap_sessions
    admin_metrics = admin_api.admin_metrics
    admin_drain = admin_api.admin_drain


if __name__ == "__main__":
    SynthEnvServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SynthEnvServer.run_webserver()  # noqa: F401
