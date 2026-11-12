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
"""dsh agent harness: wraps the dsh CLI (headless profile) as a Gym Responses-API agent.

Each rollout boots `dsh --profile headless "<task>"` as a one-shot subprocess — the same
pattern claude_code_agent uses for `claude -p`. dsh owns its model credentials and LLM
calls (no Gym model_server is wired), so token usage is not attributed through Gym.
With `workspace_root` set, each run gets its own cwd; dsh then persists a per-cwd
session journal (`<DSH_HOME|~/.dsh>/sessions/<cwd-slug>/session-*/session.v*.jsonl.zstd`)
whose `assistant/message` records carry `usage.{inputTokens, outputTokens,
totalTokens, cacheReadTokens}` — parsed and summed into the response usage.

When the resources server's seed response advertises MCP tools (nemo-gym
mcp_auto_exposure, key ``mcp``), the harness writes a dsh ``--patch`` overlay
mounting ``@deepseek-ai/dsh-mcp-client`` (streamable-http) with the per-rollout
session token — dsh's tool calls then land in the same env session that /verify
scores. Requires the mcp-client package installed into the dsh profile's home.
"""

import asyncio
import json
import logging
import os
import signal
import tempfile
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    NEMO_GYM_MCP_METADATA_KEY,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.dsh_agent.session_usage import collect_dsh_session_usage


LOG = logging.getLogger(__name__)


def resolve_dsh_workspace_root(
    configured: str | None,
    *,
    environ: dict[str, str] | None = None,
    cwd: Path | None = None,
    temp_root: Path | None = None,
) -> Path:
    """Resolve a writable DSH cwd without ever falling back to a component root."""
    env = os.environ if environ is None else environ
    workspace = env.get("AGENT_GYM_WORKSPACE")
    value = configured or workspace

    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            root = Path(workspace).expanduser() / root if workspace else (cwd or Path.cwd()) / root
    elif value:
        root = Path(value).expanduser() / "dsh"
    else:
        root = (temp_root or Path(tempfile.gettempdir())) / "agent-gym-dsh"

    resolved = root.resolve(strict=False)
    overlay = env.get("AGENT_GYM_OVERLAY")
    if overlay:
        overlay_root = Path(overlay).expanduser().resolve(strict=False)
        if resolved == overlay_root or overlay_root in resolved.parents:
            raise RuntimeError(f"DSH workspace must not be inside the read-only overlay: {resolved}")
    return resolved


def _extract_instruction(body_input, instructions: str | None = None) -> str:
    """Preserve text instructions and message roles in the one-shot DSH task."""
    sections = [f"Instructions:\n{instructions}"] if instructions else []
    for item in body_input:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role not in {"system", "developer", "user", "assistant"}:
            continue
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if isinstance(content, list):
            content = "".join(
                (part.get("text", "") if isinstance(part, dict) else getattr(part, "text", ""))
                for part in content
            )
        if content:
            sections.append(f"{role.title()}:\n{content}")
    return "\n\n".join(sections)


class DshAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    dsh_command: str = "dsh"
    profile: str = "headless"
    concurrency: int = 4
    timeout: int = 600
    # When set, each rollout runs dsh in its own `<workspace_root>/dsh_<hex>` cwd —
    # required for session-journal token attribution (dsh keys journals by cwd).
    workspace_root: Optional[str] = None
    # dsh patch layer id for the per-rollout Gym MCP connection (insert patch).
    mcp_patch_entry_id: str = "mcp-gym"


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill dsh and the sandboxed tool processes it spawned.

    DSH launches descendants such as bwrap and python. Signalling only the
    direct child can leave those descendants running after a rollout timeout.
    """
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass


class DshAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class DshAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class DshAgent(SimpleResponsesAPIAgent):
    config: DshAgentConfig
    sem: asyncio.Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)

    async def _run_dsh(self, instruction: str, cwd: Optional[Path] = None,
                       extra_args: Optional[list[str]] = None) -> str:
        argv = [self.config.dsh_command, "--profile", self.config.profile, *(extra_args or []), instruction]
        spawn_kwargs = {"start_new_session": True} if os.name == "posix" else {}
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            **spawn_kwargs,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            _kill_process_tree(proc)
            await proc.communicate()
            raise RuntimeError(f"dsh did not finish within {self.config.timeout}s")
        except asyncio.CancelledError:
            _kill_process_tree(proc)
            await proc.communicate()
            raise
        if proc.returncode != 0:
            tail = stderr.decode(errors="replace")[-500:]
            raise RuntimeError(f"dsh exited with code {proc.returncode}: {tail}")
        answer = stdout.decode(errors="replace").strip()
        if not answer:
            raise RuntimeError("dsh exited successfully without an answer")
        return answer

    def _workspace_dir(self) -> Path:
        root = resolve_dsh_workspace_root(self.config.workspace_root) / f"dsh_{uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _resources_server_base_url(self) -> str:
        cfg = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.resources_server.name,
        )
        return self.server_client._build_server_base_url(cfg)

    def _mcp_patch_document(self, seed_response_json: dict[str, Any]) -> Optional[dict]:
        """Turn the seed response's ``mcp`` metadata into a dsh ``--patch`` insert.

        The token minted by /seed_session encodes the harness's session id, so
        tool calls dsh makes over this connection land in the same session that
        /verify scores.
        """
        metadata = seed_response_json.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict):
            return None
        headers = metadata.get("headers")
        if not isinstance(headers, dict) or not headers:
            LOG.warning(
                "MCP seed metadata for %r has no headers; the tool endpoint will be called "
                "without a session token and will reject the calls.",
                self.config.resources_server.name,
            )
            headers = {}
        url_path = str(metadata.get("url_path") or "/mcp")
        url = f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}"
        return [{
            "insert": [{
                "id": self.config.mcp_patch_entry_id,
                "name": "@deepseek-ai/dsh-mcp-client",
                "config": {
                    "serverName": str(metadata.get("server_name") or self.config.resources_server.name),
                    "transport": "streamable-http",
                    "url": url,
                    "headers": {str(key): str(value) for key, value in headers.items()},
                    "toolCallTimeoutMs": 60000,
                },
            }]
        }]

    def _write_mcp_patch(self, seed_response_json: dict[str, Any], workdir: Path) -> Optional[Path]:
        document = self._mcp_patch_document(seed_response_json)
        if document is None:
            return None
        patch_path = workdir / "gym_mcp_patch.yml"
        patch_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return patch_path

    def _collect_usage(self, cwd: Optional[Path], started_at: float) -> Optional[dict]:
        if cwd is None:
            return None
        dsh_home = os.environ.get("DSH_HOME") or str(Path.home() / ".dsh")
        sessions_root = Path(dsh_home).expanduser() / "sessions"
        usage = collect_dsh_session_usage(sessions_root, cwd, started_at)
        if usage is None:
            LOG.warning(
                "no usable dsh session usage found under %s for cwd %s; reporting zero tokens",
                sessions_root,
                cwd,
            )
        return usage

    def _response_from_text(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        text: str,
        *,
        status: str = "completed",
        usage_stats: Optional[dict] = None,
    ) -> NeMoGymResponse:
        usage_stats = usage_stats or {}
        output_item = NeMoGymResponseOutputMessage(
            id=f"msg_{uuid4().hex}",
            content=[NeMoGymResponseOutputText(text=text, annotations=[])],
            role="assistant",
            status=status,
            type="message",
        )
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=f"dsh/{self.config.profile}",
            object="response",
            output=[output_item],
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=int(usage_stats.get("input_tokens", 0)),
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cached_tokens=int(usage_stats.get("cached_tokens", 0))
                ),
                output_tokens=int(usage_stats.get("output_tokens", 0)),
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=int(usage_stats.get("total_tokens", 0)),
            ),
        )

    @staticmethod
    def _initial_observation_text(seed_response_json: dict[str, Any]) -> Optional[str]:
        """The seed response's initial observation (synth_env ``initial.obs.text``).

        dsh is one-shot: without this, observation-heavy environments (office
        profile) never show the agent the workspace state at all.
        """
        initial = seed_response_json.get("initial")
        if not isinstance(initial, dict):
            return None
        obs = initial.get("obs")
        if not isinstance(obs, dict):
            return None
        text = obs.get("text")
        return text if isinstance(text, str) and text.strip() else None

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        rollout_id: Optional[str] = None,
        seed_response_json: Optional[dict] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        instruction = _extract_instruction(body.input, body.instructions)
        if seed_response_json:
            observation = self._initial_observation_text(seed_response_json)
            if observation:
                instruction = f"{instruction}\n\nInitial observation:\n{observation}"
        started_at = time()
        workdir = self._workspace_dir()
        patch_path = self._write_mcp_patch(seed_response_json or {}, workdir) if seed_response_json else None
        extra_args = ["--patch", str(patch_path)] if patch_path else []
        text = await self._run_dsh(instruction, cwd=workdir, extra_args=extra_args)
        usage_stats = self._collect_usage(workdir, started_at) or {}
        return self._response_from_text(body, text, status="completed", usage_stats=usage_stats)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        return await self._create_response(body, rollout_id=request.path_params.get("rollout_id"))

    async def run(self, request: Request, body: DshAgentRunRequest) -> DshAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            # Environments with extra="forbid" seed schemas (e.g. synth_env) reject
            # the full run body; pass only the fields a seed request may carry.
            seed_payload = {
                key: value
                for key, value in body.model_dump(mode="json").items()
                if key in {"task_id", "env_release", "max_steps", "seed", "fidelity", "budgets"}
            }
            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=seed_payload,
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies
            seed_response_json = await get_response_json(seed_resp)

            # A CLI crash or timeout is an execution failure, not a wrong answer.
            # Propagate it so Gym records a failed rollout rather than scoring
            # an empty assistant message as a valid zero-reward attempt.
            agent_resp = await self._create_response(
                body.responses_create_params, seed_response_json=seed_response_json
            )
            agent_resp_json = agent_resp.model_dump(mode="json")

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            return DshAgentVerifyResponse.model_validate(verify_json | {"turns_used": 1, "finished_naturally": True})


if __name__ == "__main__":
    DshAgent.run_webserver()
