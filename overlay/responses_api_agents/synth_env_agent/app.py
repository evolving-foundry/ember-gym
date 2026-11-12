# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SynthEnvAgent — the policy side of the synthetic environment (design §3.7).

A ``SimpleResponsesAPIAgent`` running a self-held loop over the C-SE-1 host
(v3 pattern): seed -> [model -> parse action -> step]* -> verify. The loop is
the evaluated object; the environment enforces its own budgets, so the agent
never gates anything on the environment's behalf.

Failure discipline (失败不静默, τ² pattern): a rollout that cannot proceed is
converted into a failure response tagged with ``_ng_failure_class`` —
``synth_env_malformed_action`` (after the configured retry budget),
``synth_env_budget_exceeded`` (environment forces done, 决议 9), or
``synth_env_api_error`` — and collection continues.

The simulator/model references are config-only (决议 3): ``model_server`` and
``resources_server`` are refs; no model name is hard-coded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from time import time
from typing import Any, Optional
from uuid import uuid4

from aiohttp import ClientTimeout
from fastapi import Body, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY
from nemo_gym.server_utils import (
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from resources_servers.synth_env.host_api import (
    SYNTH_ENV_BUDGET_EXCEEDED_FAILURE_CLASS,
    SYNTH_ENV_MALFORMED_ACTION_FAILURE_CLASS,
)
from responses_api_agents.synth_env_agent.env_client import SynthEnvApiError, SynthEnvClient


LOG = logging.getLogger(__name__)

SYNTH_ENV_API_FAILURE_CLASS = "synth_env_api_error"

_RESERVED_RESULT_KEYS = ("reward", "response", "error", NG_FAILURE_CLASS_KEY)

_BASH_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "One shell command to run in the synthetic terminal."},
    },
    "required": ["command"],
}


class SynthEnvMalformedActionError(ValueError):
    """A model response that cannot be turned into a legal bash action."""


class SynthEnvBudgetExceeded(RuntimeError):
    """The environment forced done because a session budget was exhausted (决议 9)."""


class SynthEnvAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 2
    request_timeout_s: float = 120.0
    # Rolling window of (command, observation) pairs re-sent to the model.
    history_window: int = Field(default=6, ge=1)
    temperature: float = 0.0
    max_output_tokens: int = 700
    malformed_action_max_retries: int = Field(default=0, ge=0)
    use_tool_call: bool = True
    # Fallback step budget when the dataset row has no max_steps; the
    # environment enforces the hard cap either way (双保险).
    default_max_steps: int = 40
    # Used when a dataset row carries no env_release pin of its own.
    default_env_release: Optional[dict[str, Any]] = None


class SynthEnvAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str
    goal: str = ""
    env_release: Optional[dict[str, Any]] = None
    max_steps: Optional[int] = None
    seed: int = 0


class SynthEnvVerifyResponse(BaseVerifyResponse):
    task_id: str
    steps: int
    truncated: bool
    budget_exceeded: bool
    violations: int
    reward_components: dict[str, float]


class SynthEnvAgentFailureResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise SynthEnvMalformedActionError("model response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return ""


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise SynthEnvMalformedActionError(f"model text contains no JSON object: {text[:200]!r}")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise SynthEnvMalformedActionError(f"model text JSON is invalid: {error}") from error
    if not isinstance(parsed, dict):
        raise SynthEnvMalformedActionError(f"model JSON must be an object, got {type(parsed).__name__}")
    return parsed


def _parse_action(response: dict[str, Any], allowed_names: Optional[set[str]] = None) -> dict[str, Any]:
    """Parse a chat completion into ``{"name": ..., "args": {...}}``.

    ``allowed_names`` is the environment's declared action set (seed response
    ``actions_available``); ``None`` accepts any declared-shape action. The
    legacy single-bash shape (``{"command": ...}`` text or a bash tool call)
    stays valid for terminal packages.
    """
    choices = response.get("choices") or []
    if not choices:
        raise SynthEnvMalformedActionError("model response has no choices")
    message = choices[0].get("message") or {}

    tool_calls = message.get("tool_calls")
    if tool_calls:
        if not isinstance(tool_calls, list):
            raise SynthEnvMalformedActionError("tool_calls must be a list")
        tool_call = tool_calls[0]
        function = tool_call.get("function") or {} if isinstance(tool_call, dict) else {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise SynthEnvMalformedActionError("tool call function.name must be a non-empty string")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise SynthEnvMalformedActionError("tool call function.arguments must be a JSON string")
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise SynthEnvMalformedActionError("tool call arguments are invalid JSON: " + str(error)) from error
        if not isinstance(parsed, dict):
            raise SynthEnvMalformedActionError("tool call arguments must decode to an object")
        args = parsed
    else:
        parsed = _extract_json_object(_message_text(response))
        if "name" in parsed and "args" in parsed:
            name, args = parsed["name"], parsed.get("args")
            if not isinstance(args, dict):
                raise SynthEnvMalformedActionError("action 'args' must be an object")
        else:
            # Legacy text shape: a bare {"command": ...} means the bash tool.
            name, args = "bash", parsed

    if allowed_names is None:
        allowed_names = {"bash"}
    if name not in allowed_names:
        if allowed_names == {"bash"}:
            raise SynthEnvMalformedActionError(f"unexpected tool call {name!r}, expected bash")
        raise SynthEnvMalformedActionError(f"unexpected action {name!r}, expected one of {sorted(allowed_names)}")
    if name == "bash" and (not isinstance(args.get("command"), str) or not args["command"].strip()):
        raise SynthEnvMalformedActionError("bash action requires a non-empty 'command' string")
    if name != "bash" and not isinstance(args, dict):
        raise SynthEnvMalformedActionError("action args must be an object")
    return {"name": name, "args": args}


class SynthEnvAgent(SimpleResponsesAPIAgent):
    config: SynthEnvAgentConfig
    sem: asyncio.Semaphore | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise NotImplementedError("SynthEnvAgent executes complete episodes through /run")

    async def run(
        self, request: Request, body: SynthEnvAgentRunRequest
    ) -> SynthEnvVerifyResponse | SynthEnvAgentFailureResponse:
        record = self._sanitized_record(body)
        try:
            return await self._run(body)
        except SynthEnvMalformedActionError as error:
            return self._failure_response(body, record, SYNTH_ENV_MALFORMED_ACTION_FAILURE_CLASS, str(error))
        except SynthEnvBudgetExceeded as error:
            return self._failure_response(body, record, SYNTH_ENV_BUDGET_EXCEEDED_FAILURE_CLASS, str(error))
        except Exception as error:  # noqa: BLE001 -- one bad rollout must not abort the full collection
            detail = error
            for nested in getattr(error, "exceptions", ()) or ():
                detail = nested
                break
            return self._failure_response(
                body, record, SYNTH_ENV_API_FAILURE_CLASS, f"{type(detail).__name__}: {detail}"
            )

    async def _run(self, body: SynthEnvAgentRunRequest) -> SynthEnvVerifyResponse:
        env_release = body.env_release or self.config.default_env_release
        if env_release is None:
            raise SynthEnvApiError(
                "no env_release pin: the dataset row and the agent config's default_env_release are both unset"
            )
        env_release = dict(env_release)
        max_steps = body.max_steps if body.max_steps is not None else self.config.default_max_steps
        system_prompt, user_instructions = self._static_prompts(body)

        assert self.sem is not None
        async with self.sem:
            client = SynthEnvClient(
                server_client=self.server_client,
                server_name=self.config.resources_server.name,
                timeout_s=self.config.request_timeout_s,
            )
            session = await client.seed_session(
                {
                    "task_id": body.task_id,
                    "env_release": env_release,
                    "max_steps": max_steps,
                    "seed": body.seed,
                    "fidelity": {},
                    "budgets": {},
                }
            )
            initial = session.get("initial") or {}
            actions_available = initial.get("actions_available") or ["bash"]
            # Mirror the environment's declared action set (§3.11): office packages
            # ship a multi-tool contract; terminal packages fall back to bash.
            tool_schemas = initial.get("action_schemas") or [
                {"name": "bash", "description": "Run exactly one command in the synthetic terminal.",
                 "parameters": _BASH_TOOL_PARAMETERS}
            ]
            obs = initial.get("obs") or {}

            usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
            trajectory: list[str] = []
            history: deque[tuple[str, str]] = deque(maxlen=self.config.history_window)
            steps = malformed_actions = 0
            goal = body.goal

            while steps < max_steps:
                attempts = self.config.malformed_action_max_retries + 1
                action: Optional[dict[str, Any]] = None
                for attempt in range(attempts):
                    response = await self._call_model(
                        body=body,
                        goal=goal,
                        obs=obs,
                        history=history,
                        steps=steps,
                        max_steps=max_steps,
                        actions_available=actions_available,
                        system_prompt=system_prompt,
                        user_instructions=user_instructions,
                        tool_schemas=tool_schemas,
                    )
                    usage = response.get("usage") or {}
                    for usage_key in usage_totals:
                        usage_totals[usage_key] += int(usage.get(usage_key, 0) or 0)
                    try:
                        action = _parse_action(response, allowed_names=set(actions_available))
                        break
                    except SynthEnvMalformedActionError:
                        malformed_actions += 1
                        if attempt == attempts - 1:
                            raise
                        LOG.warning(
                            "malformed action (attempt %d/%d) for task %s",
                            attempt + 1,
                            attempts,
                            body.task_id,
                        )
                assert action is not None

                result = await client.step(action)
                steps += 1
                action_repr = f"{action['name']}({json.dumps(action['args'], ensure_ascii=False)})"
                obs = result.get("obs") or {}
                history.append((action_repr, obs.get("text", "")))
                trajectory.append(f"{steps}. -> {action_repr}")
                if result.get("done"):
                    failure_class = (result.get("info") or {}).get("failure_class")
                    if failure_class == SYNTH_ENV_BUDGET_EXCEEDED_FAILURE_CLASS:
                        raise SynthEnvBudgetExceeded(
                            f"environment forced done at step {steps}: {failure_class}"
                        )
                    break

            verify_payload = body.model_dump(mode="json") | {
                "response": self._response_object(body, trajectory, usage_totals).model_dump(mode="json")
            }
            verify = await client.verify(verify_payload)
        return SynthEnvVerifyResponse.model_validate(verify)

    async def _call_model(
        self,
        *,
        body: SynthEnvAgentRunRequest,
        goal: str,
        obs: dict[str, Any],
        history: deque[tuple[str, str]],
        steps: int,
        max_steps: int,
        actions_available: list[str],
        system_prompt: str,
        user_instructions: str,
        tool_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        obs_text = obs.get("text", "") if isinstance(obs, dict) else str(obs)
        channel = obs.get("channel", "text") if isinstance(obs, dict) else "text"
        status = (
            f"Current status:\n"
            f"- task: {goal or '(see task prompt)'}\n"
            f"- steps used: {steps}/{max_steps}\n"
            f"- available actions: {actions_available}\n"
            f"- observation channel: {channel}"
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{user_instructions}\n\nTask: {goal or '(see task prompt)'}\n\n"
                    "Action history so far:\n"
                    + ("\n".join(f"> {action}\n{output}" for action, output in history) if history else "(none)")
                ),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"{status}\n\nLatest observation:\n{obs_text}\n\nChoose the next action.",
            }
        )

        payload: dict[str, Any] = {
            "model": str(self.config.model_server.name),
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if self.config.use_tool_call:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": schema.get("name"),
                        "description": schema.get("description", ""),
                        "parameters": schema.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
                for schema in tool_schemas
            ]
            if len(payload["tools"]) == 1:
                payload["tool_choice"] = {"type": "function", "function": {"name": payload["tools"][0]["function"]["name"]}}
            else:
                payload["tool_choice"] = "auto"

        path = self.url_path_for_run("/v1/chat/completions", body)
        response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=path,
            json=payload,
            timeout=ClientTimeout(total=self.config.request_timeout_s),
        )
        await raise_for_status(response)
        return await get_response_json(response)

    def _static_prompts(self, body: SynthEnvAgentRunRequest) -> tuple[str, str]:
        """Use the dataset prompt_config when provided, else built-in defaults."""
        system_prompt = "You are an operator working inside a synthetic terminal environment."
        user_instructions = (
            "Use the bash tool, one command at a time. Observations are command output "
            "plus a file-change summary. Never assume an unverified command succeeded."
        )
        input_items = getattr(body.responses_create_params, "input", None)
        if isinstance(input_items, str):
            user_instructions = input_items
        elif isinstance(input_items, list):
            for item in input_items:
                if isinstance(item, dict):
                    role, content = item.get("role"), item.get("content")
                elif hasattr(item, "role"):
                    role, content = item.role, item.content
                else:
                    continue
                text = (
                    content
                    if isinstance(content, str)
                    else (
                        "\n".join(
                            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                            for part in content
                        )
                        if isinstance(content, list)
                        else ""
                    )
                )
                if not text:
                    continue
                if role == "system":
                    system_prompt = text
                elif role == "user":
                    user_instructions = text
        return system_prompt, user_instructions

    def _response_object(
        self, body: SynthEnvAgentRunRequest, trajectory: list[str], usage_totals: dict[str, int]
    ) -> NeMoGymResponse:
        output_items = [
            NeMoGymResponseOutputMessage(
                id=f"msg_{uuid4().hex}",
                content=[NeMoGymResponseOutputText(type="output_text", text="\n".join(trajectory), annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )
        ]
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=f"synth_env/{self.config.model_server.name}",
            object="response",
            output=output_items,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
            usage=NeMoGymResponseUsage(
                input_tokens=usage_totals["prompt_tokens"],
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=usage_totals["cached_tokens"]),
                output_tokens=usage_totals["completion_tokens"],
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=usage_totals["total_tokens"],
            ),
        )

    def _sanitized_record(self, body: SynthEnvAgentRunRequest) -> dict[str, Any]:
        record = body.model_dump(mode="json")
        for key in _RESERVED_RESULT_KEYS:
            record.pop(key, None)
        return record

    def _failure_response(
        self, body: SynthEnvAgentRunRequest, record: dict[str, Any], failure_class: str, error: str
    ) -> SynthEnvAgentFailureResponse:
        print(f"[synth_env_agent] rollout failed for {body.task_id}: {error}", flush=True)
        return SynthEnvAgentFailureResponse.model_validate(
            record
            | {
                "reward": 0.0,
                "response": self._response_object(
                    body,
                    [],
                    {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")},
                ).model_dump(mode="json"),
                NG_FAILURE_CLASS_KEY: failure_class,
                "error": error[:500],
            }
        )


if __name__ == "__main__":
    SynthEnvAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SynthEnvAgent.run_webserver()  # noqa: F401
