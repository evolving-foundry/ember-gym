# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym-native BrowserGym runner for deterministic browser benchmark tasks.

The selected BrowserGym task owns its evaluator. This agent only runs the
BrowserGym observation/action loop and returns the reward emitted by the task.
For WebArena-Verified, BrowserGym invokes the official typed-response/network
trace evaluator; this adapter also persists a HAR and the final typed response
for official offline replay.
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field
from aiohttp import ClientTimeout

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
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


class BrowserGymAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef | None = None
    dsh_command: str | None = None
    dsh_profile: str = "headless"
    dsh_home: str | None = None
    dsh_workspace_root: str | None = None
    concurrency: int = 1
    max_steps: int = 50
    request_timeout_s: float = 180.0
    action_max_chars: int = 12000
    observation_max_chars: int = 50000
    temperature: float = 0.0
    har_dir: str = "results/browsergym_har"


class BrowserGymRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    task_name: str | None = None
    task_id: str | int | None = None
    seed: int | None = None
    intent_template_id: int | None = None
    revision: int | None = None


class BrowserGymVerifyResponse(BaseVerifyResponse):
    task_name: str
    task_id: str
    steps: int
    terminated: bool
    truncated: bool
    final_response: str | None = None
    har_path: str | None = None
    evaluator_info: dict[str, Any] = Field(default_factory=dict)


def _task_name(body: BrowserGymRunRequest) -> str:
    if body.task_name:
        return body.task_name.removeprefix("browsergym/")
    if body.intent_template_id is not None and body.task_id is not None and body.revision is not None:
        return f"webarena_verified.{body.intent_template_id}.{body.task_id}.{body.revision}"
    raise ValueError("dataset row must provide task_name or WebArena intent_template_id/task_id/revision")


def _observation_text(obs: dict[str, Any], limit: int) -> str:
    preferred = (
        "goal",
        "url",
        "open_pages_urls",
        "open_pages_titles",
        "axtree_txt",
        "dom_txt",
        "last_action",
        "last_action_error",
    )
    parts: list[str] = []
    remaining = limit
    for key in preferred:
        value = obs.get(key)
        if value is None:
            continue
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        section = f"[{key}]\n{rendered}"
        # Keep the goal and the live browser context. Large DOM snapshots are
        # truncated from their tail so the task instruction isn't lost.
        if len(section) > remaining:
            if not parts:
                section = section[:remaining]
            elif key in {"axtree_txt", "dom_txt"}:
                section = section[:remaining]
            else:
                section = section[:remaining]
        parts.append(section)
        remaining -= len(section) + 2
        if remaining <= 0:
            break
    if not any(key in obs for key in ("axtree_txt", "dom_txt")):
        for key in ("axtree_object", "dom_object"):
            if key in obs and remaining > 0:
                section = f"[{key}]\n{json.dumps(obs[key], ensure_ascii=False, default=str)}"
                parts.append(section[:remaining])
                remaining -= len(section) + 2
                if remaining <= 0:
                    break
    return "\n\n".join(parts)


def _parse_action(text: str, limit: int) -> str:
    action = text.strip()
    fenced = re.fullmatch(r"```(?:python|javascript|text)?\s*(.*?)\s*```", action, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        action = fenced.group(1).strip()
    if not action:
        raise ValueError("model returned an empty BrowserGym action")
    if len(action) > limit:
        raise ValueError(f"model action exceeded the configured {limit}-character limit")
    return action


def _normalize_assistantbench_answer(action: str) -> str:
    """Unwrap a scalar answer that the candidate put in a JSON envelope."""
    try:
        expression = ast.parse(action, mode="eval").body
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "send_msg_to_user"
            and len(expression.args) == 1
        ):
            return action
        value = ast.literal_eval(expression.args[0])
        if not isinstance(value, str):
            return action
        envelope = json.loads(value)
        if not isinstance(envelope, dict) or set(envelope) != {"answer"}:
            return action
        answer = envelope["answer"]
        if isinstance(answer, (str, int, float)) and not isinstance(answer, bool):
            return f"send_msg_to_user({str(answer)!r})"
    except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return action


def _message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("model response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    raise ValueError("model response has no text content")


def _final_response_from_obs(obs: dict[str, Any]) -> str | None:
    for message in reversed(obs.get("chat_messages") or ()):
        if isinstance(message, dict) and message.get("role") == "assistant":
            for block in reversed(message.get("content") or ()):
                if isinstance(block, dict) and block.get("type") in {"text", "message"}:
                    text = block.get("text") or block.get("message")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
    return None


def _final_response_from_action(action: str) -> str | None:
    try:
        expression = ast.parse(action.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        return None
    if expression.func.id != "send_msg_to_user" or not expression.args:
        return None
    try:
        value = ast.literal_eval(expression.args[0])
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, str) else None


class BrowserGymAgent(SimpleResponsesAPIAgent):
    config: BrowserGymAgentConfig
    sem: asyncio.Semaphore | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if not self.config.model_server and not self.config.dsh_command:
            raise ValueError("BrowserGym requires either model_server or dsh_command")
        self.sem = asyncio.Semaphore(self.config.concurrency)

    async def _dsh_action(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        if not self.config.dsh_command or not self.config.dsh_workspace_root:
            raise ValueError("DSH BrowserGym mode requires dsh_command and dsh_workspace_root")
        prompt = "\n\n".join(f"{message['role'].title()}:\n{message['content']}" for message in messages)
        workspace = Path(self.config.dsh_workspace_root).expanduser().resolve() / f"dsh_{uuid4().hex[:8]}"
        workspace.mkdir(parents=True, exist_ok=True)
        started_at = time()
        proc = await asyncio.create_subprocess_exec(
            self.config.dsh_command,
            "--profile",
            self.config.dsh_profile,
            "-",
            cwd=str(workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")), timeout=self.config.request_timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise RuntimeError(f"dsh action did not finish within {self.config.request_timeout_s}s")
        if proc.returncode != 0:
            raise RuntimeError(f"dsh action exited with code {proc.returncode}: {stderr.decode(errors='replace')[-500:]}")
        text = re.sub(r"<think\b[^>]*>.*?</think>", "", stdout.decode(errors="replace"), flags=re.I | re.S)
        dsh_home = Path(self.config.dsh_home or os.environ.get("DSH_HOME") or Path.home() / ".dsh").expanduser()
        usage = collect_dsh_session_usage(dsh_home / "sessions", workspace, started_at) or {}
        return _parse_action(text, self.config.action_max_chars), {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cached_tokens": usage.get("cached_tokens", 0),
        }

    async def _next_action(
        self,
        *,
        task_goal: str,
        observation: str,
        history: list[dict[str, str]],
        run_body: BrowserGymRunRequest,
    ) -> tuple[str, dict[str, Any]]:
        system = (
            "You are an agent operating a BrowserGym browser task. At each turn return exactly one "
            "BrowserGym high-level action, such as click('bid'), fill('bid', 'value'), scroll(0, 500), "
            "or send_msg_to_user('...'). Use only visible observation evidence. Do not return prose or "
            "markdown fences. For WebArena-Verified, finish with send_msg_to_user containing a JSON "
            "string that follows the schema included in the task goal."
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-8:])
        messages.append(
            {
                "role": "user",
                "content": f"Task goal:\n{task_goal}\n\nCurrent observation:\n{observation}\n\nReturn the next action only.",
            }
        )
        if self.config.dsh_command:
            return await self._dsh_action(messages)
        assert self.config.model_server is not None
        payload = {
            "model": str(self.config.model_server.name),
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": 1200,
        }
        path = self.url_path_for_run("/v1/chat/completions", run_body)
        response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=path,
            json=payload,
            timeout=ClientTimeout(total=self.config.request_timeout_s),
        )
        await raise_for_status(response)
        result = await get_response_json(response)
        return _parse_action(_message_text(result), self.config.action_max_chars), result.get("usage") or {}

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise NotImplementedError("BrowserGymAgent executes complete task runs through /run")

    async def run(self, request: Request, body: BrowserGymRunRequest) -> BrowserGymVerifyResponse:
        import gymnasium as gym
        import browsergym.core  # noqa: F401  # register core environments

        task_name = _task_name(body)
        if task_name.startswith("assistantbench."):
            import browsergym.assistantbench  # noqa: F401  # register AssistantBench tasks
        elif task_name.startswith("webarena_verified."):
            import browsergym.webarena_verified  # noqa: F401  # register official WA-Verified tasks
        elif task_name.startswith("workarena."):
            import browsergym.workarena  # noqa: F401  # register official WorkArena tasks
        else:
            raise ValueError(f"Unsupported BrowserGym task: {task_name}")

        assert self.sem is not None
        async with self.sem:
            task_id = str(body.task_id if body.task_id is not None else task_name)
            run_id = uuid4().hex
            artifact_dir = Path(self.config.har_dir).expanduser().resolve() / run_id / task_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            har_path = artifact_dir / f"network_{task_id}.har"
            response_path = artifact_dir / f"agent_response_{task_id}.json"

            kwargs: dict[str, Any] = {"headless": True}
            if task_name.startswith("webarena_verified."):
                kwargs["pw_context_kwargs"] = {
                    "record_har_path": str(har_path),
                    "record_har_content": "embed",
                    "record_har_mode": "full",
                }
            env = None
            all_output: list[str] = []
            usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
            history: list[dict[str, str]] = []
            task_goal = ""
            reward = 0.0
            terminated = truncated = False
            steps = 0
            final_response: str | None = None
            evaluator_info: dict[str, Any] = {}
            try:
                # Playwright's synchronous API must stay on one OS thread for the
                # entire BrowserGym episode. Model requests remain async in the
                # serving loop while all environment calls use this single worker.
                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="browsergym") as env_executor:
                    env = await loop.run_in_executor(
                        env_executor,
                        lambda: gym.make(f"browsergym/{task_name}", **kwargs),
                    )
                    obs, _reset_info = await loop.run_in_executor(
                        env_executor, lambda: env.reset(seed=body.seed or 0)
                    )
                    task_goal = str(obs.get("goal") or "")
                    for step_index in range(self.config.max_steps):
                        observation = _observation_text(obs, self.config.observation_max_chars)
                        action, usage = await self._next_action(
                            task_goal=task_goal,
                            observation=observation,
                            history=history,
                            run_body=body,
                        )
                        if task_name.startswith("assistantbench."):
                            action = _normalize_assistantbench_answer(action)
                        all_output.append(action)
                        final_response = _final_response_from_action(action) or final_response
                        history.extend(
                            [
                                {"role": "user", "content": f"Observation:\n{observation}"},
                                {"role": "assistant", "content": action},
                            ]
                        )
                        for usage_key in usage_totals:
                            usage_totals[usage_key] += int(usage.get(usage_key, 0) or 0)
                        obs, reward, terminated, truncated, evaluator_info = await loop.run_in_executor(
                            env_executor, lambda: env.step(action)
                        )
                        steps = step_index + 1
                        final_response = _final_response_from_obs(obs) or final_response
                        if terminated or truncated:
                            break
                    else:
                        truncated = True
                    await loop.run_in_executor(env_executor, env.close)
                    env = None
                if task_name.startswith("webarena_verified.") and final_response:
                    response_path.write_text(final_response + "\n", encoding="utf-8")
            except Exception:
                LOG.exception("BrowserGym rollout failed for %s", task_name)
                raise
            finally:
                if env is not None:
                    try:
                        await asyncio.to_thread(env.close)
                    except Exception:
                        LOG.exception("Failed to close BrowserGym environment %s", task_name)

            if task_name.startswith("webarena_verified.") and not har_path.exists():
                LOG.warning("BrowserGym did not persist a HAR for task %s at %s", task_id, har_path)
            final_text = "\n".join(all_output)
            output_items = [
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=final_text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ]
            response_obj = NeMoGymResponse(
                id=f"resp_{uuid4().hex}",
                created_at=int(time()),
                model=(
                    f"browsergym/dsh/{self.config.dsh_profile}"
                    if self.config.dsh_command
                    else f"browsergym/{self.config.model_server.name}"
                ),
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
            result = body.model_dump()
            result.update(
                {
                    "response": response_obj,
                    "reward": float(reward or 0.0),
                    "task_name": task_name,
                    "task_id": task_id,
                    "steps": steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "final_response": final_response,
                    "har_path": str(har_path) if har_path.exists() else None,
                    "evaluator_info": evaluator_info
                    if isinstance(evaluator_info, dict)
                    else {"info": str(evaluator_info)},
                }
            )
            return BrowserGymVerifyResponse.model_validate(result)


if __name__ == "__main__":
    BrowserGymAgent.run_webserver()
