# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym-native ARC-AGI-3 game-loop agent.

ARC-AGI-3 is not a static dataset but a set of interactive turn-based games
served by the ARC Prize API. This agent is self-contained (BrowserGym pattern):
it owns the RESET -> frame -> model -> action loop, enforces the per-game
action budget, and returns the reward the environment-side scorecard reports.

Design contract (docs/ARC-AGI_v3_基准接入设计_v0.1.md):
- Rewards aggregate the RAW upstream score; the trace also keeps per-level
  scores/actions so the M4 efficiency-weighting study can be done offline.
- Efficiency weighting is config-gated and OFF by default; the provisional
  formula below is a placeholder until it is aligned with the official
  leaderboard semantics (design decision #3, 2026-10-09).
- The model server must be a VLM; the agent only references
  ``config.model_server`` and never hard-codes a model name (decision #4).
- A single failed rollout must not abort collection: ``run`` converts
  exceptions into failure responses tagged with a failure class (tau2 pattern).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import deque
from dataclasses import dataclass
from time import time
from typing import Any
from uuid import uuid4

from aiohttp import ClientTimeout
from fastapi import Body, Request
from pydantic import ConfigDict, Field

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
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from responses_api_agents.arc_agi_v3.frame_render import grid_to_png_data_url
from responses_api_agents.arc_agi_v3.game_client import (
    ACTION_RESET_ID,
    DEFAULT_API_BASE,
    ArcAgiV3ApiError,
    ArcAgiV3GameClient,
    Frame,
    GameScore,
)


LOG = logging.getLogger(__name__)

ARC_AGI_V3_MALFORMED_ACTION_FAILURE_CLASS = "arc_agi_v3_malformed_action"
ARC_AGI_V3_API_FAILURE_CLASS = "arc_agi_v3_api_error"

_RESERVED_RESULT_KEYS = ("reward", "response", "error", NG_FAILURE_CLASS_KEY)

ACTION_NAME_TO_ID: dict[str, int] = {"RESET": ACTION_RESET_ID} | {f"ACTION{index}": index for index in range(1, 8)}
# Upstream grid is at most 64x64 with (0,0) at the top-left (docs.arcprize.org/games).
GRID_MAX_COORD = 63

DEFAULT_SYSTEM_PROMPT = (
    "You are an agent exploring an ARC-AGI-3 game. The game gives no instructions: you "
    "must discover the action semantics, the goal, and the win condition purely from "
    "observation. Each turn you receive the latest frame and a status line, and you must "
    "choose exactly one action. Prefer systematic exploration over repetition, and keep "
    "track of what each action did in earlier steps."
)

DEFAULT_USER_INSTRUCTIONS = (
    "Available actions are RESET (restarts the level or game; also the only legal action in "
    "the GAME_OVER state) and ACTION1-ACTION7 as listed in the status line. ACTION6 is a "
    "complex action that requires (x, y) coordinates on a grid whose top-left corner is "
    "(0, 0). Respond with exactly one action per turn."
)

_SUBMIT_ACTION_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTION_NAME_TO_ID)},
        "x": {"type": "integer", "minimum": 0, "maximum": GRID_MAX_COORD},
        "y": {"type": "integer", "minimum": 0, "maximum": GRID_MAX_COORD},
        "thought": {"type": "string", "description": "Brief reasoning about what to try next."},
    },
    "required": ["action"],
}


class ArcAgiV3MalformedActionError(ValueError):
    """A model response that cannot be turned into a legal game action."""


@dataclass
class _ParsedAction:
    action_id: int
    name: str
    x: int | None = None
    y: int | None = None
    thought: str | None = None


class ArcAgiV3AgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    api_base: str = DEFAULT_API_BASE
    # API keys are injected via the environment (docs/环境变量配置.md), never config files.
    api_key_env_var: str = "ARC_AGI_V3_API_KEY"
    concurrency: int = 2
    # Per HTTP/model-call timeout; the overall rollout is bounded by the action budget.
    request_timeout_s: float = 120.0
    # Fallback action budget when the dataset row has no per-game max_actions.
    default_max_actions: int = 150
    # Frames are image-heavy: keep the last N frames; the full action trajectory stays as text.
    history_window: int = Field(default=6, ge=1)
    temperature: float = 0.0
    max_output_tokens: int = 700
    malformed_action_max_retries: int = Field(default=0, ge=0)
    use_tool_call: bool = True
    # M4 decision: efficiency weighting stays off until aligned with the official leaderboard.
    efficiency_weighting: bool = False
    # Optional game_id subset filter (cost control, design §4); empty means all rows run.
    game_subset: list[str] = Field(default_factory=list)


class ArcAgiV3RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    game_id: str
    # Dataset-row overrides; None falls back to config defaults. The upstream 5x-human-baseline
    # cutoff is not published for every game, so prepare.py leaves these null when unknown.
    max_actions: int | None = None
    human_baseline_actions: int | None = None
    actions_available: list[str] | None = None


class ArcAgiV3VerifyResponse(BaseVerifyResponse):
    game_id: str
    state: str
    completed: bool
    truncated: bool
    score: float
    efficiency_weighted_score: float | None = None
    levels_completed: int
    win_levels: int | None = None
    actions_taken: int
    resets: int
    steps: int
    malformed_actions: int
    level_scores: list[float] = Field(default_factory=list)
    level_actions: list[int] = Field(default_factory=list)
    level_baseline_actions: list[int | None] = Field(default_factory=list)
    guid: str | None = None
    card_id: str | None = None


class ArcAgiV3FailureResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ArcAgiV3MalformedActionError("model response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return ""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model text response."""
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise ArcAgiV3MalformedActionError(f"model text contains no JSON object: {text[:200]!r}")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise ArcAgiV3MalformedActionError(f"model text JSON is invalid: {error}") from error
    if not isinstance(parsed, dict):
        raise ArcAgiV3MalformedActionError(f"model JSON must be an object, got {type(parsed).__name__}")
    return parsed


def _parse_action(response: dict[str, Any]) -> _ParsedAction:
    """Parse a chat completion into a game action; raises on anything unusable."""
    choices = response.get("choices") or []
    if not choices:
        raise ArcAgiV3MalformedActionError("model response has no choices")
    message = choices[0].get("message") or {}

    tool_calls = message.get("tool_calls")
    if tool_calls:
        if not isinstance(tool_calls, list):
            raise ArcAgiV3MalformedActionError("tool_calls must be a list")
        tool_call = tool_calls[0]
        function = tool_call.get("function") or {} if isinstance(tool_call, dict) else {}
        if function.get("name") != "submit_action":
            raise ArcAgiV3MalformedActionError(
                f"unexpected tool call {function.get('name')!r}, expected submit_action"
            )
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ArcAgiV3MalformedActionError("tool call function.arguments must be a JSON string")
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ArcAgiV3MalformedActionError(f"tool call arguments are invalid JSON: {error}") from error
        if not isinstance(parsed, dict):
            raise ArcAgiV3MalformedActionError("tool call arguments must decode to an object")
    else:
        parsed = _extract_json_object(_message_text(response))

    name = str(parsed.get("action", "")).strip().upper()
    if name not in ACTION_NAME_TO_ID:
        raise ArcAgiV3MalformedActionError(f"unknown action {name!r}; expected one of {sorted(ACTION_NAME_TO_ID)}")

    action = _ParsedAction(action_id=ACTION_NAME_TO_ID[name], name=name)
    if name == "ACTION6":
        x, y = parsed.get("x"), parsed.get("y")
        coords_valid = all(
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= GRID_MAX_COORD for value in (x, y)
        )
        if not coords_valid:
            raise ArcAgiV3MalformedActionError(f"ACTION6 requires integer x/y coordinates in [0, {GRID_MAX_COORD}]")
        action.x, action.y = x, y
    thought = parsed.get("thought") or parsed.get("reasoning")
    if isinstance(thought, str) and thought.strip():
        action.thought = thought.strip()
    return action


def _efficiency_weighted_score(score: GameScore) -> float | None:
    """PROVISIONAL per-level efficiency weighting, pending the M4 alignment study.

    Per level: level_score scaled by baseline/actual when the agent used more
    actions than the upstream human baseline (capped at 1, never boosted).
    Returns None when the upstream card publishes no baselines.
    """
    weighted = 0.0
    has_baseline = False
    for index, level_score in enumerate(score.level_scores):
        baseline = (
            score.level_baseline_actions[index]
            if index < len(score.level_baseline_actions) and score.level_baseline_actions[index] is not None
            else None
        )
        actual = score.level_actions[index] if index < len(score.level_actions) else None
        efficiency = 1.0
        if baseline is not None and actual is not None and actual > baseline > 0:
            efficiency = baseline / actual
            has_baseline = True
        weighted += level_score * efficiency
    return weighted if has_baseline else None


class ArcAgiV3Agent(SimpleResponsesAPIAgent):
    config: ArcAgiV3AgentConfig
    sem: asyncio.Semaphore | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise NotImplementedError("ArcAgiV3Agent executes complete game runs through /run")

    async def run(
        self, request: Request, body: ArcAgiV3RunRequest
    ) -> ArcAgiV3VerifyResponse | ArcAgiV3FailureResponse:
        record = self._sanitized_record(body)
        try:
            return await self._run(body)
        except ArcAgiV3MalformedActionError as error:
            return self._failure_response(body, record, ARC_AGI_V3_MALFORMED_ACTION_FAILURE_CLASS, str(error))
        except Exception as error:  # noqa: BLE001 -- one bad rollout must not abort the full collection
            detail = error
            for nested in getattr(error, "exceptions", ()) or ():
                detail = nested
                break
            return self._failure_response(
                body, record, ARC_AGI_V3_API_FAILURE_CLASS, f"{type(detail).__name__}: {detail}"
            )

    async def _run(self, body: ArcAgiV3RunRequest) -> ArcAgiV3VerifyResponse:
        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise ArcAgiV3ApiError(
                f"missing ARC Prize API key: set the {self.config.api_key_env_var} environment variable"
            )
        if self.config.game_subset and body.game_id not in self.config.game_subset:
            raise ValueError(f"game {body.game_id!r} is not in the configured game_subset {self.config.game_subset}")

        budget = body.max_actions if body.max_actions is not None else self.config.default_max_actions
        system_prompt, user_instructions = self._static_prompts(body)

        assert self.sem is not None
        async with self.sem:
            async with ArcAgiV3GameClient(
                api_base=self.config.api_base, api_key=api_key, timeout_s=self.config.request_timeout_s
            ) as client:
                card_id = await client.open_scorecard(
                    tags=["nemo-gym", "arc-agi-v3"],
                    opaque={"model_server": self.config.model_server.name, "game_id": body.game_id},
                )
                frame = await client.reset(body.game_id, card_id)

                usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
                trajectory: list[str] = []
                # Full action trajectory stays as text; only the last N frames are re-sent as images.
                frame_history: deque[tuple[str, str]] = deque(maxlen=self.config.history_window)
                steps = malformed_actions = 0
                try:
                    while not frame.finished and steps < budget:
                        # A malformed model answer may be retried (config-gated); each retry is a
                        # fresh model call, mirroring tau2's malformed-tool-call regeneration path.
                        attempts = self.config.malformed_action_max_retries + 1
                        for attempt in range(attempts):
                            response = await self._next_action(
                                body=body,
                                frame=frame,
                                budget=budget,
                                steps=steps,
                                system_prompt=system_prompt,
                                user_instructions=user_instructions,
                                trajectory=trajectory,
                                frame_history=frame_history,
                            )
                            usage = response.get("usage") or {}
                            for usage_key in usage_totals:
                                usage_totals[usage_key] += int(usage.get(usage_key, 0) or 0)
                            try:
                                action = _parse_action(response)
                            except ArcAgiV3MalformedActionError:
                                malformed_actions += 1
                                if attempt == attempts - 1:
                                    raise
                                LOG.warning(
                                    "malformed action (attempt %d/%d) for game %s: %s",
                                    attempt + 1,
                                    attempts,
                                    body.game_id,
                                    response.get("choices", [{}])[0].get("message", {}).get("content", "")[:200],
                                )
                                continue
                            if action.action_id != ACTION_RESET_ID and action.action_id not in frame.available_actions:
                                malformed_actions += 1
                                raise ArcAgiV3MalformedActionError(
                                    f"{action.name} is not in the game's available actions {frame.available_actions}"
                                )
                            break

                        steps += 1
                        if action.action_id == ACTION_RESET_ID:
                            next_frame = await client.reset(body.game_id, card_id, guid=frame.guid)
                        else:
                            next_frame = await client.action(
                                action.action_id,
                                frame.game_id,
                                frame.guid,
                                x=action.x,
                                y=action.y,
                                reasoning={"thought": action.thought} if action.thought else None,
                            )
                        trajectory.append(
                            f"{steps}. {action.name}"
                            + (f"({action.x}, {action.y})" if action.name == "ACTION6" else "")
                            + f" -> state={next_frame.state}, levels={next_frame.levels_completed}/{next_frame.win_levels}"
                        )
                        frame_history.append(
                            (f"Step {steps}: {trajectory[-1]}", grid_to_png_data_url(next_frame.latest_frame))
                        )
                        frame = next_frame

                    score = await client.game_score(card_id, body.game_id)
                finally:
                    try:
                        await client.close_scorecard(card_id)
                    except Exception:  # noqa: BLE001 -- closing is bookkeeping; never mask the outcome
                        LOG.exception("failed to close scorecard %s for game %s", card_id, body.game_id)

        truncated = not score.completed and frame.state not in ("WIN", "GAME_OVER")
        weighted = _efficiency_weighted_score(score) if self.config.efficiency_weighting else None
        return ArcAgiV3VerifyResponse(
            **self._sanitized_record(body),
            response=self._response_object(body, trajectory, usage_totals),
            reward=float(score.score),
            state=score.state or frame.state,
            completed=bool(score.completed or frame.state == "WIN"),
            truncated=truncated,
            score=float(score.score),
            efficiency_weighted_score=weighted,
            levels_completed=score.levels_completed,
            win_levels=frame.win_levels,
            actions_taken=score.actions,
            resets=score.resets,
            steps=steps,
            malformed_actions=malformed_actions,
            level_scores=score.level_scores,
            level_actions=score.level_actions,
            level_baseline_actions=score.level_baseline_actions,
            guid=frame.guid,
            card_id=card_id,
        )

    async def _next_action(
        self,
        *,
        body: ArcAgiV3RunRequest,
        frame: Frame,
        budget: int,
        steps: int,
        system_prompt: str,
        user_instructions: str,
        trajectory: list[str],
        frame_history: deque[tuple[str, str]],
    ) -> dict[str, Any]:
        status = (
            f"Current game status:\n"
            f"- state: {frame.state}\n"
            f"- levels completed: {frame.levels_completed}/{frame.win_levels}\n"
            f"- actions used: {steps}/{budget}\n"
            f"- available actions: {frame.available_actions}"
            f" (ACTION6 needs x/y coordinates 0-{GRID_MAX_COORD})"
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{user_instructions}\n\nFull action trajectory so far:\n"
                    + ("\n".join(trajectory) if trajectory else "(no actions taken yet)")
                ),
            }
        )
        for caption, data_url in list(frame_history)[:-1]:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{status}\n\nLatest frame below. Choose the next action."},
                    {"type": "image_url", "image_url": {"url": grid_to_png_data_url(frame.latest_frame)}},
                ],
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
                        "name": "submit_action",
                        "description": "Submit exactly one ARC-AGI-3 game action for this turn.",
                        "parameters": _SUBMIT_ACTION_PARAMETERS,
                    },
                }
            ]
            payload["tool_choice"] = {"type": "function", "function": {"name": "submit_action"}}

        path = self.url_path_for_run("/v1/chat/completions", body)
        response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=path,
            json=payload,
            timeout=ClientTimeout(total=self.config.request_timeout_s),
        )
        await raise_for_status(response)
        return await get_response_json(response)

    def _static_prompts(self, body: ArcAgiV3RunRequest) -> tuple[str, str]:
        """Use the dataset prompt_config when provided, else the built-in defaults."""
        system_prompt, user_instructions = DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_INSTRUCTIONS
        input_items = getattr(body.responses_create_params, "input", None)
        if isinstance(input_items, str):
            user_instructions = input_items
        elif isinstance(input_items, list):
            for item in input_items:
                # input items arrive as message dicts or as pydantic message objects.
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
        self, body: ArcAgiV3RunRequest, trajectory: list[str], usage_totals: dict[str, int]
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
            model=f"arc_agi_v3/{self.config.model_server.name}",
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

    def _sanitized_record(self, body: ArcAgiV3RunRequest) -> dict[str, Any]:
        record = body.model_dump(mode="json")
        for key in _RESERVED_RESULT_KEYS:
            record.pop(key, None)
        return record

    def _failure_response(
        self, body: ArcAgiV3RunRequest, record: dict[str, Any], failure_class: str, error: str
    ) -> ArcAgiV3FailureResponse:
        print(f"[arc_agi_v3] rollout failed for {body.game_id}: {error}", flush=True)
        return ArcAgiV3FailureResponse.model_validate(
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
    ArcAgiV3Agent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ArcAgiV3Agent.run_webserver()  # noqa: F401
