# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ARC-AGI-3 game-loop agent.

The agent is exercised end-to-end against a local mock of the ARC-AGI-3 REST
API (scorecards + RESET/ACTION + per-game score), with the model server stubbed
at ``ServerClient.post`` — the same seam the real rollout traffic uses.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web


pytest.importorskip("resource", reason="nemo-gym server_utils requires the Unix resource module")

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY
from nemo_gym.server_utils import ServerClient
from responses_api_agents.arc_agi_v3.app import (
    ARC_AGI_V3_API_FAILURE_CLASS,
    ARC_AGI_V3_MALFORMED_ACTION_FAILURE_CLASS,
    ArcAgiV3Agent,
    ArcAgiV3AgentConfig,
    ArcAgiV3MalformedActionError,
    ArcAgiV3RunRequest,
    _efficiency_weighted_score,
    _parse_action,
)
from responses_api_agents.arc_agi_v3.frame_render import grid_to_png_data_url, validate_grid
from responses_api_agents.arc_agi_v3.game_client import ArcAgiV3ApiError, ArcAgiV3GameClient


def _grid(fill: int = 0) -> list[list[int]]:
    return [[fill] * 4 for _ in range(4)]


def _frame_response(state: str = "NOT_FINISHED", levels_completed: int = 0) -> dict:
    return {
        "game_id": "ls20-016295f7601e",
        "guid": "guid-1",
        "frame": [_grid()],
        "state": state,
        "levels_completed": levels_completed,
        "win_levels": 1,
        "action_input": {"id": 0, "data": {}},
        "full_reset": state != "NOT_FINISHED",
        "available_actions": [0, 1, 2, 3, 4, 6],
    }


def _chat_completion(tool_arguments: dict | str | None) -> dict:
    """A chat completion with a submit_action tool call (or malformed arguments)."""
    if isinstance(tool_arguments, dict):
        arguments = json.dumps(tool_arguments)
    elif tool_arguments is None:
        arguments = "not json at all"
    else:
        arguments = tool_arguments
    return {
        "id": "chatcmpl-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "submit_action", "arguments": arguments},
                        }
                    ],
                },
            }
        ],
        "created": 0,
        "model": "vlm",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


class TestParseAction:
    def test_tool_call_simple_action(self):
        action = _parse_action(_chat_completion({"action": "ACTION1", "thought": "move"}))
        assert (action.action_id, action.name) == (1, "ACTION1")
        assert action.thought == "move"

    def test_tool_call_complex_action_with_coords(self):
        action = _parse_action(_chat_completion({"action": "ACTION6", "x": 10, "y": 63}))
        assert (action.action_id, action.x, action.y) == (6, 10, 63)

    def test_tool_call_action6_without_coords_is_malformed(self):
        with pytest.raises(ArcAgiV3MalformedActionError, match="coordinates"):
            _parse_action(_chat_completion({"action": "ACTION6"}))

    def test_tool_call_action6_with_out_of_range_coord_is_malformed(self):
        with pytest.raises(ArcAgiV3MalformedActionError, match="coordinates"):
            _parse_action(_chat_completion({"action": "ACTION6", "x": 64, "y": 0}))

    def test_unknown_action_is_malformed(self):
        with pytest.raises(ArcAgiV3MalformedActionError, match="unknown action"):
            _parse_action(_chat_completion({"action": "ACTION8"}))

    def test_invalid_json_arguments_is_malformed(self):
        with pytest.raises(ArcAgiV3MalformedActionError, match="invalid JSON"):
            _parse_action(_chat_completion("{not json"))

    def test_json_text_fallback_without_tool_calls(self):
        response = {
            "choices": [
                {"message": {"role": "assistant", "content": 'Let me click. {"action": "ACTION6", "x": 3, "y": 4}'}}
            ]
        }
        action = _parse_action(response)
        assert (action.action_id, action.x, action.y) == (6, 3, 4)

    def test_text_without_json_is_malformed(self):
        response = {"choices": [{"message": {"role": "assistant", "content": "I will move up."}}]}
        with pytest.raises(ArcAgiV3MalformedActionError, match="no JSON object"):
            _parse_action(response)

    def test_reset_maps_to_id_zero(self):
        action = _parse_action(_chat_completion({"action": "reset"}))
        assert (action.action_id, action.name) == (0, "RESET")


class TestFrameRender:
    def test_grid_to_png_data_url(self):
        data_url = grid_to_png_data_url(_grid(9))
        assert data_url.startswith("data:image/png;base64,")
        header = base64.b64decode(data_url.split(",", 1)[1])[:8]
        assert header == b"\x89PNG\r\n\x1a\n"  # PNG magic

    def test_validate_grid_rejects_ragged_and_out_of_range(self):
        with pytest.raises(ValueError, match="same width"):
            validate_grid([[0, 0], [0]])
        with pytest.raises(ValueError, match="0, 15"):
            validate_grid([[16]])
        with pytest.raises(ValueError, match="non-empty"):
            validate_grid([])


class TestEfficiencyWeightedScore:
    def _score(self, level_scores, level_actions, level_baseline_actions):
        from responses_api_agents.arc_agi_v3.game_client import GameScore

        return GameScore(
            game_id="ls20-016295f7601e",
            guid="guid-1",
            score=sum(level_scores),
            levels_completed=len(level_scores),
            actions=sum(level_actions),
            resets=0,
            state="WIN",
            completed=True,
            level_scores=level_scores,
            level_actions=level_actions,
            level_baseline_actions=level_baseline_actions,
        )

    def test_weighted_when_agent_used_more_actions_than_baseline(self):
        # level 0: 100 points at 10 actions vs baseline 5 -> 50; level 1: untouched (faster).
        score = self._score([100.0, 100.0], [10, 2], [5, 5])
        assert _efficiency_weighted_score(score) == pytest.approx(50.0 + 100.0)

    def test_none_without_published_baselines(self):
        assert _efficiency_weighted_score(self._score([100.0], [10], [None])) is None


class _MockArcServer:
    """Minimal stand-in for three.arcprize.org: 2x ACTION1 wins the game."""

    def __init__(self):
        self.action_count = 0
        self.closed_cards: list[str] = []

    async def _frame(self, state: str, levels_completed: int) -> web.Response:
        return web.json_response(_frame_response(state=state, levels_completed=levels_completed))

    async def open_scorecard(self, request: web.Request) -> web.Response:
        return web.json_response({"card_id": "card-1"})

    async def close_scorecard(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.closed_cards.append(body["card_id"])
        return web.json_response({"card_id": body["card_id"]})

    async def reset(self, request: web.Request) -> web.Response:
        self.action_count = 0
        return await self._frame("NOT_FINISHED", 0)

    async def action(self, request: web.Request) -> web.Response:
        self.action_count += 1
        if self.action_count >= 2:
            return await self._frame("WIN", 1)
        return await self._frame("NOT_FINISHED", 0)

    async def game_score(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "id": "ls20-016295f7601e",
                "guid": "guid-1",
                "score": 100.0,
                "levels_completed": 1,
                "actions": 2,
                "resets": 1,
                "state": "WIN",
                "completed": True,
                "level_scores": [100.0],
                "level_actions": [2],
                "level_baseline_actions": [10],
                "number_of_levels": 1,
            }
        )

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/api/scorecard/open", self.open_scorecard)
        app.router.add_post("/api/scorecard/close", self.close_scorecard)
        app.router.add_post("/api/cmd/RESET", self.reset)
        app.router.add_post("/api/cmd/ACTION1", self.action)
        app.router.add_get("/api/scorecard/card-1/ls20", self.game_score)
        return app


async def _start_mock_arc_server(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


class TestAgentEndToEnd:
    def _agent(self, api_base: str) -> ArcAgiV3Agent:
        config = ArcAgiV3AgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            name="arc_agi_v3_agent",
            model_server={"type": "responses_api_models", "name": "vlm"},
            api_base=api_base,
            request_timeout_s=10.0,
            default_max_actions=5,
            history_window=4,
        )
        server_client = MagicMock(spec=ServerClient)
        # Empty mapping => observability/token-capture correlation stays disabled.
        server_client.global_config_dict = {}
        server_client.post = AsyncMock(return_value=None)
        agent = ArcAgiV3Agent(config=config, server_client=server_client)
        return agent

    @staticmethod
    def _stub_model_responses(agent: ArcAgiV3Agent, responses: list[dict]) -> None:
        # AsyncMock returns each side_effect value as-is; the agent expects the
        # post() awaitable to resolve to an aiohttp-like response object.
        agent.server_client.post.side_effect = [_FakeResponse(payload) for payload in responses]

    def _run_body(self, input_messages: list[dict]) -> ArcAgiV3RunRequest:
        return ArcAgiV3RunRequest(
            game_id="ls20",
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(model="vlm", input=input_messages),
        )

    def test_full_loop_wins_game_and_returns_raw_score(self, monkeypatch):
        monkeypatch.setenv("ARC_AGI_V3_API_KEY", "test-key")  # pragma: allowlist secret

        async def scenario():
            server = _MockArcServer()
            runner, api_base = await _start_mock_arc_server(server.app())
            try:
                agent = self._agent(api_base)
                # Two model turns: ACTION1, ACTION1 -> the mock declares WIN.
                self._stub_model_responses(
                    agent,
                    [
                        _chat_completion({"action": "ACTION1", "thought": "try up"}),
                        _chat_completion({"action": "ACTION1"}),
                    ],
                )
                response = await agent.run(
                    request=None,
                    body=self._run_body([{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]),
                )
                model_payload = agent.server_client.post.await_args_list[0].kwargs["json"]
            finally:
                await runner.cleanup()
            return response, server, model_payload

        response, server, model_payload = asyncio.run(scenario())

        assert response.reward == 100.0  # raw score, decision #3
        assert response.completed and not response.truncated
        assert response.state == "WIN"
        assert response.steps == 2
        assert response.malformed_actions == 0
        assert response.level_scores == [100.0]
        assert response.efficiency_weighted_score is None  # default off
        assert server.closed_cards == ["card-1"]  # scorecard closed for bookkeeping

        # The tool schema is attached and the dataset prompt_config's system text is used.
        assert model_payload["tools"][0]["function"]["name"] == "submit_action"
        system_texts = [m for m in model_payload["messages"] if m["role"] == "system"]
        assert [m["content"] for m in system_texts] == ["sys"]

    def test_malformed_action_fails_rollout_with_failure_class(self, monkeypatch):
        monkeypatch.setenv("ARC_AGI_V3_API_KEY", "test-key")  # pragma: allowlist secret

        async def scenario():
            runner, api_base = await _start_mock_arc_server(_MockArcServer().app())
            try:
                agent = self._agent(api_base)
                self._stub_model_responses(agent, [_chat_completion(None)])  # garbage arguments
                return await agent.run(request=None, body=self._run_body([]))
            finally:
                await runner.cleanup()

        response = asyncio.run(scenario())

        assert response.reward == 0.0
        assert getattr(response, NG_FAILURE_CLASS_KEY) == ARC_AGI_V3_MALFORMED_ACTION_FAILURE_CLASS
        assert "JSON" in response.error

    def test_api_error_fails_rollout_with_failure_class(self, monkeypatch):
        monkeypatch.setenv("ARC_AGI_V3_API_KEY", "test-key")  # pragma: allowlist secret

        async def scenario():
            async def broken_reset(request: web.Request) -> web.Response:
                return web.json_response({"error": "boom"}, status=500)

            app = web.Application()
            app.router.add_post("/api/scorecard/open", _MockArcServer().open_scorecard)
            app.router.add_post("/api/cmd/RESET", broken_reset)
            runner, api_base = await _start_mock_arc_server(app)
            try:
                agent = self._agent(api_base)
                return await agent.run(request=None, body=self._run_body([]))
            finally:
                await runner.cleanup()

        response = asyncio.run(scenario())
        assert response.reward == 0.0
        assert getattr(response, NG_FAILURE_CLASS_KEY) == ARC_AGI_V3_API_FAILURE_CLASS

    def test_missing_api_key_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("ARC_AGI_V3_API_KEY", raising=False)
        agent = self._agent("http://127.0.0.1:1")
        failure = asyncio.run(agent.run(request=None, body=self._run_body([])))
        assert getattr(failure, NG_FAILURE_CLASS_KEY) == ARC_AGI_V3_API_FAILURE_CLASS
        assert "ARC_AGI_V3_API_KEY" in failure.error


class TestGameClient:
    def test_api_error_includes_status_and_body(self):
        async def scenario():
            async def forbidden(request: web.Request) -> web.Response:
                return web.json_response({"detail": "bad key"}, status=403)

            app = web.Application()
            app.router.add_post("/api/scorecard/open", forbidden)
            runner, api_base = await _start_mock_arc_server(app)
            try:
                async with ArcAgiV3GameClient(api_base=api_base, api_key="k", timeout_s=5.0) as client:
                    with pytest.raises(ArcAgiV3ApiError, match="bad key") as exc_info:
                        await client.open_scorecard()
                    assert exc_info.value.status == 403
            finally:
                await runner.cleanup()

        asyncio.run(scenario())

    def test_action_rejects_action6_without_coordinates(self):
        async def scenario():
            async with ArcAgiV3GameClient(api_base="http://127.0.0.1:1", api_key="k", timeout_s=1.0) as client:
                with pytest.raises(ArcAgiV3ApiError, match="requires x and y"):
                    await client.action(6, "ls20", "guid-1")

        asyncio.run(scenario())


class _FakeResponse:
    """Just enough of aiohttp.ClientResponse for raise_for_status/get_response_json."""

    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()
        self.ok = True

    def raise_for_status(self):
        return None

    async def read(self) -> bytes:
        return self._payload
