# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async client for the ARC-AGI-3 REST API (https://three.arcprize.org).

The interface surface is small (scorecards, RESET, ACTION1-7, per-game score),
so the Gym rolls its own aiohttp client instead of pulling the upstream SDK —
the official OpenAPI spec (docs.arcprize.org/arc3v1.yaml) is the contract:

- Every request carries the ``X-API-Key`` header; the key is injected from an
  environment variable, never from config.
- Games are stateful server-side, so the client MUST keep and replay the
  ``AWSALB*`` session-affinity cookies the server sets on RESET/ACTION
  responses; the shared ``aiohttp.CookieJar`` handles that.
- Non-2xx responses raise :class:`ArcAgiV3ApiError` with the response body
  attached — upstream returns 400 for actions in a game-over state, which the
  agent loop is expected to avoid by honouring ``state`` transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from responses_api_agents.arc_agi_v3.frame_render import validate_grid


LOG = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://three.arcprize.org"

# FrameResponse.state / scorecard run state (OpenAPI enum).
STATE_NOT_PLAYED = "NOT_PLAYED"
STATE_NOT_FINISHED = "NOT_FINISHED"
STATE_WIN = "WIN"
STATE_GAME_OVER = "GAME_OVER"

# Numeric GameAction ids: 0 is RESET (OpenAPI action_input.id echo), 1-7 are
# the game actions. ACTION6 additionally requires (x, y) coordinates.
ACTION_RESET_ID = 0
MAX_GAME_ACTION_ID = 7
COMPLEX_ACTION_ID = 6


class ArcAgiV3ApiError(RuntimeError):
    """A non-2xx or unparseable response from the ARC-AGI-3 API."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(f"{message} (status={status})" if status is not None else message)
        self.status = status


@dataclass
class Frame:
    """Parsed FrameResponse returned by every RESET/ACTION command."""

    game_id: str
    guid: str
    frames: list[list[list[int]]]
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    full_reset: bool = False
    action_input: dict[str, Any] = field(default_factory=dict)

    @property
    def latest_frame(self) -> list[list[int]]:
        return self.frames[-1]

    @property
    def finished(self) -> bool:
        return self.state in (STATE_WIN, STATE_GAME_OVER)


@dataclass
class GameScore:
    """Per-game scorecard run stats (upstream ``RunScore``)."""

    game_id: str
    guid: str
    score: float
    levels_completed: int
    actions: int
    resets: int
    state: str
    completed: bool
    level_scores: list[float] = field(default_factory=list)
    level_actions: list[int] = field(default_factory=list)
    level_baseline_actions: list[int | None] = field(default_factory=list)
    number_of_levels: int = 0


class ArcAgiV3GameClient:
    """One client per rollout; holds the session-affinity cookie jar."""

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = "",
        timeout_s: float = 120.0,
    ):
        self._api_base = api_base.rstrip("/")
        # unsafe=True keeps IP-host cookies (mock servers on 127.0.0.1); harmless
        # for the https production base URL.
        self._session = aiohttp.ClientSession(
            headers={"X-API-Key": api_key},  # pragma: allowlist secret
            timeout=aiohttp.ClientTimeout(total=timeout_s),
            cookie_jar=aiohttp.CookieJar(unsafe=not self._api_base.startswith("https")),
        )

    async def __aenter__(self) -> "ArcAgiV3GameClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._session.close()

    async def _request(self, method: str, path: str, json_body: dict[str, Any] | None = None) -> Any:
        url = f"{self._api_base}{path}"
        try:
            async with self._session.request(method, url, json=json_body) as response:
                text = await response.text()
                if response.status >= 300:
                    raise ArcAgiV3ApiError(f"{method} {path} failed: {text[:500]}", status=response.status)
                try:
                    return await response.json(content_type=None)
                except aiohttp.ContentTypeError:
                    return text
        except aiohttp.ClientError as error:
            raise ArcAgiV3ApiError(f"{method} {path} transport error: {error}") from error

    async def list_games(self) -> list[dict[str, Any]]:
        result = await self._request("GET", "/api/games")
        if not isinstance(result, list):
            raise ArcAgiV3ApiError("/api/games returned a non-list payload")
        return result

    async def open_scorecard(
        self,
        source_url: str | None = None,
        tags: list[str] | None = None,
        opaque: dict[str, Any] | None = None,
    ) -> str:
        body = {key: value for key, value in (("source_url", source_url), ("tags", tags), ("opaque", opaque)) if value}
        result = await self._request("POST", "/api/scorecard/open", body)
        card_id = result.get("card_id") if isinstance(result, dict) else None
        if not card_id:
            raise ArcAgiV3ApiError("/api/scorecard/open response missing card_id")
        return str(card_id)

    async def close_scorecard(self, card_id: str) -> dict[str, Any]:
        return await self._request("POST", "/api/scorecard/close", {"card_id": card_id})

    async def reset(self, game_id: str, card_id: str, guid: str | None = None) -> Frame:
        """Start a new game instance (guid=None) or reset an existing session."""
        return self._parse_frame(
            await self._request("POST", "/api/cmd/RESET", {"game_id": game_id, "card_id": card_id, "guid": guid})
        )

    async def action(
        self,
        action_id: int,
        game_id: str,
        guid: str,
        x: int | None = None,
        y: int | None = None,
        reasoning: dict[str, Any] | str | None = None,
    ) -> Frame:
        """Execute ACTION1-7 (ACTION6 requires x/y on the 64x64 grid)."""
        if not 1 <= action_id <= MAX_GAME_ACTION_ID:
            raise ArcAgiV3ApiError(f"invalid game action id {action_id}; expected 1-{MAX_GAME_ACTION_ID}")
        body: dict[str, Any] = {"game_id": game_id, "guid": guid}
        if action_id == COMPLEX_ACTION_ID:
            if x is None or y is None:
                raise ArcAgiV3ApiError("ACTION6 requires x and y coordinates")
            body |= {"x": x, "y": y}
        if reasoning is not None:
            body["reasoning"] = reasoning
        return self._parse_frame(await self._request("POST", f"/api/cmd/ACTION{action_id}", body))

    async def game_score(self, card_id: str, game_id: str) -> GameScore:
        result = await self._request("GET", f"/api/scorecard/{card_id}/{game_id}")
        return self._parse_game_score(game_id, result)

    @staticmethod
    def _parse_frame(payload: Any) -> Frame:
        if not isinstance(payload, dict):
            raise ArcAgiV3ApiError("frame response is not a JSON object")
        required = ("game_id", "guid", "frame", "state", "levels_completed", "win_levels", "available_actions")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ArcAgiV3ApiError(f"frame response missing fields: {missing}")
        frames = payload["frame"]
        if not isinstance(frames, list) or not frames:
            raise ArcAgiV3ApiError("frame response contains no frames")
        try:
            for frame in frames:
                validate_grid(frame)
        except ValueError as error:
            raise ArcAgiV3ApiError(f"frame response has a malformed grid: {error}") from error
        return Frame(
            game_id=str(payload["game_id"]),
            guid=str(payload["guid"]),
            frames=frames,
            state=str(payload["state"]),
            levels_completed=int(payload["levels_completed"]),
            win_levels=int(payload["win_levels"]),
            available_actions=[int(action) for action in payload["available_actions"]],
            full_reset=bool(payload.get("full_reset", False)),
            action_input=payload.get("action_input") or {},
        )

    @staticmethod
    def _parse_game_score(game_id: str, payload: Any) -> GameScore:
        if not isinstance(payload, dict):
            raise ArcAgiV3ApiError("scorecard response is not a JSON object")
        # The endpoint returns a summary wrapping per-environment runs, or a bare
        # run object when queried per game; find the run matching game_id first.
        runs = payload.get("environments") if isinstance(payload.get("environments"), list) else [payload]
        run = next(
            (entry for entry in runs if isinstance(entry, dict) and str(entry.get("id", "")).startswith(game_id)),
            None,
        )
        if run is None:
            raise ArcAgiV3ApiError(f"scorecard response has no run for game {game_id}")
        return GameScore(
            game_id=str(run.get("id", game_id)),
            guid=str(run.get("guid", "")),
            score=float(run.get("score", 0.0)),
            levels_completed=int(run.get("levels_completed", 0)),
            actions=int(run.get("actions", 0)),
            resets=int(run.get("resets", 0)),
            state=str(run.get("state", STATE_NOT_PLAYED)),
            completed=bool(run.get("completed", False)),
            level_scores=[float(score) for score in run.get("level_scores") or []],
            level_actions=[int(actions) for actions in run.get("level_actions") or []],
            level_baseline_actions=list(run.get("level_baseline_actions") or []),
            number_of_levels=int(run.get("number_of_levels", 0) or 0),
        )
