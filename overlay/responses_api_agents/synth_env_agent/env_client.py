# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""aiohttp client for the synth_env host's C-SE-1 API (design §3.1).

The session cookie minted by ``/seed_session`` must flow through every
subsequent call (step/observe/verify) — that cookie *is* the session. API
failures raise :class:`SynthEnvApiError` with the HTTP status and body so the
agent can classify them (``synth_env_api_error``) instead of failing silently.
"""

from __future__ import annotations

from typing import Any, Optional

from aiohttp import ClientTimeout

from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


class SynthEnvApiError(Exception):
    """A non-OK response from the synth_env host."""

    def __init__(self, status: int, body: str):
        super().__init__(f"synth_env host returned HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class SynthEnvClient:
    """Cookie-carrying client over the C-SE-1 routes (seed/reset/observe/step/verify)."""

    def __init__(self, server_client: ServerClient, server_name: str, timeout_s: float = 120.0):
        self._server_client = server_client
        self._server_name = server_name
        self._timeout = ClientTimeout(total=timeout_s)
        self.cookies: Any = None

    async def _post(self, url_path: str, json: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        response = await self._server_client.post(
            server_name=self._server_name,
            url_path=url_path,
            json=json,
            cookies=self.cookies,
            timeout=self._timeout,
        )
        await self._raise(response)
        if getattr(response, "cookies", None):
            self.cookies = response.cookies
        return await get_response_json(response)

    async def _get(self, url_path: str) -> dict[str, Any]:
        response = await self._server_client.get(
            server_name=self._server_name,
            url_path=url_path,
            cookies=self.cookies,
            timeout=self._timeout,
        )
        await self._raise(response)
        return await get_response_json(response)

    @staticmethod
    async def _raise(response: Any) -> None:
        if not response.ok:
            body = (await response.read()).decode(errors="replace")
            raise SynthEnvApiError(response.status, body)
        await raise_for_status(response)

    async def seed_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/seed_session", payload)

    async def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/step", {"action": action})

    async def observe(self) -> dict[str, Any]:
        return await self._get("/observe")

    async def verify(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/verify", payload)
