# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic GAIA quasi-exact answer verifier for the public validation split."""

from __future__ import annotations

import re
import string
from typing import Any

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class GaiaResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GaiaRunRequest(BaseRunRequest):
    question: str
    expected_answer: str


class GaiaVerifyRequest(GaiaRunRequest, BaseVerifyRequest):
    pass


class GaiaVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: str | None


def _normalize_string(value: str, *, remove_punctuation: bool = True) -> str:
    value = re.sub(r"\s", "", value).lower()
    if remove_punctuation:
        value = value.translate(str.maketrans("", "", string.punctuation))
    return value


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_number(value: str) -> float:
    for char in ("$", "%", ","):
        value = value.replace(char, "")
    try:
        return float(value)
    except ValueError:
        return float("inf")


def gaia_quasi_exact_match(model_answer: str | None, expected_answer: str) -> bool:
    """Mirror GAIA leaderboard scoring: number, ordered list, then normalized text."""
    model_answer = "None" if model_answer is None else model_answer
    if _is_float(expected_answer):
        return _normalize_number(model_answer) == float(expected_answer)

    if any(char in expected_answer for char in (",", ";")):
        expected_items = re.split(r"[,;]", expected_answer)
        answer_items = re.split(r"[,;]", model_answer)
        if len(expected_items) != len(answer_items):
            return False
        for answer_item, expected_item in zip(answer_items, expected_items):
            if _is_float(expected_item):
                if _normalize_number(answer_item) != float(expected_item):
                    return False
            elif _normalize_string(answer_item, remove_punctuation=False) != _normalize_string(
                expected_item, remove_punctuation=False
            ):
                return False
        return True

    return _normalize_string(model_answer) == _normalize_string(expected_answer)


def _extract_final_answer(text: str, expected_answer: str) -> str:
    """Select an explicitly marked answer without consulting the answer value."""
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    marked_lines = re.findall(
        r"(?im)^\s*(?:\*{1,2})?\s*(?:final\s+)?answer\s*:\s*(.+)$", text
    )
    if not marked_lines:
        return text
    answer = marked_lines[-1].strip()
    if _is_float(expected_answer):
        # A marked line such as "Answer: 10 stops" is an unambiguous numeric
        # answer even when the agent adds units or explanation after it.
        number = re.match(r"^\*{0,2}\s*(\$?[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?)(?=\s|\*|$)", answer)
        if number:
            return number.group(1)
    return answer.strip("* ")


class GaiaResourcesServer(SimpleResourcesServer):
    config: GaiaResourcesServerConfig

    async def verify(self, body: GaiaVerifyRequest) -> GaiaVerifyResponse:
        answer_parts: list[str] = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type == "output_text":
                    answer_parts.append(content_item.text)
        answer = "\n".join(answer_parts).strip() or None
        if answer is not None:
            answer = _extract_final_answer(answer, body.expected_answer)
        return GaiaVerifyResponse(
            **body.model_dump(),
            reward=float(gaia_quasi_exact_match(answer, body.expected_answer)),
            extracted_answer=answer,
        )


if __name__ == "__main__":
    GaiaResourcesServer.run_webserver()
