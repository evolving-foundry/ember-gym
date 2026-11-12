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
"""ARC-AGI resources server.

Each task presents a few example input/output grid pairs and one test input
grid; the model must apply the induced transformation and return the output
grid. Verification is a deterministic exact match against the expected grid
(no judge, no sandbox), following the ARC Prize scoring rule: the predicted
grid must equal the ground-truth grid cell for cell, including dimensions.
"""
import json
import logging
import re
from typing import Any, List, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class ARCAgiResourcesServerConfig(BaseResourcesServerConfig):
    pass


class ARCAgiRunRequest(BaseRunRequest):
    # Pre-rendered task text (examples + test input) shown to the agent.
    question: str
    train_pairs: List[dict]
    test_input: List[List[int]]
    expected_output_grid: List[List[int]]


class ARCAgiVerifyRequest(ARCAgiRunRequest, BaseVerifyRequest):
    pass


class ARCAgiVerifyResponse(BaseVerifyResponse):
    expected_output_grid: List[List[int]]
    extracted_grid: Optional[List[List[int]]]


def _coerce_grid(value: Any) -> Optional[List[List[int]]]:
    """Return ``value`` as a rectangular list-of-lists of ints, or None.

    Cell values may be ints or single-digit strings (models sometimes emit
    ``"1"`` instead of ``1``); anything else rejects the candidate.
    """
    if not isinstance(value, list) or not value:
        return None

    grid: List[List[int]] = []
    width: Optional[int] = None
    for row in value:
        if not isinstance(row, list) or not row:
            return None

        if width is None:
            width = len(row)
        elif len(row) != width:
            return None

        coerced_row: List[int] = []
        for cell in row:
            if isinstance(cell, bool):
                return None
            if isinstance(cell, int):
                coerced_row.append(cell)
            elif isinstance(cell, str) and cell.strip().lstrip("-").isdigit():
                coerced_row.append(int(cell))
            else:
                return None
        grid.append(coerced_row)

    return grid


def _last_grid_in_text(text: str) -> Optional[List[List[int]]]:
    """Return the rightmost parsable grid in ``text``, or None.

    Scans every ``[`` position with ``json.JSONDecoder.raw_decode`` so a grid
    embedded in surrounding prose is still found; later candidates win so
    that a model that self-corrects is scored on its final answer.
    """
    decoder = json.JSONDecoder()
    best: Optional[List[List[int]]] = None
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        grid = _coerce_grid(value)
        if grid is not None:
            best = grid
    return best


def _extract_last_grid(text: str) -> Optional[List[List[int]]]:
    """Extract the predicted output grid from a free-form response.

    Fenced code blocks are preferred (in reverse order), then any JSON
    array-of-arrays in the raw text. This tolerates common formats: a bare
    JSON grid, a grid in a ```json fence, or a JSON grid followed by
    commentary.
    """
    for match in reversed(list(re.finditer(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL))):
        grid = _last_grid_in_text(match.group(1))
        if grid is not None:
            return grid

    return _last_grid_in_text(text)


class ARCAgiResourcesServer(SimpleResourcesServer):
    config: ARCAgiResourcesServerConfig

    async def verify(self, body: ARCAgiVerifyRequest) -> ARCAgiVerifyResponse:
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)
        extracted_grid = _extract_last_grid(combined_response)
        reward = 1.0 if extracted_grid == body.expected_output_grid else 0.0

        logging.debug("ARC-AGI task: reward=%s extracted=%s", reward, extracted_grid)

        return ARCAgiVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_grid=extracted_grid,
        )


if __name__ == "__main__":
    ARCAgiResourcesServer.run_webserver()
