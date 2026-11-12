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
import asyncio
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.arc_agi.app import (
    ARCAgiResourcesServer,
    ARCAgiResourcesServerConfig,
    ARCAgiVerifyRequest,
    _coerce_grid,
    _extract_last_grid,
)


EXPECTED_GRID = [[6, 8, 6, 8], [4, 6, 4, 6]]


def _make_server() -> ARCAgiResourcesServer:
    config = ARCAgiResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="arc_agi",
    )
    return ARCAgiResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_verify_request(text: str) -> ARCAgiVerifyRequest:
    return ARCAgiVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Predict the output grid."}]},
        response={
            "id": "response_1",
            "created_at": 1.0,
            "model": "model_1",
            "object": "response",
            "output": [
                {
                    "id": "message_1",
                    "content": [{"annotations": [], "text": text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
        question="Example 1:\nInput: [[8, 6], [6, 4]]\nOutput: [[8, 6, 8, 6], [6, 4, 6, 4]]\n\nTest input: [[6, 8], [4, 6]]",
        train_pairs=[{"input": [[8, 6], [6, 4]], "output": [[8, 6, 8, 6], [6, 4, 6, 4]]}],
        test_input=[[6, 8], [4, 6]],
        expected_output_grid=EXPECTED_GRID,
    )


def _verify_text(text: str) -> tuple[float, object]:
    response = asyncio.run(_make_server().verify(_make_verify_request(text)))
    return response.reward, response.extracted_grid


def test_coerce_grid_accepts_rectangular_int_grid():
    assert _coerce_grid([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


def test_coerce_grid_coerces_string_digits():
    assert _coerce_grid([["1", "2"], ["3", "4"]]) == [[1, 2], [3, 4]]


def test_coerce_grid_rejects_malformed_values():
    assert _coerce_grid([]) is None
    assert _coerce_grid([[]]) is None
    assert _coerce_grid([[1, 2], [3]]) is None
    assert _coerce_grid([[1, "x"], [2, 3]]) is None
    assert _coerce_grid([[True, 2], [3, 4]]) is None
    assert _coerce_grid([1, 2, 3]) is None


def test_extract_last_grid_bare_json():
    assert _extract_last_grid("[[6, 8, 6, 8], [4, 6, 4, 6]]") == EXPECTED_GRID


def test_extract_last_grid_json_with_preface_and_trailing_text():
    text = "The transformation tiles the grid 2x2. Here is my answer:\n[[6, 8, 6, 8], [4, 6, 4, 6]]\nLet me know if you need anything else."
    assert _extract_last_grid(text) == EXPECTED_GRID


def test_extract_last_grid_fenced_code_block():
    text = "Answer:\n```json\n[[6, 8, 6, 8], [4, 6, 4, 6]]\n```"
    assert _extract_last_grid(text) == EXPECTED_GRID


def test_extract_last_grid_prefers_fenced_block_over_raw_text():
    text = "[[1, 1], [1, 1]]\nCorrected answer:\n```json\n[[6, 8, 6, 8], [4, 6, 4, 6]]\n```"
    assert _extract_last_grid(text) == EXPECTED_GRID


def test_extract_last_grid_takes_rightmost_candidate():
    text = "[[1, 1], [1, 1]] then [[6, 8, 6, 8], [4, 6, 4, 6]]"
    assert _extract_last_grid(text) == EXPECTED_GRID


def test_extract_last_grid_ignores_flat_lists():
    assert _extract_last_grid("The colors are [6, 8] and [[6, 8, 6, 8], [4, 6, 4, 6]].") == EXPECTED_GRID


def test_extract_last_grid_returns_none_when_absent():
    assert _extract_last_grid("I cannot solve this puzzle.") is None


def test_verify_exact_match_scores_one():
    reward, extracted = _verify_text("[[6, 8, 6, 8], [4, 6, 4, 6]]")
    assert reward == 1.0
    assert extracted == EXPECTED_GRID


def test_verify_string_digit_match_scores_one():
    reward, extracted = _verify_text('[["6", "8", "6", "8"], ["4", "6", "4", "6"]]')
    assert reward == 1.0
    assert extracted == EXPECTED_GRID


def test_verify_wrong_cells_score_zero():
    reward, extracted = _verify_text("[[6, 8, 6, 8], [4, 6, 4, 4]]")
    assert reward == 0.0
    assert extracted == [[6, 8, 6, 8], [4, 6, 4, 4]]


def test_verify_wrong_dimensions_score_zero():
    reward, _ = _verify_text("[[6, 8, 6, 8, 6, 8], [4, 6, 4, 6, 4, 6]]")
    assert reward == 0.0


def test_verify_unparsable_response_scores_zero():
    reward, extracted = _verify_text("I think the answer is a 2x4 grid of alternating colors.")
    assert reward == 0.0
    assert extracted is None
