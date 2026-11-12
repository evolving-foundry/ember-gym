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
import json

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()
task = server_client.post(
    server_name="arc_agi",
    url_path="/verify",
    json={
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": "Predict the output grid.",
                },
            ]
        },
        "response": {
            "id": "response_1",
            "created_at": 1.0,
            "model": "model_1",
            "object": "response",
            "output": [
                {
                    "id": "message_1",
                    "content": [
                        {
                            "annotations": [],
                            "text": "[[6, 8], [4, 6]]",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
        "question": "Example 1:\nInput: [[8, 6], [6, 4]]\nOutput: [[8, 6, 8, 6], [6, 4, 6, 4]]\n\nTest input: [[6, 8], [4, 6]]",
        "train_pairs": [
            {"input": [[8, 6], [6, 4]], "output": [[8, 6, 8, 6], [6, 4, 6, 4]]}
        ],
        "test_input": [[6, 8], [4, 6]],
        "expected_output_grid": [[6, 8, 6, 8], [4, 6, 4, 6]],
    },
)
result = asyncio.run(task)
print(json.dumps(asyncio.run(result.json()), indent=4))
