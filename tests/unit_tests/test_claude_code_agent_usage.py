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
"""Unit tests for claude_code_agent parse_stream_json token accounting.

nemo_gym is stubbed at the sys.modules level so the harness module imports on a
bare interpreter; parse_stream_json itself only touches plain dicts.
"""

import importlib
import json
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    saved = dict(sys.modules)
    yield
    sys.modules.clear()
    sys.modules.update(saved)


def _install_nemo_gym_stubs() -> None:
    class _LooseModule(types.ModuleType):
        """Yields a permissive stub object for any missing attribute.

        parse_stream_json itself only touches plain dicts, so the imported
        nemo_gym names are never exercised — they just need to exist.
        """

        def __getattr__(self, name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            stub = type(name, (), {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)})
            setattr(self, name, stub)
            return stub

    def _stub_module(name: str, **attrs) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        module = _LooseModule(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    nemo = _stub_module("nemo_gym")
    nemo.__path__ = []  # mark as package so submodule imports resolve through sys.modules
    for sub in (
        "nemo_gym.base_resources_server",
        "nemo_gym.base_responses_api_agent",
        "nemo_gym.config_types",
        "nemo_gym.global_config",
        "nemo_gym.openai_utils",
        "nemo_gym.rollout_observability",
        "nemo_gym.server_utils",
        "nemo_gym.skills",
    ):
        _stub_module(sub)


def _load_app():
    _install_nemo_gym_stubs()
    return importlib.import_module("responses_api_agents.claude_code_agent.app")


def test_parse_stream_json_accumulates_cache_read_tokens():
    app = _load_app()
    stdout = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hello"}],
                        "usage": {"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 20},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "num_turns": 1,
                    "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 80},
                }
            ),
        ]
    )
    output_items, metadata = app.parse_stream_json(stdout)
    assert metadata["input_tokens"] == 110
    assert metadata["output_tokens"] == 54
    assert metadata["cached_tokens"] == 100
    assert any(getattr(item, "role", None) == "assistant" for item in output_items)


def test_parse_stream_json_tolerates_missing_cache_fields():
    app = _load_app()
    _, metadata = app.parse_stream_json(
        json.dumps({"type": "result", "subtype": "success", "usage": {"input_tokens": 3, "output_tokens": 2}})
    )
    assert metadata["cached_tokens"] == 0
    assert metadata["input_tokens"] == 3
    assert metadata["output_tokens"] == 2
