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
"""Unit tests for hermes_agent usage derivation from training output items.

nemo_gym (and the heavy hermes runtime) are stubbed at the sys.modules level;
_usage_from_output_items itself only inspects token-id lists.
"""

import importlib
import importlib.metadata
import sys
import types
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    saved = dict(sys.modules)
    yield
    sys.modules.clear()
    sys.modules.update(saved)


def _load_app():
    class LooseModule(types.ModuleType):
        def __getattr__(self, name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            stub_type = type(name, (), {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)})
            setattr(self, name, stub_type)
            return stub_type

    def stub(name: str) -> types.ModuleType:
        module = LooseModule(name)
        sys.modules[name] = module
        return module

    nemo = stub("nemo_gym")
    nemo.__path__ = []
    # Overlay tests may have imported the real module first; this test deliberately stubs its
    # heavy runtime imports, so load a fresh module rather than reusing that cached instance.
    sys.modules.pop("responses_api_agents.hermes_agent.app", None)
    for sub in (
        "nemo_gym.base_resources_server",
        "nemo_gym.base_responses_api_agent",
        "nemo_gym.config_types",
        "nemo_gym.global_config",
        "nemo_gym.openai_utils",
        "nemo_gym.responses_converter",
        "nemo_gym.rollout_observability",
        "nemo_gym.sandbox",
        "nemo_gym.sandbox.access",
        "nemo_gym.sandbox.config",
        "nemo_gym.sandbox.providers",
        "nemo_gym.server_utils",
        "nemo_gym.tool_access",
    ):
        stub(sub)
    sys.modules["nemo_gym.sandbox"].__path__ = []
    stub("model_tools")
    stub("toolsets")

    class HermesDistribution:
        version = "test"

        def read_text(self, name: str) -> None:
            return None

    with patch.object(importlib.metadata, "distribution", return_value=HermesDistribution()):
        return importlib.import_module("responses_api_agents.hermes_agent.app")


def test_usage_sums_token_id_lengths_across_training_items():
    app = _load_app()
    training_type = sys.modules["nemo_gym.openai_utils"].NeMoGymResponseOutputMessageForTraining
    items = [
        training_type(prompt_token_ids=[1, 2, 3], generation_token_ids=[4, 5]),
        object(),  # non-training items are ignored
        training_type(prompt_token_ids=[6, 7], generation_token_ids=[8]),
    ]
    usage = app._usage_from_output_items(items)
    assert usage.input_tokens == 5
    assert usage.output_tokens == 3
    assert usage.total_tokens == 8
    assert usage.input_tokens_details.cached_tokens == 0


def test_usage_is_zero_without_token_ids():
    app = _load_app()
    usage = app._usage_from_output_items([])
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0
