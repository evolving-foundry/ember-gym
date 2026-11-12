from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf
from responses_api_agents.browsergym_agent.app import (
    BrowserGymAgent,
    BrowserGymAgentConfig,
    BrowserGymRunRequest,
    _task_name,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "overlay" / "responses_api_agents" / "browsergym_agent" / "configs"


def _load_agent_block(filename: str) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(CONFIG_ROOT / filename), resolve=True)
    return loaded["browsergym_agent"]["responses_api_agents"]["browsergym_agent"]


def _runtime_config(block: dict[str, Any], tmp_path: Path) -> BrowserGymAgentConfig:
    return BrowserGymAgentConfig.model_validate(
        {
            "name": "browsergym_agent",
            "host": "127.0.0.1",
            "port": 8000,
            "entrypoint": block["entrypoint"],
            "model_server": block.get("model_server"),
            "dsh_command": block.get("dsh_command"),
            "dsh_profile": block.get("dsh_profile", "headless"),
            "dsh_home": block.get("dsh_home"),
            "dsh_workspace_root": block.get("dsh_workspace_root"),
            "concurrency": block["concurrency"],
            "max_steps": block["max_steps"],
            "request_timeout_s": block["request_timeout_s"],
            "action_max_chars": block["action_max_chars"],
            "observation_max_chars": block["observation_max_chars"],
            "temperature": block.get("temperature", 0.0),
            "har_dir": str(tmp_path / "har"),
        }
    )


def _agent(block: dict[str, Any], tmp_path: Path) -> BrowserGymAgent:
    agent = BrowserGymAgent.model_construct(config=_runtime_config(block, tmp_path))
    agent.__dict__["sem"] = asyncio.Semaphore(1)
    return agent


class _FakeEnv:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.reset_seed: int | None = None
        self.closed = False

    def reset(self, seed: int | None = None) -> tuple[dict[str, str], dict[str, Any]]:
        self.reset_seed = seed
        return {"goal": "complete the test task", "url": "https://example.test"}, {}

    def step(self, action: str) -> tuple[dict[str, str], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        return (
            {"goal": "complete the test task", "url": "https://example.test"},
            1.0,
            True,
            False,
            {"source": "fake-browsergym"},
        )

    def close(self) -> None:
        self.closed = True


def _install_fake_browser_modules(
    monkeypatch: pytest.MonkeyPatch,
    task_family: str | None,
) -> tuple[list[tuple[str, dict[str, Any]]], list[_FakeEnv]]:
    made: list[tuple[str, dict[str, Any]]] = []
    envs: list[_FakeEnv] = []

    gymnasium = types.ModuleType("gymnasium")

    def make(task_name: str, **kwargs: Any) -> _FakeEnv:
        made.append((task_name, kwargs))
        env = _FakeEnv()
        envs.append(env)
        return env

    gymnasium.make = make  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    browsergym = types.ModuleType("browsergym")
    browsergym.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "browsergym", browsergym)

    core = types.ModuleType("browsergym.core")
    monkeypatch.setitem(sys.modules, "browsergym.core", core)
    browsergym.core = core  # type: ignore[attr-defined]

    if task_family is not None:
        feature = types.ModuleType(f"browsergym.{task_family}")
        monkeypatch.setitem(sys.modules, f"browsergym.{task_family}", feature)
        setattr(browsergym, task_family, feature)

    return made, envs


def test_browsergym_configs_resolve_to_the_agent_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BROWSERGYM_HAR_DIR", raising=False)

    model_block = _load_agent_block("browsergym_agent.yaml")
    assert model_block["entrypoint"] == "app.py"
    assert model_block["model_server"] == {"type": "responses_api_models", "name": "policy_model"}
    assert model_block["concurrency"] == 1
    assert model_block["max_steps"] == 50
    assert model_block["request_timeout_s"] == 180
    assert model_block["action_max_chars"] == 12000
    assert model_block["observation_max_chars"] == 50000
    assert model_block["temperature"] == 0.0
    assert model_block["har_dir"] == "results/browsergym_har"

    model_config = _runtime_config(model_block, tmp_path)
    assert model_config.model_server is not None
    assert model_config.model_server.name == "policy_model"
    assert model_config.dsh_command is None

    monkeypatch.setenv("RSI_DSH_COMMAND", "/opt/rsi/bin/dsh")
    monkeypatch.setenv("RSI_DSH_HOME", str(tmp_path / "dsh-home"))
    monkeypatch.setenv("RSI_DSH_WORKSPACE_ROOT", str(tmp_path / "dsh-work"))
    monkeypatch.setenv("RSI_BROWSERGYM_MAX_STEPS", "7")

    dsh_block = _load_agent_block("browsergym_dsh_agent.yaml")
    assert dsh_block["dsh_command"] == "/opt/rsi/bin/dsh"
    assert dsh_block["dsh_home"] == str(tmp_path / "dsh-home")
    assert dsh_block["dsh_workspace_root"] == str(tmp_path / "dsh-work")
    assert dsh_block["dsh_profile"] == "headless"
    assert dsh_block["max_steps"] == "7"

    dsh_config = _runtime_config(dsh_block, tmp_path)
    assert dsh_config.dsh_command == "/opt/rsi/bin/dsh"
    assert dsh_config.max_steps == 7
    assert dsh_config.model_server is None


@pytest.mark.parametrize(
    ("task_name", "task_family"),
    [
        ("assistantbench.smoke.1", "assistantbench"),
        ("webarena_verified.42.17.1", "webarena_verified"),
        ("workarena.servicenow.smoke.1", "workarena"),
    ],
)
def test_browsergym_run_routes_task_families_to_registered_environments(
    task_name: str,
    task_family: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made, envs = _install_fake_browser_modules(monkeypatch, task_family)
    block = _load_agent_block("browsergym_agent.yaml")
    agent = _agent(block, tmp_path)
    captured_actions: list[str] = []

    async def fake_next_action(self: BrowserGymAgent, **kwargs: Any) -> tuple[str, dict[str, int]]:
        captured_actions.append(kwargs["task_goal"])
        return "send_msg_to_user('done')", {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
            "cached_tokens": 1,
        }

    monkeypatch.setattr(BrowserGymAgent, "_next_action", fake_next_action)
    body = BrowserGymRunRequest.model_validate(
        {
            "responses_create_params": {"input": "run the BrowserGym smoke task"},
            "task_name": task_name,
            "task_id": "task-1",
            "seed": 7,
        }
    )

    result = asyncio.run(agent.run(None, body))

    assert len(made) == 1
    made_task_name, kwargs = made[0]
    assert made_task_name == f"browsergym/{task_name}"
    assert kwargs["headless"] is True
    if task_family == "webarena_verified":
        assert kwargs["pw_context_kwargs"]["record_har_content"] == "embed"
        assert kwargs["pw_context_kwargs"]["record_har_mode"] == "full"
    else:
        assert "pw_context_kwargs" not in kwargs

    assert len(envs) == 1
    assert envs[0].reset_seed == 7
    assert envs[0].actions == ["send_msg_to_user('done')"]
    assert envs[0].closed is True
    assert captured_actions == ["complete the test task"]
    assert result.task_name == task_name
    assert result.task_id == "task-1"
    assert result.reward == 1.0
    assert result.steps == 1
    assert result.terminated is True
    assert result.truncated is False
    assert result.final_response == "done"
    assert result.response.usage.total_tokens == 5


def test_browsergym_run_rejects_unknown_task_family_before_creating_an_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made, _ = _install_fake_browser_modules(monkeypatch, None)
    agent = _agent(_load_agent_block("browsergym_agent.yaml"), tmp_path)
    body = BrowserGymRunRequest.model_validate(
        {
            "responses_create_params": {"input": "unsupported task"},
            "task_name": "unknown_family.task.1",
        }
    )

    with pytest.raises(ValueError, match="Unsupported BrowserGym task"):
        asyncio.run(agent.run(None, body))

    assert made == []


def test_task_name_strips_browsergym_prefix_and_reconstructs_webarena_verified_tasks() -> None:
    explicit = BrowserGymRunRequest.model_validate(
        {
            "responses_create_params": {"input": "task"},
            "task_name": "browsergym/workarena.servicenow.smoke.1",
        }
    )
    reconstructed = BrowserGymRunRequest.model_validate(
        {
            "responses_create_params": {"input": "task"},
            "intent_template_id": 42,
            "task_id": 17,
            "revision": 1,
        }
    )
    missing_fields = BrowserGymRunRequest.model_validate(
        {
            "responses_create_params": {"input": "task"},
            "intent_template_id": 42,
            "task_id": 17,
        }
    )

    assert _task_name(explicit) == "workarena.servicenow.smoke.1"
    assert _task_name(reconstructed) == "webarena_verified.42.17.1"
    with pytest.raises(ValueError, match="dataset row must provide task_name"):
        _task_name(missing_fields)
