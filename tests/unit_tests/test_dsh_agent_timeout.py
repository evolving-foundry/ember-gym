import asyncio
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import responses_api_agents.dsh_agent.app as dsh_app


class _FakeProcess:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.stopped = False
        self.killed = False

    async def communicate(self):
        if self.stopped:
            return b"", b""
        await asyncio.sleep(10)
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.stopped = True


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_kill_process_tree_uses_process_group(monkeypatch):
    calls = []
    monkeypatch.setattr(dsh_app.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    proc = _FakeProcess(pid=987)

    dsh_app._kill_process_tree(proc)

    assert calls == [(987, signal.SIGKILL)]
    assert proc.killed is False


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_kill_process_tree_falls_back_to_direct_kill(monkeypatch):
    def deny(pid, sig):
        raise PermissionError

    monkeypatch.setattr(dsh_app.os, "killpg", deny)
    proc = _FakeProcess()

    dsh_app._kill_process_tree(proc)

    assert proc.killed is True


@pytest.mark.asyncio
async def test_run_dsh_timeout_kills_process_tree(monkeypatch):
    proc = _FakeProcess()
    killed = []

    async def create_process(*args, **kwargs):
        assert kwargs.get("start_new_session") is (os.name == "posix")
        return proc

    def kill_tree(process):
        killed.append(process)
        process.stopped = True

    monkeypatch.setattr(dsh_app.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(dsh_app, "_kill_process_tree", kill_tree)
    agent = SimpleNamespace(
        config=SimpleNamespace(dsh_command="dsh", profile="headless", timeout=0.01),
    )

    with pytest.raises(RuntimeError, match="did not finish within"):
        await dsh_app.DshAgent._run_dsh(agent, "task")

    assert killed == [proc]


def test_gaia_dsh_config_counts_failed_agent_runs_as_zero():
    config_path = Path(__file__).resolve().parents[2] / "overlay" / "benchmarks" / "gaia" / "config_dsh.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["route_failures_to_sidecar"] is True
    assert config["count_failure_classes_as_zero"] == ["agent_run_error"]
    assert config["count_missing_rollouts_as_zero"] is True
