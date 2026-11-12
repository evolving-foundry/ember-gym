from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import nemo_gym
import pytest
from nemo_gym.sandbox import (
    Sandbox,
    SandboxSpec,
    SandboxStatus,
    resolve_provider_config,
    resolve_provider_metadata,
)
from omegaconf import OmegaConf


IMAGE = "busybox:1.36"
CONTAINER_LABEL = "nemo-gym.sandbox=1"


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _sandbox_container_ids() -> set[str]:
    output = _docker("ps", "-aq", "--filter", f"label={CONTAINER_LABEL}").stdout
    return {line.strip() for line in output.splitlines() if line.strip()}


def _require_docker() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not available on PATH")
    probe = _docker("info", "--format", "{{.ServerVersion}}", check=False)
    if probe.returncode != 0:
        pytest.skip(f"Docker daemon is unavailable: {probe.stderr.strip()[-300:]}")


@pytest.mark.integration
@pytest.mark.sandbox
def test_nemo_gym_sandbox_contract_against_real_docker() -> None:
    _require_docker()
    if _docker("image", "inspect", IMAGE, check=False).returncode != 0:
        pytest.skip(f"{IMAGE} is not available locally; run 'docker pull {IMAGE}'")

    provider_path = (
        Path(nemo_gym.__file__).resolve().parent / "sandbox" / "providers" / "docker" / "configs" / "docker.yaml"
    )
    named_configs = OmegaConf.to_container(OmegaConf.load(provider_path), resolve=True)
    provider = resolve_provider_config("sandbox", named_configs)
    metadata = resolve_provider_metadata("sandbox", named_configs)

    assert list(provider) == ["docker"]
    assert metadata == {"sandbox-api": "docker-cli"}

    containers_before = _sandbox_container_ids()
    sandbox = Sandbox(
        provider,
        SandboxSpec(
            image=IMAGE,
            ttl_s=60,
            ready_timeout_s=60,
            env={"SMOKE": "ok"},
        ),
    )
    try:
        sandbox.start()
        assert sandbox.status() is SandboxStatus.RUNNING
        assert _sandbox_container_ids() - containers_before

        env_result = sandbox.exec("printf '%s' \"$SMOKE\"")
        assert env_result.return_code == 0
        assert env_result.error_type is None
        assert env_result.stdout == "ok"
        assert not env_result.stderr

        file_result = sandbox.exec("printf 'sandbox-file' > /tmp/nemo-gym-smoke && cat /tmp/nemo-gym-smoke")
        assert file_result.return_code == 0
        assert file_result.error_type is None
        assert file_result.stdout == "sandbox-file"
    finally:
        sandbox.stop()

    assert sandbox.status() is SandboxStatus.STOPPED
    assert _sandbox_container_ids() == containers_before
