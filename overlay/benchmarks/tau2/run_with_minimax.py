"""Launch a Gym command with the τ simulated-user key from MiniMax settings.

The key is read at runtime and passed only in the child environment. It is
never written to a repository config or included in the child command line.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


MINIMAX_API_HOSTS = {"api.minimaxi.com", "api.minimax.cn", "api.minimax.io"}


def simulator_environment(settings_path: Path) -> dict[str, str]:
    settings = json.loads(settings_path.expanduser().read_text(encoding="utf-8"))
    env = settings.get("env")
    if not isinstance(env, dict):
        raise ValueError("MiniMax settings must contain an env object")
    token = env.get("ANTHROPIC_AUTH_TOKEN")
    base_url = env.get("ANTHROPIC_BASE_URL")
    model = env.get("ANTHROPIC_DEFAULT_SONNET_MODEL") or env.get("ANTHROPIC_DEFAULT_OPUS_MODEL")
    if not all(isinstance(value, str) and value for value in (token, base_url, model)):
        raise ValueError("MiniMax settings lack the auth token, base URL, or model")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in MINIMAX_API_HOSTS
        or parsed.path.rstrip("/") != "/anthropic"
    ):
        raise ValueError("MiniMax Anthropic base URL must use an official HTTPS MiniMax API host")
    return {
        "TAU_USER_BASE_URL": urlunsplit((parsed.scheme, parsed.netloc, "/v1", "", "")),
        "TAU_USER_API_KEY": token,
        "TAU_USER_MODEL_NAME": model.removesuffix("[1m]"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Gym command after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a Gym command after --")
    settings_path = os.environ.get("MINIMAX_SETTINGS_PATH")
    if not settings_path:
        parser.error("MINIMAX_SETTINGS_PATH must point to the MiniMax settings file")
    child_env = os.environ.copy()
    child_env.update(simulator_environment(Path(settings_path)))
    return subprocess.run(command, env=child_env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
