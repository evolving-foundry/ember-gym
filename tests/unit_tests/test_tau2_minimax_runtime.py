import json
from pathlib import Path

import pytest
from benchmarks.tau2.run_with_minimax import simulator_environment


def test_reads_minimax_settings_without_storing_key(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json.minimax"
    settings.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": "https://api.minimaxi.com/anthropic",
                    "ANTHROPIC_AUTH_TOKEN": "private-test-key",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M3",
                }
            }
        ),
        encoding="utf-8",
    )
    resolved = simulator_environment(settings)
    assert resolved == {
        "TAU_USER_BASE_URL": "https://api.minimaxi.com/v1",
        "TAU_USER_API_KEY": "private-test-key",
        "TAU_USER_MODEL_NAME": "MiniMax-M3",
    }


def test_rejects_non_minimax_host_before_forwarding_key(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json.minimax"
    settings.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": "https://example.invalid/anthropic",
                    "ANTHROPIC_AUTH_TOKEN": "private-test-key",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M3",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official HTTPS MiniMax API host"):
        simulator_environment(settings)
