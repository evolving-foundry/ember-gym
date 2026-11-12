from __future__ import annotations

from pathlib import Path

import pytest

from agent_gym import doctor


def test_entrypoint_probe_prefers_active_interpreter_scripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    interpreter = tmp_path / "bin" / "python"
    calls: list[str | None] = []
    expected = str(interpreter.parent / "gym")

    def fake_which(name: str, path: str | None = None) -> str | None:
        calls.append(path)
        if path == str(interpreter.parent):
            return expected
        return "/usr/bin/gym"

    monkeypatch.setattr(doctor.sys, "executable", str(interpreter))
    monkeypatch.setattr(doctor.shutil, "which", fake_which)

    assert doctor._entrypoint_script("gym") == expected
    assert calls == [str(interpreter.parent)]


def test_entrypoint_probe_falls_back_to_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    interpreter = tmp_path / "bin" / "python"

    def fake_which(name: str, path: str | None = None) -> str | None:
        return None if path is not None else "/usr/bin/gym"

    monkeypatch.setattr(doctor.sys, "executable", str(interpreter))
    monkeypatch.setattr(doctor.shutil, "which", fake_which)

    assert doctor._entrypoint_script("gym") == "/usr/bin/gym"
