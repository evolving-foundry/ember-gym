# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from responses_api_agents.dsh_agent.app import resolve_dsh_workspace_root


def test_explicit_workspace_wins(tmp_path):
    root = resolve_dsh_workspace_root(
        str(tmp_path / "candidate"),
        environ={"AGENT_GYM_WORKSPACE": str(tmp_path / "workspace")},
    )
    assert root == (tmp_path / "candidate").resolve()


def test_relative_workspace_is_resolved_under_gym_workspace(tmp_path):
    root = resolve_dsh_workspace_root(
        "candidate",
        environ={"AGENT_GYM_WORKSPACE": str(tmp_path / "workspace")},
    )
    assert root == (tmp_path / "workspace" / "candidate").resolve()


def test_missing_config_uses_gym_workspace_instead_of_component_cwd(tmp_path):
    component = tmp_path / "overlay" / "responses_api_agents" / "dsh_agent"
    root = resolve_dsh_workspace_root(
        None,
        environ={"AGENT_GYM_WORKSPACE": str(tmp_path / "workspace")},
        cwd=component,
    )
    assert root == (tmp_path / "workspace" / "dsh").resolve()
    assert component.resolve() not in root.parents


def test_missing_all_roots_uses_temp_directory(tmp_path):
    root = resolve_dsh_workspace_root(
        None,
        environ={},
        cwd=tmp_path / "component",
        temp_root=tmp_path / "tmp",
    )
    assert root == (tmp_path / "tmp" / "agent-gym-dsh").resolve()


def test_overlay_workspace_is_rejected(tmp_path):
    overlay = tmp_path / "overlay"
    with pytest.raises(RuntimeError, match="read-only overlay"):
        resolve_dsh_workspace_root(
            str(overlay / "responses_api_agents" / "dsh_agent"),
            environ={"AGENT_GYM_OVERLAY": str(overlay)},
        )
