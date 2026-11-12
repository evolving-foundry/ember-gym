"""Shared runtime bootstrap for the AgentGym launcher and direct service entry points."""

from __future__ import annotations

import os
from pathlib import Path

from agent_gym.layout import GymLayout, LayoutError
from agent_gym.overlay import verify_manifest


def shadowing_component_root(framework_file: Path, component_roots: tuple[Path, ...]) -> Path | None:
    resolved = framework_file.resolve(strict=False)
    for root in component_roots:
        shadow = (root / "nemo_gym").resolve(strict=False)
        if resolved == shadow or shadow in resolved.parents:
            return shadow
    return None


def prepare_runtime(
    *,
    repo_root: str | Path | None = None,
    overlay_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
    data_root: str | Path | None = None,
    require_overlay: bool = True,
    verify_trees: bool = True,
    chdir: bool = True,
    import_framework: bool = True,
) -> GymLayout:
    """Resolve, validate and activate one runtime contract.

    Validation happens before ``chdir`` so a bad deployment fails without changing process
    state. ``nemo_gym`` is imported only after the workspace is active, preserving the
    upstream cache-location contract.
    """
    layout = GymLayout.discover(
        repo_root=repo_root,
        overlay_root=overlay_root,
        workspace_root=workspace_root,
        data_root=data_root,
    )
    if require_overlay:
        layout.require_overlay()
        layout.assert_framework_not_shadowed()
        verify_manifest(layout, verify_trees=verify_trees)
    else:
        layout.assert_framework_not_shadowed()

    if chdir:
        layout.workspace_root.mkdir(parents=True, exist_ok=True)
        os.chdir(layout.workspace_root)
    os.environ.update(layout.gym_env())

    if import_framework:
        import nemo_gym

        augment = getattr(nemo_gym, "_augment_sys_path", None)
        if callable(augment):
            augment()
        framework_file = Path(nemo_gym.__file__).resolve(strict=False)
        shadow = shadowing_component_root(framework_file, layout.component_roots())
        if shadow is not None:
            raise LayoutError(f"nemo_gym resolved from a component shadow: {framework_file}")
    return layout
