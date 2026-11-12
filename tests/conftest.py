from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = Path(os.environ.get("AGENT_GYM_OVERLAY", REPO_ROOT / "overlay")).resolve()

if OVERLAY_ROOT.is_dir():
    overlay_entry = str(OVERLAY_ROOT)
    if overlay_entry not in sys.path:
        sys.path.insert(0, overlay_entry)
    extra_roots = [overlay_entry, str(REPO_ROOT)]
    configured = os.environ.get("NEMO_GYM_EXTRA_ROOTS")
    if configured:
        extra_roots.extend(item for item in configured.split(os.pathsep) if item)
    os.environ["NEMO_GYM_EXTRA_ROOTS"] = os.pathsep.join(dict.fromkeys(extra_roots))
