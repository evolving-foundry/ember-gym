# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the synth-terminal benchmark dataset from the demo asset package.

The package is the single source of truth (design §3.6): this script extracts
``tasks.jsonl`` from the package zip into ``data/synth_terminal.jsonl`` — the
published thin shell the benchmark config points at. Under the wheel+overlay
layout (design 决议 10) generated data is never committed: ``overlay/**/data/``
is git-ignored and regenerated on demand (``gym eval prepare`` stages this
component into the writable workspace, so the output lands there too); re-run
after rebuilding the package:

    uv run python overlay/benchmarks/synth-terminal/build_package.py
    uv run python overlay/benchmarks/synth-terminal/prepare.py
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent
PACKAGE_ZIP = BENCH_DIR / "assets" / "synth_terminal_demo_v1.zip"
OUTPUT_JSONL = BENCH_DIR / "data" / "synth_terminal.jsonl"


def prepare(package_zip: Path = PACKAGE_ZIP, output_jsonl: Path = OUTPUT_JSONL) -> Path:
    with zipfile.ZipFile(package_zip) as archive:
        raw = archive.read("tasks.jsonl").decode("utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"package {package_zip} contains no tasks")
    pins = {(row["env_release"]["ref_id"], row["env_release"]["version"], row["env_release"]["fingerprint"]) for row in rows}
    if len(pins) != 1:
        raise ValueError(f"tasks pin {len(pins)} different package versions; a benchmark needs exactly one")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {output_jsonl} ({len(rows)} tasks, pin {next(iter(pins))})")
    return output_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-zip", type=Path, default=PACKAGE_ZIP)
    parser.add_argument("--output", type=Path, default=OUTPUT_JSONL)
    args = parser.parse_args()
    try:
        prepare(args.package_zip, args.output)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as error:
        print(f"prepare failed: {error}", file=sys.stderr)
        raise SystemExit(1)
