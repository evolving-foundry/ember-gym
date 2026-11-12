#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize BrowserGym's public AssistantBench validation task catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
BROWSERGYM_COMMIT = "c05d7f38f4f788d6e4ee960512c3d5c8aabc6e42"
DATASET_REVISION = "482cbbc0400f6d048438c4021727f21a10cbff49"
VALIDATION_TASKS = 33


def prepare(output_fpath: Path | None = None) -> Path:
    """Write only public validation task IDs; answers stay in BrowserGym's evaluator."""
    output_fpath = (output_fpath or DATA_DIR / "assistantbench_validation.jsonl").expanduser().resolve()
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    task_ids = [f"assistantbench.validation.{index}" for index in range(VALIDATION_TASKS)]
    digest = hashlib.sha256()
    with output_fpath.open("w", encoding="utf-8", newline="\n") as stream:
        for task_name in task_ids:
            encoded = (
                json.dumps(
                    {
                        "task_id": task_name,
                        "task_name": task_name,
                        # BaseRunRequest requires Responses params even though the
                        # BrowserGym agent derives its goal from the task env.
                        "responses_create_params": {"input": []},
                        "dataset_revision": DATASET_REVISION,
                        "browsergym_commit": BROWSERGYM_COMMIT,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            stream.write(encoded.decode("utf-8"))
            digest.update(encoded)

    manifest = {
        "schema_version": 1,
        "benchmark": "AssistantBench",
        "split": "public-validation",
        "dataset_revision": DATASET_REVISION,
        "browsergym_commit": BROWSERGYM_COMMIT,
        "row_count": len(task_ids),
        "task_ids_sha256": hashlib.sha256("\n".join(task_ids).encode()).hexdigest(),
        "materialized_jsonl_sha256": digest.hexdigest(),
        "file": output_fpath.name,
        "note": "Task IDs only; no public test prediction or answer data copied.",
    }
    output_fpath.with_suffix(output_fpath.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(task_ids)} AssistantBench validation task IDs: {output_fpath}")
    return output_fpath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
