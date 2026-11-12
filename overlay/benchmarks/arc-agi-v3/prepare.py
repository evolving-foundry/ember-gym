# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the ARC-AGI-3 benchmark (official preview games).

ARC-AGI-3 is an interactive benchmark: each JSONL row is an evaluation unit
(game x action budget), not a static puzzle. Nothing is downloaded here — the
games run on the ARC Prize API (three.arcprize.org) and need an API key.

The game list comes from the pinned manifest (games_manifest.json), following
the design's decision to restrict the first phase to the official preview
games. With --probe (and ARC_AGI_V3_API_KEY set in the environment), each
manifest slug is checked against the live /api/games listing; a slug the API
no longer offers fails the run instead of silently producing a stale row.

Re-pinning the manifest changes the benchmark definition and must be recorded
explicitly in the commit message.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "arc_agi_v3_games.jsonl"
MANIFEST_FPATH = BENCHMARK_DIR / "games_manifest.json"

# Guard against silent empty/partial output if the manifest is trimmed by
# accident or a probe removes every game.
EXPECTED_MIN_GAMES = 3


def _repo_root() -> Path:
    return BENCHMARK_DIR.parent.parent


def _probe_manifest(api_base: str, games: list[dict]) -> list[dict]:
    """Cross-check manifest slugs against the live API; drop nothing, fail loudly."""
    sys.path.insert(0, str(_repo_root()))
    from responses_api_agents.arc_agi_v3.game_client import ArcAgiV3GameClient

    api_key = os.environ.get("ARC_AGI_V3_API_KEY")
    if not api_key:
        raise SystemExit(
            "--probe requires the ARC_AGI_V3_API_KEY environment variable "
            "(or run without --probe to use the pinned manifest only)"
        )

    async def _list() -> list[dict]:
        async with ArcAgiV3GameClient(api_base=api_base, api_key=api_key) as client:
            return await client.list_games()

    live = asyncio.run(_list())
    live_ids = [str(entry.get("game_id", "")) for entry in live]
    for game in games:
        slug = game["game_id"]
        if not any(game_id == slug or game_id.startswith(f"{slug}-") for game_id in live_ids):
            raise RuntimeError(
                f"manifest game {slug!r} not offered by the live API "
                f"(available: {live_ids[:20]}{'...' if len(live_ids) > 20 else ''}) — "
                "re-pin the manifest deliberately and record it in the commit message"
            )
    print(f"Probed {len(games)} manifest games against {len(live_ids)} live games: OK")
    return games


def prepare(
    games_filter: list[str] | None = None,
    probe: bool = False,
    api_base: str = "https://three.arcprize.org",
) -> Path:
    manifest = json.loads(MANIFEST_FPATH.read_text(encoding="utf-8"))
    games = manifest["games"]
    if games_filter:
        wanted = set(games_filter)
        missing = wanted - {game["game_id"] for game in games}
        if missing:
            raise SystemExit(f"--games references unknown manifest games: {sorted(missing)}")
        games = [game for game in games if game["game_id"] in wanted]

    if probe:
        games = _probe_manifest(api_base, games)

    if len(games) < EXPECTED_MIN_GAMES:
        raise RuntimeError(
            f"only {len(games)} games selected (minimum {EXPECTED_MIN_GAMES}); "
            "a trimmed game list silently shrinks the benchmark — re-check the manifest/filter"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as fout:
        for game in games:
            # max_actions/human_baseline_actions stay null: upstream cutoffs are not
            # published per game yet, so the agent config default budget applies.
            row = {
                "game_id": game["game_id"],
                "title": game.get("title"),
                "actions_available": None,
                "max_actions": None,
                "human_baseline_actions": None,
            }
            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(games)} evaluation units to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="*", help="Optional game_id subset filter (cost control).")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Check manifest slugs against the live /api/games listing (needs ARC_AGI_V3_API_KEY).",
    )
    parser.add_argument("--api-base", default="https://three.arcprize.org", help="ARC-AGI-3 API base URL.")
    args = parser.parse_args()
    prepare(games_filter=args.games, probe=args.probe, api_base=args.api_base)
