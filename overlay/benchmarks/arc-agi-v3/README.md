# ARC-AGI-3

Interactive reasoning benchmark (preview games first): turn-based games served
by the ARC Prize API where agents must discover action semantics, goals, and
win conditions from observation alone. Unlike v1/v2 this is **not a static
dataset** — each JSONL row is an evaluation unit (game x action budget), and
the games run online at `three.arcprize.org`, requiring an `X-API-Key`.

Design: `docs/ARC-AGI_v3_基准接入设计_v0.1.md`.

## How it runs

The benchmark uses a **self-contained game-loop agent** (BrowserGym pattern,
`responses_api_agents/arc_agi_v3/`): the agent owns the
RESET → frame → model → action loop, enforces the per-game action budget, and
returns the environment-side scorecard score as the reward. There is no
separate resources server and no judge.

Key behaviors:

- Frames arrive as 64x64 grids of 4-bit colour indices and are rendered to
  PNG (official 16-colour palette) for the VLM via `image_url` content.
- Actions are constrained to `RESET`/`ACTION1`-`ACTION7` (`ACTION6` takes
  x/y coordinates on the 64x64 grid) via a `submit_action` tool call, with a
  JSON-text fallback; unparsable answers count as
  `arc_agi_v3_malformed_action` failures and never silently pass.
- Per design decision #3, the reward is the **raw upstream score**; per-level
  scores/actions/baselines are kept in the trace, and efficiency weighting is
  a config flag (`efficiency_weighting`, default off) pending the M4
  alignment with the official leaderboard.
- The image-heavy history keeps the last `history_window` frames plus the
  full action trajectory as text.

## Requirements

- **A VLM model server**: text-only models cannot see the frames. Switching
  models is a config change on `model_server` — no code changes (decision #4).
- `ARC_AGI_V3_API_KEY` in the environment (key acquisition is a pending
  design item; mock-based development does not need it).
- Optional: `ARC_AGI_V3_API_BASE` to point the client at a mock server.

## Game list

Pinned to the official preview games in `games_manifest.json` (decision #2):
`ls20`, `ft09`, `vc33`. The Kaggle full list is intentionally deferred until
M4 settles the scoring semantics. `prepare.py --probe` cross-checks the
manifest against the live `/api/games` listing when a key is available.

## Score comparability

Gym-internal scores are **not comparable** with the official leaderboard
until M4 aligns the action-cutoff/efficiency-weighting semantics. Expect
near-zero scores from frontier models: upstream reported ~1% for frontier AI
at release.

## Cost

Upstream warns that full-scale evaluation can reach the tens-of-thousands-of-
USD range on high-tier inference. Start with a `--games` subset, low
concurrency (2), and a small repeat count; monitor token usage per rollout.

## Example usage

```bash
# Prepare benchmark data (no network needed without --probe)
gym eval prepare --benchmark arc-agi-v3

# Running servers (VLM required)
gym env start \
    --model-type vllm_model \
    --benchmark arc-agi-v3

# Collecting rollouts
gym eval run --no-serve \
    --agent arc_agi_v3_agent \
    --input benchmarks/arc-agi-v3/data/arc_agi_v3_games.jsonl \
    --output results/arc_agi_v3_rollouts.jsonl \
    --num-repeats 3
```

## Upstream pin

- REST contract: official OpenAPI spec `docs.arcprize.org/arc3v1.yaml`
  (base URL `https://three.arcprize.org`, `X-API-Key` header, `AWSALB*`
  cookie affinity).
- Frame palette mirrored from `arcprize/ARC-AGI-3-Agents`
  (`agents/templates/multimodal.py`).
- Manifest pinned at `2026-10-09`; re-pinning changes the benchmark
  definition and must be recorded explicitly in the commit message.
