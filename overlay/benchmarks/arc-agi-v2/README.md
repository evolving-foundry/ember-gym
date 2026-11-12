# ARC-AGI-2

Abstraction and Reasoning Corpus 2: few-shot grid transformation puzzles
(public evaluation split, 120 tasks). Sourced from the arcprize/ARC-AGI-2
repo (Apache-2.0) at prepare time.

Verification is a deterministic exact match against the expected output grid
(cell-for-cell, dimensions included), shared with the v1 benchmark and
implemented in `resources_servers/arc_agi/app.py` — no judge and no sandbox.
The task JSON schema is identical to ARC-AGI-1, so the prompt template
(`benchmarks/prompts/generic/arc.yaml`) is reused as-is.

The semi-private and private evaluation sets are not distributed upstream
and therefore not included.

## Score comparability

Gym runs the **public evaluation split**; the official ARC-AGI-2 leaderboard
is scored on the semi-private set. Scores produced here are **not directly
comparable** with official leaderboard numbers (upstream calibration between
the two sets is close — ±3 percentage points — but not identical).

The public evaluation set must not leak into any training or prompt-optimization
flow (an upstream requirement as well).

## Upstream pin

Data is pinned to `arcprize/ARC-AGI-2` commit
`f3283f727488ad98fe575ea6a5ac981e4a188e49` (upstream main, last pushed
2025-05-22). Re-pinning changes the benchmark definition and must be
recorded explicitly in the commit message and here.

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark arc-agi-v2

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark arc-agi-v2

# Collecting rollouts
gym eval run --no-serve \
    --agent arc_agi_simple_agent \
    --input benchmarks/arc-agi-v2/data/arc_agi_v2_benchmark.jsonl \
    --output results/arc_agi_v2_rollouts.jsonl \
    --num-repeats 4
```

For the DSH agent variant, run with `--agent arc_agi_v2_dsh_agent` against
the config in `config_dsh.yaml` (same chain as `benchmarks/arc-agi/config_dsh.yaml`).
