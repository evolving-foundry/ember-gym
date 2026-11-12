# ARC-AGI

Abstraction and Reasoning Corpus: few-shot grid transformation puzzles
(public evaluation split, 400 tasks). Sourced from the fchollet/ARC repo
(Apache-2.0) at prepare time.

Verification is a deterministic exact match against the expected output grid
(cell-for-cell, dimensions included) implemented in
`resources_servers/arc_agi/app.py` — no judge and no sandbox. The hidden
ARC test set is not distributed upstream and therefore not included.

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark arc-agi

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark arc-agi

# Collecting rollouts
gym eval run --no-serve \
    --agent arc_agi_simple_agent \
    --input benchmarks/arc-agi/data/arc_agi_benchmark.jsonl \
    --output results/arc_agi_rollouts.jsonl \
    --num-repeats 4
```
