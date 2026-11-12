# Terminal-Bench 2.1

The task pack is pinned to `harbor-framework/terminal-bench-2-1` commit
`7131e4375048a0e408a8fb404b5f499d726b695b`. The preparation script emits only
agent-facing instructions, image names, task IDs, and paths; it does not copy
reference solutions into the prompt JSONL. It writes a source manifest with the
materialized data hash.

Prepare the task pack and an RSI search/holdout split with task-level isolation:

```powershell
python scripts/run_component.py benchmarks/terminal_bench_2_1/prepare.py
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/terminal_bench_2_1/data/benchmark.jsonl `
  --train benchmarks/terminal_bench_2_1/data/rsi/terminal_bench_2_1_train.jsonl `
  --holdout benchmarks/terminal_bench_2_1/data/rsi/terminal_bench_2_1_holdout.jsonl `
  --id-field task_id `
  --group-field task_name
```

Use [`rsi_v1_terminal_bench_2_1.yaml`](../../../eval_service/profiles/rsi_v1_terminal_bench_2_1.yaml)
with the Terminus 2 sandboxed agent or OpenCode composition. The run requires
the NeMo Gym sandbox provider and pinned Terminal-Bench task Docker images.
