# SWE-bench Verified and Lite

This directory includes the official public Verified and Lite task packs, their Gym compositions,
and deterministic RSI project splits. The RSI splits are task subsets for candidate search and local
promotion; neither is the official benchmark leaderboard protocol.

## Prepare data and RSI splits

Run from the repository root. The sources are public Hugging Face datasets; install the repository's
`datasets` dependency first if it is not already available.

```powershell
python scripts/run_component.py benchmarks/swebench/verified/prepare.py
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/swebench/data/swebench_verified_benchmark.jsonl `
  --train benchmarks/swebench/data/rsi/swebench_verified_train.jsonl `
  --holdout benchmarks/swebench/data/rsi/swebench_verified_holdout.jsonl `
  --id-field instance_id `
  --group-field repo

python scripts/run_component.py benchmarks/swebench/lite/prepare.py
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/swebench/data/swebench_lite_benchmark.jsonl `
  --train benchmarks/swebench/data/rsi/swebench_lite_train.jsonl `
  --holdout benchmarks/swebench/data/rsi/swebench_lite_holdout.jsonl `
  --id-field instance_id `
  --group-field repo
```

Repository-level grouping prevents issues from the same codebase crossing the search and holdout
boundary. The splitter writes source/output hashes and ID/group hashes next to the train split.

## Run through Gym

Use the matching RSI profile in `eval_service/profiles/` and point the RSI service at the desired
benchmark key. For standalone Gym runs, the original full task pack can be run with the upstream
composition:

```bash
gym eval prepare --config benchmarks/swebench/verified/opencode.yaml
gym env start --model-type vllm_model --config benchmarks/swebench/verified/opencode.yaml
gym eval run --no-serve \
  --agent mini_swe_agent_2_swebench_verified \
  --input benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
  --output results/swebench_verified/rollouts.jsonl
```

SWE-bench requires the Linux sandbox/resource-server setup and benchmark task images. Give every
candidate/repeat a unique run ID so image caches, grader state, and output files do not collide.
The direct RSI transfer suite EvoAgentBench also includes a separate SWE-Bench Verified domain;
use its own official train/test protocol and do not reuse these project-created splits for that score.
