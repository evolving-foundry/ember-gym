# RSI split preparation

`split_jsonl.py` turns one already prepared NeMo Gym task JSONL into a stable, disjoint search set
and promotion holdout. The outputs and manifest are local data artifacts. A project-created split is
not an official benchmark split and must not be reported as the benchmark's leaderboard score.

The task pack is pinned by its source hash, split seed, task ID/group fields, output hashes, and an
emitted manifest. By default each task ID is its own group. Use `--group-field` when repeated rows
share a template, repository, or task class that must not cross the search/holdout boundary. Keep
the manifest beside the files and use the same seed for every candidate in an experiment profile.

Examples after the benchmark-specific prepare step:

```powershell
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/osworld/data/osworld_1x_verified_source.jsonl `
  --train benchmarks/osworld/data/rsi/osworld_train.jsonl `
  --holdout benchmarks/osworld/data/rsi/osworld_holdout.jsonl `
  --id-field task_id

python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/swebench/data/swebench_verified_benchmark.jsonl `
  --train benchmarks/swebench/data/rsi/swebench_verified_train.jsonl `
  --holdout benchmarks/swebench/data/rsi/swebench_verified_holdout.jsonl `
  --id-field instance_id `
  --group-field repo

python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/tau2/data/tau2_benchmark.jsonl `
  --train benchmarks/tau2/data/rsi/tau2_train.jsonl `
  --holdout benchmarks/tau2/data/rsi/tau2_holdout.jsonl `
  --id-field task_id

python scripts/run_component.py benchmarks/tau2/prepare.py banking_knowledge --retrieval-config bm25_grep

python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/tau2/data/tau2_banking_knowledge_bm25_grep.jsonl `
  --train benchmarks/tau2/data/rsi/tau3_banking_train.jsonl `
  --holdout benchmarks/tau2/data/rsi/tau3_banking_holdout.jsonl `
  --id-field task_id

python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/webarena_verified/data/webarena_verified_hard.jsonl `
  --train benchmarks/webarena_verified/data/rsi/webarena_verified_hard_train.jsonl `
  --holdout benchmarks/webarena_verified/data/rsi/webarena_verified_hard_holdout.jsonl `
  --id-field task_id `
  --group-field intent_template_id

python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/workarena/data/workarena_l1_train.jsonl `
  --train benchmarks/workarena/data/rsi/workarena_l1_train.jsonl `
  --holdout benchmarks/workarena/data/rsi/workarena_l1_holdout.jsonl `
  --id-field task_id `
  --group-field task_name

python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/assistantbench/data/assistantbench_validation.jsonl `
  --train benchmarks/assistantbench/data/rsi/assistantbench_validation_train.jsonl `
  --holdout benchmarks/assistantbench/data/rsi/assistantbench_validation_holdout.jsonl `
  --id-field task_id

python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/terminal_bench_2_1/data/benchmark.jsonl `
  --train benchmarks/terminal_bench_2_1/data/rsi/terminal_bench_2_1_train.jsonl `
  --holdout benchmarks/terminal_bench_2_1/data/rsi/terminal_bench_2_1_holdout.jsonl `
  --id-field task_id `
  --group-field task_name

python scripts/run_component.py benchmarks/gaia/prepare.py --levels 1 2 3
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/gaia/data/gaia_text_validation.jsonl `
  --train benchmarks/gaia/data/rsi/gaia_train.jsonl `
  --holdout benchmarks/gaia/data/rsi/gaia_holdout.jsonl `
  --id-field task_id
```

The splitter requires unique, non-null IDs and at least two task groups. It reserves the lowest
SHA-256 ranks of `seed + group_id` for holdout, independent of source row order, and never splits a
group. To use the promotion-only split, set the generated holdout JSONL as `holdout_dataset_fpath`
in the profile.

## Data access and environment prerequisites

- Run each benchmark's documented preparation workflow first. The RSI profile does not download
  assets, build containers, start a browser, provision a VM, or configure API credentials.
- For OSWorld, use the same verified task JSONL and VM asset revision across agent/control hosts;
  provision the provider and persistent control process according to `overlay/benchmarks/osworld/README.md`.
- For SWE-bench, use the existing pinned Gym preparation and resource server; keep task image and
  grading-run IDs unique between candidate evaluations.
- For τ²/τ³, pin the upstream revision, domain, user simulator, judge, retrieval config, and repeats.
- For GAIA, prepare the public text-only validation rows first; the official test set is excluded.
- For WebArena-Verified, run `python scripts/run_component.py benchmarks/webarena_verified/prepare.py` before splitting. Its
  Gym adapter is `responses_api_agents/browsergym_agent/` (source in `overlay/`); use the pinned BrowserGym runtime and keep HAR
  captures with rollout artifacts for offline scoring.
