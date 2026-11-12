# WebArena-Verified task data

The preparation script materializes the upstream `full` (812 tasks) or `hard`
(258 tasks) split from the pinned official Git commit in `prepare.py`. The
Hard subset task IDs are checked against the pinned full task data. Native task
and evaluator structures are retained, and a content manifest is written next
to the JSONL.

Prepare the Hard subset and make the project-local RSI search/holdout split:

```powershell
python scripts/run_component.py benchmarks/webarena_verified/prepare.py --split hard
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/webarena_verified/data/webarena_verified_hard.jsonl `
  --train benchmarks/webarena_verified/data/rsi/webarena_verified_hard_train.jsonl `
  --holdout benchmarks/webarena_verified/data/rsi/webarena_verified_hard_holdout.jsonl `
  --id-field task_id `
  --group-field intent_template_id
```

The project split is for internal RSI search and promotion only. It is not an
official WebArena-Verified split or leaderboard result. Keep official task
revision and the generated manifests with all results.

The official offline evaluator is wired through `score_offline.py`. It expects
the official artifact layout (`<task_id>/agent_response_<task_id>.json` and
`<task_id>/network_<task_id>.har`), verifies artifacts exist and are parseable,
then delegates every score to the pinned upstream CLI. It stages the latest
complete attempt for each task in the official directory layout, then writes a
JSON summary with per-task evaluator output and SHA-256 hashes for the dataset,
config, and selected input artifacts.

```powershell
uv pip install -r overlay/benchmarks/webarena_verified/requirements.txt
python scripts/run_component.py benchmarks/webarena_verified/score_offline.py `
  --dataset benchmarks/webarena_verified/data/rsi/webarena_verified_hard_train.jsonl `
  --artifacts runs/webarena_verified `
  --config path/to/webarena-verified-config.json `
  --output runs/webarena_verified/rsi_summary.json
```

The official evaluator consumes the structured final response and HAR network
trace. See the [official artifact and scoring contract](https://github.com/ServiceNow/webarena-verified#evaluate-a-task).
Online tasks are registered through `browsergym_agent` and the
[`rsi_v1_webarena_verified` profile](../../../eval_service/profiles/rsi_v1_webarena_verified.yaml).
Install the pinned BrowserGym task packages with
`uv pip install --override responses_api_agents/browsergym_agent/overrides.txt -r responses_api_agents/browsergym_agent/requirements.txt`,
then install Chromium with `playwright install chromium`. Configure the official site
URLs (`WA_SHOPPING`, `WA_SHOPPING_ADMIN`, `WA_REDDIT`, `WA_GITLAB`,
`WA_WIKIPEDIA`, `WA_MAP`) on the Gym agent runtime. Set `BROWSERGYM_HAR_DIR` to
capture each rollout under `<har-dir>/<run-id>/<task-id>/`. The task itself runs
BrowserGym's WebArena-Verified evaluator; the saved typed response/HAR can be
rescored by pointing `score_offline.py --artifacts` at the matching `<run-id>`.
