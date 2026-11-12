# WorkArena-L1 RSI task catalog

WorkArena generates task instances against an authenticated ServiceNow
environment; its task catalog is not a standalone fixed QA JSONL. The
preparation script pins the BrowserGym catalog revision, selects only its
official L1 `train` task classes, and expands each to a deterministic range of
seeds. It does not copy the gated ServiceNow instance dataset or claim to
materialize the upstream 19,912 instance states.

Create the default 10-seed task pack and split it for internal RSI search and
promotion:

```powershell
python scripts/run_component.py benchmarks/workarena/prepare.py
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/workarena/data/workarena_l1_train.jsonl `
  --train benchmarks/workarena/data/rsi/workarena_l1_train.jsonl `
  --holdout benchmarks/workarena/data/rsi/workarena_l1_holdout.jsonl `
  --id-field task_id `
  --group-field task_name
```

Both outputs are project-local task-seed splits, not official WorkArena
leaderboard splits. The generated rows pin `task_name`, `seed`, category, and
BrowserGym commit. Run the profile at
[`rsi_v1_workarena.yaml`](../../../eval_service/profiles/rsi_v1_workarena.yaml).
Install the pinned packages with
`uv pip install --override responses_api_agents/browsergym_agent/overrides.txt -r responses_api_agents/browsergym_agent/requirements.txt`, then install Chromium
with `playwright install chromium`. The runtime also needs approved ServiceNow
instance URLs/credentials and a versioned ServiceNow state.

Upstream setup requires approval for the
[`ServiceNow/WorkArena-Instances` dataset](https://huggingface.co/datasets/ServiceNow/WorkArena-Instances)
and installation of [`browsergym-workarena`](https://github.com/ServiceNow/WorkArena).
