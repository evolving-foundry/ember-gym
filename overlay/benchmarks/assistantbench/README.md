# AssistantBench BrowserGym integration

This integration materializes the 33 public validation task IDs registered by
BrowserGym 0.14.4. Task instructions and answers are loaded by the pinned
BrowserGym task implementation; this project stores no answer-bearing test
rows. Prepare the task catalog and RSI project split with:

```powershell
python scripts/run_component.py benchmarks/assistantbench/prepare.py
python scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py `
  benchmarks/assistantbench/data/assistantbench_validation.jsonl `
  --train benchmarks/assistantbench/data/rsi/assistantbench_validation_train.jsonl `
  --holdout benchmarks/assistantbench/data/rsi/assistantbench_validation_holdout.jsonl `
  --id-field task_id
```

Run the `rsi_v1_assistantbench.yaml` profile after installing the pinned
BrowserGym agent dependencies and Chromium. Tasks use live public websites, so
the task pack is reproducible but web content may change between runs. BrowserGym
0.14.4 has a reported thousands-separator parsing issue in its answer scorer;
record the exact package/evaluator version and inspect affected numeric tasks
before comparing scores.
