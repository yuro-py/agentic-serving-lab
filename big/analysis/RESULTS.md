# Results

**Status: pending real hardware experiments.** No benchmark numbers are populated by the scaffold. Mock runs verify harness plumbing only and must not be presented as serving performance.

## Setup and hardware

- Host/provider and host ID: pending
- GPUs, driver, CUDA, PCIe topology: pending (`env/` and each run's `host_facts.txt`)
- Pinned vLLM build/commit, PyTorch, Transformers, FlashInfer: pending
- Model checkpoint and exact revision: pending

## Workload and method

- Trace path, SHA-256 and seed: pending
- Preset/mix, session count, arrival model and concurrency: pending
- Replay record source: pending
- Warm-up, steady-state duration, repeats: pending
- SLOs fixed before measurement: TTFT p99 ≤ 3 s; mean ITL p95 ≤ 50 ms; valid tool JSON ≥ 99%

## Experiment table

| ID | Hypothesis | Configuration | Result (median and range) | Explanation |
|---|---|---|---|---|
| E0–E14 | See experiment catalog | Pending | Pending | Pending |

## Recommended configuration

Pending reproducible runs and leave-one-out ablation. Link every reported number to its run directory and raw outputs.

## Limitations

Record workload approximation, tokenizer accounting, client utilization, host throttling, metric availability, and any deviations from the prescribed protocol.

## Reproduction

```bash
python -m harness.cli trace --preset M --sessions 200 --seed 2026 --output traces/mixed.jsonl
python -m harness.cli run-config configs/E0-tp2.yaml --trace traces/mixed.jsonl --mode replay --repeat 3
python -m harness.cli analyze runs --output analysis/results
```
