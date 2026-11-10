# Agentic Serving Lab

A reproducible harness for measuring multi-turn tool-using agent workloads on vLLM. It creates seeded traces, runs live or teacher-forced replay traffic, captures request/session/server/GPU data, and regenerates analysis plots from raw run folders.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m harness.cli trace --preset M --sessions 12 --seed 2026 --output traces/smoke.jsonl --small
python -m harness.cli run --trace traces/smoke.jsonl --mode mock --output runs/smoke --concurrency 2
python -m harness.cli analyze runs/smoke --output analysis/smoke
```

The mock backend checks trace, scheduling, logging, and plotting locally; it does not produce model performance evidence. For real measurements, set `BASE_URL`, `API_KEY`, and (optionally) `MODEL`, then use `--mode live` or `--mode replay`.

## Real vLLM run

Use a pinned, tested environment and record `vllm --version`, `pip freeze`, driver/CUDA/GPU and topology in `env/`. The model card and vLLM recipe may require gated model access. On the two-GPU TP2 setup, the baseline command is in `configs/E0-tp2.yaml`. Never expose port 8000 publicly; set an API key. Run the client on the GPU host, not through an SSH tunnel.

```bash
export API_KEY='local-secret'
python -m harness.cli run-config configs/E0-tp2.yaml --mode replay --trace traces/mixed.jsonl --repeat 1
```

`run-config` validates flags against that installed vLLM's `serve --help`, writes the exact resolved command/environment, keeps API keys out of the resolved command record, starts the server, waits for health, captures startup and metric-name snapshots, starts GPU loggers when `nvidia-smi` exists, executes the workload, and shuts down the server. It does not claim recipe results until the hardware run actually produces them. The recipe records vLLM `0.26.1rc1` dev and `transformers>=5.8.0`; pin a compatible build rather than assuming that build is available from ordinary package indexes. Harness dependencies have a tested lock snapshot in `requirements-lock.txt`; the serving host must separately pin a compatible PyTorch/vLLM/CUDA combination.

## Reproducible experiment flow

1. Generate a trace once with a fixed seed and commit it (or store its SHA-256 with results). Trace rows include arrival offsets, project, preset, thinking setting, and per-step seeded tool delays.
2. Use the seeded trace with `--mode replay` for ablations. Replay sends each recorded prompt, discards the generated assistant response, and advances using the trace's deterministic assistant/tool messages. The current CLI does not capture a live conversation into a trace yet.
3. Define each experiment as YAML in `configs/`. Every configuration is a fresh server process; repeat runs into separate folders and preserve raw JSONL/CSV/logs.
4. Run analysis once across all result directories: `python -m harness.cli analyze runs --output analysis/results`.

Recommended protocol from the brief: discard a two-minute warm-up, collect at least five minutes of steady state or 200 sessions per data point, repeat three times, and report median/range. Repeat directories are supported. For rigorous hardware runs, handle a dedicated warm-up batch before measured traffic; `run-config` does not automatically schedule a separate warm-up trace. Record SLOs before reviewing measurements.

## Commands

```text
agentic-lab trace --help       seeded trace generation
agentic-lab run --help         mock/live/replay workload execution
agentic-lab run-config --help  managed vLLM experiment lifecycle
agentic-lab analyze --help     tables and reproducible plots
agentic-lab env                capture host environment facts
```

## Layout

`harness/` runner, tools, trace schema, collectors and analysis; `configs/` experiment definitions; `traces/` immutable JSONL inputs; `runs/<experiment>/<config>/<repeat>/` raw evidence; `env/` machine facts; `analysis/` generated plots and `RESULTS.md`.

## Metrics and interpretation

Requests include prompt/completion/reasoning/cached tokens when returned, TTFT, time-to-answer, mean ITL, E2E, finish reason, tool JSON/schema validity, error and class/session identifiers. Session rows include duration, steps, tokens, LLM/tool time, task check and success. Server metrics are captured without hardcoding version-specific names; `metric_names.txt` is saved per run. GPU CSV and dmon are recorded when available. Analysis derives p50/p95/p99, throughput, SLO goodput, cache efficiency when both measured and ideal counts exist, cost per million output tokens when hourly cost is supplied, and speculative acceptance where metrics are present.

Plots are generated only from raw run files. No performance or quality result is asserted by this starter scaffold; the `RESULTS.md` template labels missing results as pending.
