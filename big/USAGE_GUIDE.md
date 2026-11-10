# Using the Agentic Serving Lab

This is the operator guide for the repository scaffold built from `agentic_project.pdf`. The PDF is the source for the hardware assumptions, experiment questions, and measurement protocol; this guide maps those requirements to the files and commands that exist here.

## 1. What the harness does

- `harness/traces.py` creates seeded synthetic agent sessions for coding (A), research (B), support/tool-router (C), long-document (D), and the weighted mixed workload (M = 40/20/30/10%). It records arrival offsets, project grouping, tool-turn records, and delay samples.
- `harness/runner.py` sends live requests or advances teacher-forced replay steps. Replay sends the recorded prompt at each step, discards the generated response, and advances with the recorded assistant/tool messages.
- `harness/cli.py run-config` starts one configured vLLM server, waits for `/health`, samples `/metrics`, starts GPU logging, runs a trace, and shuts the server down.
- `harness/analysis.py` reads raw request/session/metric logs and generates CSV summaries, `RESULTS.md`, and eight chart files.

The mock mode is for checking the harness plumbing only. It does not invoke a model and its timings are synthetic. The starter trace has deterministic assistant/tool records, but the current CLI does not yet save a conversation produced by a live model; replay therefore uses the seeded trace records, not a live-captured conversation. Do not describe it as recorded-live replay in results.

## 2. First run on a laptop (plumbing only)

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps

python -m harness.cli trace --preset M --sessions 12 --seed 2026 \
  --small --output traces/plumbing.jsonl
python -m harness.cli run --trace traces/plumbing.jsonl --mode mock \
  --output runs/plumbing --concurrency 2
python -m harness.cli analyze runs/plumbing --output analysis/plumbing
```

`--small` shortens generated contexts/step counts; it is not suitable for measuring performance. Review `runs/plumbing/requests.jsonl`, `sessions.jsonl`, and all eight plots. The trace command prints a SHA-256 and writes a `.meta.json` sidecar. Keep that hash with any result.

## 3. Prepare the inference host

Use the project brief’s two RTX 5090 machine or document any differences. In a persistent GPU-host checkout:

1. Keep the server, client, models, and logs on that host. Do not benchmark through an SSH tunnel.
2. Ensure the machine has a suitable Blackwell driver/CUDA stack, enough disk and RAM for model files and any offload experiment, and a persistent Hugging Face cache. Set `HF_HOME` and `HF_HUB_ENABLE_HF_TRANSFER=1` where appropriate.
3. Activate the tested vLLM environment. The brief’s recipe was verified on a vLLM `0.26.1rc1` development build and requires `transformers>=5.8.0`; record the exact working vLLM commit/build, PyTorch, CUDA, Transformers, FlashInfer, and driver versions. Do not assume ordinary package indexes contain that exact development build.
4. Capture machine details:

   ```bash
   python -m harness.cli env
   nvidia-smi topo -m
   ```

   Confirm the two GPUs, PCIe topology, GPU memory, clocks, and host ID. The automated runner also saves `host_facts.txt`, `versions.txt`, `gpu.csv`, and `dmon.csv` in each run when utilities are available.
5. Keep the API listener on localhost or a private network. Set an API key; do not expose port 8000 publicly. Make sure the model repository is accessible and any gated model access is configured.

The baseline config uses the brief’s key inference settings: TP2, 262144 max model length, FP8 KV cache, Qwen3 reasoning parser, automatic tool choice, and `qwen3_coder` tool parsing. On startup, inspect `server.log` for the FlashInfer Cutlass NVFP4 kernel and reported KV pool. Then verify plain chat, tool-call parsing, reasoning in the reasoning field, and nonzero accepted-draft counters when MTP is enabled before benchmarking.

## 4. Create the experiment trace once

Generate a full trace and reuse the same file/hash for each ablation:

```bash
python -m harness.cli trace --preset M --sessions 200 --seed 2026 \
  --rate 0.1 --output traces/mixed-seed2026.jsonl
sha256sum traces/mixed-seed2026.jsonl
```

For quick targeted checks, choose `A`, `B`, `C`, or `D`. `--rate` sets the average Poisson session arrival rate used by trace generation. The trace uses deterministic synthetic text and whitespace-scale prompt sizes, so the recorded prompt length is an approximation; use actual server tokenizer usage for reporting token counts. Tool delays are sampled and stored in the trace.

The present trace generator encodes a fixed small project range and a deterministic tool-turn script. It is a useful repeatable workload starter, but implement/extend its generator before claiming it exactly matches every PDF target (long-tail prompt distributions, arbitrary project counts, burst arrivals, checksummed realistic tasks, or tokenizer-exact fabrication).

## 5. Run the baseline inference experiment (E0)

Set the API key in the environment; the runner passes it to vLLM without writing its value into the saved command:

```bash
export API_KEY='choose-a-private-key'
python -m harness.cli run-config configs/E0-tp2.yaml \
  --trace traces/mixed-seed2026.jsonl --mode replay --repeat 3 \
  --concurrency 8
```

Each repeat starts a fresh server and writes under:

```text
runs/E0/baseline-unsloth-tp2-mtp-off/repeat-01/
```

Key files: `config.resolved.json` (flags/env/trace), `server.log`, `startup.log`, `metric_names.txt`, `metrics.jsonl`, `requests.jsonl`, `sessions.jsonl`, `gpu.csv`, `dmon.csv`, `host_facts.txt`, and `versions.txt`. Check the startup log and exact metric names; vLLM metric names vary between builds.

For E0 with MTP depth three, use `configs/E0-tp2-mtp3.yaml`. Keep the checkpoint and all other settings fixed. Verify the accepted-draft metric is above zero: a speed change alone does not prove MTP was active. The current analyzer doesn’t yet normalize version-specific speculative counters into request rows, so calculate/document MTP acceptance from the raw metrics that your vLLM build exposes.

`--repeat 3` here means three independent server starts. To satisfy the PDF protocol, each measured point should have at least five minutes of steady-state data or 200 sessions, after a two-minute warm-up. The runner currently labels and excludes warm-up rows for analysis when used with `run --warmup`, but the managed `run-config` path does not yet isolate a dedicated warm-up trace; for rigorous GPU runs, run an explicit warm-up request batch before the measured trace or add that orchestration before treating this as protocol-complete.

## 6. Run a direct live loop or replay

`run-config` starts/stops the server and runs a trace. For a server already started by you, call the client directly:

```bash
export BASE_URL=http://127.0.0.1:8000
export API_KEY='choose-a-private-key'
python -m harness.cli run --trace traces/mixed-seed2026.jsonl --mode live \
  --output runs/manual/live-01 --concurrency 8
```

- `live`: the model’s assistant responses drive tool execution and later turns. Use for end-to-end behavior and task success.
- `replay`: the model is called for each stored step, but its output is discarded; the trace’s recorded assistant/tool messages advance the conversation. Use for comparing settings with identical prompt histories.
- `mock`: no server is contacted; use only for harness plumbing.

Use the identical trace, seed, arrival schedule, and workload settings for an ablation. Do not compare one live trace to another and attribute all differences to the server flag.

## 7. Translate the PDF experiment plan into runs

Create a YAML config per measured point, copying `configs/E0-tp2.yaml` and changing one factor at a time. Run each config through `run-config`; unknown server flags are rejected against the installed `vllm serve --help` before startup.

| Brief section | Experiments | Change and compare |
|---|---|---|
| E0–E1 | Baseline / checkpoints | TP2 baseline, MTP off/on; FP8, Inferact NVFP4, Unsloth NVFP4, with the same trace and MTP conditions. Record KV tokens, task success, tool validity, and acceptance. |
| E2–E2b | Topology / routing | TP2 vs two independent TP1 replicas; then round-robin, least-loaded, session-sticky. `harness/router.py` supplies a routing helper, but there is not yet a complete multi-server launcher/router CLI. Run two servers and a routing driver as a separately controlled extension. |
| E3 | Prefix cache | Prefix caching enabled vs disabled, only after checking whether the installed vLLM supports the chosen flag and startup confirms the state. Compare measured/ideal reuse. |
| E4 | Batching | `--max-num-batched-tokens` 2048/4096/8192/16384 and `--max-num-seqs` 16/32/64/128; factorial sweep can be large, so hold all other factors fixed. |
| E5 | Memory | GPU memory utilization, FP8 vs auto KV, max context 32k/64k/128k/262k, language-model-only. Separate each factor before combinations. |
| E6 | Speculation | MTP off and 1–5 draft tokens, sweep concurrency; collect draft and accepted counts, speed and quality. |
| E7 | Thinking | Off/low/medium/xhigh in trace settings; compare reasoning tokens, time-to-answer, session duration, success. |
| E8–E12 | Runtime settings | Graphs/eager, all-reduce, KV offload, scheduling and frontend; verify every flag/environment option with the installed build. |
| E13 | Stress/failure | Overload, bursts, disconnects, long prompts, soak. The current CLI has no built-in burst/disconnect generator; add a driver or use controlled external clients and preserve their exact scripts and logs. |
| E14 | Final | Combine the best supported settings, then remove each one in turn. Keep raw runs for the waterfall. |

The full hypothesis/compare catalog is in `configs/experiments.yaml`. That file is an experiment checklist, not directly executable. The brief’s optional DFlash2/DSpark work is excluded from the initial setup.

Example config editing: copy a config, change one field, and give it a unique `name`, e.g. `configs/E4-batch-4096.yaml`. For an argument, add YAML list items under `args` exactly as in the baseline. Always run `vllm serve --help` on the target environment to confirm the flag and semantics.

## 8. Analyze and report

After runs complete:

```bash
python -m harness.cli analyze runs --output analysis/results \
  --slo-ttft 3 --slo-itl-ms 50 --hourly-cost 2.50
```

The output includes `summary.csv`, flattened raw tables, `RESULTS.md`, and the plot files. The hourly cost is an input example; replace it with the actual instance rate. The PDF’s desired chart set includes offered-load curves, goodput, latency split, ablation waterfall, cache reuse, speculative acceptance, KV/preemption, and cost vs success. Some plots remain placeholders unless the required data exists; in particular, current latency “split” is a TTFT/E2E summary, not queue/prefill/decode decomposition. Current request arrival replay is open-loop by stored arrival offsets; the harness does not yet provide closed-loop N-agent or burst modes through its CLI.

Before publishing a value:

1. Confirm the run used the intended trace SHA-256, checkpoint, flags, and host/software metadata.
2. Exclude warm-up and failed/error requests consistently; inspect raw logs and `metric_names.txt`.
3. Run at least three repeats, each with ≥5 minutes steady state or ≥200 sessions; report median and range.
4. Check client CPU stays below 70%, inspect GPU throttling and PCIe counters, and rerun suspicious points.
5. Keep negative results. Link every chart and claim back to the raw `runs/...` data and exact YAML.

Use `analysis/RESULTS.md` as the results template, `analysis/blog-outline.md` for the write-up, and `analysis/resume-bullets.md` only after real results exist. Never use mock timings as resume or serving claims.

## 9. Relation to the attached brief

The scaffold follows the brief’s workload presets, TP2 baseline intent, E0–E14 comparison topics, metrics collection, one-command managed run, raw-log retention, and analysis/write-up layout. The guide also marks implementation gaps explicitly: live-conversation trace capture, exact tokenizer-based synthetic prompt sizing, configurable project-count sweeps, closed-loop/bursty workload modes, full replica launcher/router, long-run/load-failure drivers, queue/prefill/decode decomposition, and server metric normalization for cache/MTP/preemption. Those are required extensions before claiming every PDF feature is fully realized. The attached PDF remains the authority for experiment goals and the measurement protocol.
