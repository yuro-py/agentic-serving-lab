# Phase 2: Qwen3.8-27B (NVFP4) on 2x RTX 5090

Agentic-workload serving with vLLM, driven by Applied Compute's [`trie`](https://github.com/Applied-Compute/trie) harness.
Same method as [phase 1](../phase-1) (Qwen3-0.6B on a 4 GB laptop GPU), scaled to a 27B hybrid-attention model on two rented RTX 5090s (~$20 / ~10 GPU-hours budget).

## Stack

| | |
|---|---|
| GPUs | 2x RTX 5090 (32 GB each, PCIe, no NVLink), vast.ai |
| Model | `unsloth/Qwen3.8-27B-NVFP4` (dense, 16 of 64 layers full attention, 48 linear attention) |
| Serving | vLLM 0.32.0, FlashInfer 0.7.0.post1, tensor-parallel 2 |
| Harness | trie 0.1.0 (Apache-2.0, vendored in `trie/`) |
| Baseline flags | `--max-model-len 262144 --kv-cache-dtype fp8 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --linear-backend marlin` |
| Baseline KV pool | 934,076 tokens |

> **Kernel note.** The baseline uses `--linear-backend marlin`, which vLLM logs as the *weight-only* FP4 path (`MarlinNvFp4LinearKernel`), not native FP4 tensor-core math. Trying `--linear-backend flashinfer_cutlass` failed at startup on this stack (`Failed to find a kernel that can implement the ScaledMM linear layer`). All numbers below are for the Marlin path.

## Setup

```bash
git clone https://github.com/yuro-py/agentic-serving-lab && cd agentic-serving-lab/phase-2
pip install vllm==0.32.0
cd trie && uv pip install -e . && cd ..
source env.sh
```

`run_trie.sh` starts vLLM with the baseline flags, waits for `/health`, starts `nvidia-smi dmon`, snapshots `/metrics` before and after, runs trie, and writes everything to `runs/<name>/`. Experiments only change environment variables:

```bash
EXTRA_ARGS="--enable-prefix-caching" WORKLOAD=workloads/burst.jsonl \
CONCURRENCY=16 DURATION=60 OUT_DIR=runs/my-run ./run_trie.sh
python tools/extract_run.py runs/my-run      # writes summary-row.csv
```

Each run directory holds `vllm.log`, `trie.log`, `dmon.csv`, `metrics-before.txt`, `metrics-after.txt`, `summary-row.csv`.

## Workloads

`ls workloads/` -> `light  grow  burst  slowtool` (2 templates each). Keys are the same as phase 1 (`num_turns`, `input_prompt_length`, `assistant_response_length`, `tool_call_output_length`, `tool_call_latency`, `final_assistant_response_length`).

| Class | Initial prompt | Turns | Tool output / turn | Tool latency | Peak context | What it stresses |
|---|---|---|---|---|---|---|
| light | 8-10k | 3-4 | 1.5-4k | 0.3-2.1 s | ~17-20k | baseline, speculative decoding |
| grow | 12-16k | 5-6 | 4-8k | 1.8-6 s | ~48-50k | prefix reuse across idle gaps |
| burst | 16-20k | 4-5 | 6-8k | 0.1-0.25 s | ~51-52k | KV pressure, overload |
| slowtool | 0.6-0.7k | 5-6 | 0.15-0.25k | 1.8-6 s | ~2k | scheduler idle gaps |

Peak context is computed from the jsonl files. trie *forces* the assistant output lengths, so replies are synthetic.

## Experiments

1. Baseline, MTP speculative decoding (on/off, depth 1/2/3/5)
2. Linear backend (Marlin vs FlashInfer-CUTLASS)
3. KV dtype (fp8 vs nvfp4)
4. Concurrency sweep c8 / c16 / c32
5. Reasoning effort (low vs xhigh)
6. Prefix caching on/off
7. Scheduling policy (fcfs vs priority) and async scheduling
8. KV offload to host RAM
9. `max-model-len` (32k / 64k / 128k / 262k)
10. `gpu-memory-utilization` (0.85 / 0.90 / 0.93)
11. `--language-model-only`
12. `max-num-batched-tokens` (2048 / 4096 / 8192 / 16384)
13. `max-num-seqs` vs concurrency (16 / 32 / 64 / 128)
14. Chunked prefill off
15. Burst overload (c64 / c128)

## Results

Single runs unless noted. Two identical configs (`max-num-batched-tokens 16384`, c16) differed by about 9% (269 s vs 293 s), so treat differences under ~10% as noise. "Server" metrics are means from vLLM's own histograms, because trie ran with `stream=False` (no client-side TTFT/ITL).

| Question | Result |
|---|---|
| Does prefix caching work on this hybrid model? | Yes. Eligible cache-hit rate is 92-98% on multi-turn traces whenever KV is not under pressure. |
| What does it buy? (c4) | Off: 161 s vs on: 65.5 s per trace (2.5x slower). Server TTFT 13.8 s vs 3.9 s, ITL 108 ms vs 42 ms, hit rate 0% vs 72%. |
| Where does concurrency break? (c8 / c16 / c32) | Server ITL 81 / 299 / 1330 ms, queue time 1.9 / 6.3 / 47 s. At c32 the cache hit rate drops to 0% and 11 preemptions appear. |
| Same cliff via `max-num-seqs` = concurrency | c16: hit rate 70%, 204 s. c32: 11.5%, 888 s. c64: 0%, 1829 s. c128: request timed out. |
| Is the cliff explained by KV? | Consistent with it: 16 sessions x ~51k tokens is about 0.82M tokens, versus a ~0.93-0.96M-token pool. 32 sessions need about 1.6M. (Inference, not directly measured.) |
| Does MTP help? | No, in this setup. c1: 29.3 vs 29.2 tok/s (flat). c8: 47.7 vs 67.0 tok/s (-29%); server ITL 147 vs 57 ms. Acceptance 0.38-0.44 at depth 2. |
| MTP depth (c1, own workload, no non-MTP baseline) | Depth 1 / 2 / 3 / 5: throughput 8.45 / 8.39 / 7.87 / 8.48 tok/s, acceptance 0.56 / 0.60 / 0.26 / 0.29, mean accepted length 1.56 / 2.20 / 1.77 / 2.47. |
| KV dtype | nvfp4 KV holds 1.61M tokens vs 0.93M for fp8 (+73%) at about +7% latency, -7% throughput. |
| `--language-model-only` | KV pool +10% (1.03M vs 0.93M tokens), same latency. |
| `gpu-memory-utilization` | 0.85 / 0.90 / 0.93 gives 795k / 895k / 954k KV tokens, identical latency (~101 s): KV was not the bottleneck at c8. |
| `max-model-len` | 32k rejected every request (prompts exceed 32,768 tokens). 64k / 128k / 262k: same latency (~65 s). |
| `max-num-batched-tokens` (c16) | 2048 / 4096 / 8192 / 16384 gives 226 / 235 / 206 / 293 s per trace. 8192 was best, 16384 worst. |
| Async scheduling (c8) | 204.0 vs 205.6 s. No effect. |
| Priority scheduling | c8 identical. At c16 it looked 14% faster, but the FCFS run's cache hit rate was an outlier (42% vs 70% in every other c16 run), so this is cache noise. All requests had one priority, so the policy had nothing to reorder. |
| KV offload (32 GB host pool, c8) | 108.2 vs 108.1 s. No effect, because KV never filled. |
| Reasoning effort low vs xhigh | Identical (14.3 s vs 14.4 s). trie forces output lengths, so this does not measure thinking cost. |
| Chunked prefill off | vLLM refuses to start: chunked prefill is required for mamba cache mode `align`. |
| Native FP4 backend | `flashinfer_cutlass` failed at startup (see kernel note). |

## Takeaways

- **Prefix cache hit rate is the number that predicts everything else.** It is the biggest lever, and when KV overflows it collapses first.
- **For ~50k-token agent sessions, 2x RTX 5090 holds about 16 concurrent sessions.** Beyond that, latency grows by an order of magnitude.
- **Memory knobs buy capacity, not speed, until KV is the bottleneck.** `language-model-only` and nvfp4 KV are free capacity; offload and gpu-memory-utilization only matter under pressure.
- **MTP was a net loss here**, though acceptance is probably understated because trie's replies are synthetic (the vLLM recipe reports ~0.79 on real text).
- Tentative config: prefix caching on, `max-num-batched-tokens 8192`, `--language-model-only`, MTP off, concurrency cap ~16 per box.

## Limitations

- Mostly n=1, with 1-9 traces per run. The 2-template workloads give very few data points.
- Different experiments used different workloads and concurrencies, so cross-experiment comparisons are invalid. Compare only within a row.
- No client-side TTFT/ITL (`stream=False`). Server-side means are used instead; there are no per-request percentiles.
- GPU monitoring (`dmon`) covers GPU 0 only.
- Synthetic, forced-length replies understate real speculative-decoding acceptance and say nothing about thinking cost.
- Marlin weight-only FP4, not native FP4 (see kernel note).

## Attribution

Built on Applied Compute's `trie` (Apache-2.0); see `trie/LICENSE`.
