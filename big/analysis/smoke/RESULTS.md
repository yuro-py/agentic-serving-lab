# Results

Source run root: `runs/smoke`

SLO used for this report: request TTFT p99 target ≤ 3s; per-request mean ITL target ≤ 50ms; tool JSON valid.

## Run summary

No raw request logs found. Results are pending actual runs.

## Plots

Plots are regenerated from the raw request/session/server metrics collected in the source run folders.

![throughput_latency.png](throughput_latency.png)
![goodput.png](goodput.png)
![latency_split.png](latency_split.png)
![ablation_waterfall.png](ablation_waterfall.png)
![cache_reuse.png](cache_reuse.png)
![speculation.png](speculation.png)
![kv_preemptions.png](kv_preemptions.png)
![cost_quality.png](cost_quality.png)

## Interpretation

No recommendation is generated from an empty or mock-only dataset. Label mock results as harness plumbing only. The trace metadata and per-run metadata are authoritative for exact inputs and environment.
