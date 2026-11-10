# Blog post outline

1. Question: serving multi-turn agents on two PCIe-connected RTX 5090s.
2. Hardware, software pins, checkpoint revisions, and vLLM recipe baseline.
3. Why agent traces include growing conversations, shared prefixes, tools, and idle gaps.
4. Trace generation and teacher-forced replay; show trace hash and what replay controls.
5. Baseline validation: kernel, cache capacity, reasoning parsing, tools, and MTP counters.
6. Experiments E0–E14: hypothesis, controlled variable, raw run links, result median/range, explanation; include negative results.
7. Latency/throughput, goodput, cache reuse, topology, speculative acceptance, KV, and cost plots.
8. Final setup and leave-one-out waterfall.
9. Limitations and one-command reproduction.

Only insert quantitative claims that can be regenerated from committed scripts/configs and retained raw run output.
