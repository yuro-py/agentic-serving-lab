# Qwen3-0.6B locally

The work below is dependent upon:-
vllm=0.32.0
trie-0.1.0

The model and hardware are not hardcoded in the config.
The experiments are limited to Ampere/Turing gpu architecture and a non-MoE dense model architecture(sparse attention works).

1. git clone https://github.com/yuro-py/agentic-serving-lab;cd agentic-serving-lab
2. pip install vllm=0.32.0
3. cd trie;uv pip install -e .;cd ..
4. source env.sh

"run_trie.sh" script for running vllm with hardcoded/custom flags with options.

```
$ ls workloads/
burst.jsonl  grow.jsonl  light.jsonl

workloads needs these keys defined before-hand depending upon the nature:-
1. num_turns -> Number of tool-use turns before the final answer. The trace produces N tool-call requests + 1 final-answer request = N+1 API calls. controls how many requests per session; drives scheduler pressure and prefix-cache reuse opportunities.
2. input_prompt_length -> Token length of the initial user/system prompt (the task itself). Appears in every request of the trace. baseline prefill cost paid on every turn.
3. assistant_response_length -> Token length the assistant is forced to generate on tool-use turn i. This is the "thinking + tool-call JSON" budget. decode work per turn; small values stress TTFT, large values stress decode TPS.
4. tool_call_output_length -> Token length of the mock tool result returned after turn i. This is what gets appended to context and grows the prompt for later turns. the biggest lever for prefix caching and KV growth. Large values mean later turns have long prefills that only the cache can make cheap.
5. tool_call_latency -> Seconds trie sleeps between the assistant's tool-call and the simulated tool result for turn i. Models real tool round-trip time (search, DB, file read). creates idle gaps between requests; separates "server-bound" load from "agent-paced" load. With high latency, the server is often idle and concurrency matters less.
6. final_assistant_response_length -> Token length of the final answer turn, after all tool use is done. decode cost of the answer step; usually what the user perceives as latency.

Experiments(details and commands in experiments.xlsx):-
1. Baseline with increasing concurrency.
2. prefix cache on/off
3. batching, max sequences, chunked prefill off
4. kv capacity expansion with gpu-mem-util progression, quantization
5. cuda graphs
6. speculative decoding
7. burst workload with growing concurrency


