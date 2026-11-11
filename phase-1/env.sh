# env.sh
#export GPU=2050
export MODEL=Qwen/Qwen3-0.6B
export PORT=8000

# trie uses AsyncOpenAI; OPENAI_API_KEY MUST be non-empty.
# If vLLM is launched with --api-key, the two values must be byte-identical.
export API_KEY=lab-secret-key
export OPENAI_API_KEY="$API_KEY"

export HF_HOME="$HOME/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRIE_BIN="uv run trie"

# Safe 4 GB baseline: NO reasoning parser, NO tool-call parser, NO auto-tool-choice.
# Those three flags trigger the vLLM streaming bug where trie sees empty content.
export BASE_SAFE="--max-model-len 4096 --gpu-memory-utilization 0.80 --max-num-seqs 8 --enforce-eager"
export VLLM_ARGS="${VLLM_ARGS:-$BASE_SAFE}"
