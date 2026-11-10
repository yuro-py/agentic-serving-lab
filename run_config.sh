#!/usr/bin/env bash
set -uo pipefail
source env/vars.sh

CFG=${1:?usage: run_config.sh <config-name>}
source configs/${CFG}.env
OUT=runs/${GPU}/${EXP}/${CFG}
mkdir -p "$OUT"
cp configs/${CFG}.env "$OUT/config.env"

cleanup() {
    kill ${P1:-} ${P2:-} ${P3:-} ${SP:-} 2>/dev/null
    pkill -f 'vllm serve' 2>/dev/null
    sleep 3
}
trap cleanup EXIT

# GPU + dmon loggers
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > "$OUT/gpu.csv" &
P1=$!
nvidia-smi dmon -s pucvmet -d 1 > "$OUT/dmon.csv" &
P2=$!

T0=$(date +%s)

# start server (eval so quoted JSON flags survive)
export SERVE_ARGS WL_ARGS ENVV GPU
eval "exec env $ENVV vllm serve $MODEL $SERVE_ARGS --port 8000 --api-key $KEY" > "$OUT/server.log" 2>&1 &
SP=$!

# wait for health
for i in $(seq 1 400); do
    curl -sf localhost:8000/health > /dev/null && break
    kill -0 $SP 2>/dev/null || { echo "server died, see $OUT/server.log"; exit 1; }
    sleep 2
done
STARTUP=$(( $(date +%s) - T0 ))
export STARTUP

# metrics scraper
python harness/scrape.py http://localhost:8000/metrics "$OUT/metrics.csv" &
P3=$!

# metadata
python - > "$OUT/meta.json" <<'PYEOF'
import json, os, vllm, torch, hashlib
print(json.dumps(dict(
    vllm=vllm.__version__,
    torch=torch.__version__,
    cuda=torch.version.cuda,
    gpu=os.environ['GPU'],
    startup_s=int(os.environ['STARTUP']),
    serve_args=os.environ['SERVE_ARGS'],
    env=os.environ['ENVV'],
    wl=os.environ['WL_ARGS'],
    trace_sha=hashlib.sha256(open('traces/M.jsonl','rb').read()).hexdigest(),
)))
PYEOF

grep -iE 'kv cache size|maximum concurrency|model loading took' "$OUT/server.log" > "$OUT/startup-facts.txt"

# workload
python harness/run.py $WL_ARGS --out "$OUT/requests.jsonl" 2>&1 | tee "$OUT/client.log"

echo done $CFG
