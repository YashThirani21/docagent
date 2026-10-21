#!/usr/bin/env bash
# Serve Qwen2.5-VL-7B with vLLM on one GPU and wait until it answers.
#   >= 20 GB and bf16-capable (L4, A10, A100...): full-precision weights in bf16 (~16.5 GB)
#   smaller / older (T4 16 GB, no bf16):          AWQ 4-bit weights (~6 GB) in fp16
# Override with MODEL=..., QUANT=awq|none, DTYPE=half|bfloat16, GPU_UTIL=0.80, MAX_LEN=16384,
# SERVED=<name the config's llm.model uses> (e.g. qwen2.5-vl-7b-awq: own name -> own cache, own eval record).
# GPU_UTIL leaves the rest of the GPU for the bge-m3 embedder (device: cuda).
# With CONFIG=<yaml> (or DOCAGENT_CONFIG) the served name is the config's llm.model, so the two always match;
# a name ending in -awq also selects the AWQ weights:
#   DOCAGENT_CONFIG=configs/colab_vllm_awq.yaml bash scripts/start_vllm.sh
# The image limit per prompt is the config's llm.max_images (default 4), the same budget the agent keeps to.
set -euo pipefail
CONFIG="${CONFIG:-${DOCAGENT_CONFIG:-}}"
MAX_IMAGES=4                                                     # = LLMCfg.max_images default
if [ -n "$CONFIG" ]; then
  [ -f "$CONFIG" ] || CONFIG="$(dirname "$0")/../$CONFIG"      # relative to the project, like the CLI
  read -r CFG_BACKEND CFG_MODEL MAX_IMAGES < <(python -c 'import sys, yaml; l = yaml.safe_load(open(sys.argv[1]))["llm"]; print(l.get("backend", "ollama"), l["model"], l.get("max_images", 4))' "$CONFIG")
  if [ "$CFG_BACKEND" != "vllm" ]; then
    echo "$CONFIG uses backend '$CFG_BACKEND', not vllm: nothing to start here (Ollama: scripts/setup_ollama.sh)"; exit 1
  fi
  if [ -n "${SERVED:-}" ] && [ "$SERVED" != "$CFG_MODEL" ]; then
    echo "SERVED=$SERVED but $CONFIG asks for llm.model=$CFG_MODEL: every request would get a 404"; exit 1
  fi
  SERVED="$CFG_MODEL"
fi
MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
if [[ "${SERVED:-}" == *-awq ]]; then
  DEF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ; DEF_QUANT=awq; DEF_DTYPE=half; DEF_UTIL=0.80   # the name promises AWQ
elif [ "$MEM_MB" -ge 20000 ] && [ "$CC" -ge 80 ]; then
  DEF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct; DEF_QUANT=none; DEF_DTYPE=bfloat16; DEF_UTIL=0.85
else
  DEF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ; DEF_QUANT=awq; DEF_DTYPE=half; DEF_UTIL=0.80
fi
MODEL="${MODEL:-$DEF_MODEL}"; QUANT="${QUANT:-$DEF_QUANT}"; DTYPE="${DTYPE:-$DEF_DTYPE}"
GPU_UTIL="${GPU_UTIL:-$DEF_UTIL}"; MAX_LEN="${MAX_LEN:-16384}"; LOG="${LOG:-vllm.log}"
SERVED="${SERVED:-qwen2.5-vl-7b}"
QUANT_ARGS=(); [ "$QUANT" != "none" ] && QUANT_ARGS=(--quantization "$QUANT")
echo "GPU ${MEM_MB} MB, compute ${CC}: serving $MODEL as $SERVED (quant=$QUANT dtype=$DTYPE util=$GPU_UTIL images=$MAX_IMAGES)"

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" "${QUANT_ARGS[@]}" \
  --served-model-name "$SERVED" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt "{\"image\": $MAX_IMAGES}" \
  --max-num-seqs 8 \
  --host 127.0.0.1 --port 8000 \
  > "$LOG" 2>&1 &
echo "vLLM starting (pid $!), log: $LOG"

for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8000/v1/models > /dev/null; then echo "vLLM ready"; exit 0; fi
  if ! kill -0 $! 2>/dev/null; then echo "vLLM exited:"; tail -40 "$LOG"; exit 1; fi
  sleep 5
done
echo "vLLM did not come up in 15 min - see $LOG"; tail -40 "$LOG"; exit 1
