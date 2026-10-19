#!/usr/bin/env bash
# One-time: pull the VLM and create the 16k-context variant used by configs/default.yaml.
set -euo pipefail
cd "$(dirname "$0")"
ollama pull qwen2.5vl:7b
ollama create qwen2.5vl-16k -f Modelfile.qwen2.5vl-16k
ollama list | grep qwen2.5vl
