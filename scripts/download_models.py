"""One-time model download (the only step that needs internet). Afterwards the system runs offline:
configs set HF_HUB_OFFLINE=1 and the VLM is served by the local Ollama."""
from huggingface_hub import snapshot_download

for repo in ["BAAI/bge-m3"]:
    print("downloading", repo, "->", snapshot_download(repo))
print("now run scripts/setup_ollama.sh for the VLM")
