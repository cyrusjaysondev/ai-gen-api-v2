#!/bin/bash
# =============================================================
# AI Gen API v2 — Setup
# FLUX.2 Klein 9B (face swap + text-to-image)
# LTX 2.3 22B (image-to-video, text-to-video, face-animate pipeline)
#
# Works with RunPod ComfyUI template (runpod/comfyui:latest)
# ComfyUI location: /workspace/runpod-slim/ComfyUI/
# Python venv: /workspace/runpod-slim/ComfyUI/.venv-cu128/
#
# Set as start command in template overrides:
#   bash -c "wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh && bash /tmp/setup.sh &"
#
# Required env var (set in RunPod template):
#   HF_TOKEN = your Hugging Face token
#     - needs access to: black-forest-labs/FLUX.2-klein-9B
#     - needs access to: Lightricks/LTX-2.3-fp8
#   Accept licenses at:
#     https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
#     https://huggingface.co/Lightricks/LTX-2.3-fp8
# =============================================================

LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

# ─────────────────────────────────────────────
# HF Token (required for gated model downloads)
# ─────────────────────────────────────────────
TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
if [ -z "$TOKEN" ]; then
  log "ERROR: HF_TOKEN env var is not set."
  log "  Set it in your RunPod template environment variables."
  log "  Get a token at: https://huggingface.co/settings/tokens"
  log "  Then accept licenses at:"
  log "    https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
  log "    https://huggingface.co/Lightricks/LTX-2.3-fp8"
  exit 1
fi

# ─────────────────────────────────────────────
# Auto-detect ComfyUI location
# ─────────────────────────────────────────────
if [ -d "/workspace/runpod-slim/ComfyUI" ]; then
  COMFY_ROOT="/workspace/runpod-slim/ComfyUI"
elif [ -d "/workspace/ComfyUI" ]; then
  COMFY_ROOT="/workspace/ComfyUI"
else
  log "ERROR: ComfyUI not found. Searching..."
  COMFY_ROOT=$(find /workspace -name "main.py" -path "*/ComfyUI/*" -exec dirname {} \; 2>/dev/null | head -1)
  if [ -z "$COMFY_ROOT" ]; then
    log "ERROR: ComfyUI not found anywhere. Exiting."
    exit 1
  fi
fi

# ─────────────────────────────────────────────
# Auto-detect Python
# ─────────────────────────────────────────────
if [ -f "$COMFY_ROOT/.venv-cu128/bin/python" ]; then
  PYTHON="$COMFY_ROOT/.venv-cu128/bin/python"
  PIP="$COMFY_ROOT/.venv-cu128/bin/pip"
elif [ -f "/opt/venv/bin/python" ]; then
  PYTHON="/opt/venv/bin/python"
  PIP="/opt/venv/bin/pip"
else
  PYTHON=$(which python3)
  PIP=$(which pip3)
fi

MODELS="$COMFY_ROOT/models"
NODES="$COMFY_ROOT/custom_nodes"

log "=========================================="
log "AI Gen API v2 Setup Started"
log "Pod ID: $RUNPOD_POD_ID"
log "ComfyUI: $COMFY_ROOT"
log "Python: $PYTHON"
log "=========================================="

# Helper: download with size check and cleanup on failure
# Usage: download_file <url> <dest_path> <label> [hf_token]
download_file() {
  local URL="$1"
  local DEST="$2"
  local LABEL="$3"
  local HF_TOKEN="$4"

  if [ -s "$DEST" ]; then
    log "  $LABEL already exists, skipping"
    return 0
  fi

  # Remove empty/partial file if present
  [ -f "$DEST" ] && rm -f "$DEST"

  log "  Downloading $LABEL..."
  if [ -n "$HF_TOKEN" ]; then
    wget -q --show-progress --header="Authorization: Bearer $HF_TOKEN" -O "$DEST" "$URL"
  else
    wget -q --show-progress -O "$DEST" "$URL"
  fi

  if [ ! -s "$DEST" ]; then
    log "  ERROR: $LABEL download failed or produced empty file"
    rm -f "$DEST"
    return 1
  fi
  log "  $LABEL downloaded"
}

# ─────────────────────────────────────────────
# 1. Pip dependencies
# ─────────────────────────────────────────────
log "[1/8] Installing pip dependencies..."
$PIP install -q fastapi uvicorn httpx websockets python-multipart pillow 2>&1 | tail -1
log "  Done"

# ─────────────────────────────────────────────
# 2. FLUX.2 Klein 9B UNET (~18GB) — requires HF token + license
# ─────────────────────────────────────────────
log "[2/8] FLUX.2 Klein 9B UNET..."
mkdir -p "$MODELS/diffusion_models"
FLUX_UNET="$MODELS/diffusion_models/flux2-klein-9b.safetensors"
download_file \
  "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors" \
  "$FLUX_UNET" \
  "FLUX Klein 9B UNET (18GB)" \
  "$TOKEN"

if [ $? -ne 0 ]; then
  log "  FATAL: FLUX UNET download failed. Check your HF_TOKEN has access to black-forest-labs/FLUX.2-klein-9B"
  exit 1
fi

# Symlink so both filenames resolve (workflow references flux-2-klein-9b.safetensors)
ln -sf "$FLUX_UNET" "$MODELS/diffusion_models/flux-2-klein-9b.safetensors"

# ─────────────────────────────────────────────
# 3. FLUX VAE + Qwen text encoder
# ─────────────────────────────────────────────
log "[3/8] FLUX VAE + Qwen text encoder..."
mkdir -p "$MODELS/vae" "$MODELS/text_encoders"

download_file \
  "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors" \
  "$MODELS/vae/flux2-vae.safetensors" \
  "FLUX VAE (321MB)"

download_file \
  "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" \
  "$MODELS/text_encoders/qwen_3_8b_fp8mixed.safetensors" \
  "Qwen 3 8B text encoder (8.1GB)"

# ─────────────────────────────────────────────
# 4. BFS Head Swap LoRA
# ─────────────────────────────────────────────
log "[4/8] BFS Head Swap LoRA..."
mkdir -p "$MODELS/loras"

download_file \
  "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors" \
  "$MODELS/loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors" \
  "BFS head swap LoRA (633MB)"

# ─────────────────────────────────────────────
# 5. LanPaint custom node (required for face swap)
# ─────────────────────────────────────────────
mkdir -p "$NODES"
if [ ! -d "$NODES/LanPaint" ]; then
  log "[5/8] Installing LanPaint custom node..."
  (
    cd "$NODES"
    git clone -q https://github.com/scraed/LanPaint
    if [ -f "LanPaint/requirements.txt" ]; then
      $PIP install -q -r LanPaint/requirements.txt 2>&1 | tail -1
    fi
  )
  log "  LanPaint installed"
else
  log "[5/8] LanPaint already exists"
fi

# ─────────────────────────────────────────────
# 6. LTX-2.3 Checkpoint (~27GB) — requires HF token + license
# ─────────────────────────────────────────────
log "[6/8] LTX-2.3 checkpoint..."
mkdir -p "$MODELS/checkpoints"
download_file \
  "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors" \
  "$MODELS/checkpoints/ltx-2.3-22b-dev-fp8.safetensors" \
  "LTX-2.3 22B dev fp8 (27GB)" \
  "$TOKEN"

if [ $? -ne 0 ]; then
  log "  FATAL: LTX-2.3 checkpoint download failed. Check your HF_TOKEN has access to Lightricks/LTX-2.3-fp8"
  exit 1
fi

# ─────────────────────────────────────────────
# 7. LTX-2.3 LoRAs + text encoder + spatial upscaler
# ─────────────────────────────────────────────
log "[7/8] LTX-2.3 LoRAs + text encoder + upscaler..."
mkdir -p "$MODELS/loras" "$MODELS/text_encoders" "$MODELS/latent_upscale_models"

download_file \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors" \
  "$MODELS/loras/ltx-2.3-22b-distilled-lora-384.safetensors" \
  "LTX-2.3 distilled LoRA (7.1GB)"

download_file \
  "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" \
  "$MODELS/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" \
  "Gemma abliterated LoRA (599MB)"

download_file \
  "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" \
  "$MODELS/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" \
  "Gemma 3 12B text encoder (8.8GB)"

download_file \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" \
  "$MODELS/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" \
  "LTX-2.3 spatial upscaler (950MB)"

# ─────────────────────────────────────────────
# 8. Download main.py + start API
# ─────────────────────────────────────────────
log "[8/8] Setting up API..."
mkdir -p /workspace/api

wget -q -O /workspace/api/main.py \
  "https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/main.py"

if [ ! -s "/workspace/api/main.py" ]; then
  log "  ERROR: Failed to download main.py"
  exit 1
fi
log "  main.py downloaded"

# Write detected paths so start_api.sh can use the correct Python
cat > /workspace/api/config.env << CONFEOF
COMFY_ROOT=$COMFY_ROOT
PYTHON=$PYTHON
PIP=$PIP
CONFEOF

# Create startup script (runs on every pod restart)
cat > /workspace/start_api.sh << 'EOF'
#!/bin/bash
LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

# Load detected Python/pip paths
source /workspace/api/config.env

# Reinstall pip deps (lost on pod restart)
$PIP install -q fastapi uvicorn httpx websockets python-multipart pillow 2>&1 | tail -1

# Wait for ComfyUI to be ready
log "Waiting for ComfyUI..."
MAX_WAIT=300; WAITED=0
until curl -s http://localhost:8188/system_stats > /dev/null 2>&1; do
  sleep 3; WAITED=$((WAITED + 3))
  if [ $WAITED -ge $MAX_WAIT ]; then log "ERROR: ComfyUI did not start within 5 min"; exit 1; fi
done
log "ComfyUI ready after ${WAITED}s"

# Kill any stale API process on port 7860
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1

# Start API
cd /workspace/api || exit 1
log "Starting API on port 7860..."
exec $PYTHON -m uvicorn main:app --host 0.0.0.0 --port 7860 >> /workspace/api.log 2>&1
EOF

chmod +x /workspace/start_api.sh
nohup bash /workspace/start_api.sh >> /workspace/api_setup.log 2>&1 & disown

log "=========================================="
log "Setup Complete!"
log "  API docs: https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/docs"
log "  Setup log: tail -f /workspace/api_setup.log"
log "  API log:   tail -f /workspace/api.log"
log "=========================================="
