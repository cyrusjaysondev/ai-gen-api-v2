#!/bin/bash
# =============================================================
# AI Gen API v2 — Minimal Setup
# FLUX.2 Klein 9B (head swap) + JuggernautXL (image gen)
#
# Set in RunPod template env vars:
#   SETUP_SCRIPT_URL = https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh
# =============================================================

LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

log "=========================================="
log "AI Gen API v2 Setup Started"
log "Pod ID: $RUNPOD_POD_ID"
log "=========================================="

MODELS="/workspace/ComfyUI/models"
NODES="/workspace/ComfyUI/custom_nodes"

# ─────────────────────────────────────────────
# 1. Pip dependencies
# ─────────────────────────────────────────────
log "[1/6] Installing pip dependencies..."
pip install -q fastapi uvicorn httpx websockets python-multipart 2>&1 | tail -1
log "  Done"

# ─────────────────────────────────────────────
# 2. FLUX.2 Klein 9B UNET (~18GB)
# ─────────────────────────────────────────────
mkdir -p "$MODELS/diffusion_models"
FLUX_UNET="$MODELS/diffusion_models/flux2-klein-9b.safetensors"
if [ ! -f "$FLUX_UNET" ]; then
  log "[2/6] Downloading FLUX.2 Klein 9B UNET (18GB)..."
  wget -q --show-progress -O "$FLUX_UNET" \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/flux2-klein-9b.safetensors"
  log "  Downloaded"
else
  log "[2/6] FLUX Klein 9B already exists"
fi
# Symlink (workflow references flux-2-klein-9b with dash)
ln -sf "$FLUX_UNET" "$MODELS/diffusion_models/flux-2-klein-9b.safetensors"

# ─────────────────────────────────────────────
# 3. FLUX VAE + Qwen text encoder
# ─────────────────────────────────────────────
mkdir -p "$MODELS/vae" "$MODELS/text_encoders"

FLUX_VAE="$MODELS/vae/flux2-vae.safetensors"
if [ ! -f "$FLUX_VAE" ]; then
  log "[3/6] Downloading FLUX VAE (336MB)..."
  wget -q --show-progress -O "$FLUX_VAE" \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors"
  log "  Downloaded"
else
  log "[3/6] FLUX VAE already exists"
fi

QWEN="$MODELS/text_encoders/qwen_3_8b_fp8mixed.safetensors"
if [ ! -f "$QWEN" ]; then
  log "[3/6] Downloading Qwen 3 8B text encoder (8.7GB)..."
  wget -q --show-progress -O "$QWEN" \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"
  log "  Downloaded"
else
  log "[3/6] Qwen 3 8B already exists"
fi

# ─────────────────────────────────────────────
# 4. BFS Head Swap LoRA
# ─────────────────────────────────────────────
mkdir -p "$MODELS/loras"

BFS_HEAD="$MODELS/loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"
if [ ! -f "$BFS_HEAD" ]; then
  log "[4/6] Downloading BFS head swap LoRA (663MB)..."
  wget -q --show-progress -O "$BFS_HEAD" \
    "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"
  log "  Downloaded"
else
  log "[4/6] BFS head LoRA already exists"
fi

# ─────────────────────────────────────────────
# 5. LanPaint custom node (FLUX head swap)
# ─────────────────────────────────────────────
if [ ! -d "$NODES/LanPaint" ]; then
  log "[5/6] Installing LanPaint custom node..."
  cd "$NODES"
  git clone -q https://github.com/scraed/LanPaint
  if [ -f "$NODES/LanPaint/requirements.txt" ]; then
    cd LanPaint && pip install -q -r requirements.txt 2>&1 | tail -1
  fi
  log "  LanPaint installed"
else
  log "[5/6] LanPaint already exists"
fi

# ─────────────────────────────────────────────
# 7. Download main.py + start API
# ─────────────────────────────────────────────
log "[7/6] Setting up API..."
mkdir -p /workspace/api

wget -q -O /workspace/api/main.py \
  "https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/main.py"

if [ ! -f "/workspace/api/main.py" ] || [ ! -s "/workspace/api/main.py" ]; then
  log "  WARNING: Failed to download main.py — download manually"
fi

# Create startup script (survives SSH disconnect + pod restart)
cat > /workspace/start_api.sh << 'EOF'
#!/bin/bash
LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

# Reinstall pip deps (lost on restart)
pip install -q fastapi uvicorn httpx websockets python-multipart 2>&1 | tail -1

# Wait for ComfyUI
log "Waiting for ComfyUI..."
MAX_WAIT=300; WAITED=0
until curl -s http://localhost:8188/system_stats > /dev/null 2>&1; do
  sleep 3; WAITED=$((WAITED + 3))
  if [ $WAITED -ge $MAX_WAIT ]; then log "ComfyUI timeout"; exit 1; fi
done
log "ComfyUI ready after ${WAITED}s"

# Start API
cd /workspace/api || exit 1
log "Starting API on port 7860..."
exec /opt/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 7860 >> /workspace/api.log 2>&1
EOF

chmod +x /workspace/start_api.sh
nohup bash /workspace/start_api.sh >> /workspace/api_setup.log 2>&1 & disown

log "=========================================="
log "Setup Complete!"
log "  API: https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/docs"
log "  Logs: tail -f /workspace/api.log"
log "=========================================="
