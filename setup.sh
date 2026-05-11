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
# Single-instance lock. Prevents two setups colliding on aria2 partial
# files, duplicate /start.sh patches, or racing supervisor launches when
# e.g. an SSH session re-runs setup.sh while the template boot is still
# running. Exits 0 (no-op) if another setup is already in progress.
# ─────────────────────────────────────────────
exec 8>/var/lock/ai-gen-api-v2-setup.lock
if ! flock -n 8; then
  log "setup.sh: another setup already in progress — exiting"
  exit 0
fi

API_REPO="https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main"

# ─────────────────────────────────────────────
# Bind :7860 IMMEDIATELY with an install-progress server so the proxy
# returns 503 + a useful JSON body during setup — not a silent Cloudflare
# 502. start_api.sh's STALE_PID logic will take the port cleanly when
# it's ready to bind uvicorn.
#
# Skip if :7860 is already held by the real API supervisor (pod restart
# with /start.sh hook) or if another status server is still alive.
# ─────────────────────────────────────────────
STATUS_PID_FILE=/var/run/ai-gen-api-v2-status.pid
if [ -f "$STATUS_PID_FILE" ]; then
  kill "$(cat "$STATUS_PID_FILE" 2>/dev/null)" 2>/dev/null || true
  rm -f "$STATUS_PID_FILE"
  sleep 0.5
fi

if pgrep -xf "bash /workspace/start_api.sh" >/dev/null 2>&1; then
  log "start_api.sh supervisor already running — skipping status server"
elif netstat -tln 2>/dev/null | grep -q ":7860 "; then
  log ":7860 already bound — skipping status server"
else
  cat > /tmp/ai-gen-api-v2-status.py <<'PYEOF'
import http.server, socketserver, json, os, subprocess, signal, sys
socketserver.ThreadingTCPServer.allow_reuse_address = True

def recent_log():
    try:
        return subprocess.check_output(
            ['tail', '-40', '/workspace/api_setup.log'],
            text=True, timeout=2
        ).splitlines()[-30:]
    except Exception:
        return []

class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        payload = {
            "status": "installing",
            "message": (
                "AI Gen API v2 is still setting up. First deploy downloads "
                "~72 GB of models (3-10 min on warm HF CDN). /health will "
                "return HTTP 200 once uvicorn is bound."
            ),
            "pod_id": os.environ.get('RUNPOD_POD_ID', 'unknown'),
            "hint": "tail -f /workspace/api_setup.log",
            "recent_log": recent_log(),
        }
        self._send(503, json.dumps(payload, indent=2).encode())

signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
try:
    with socketserver.ThreadingTCPServer(("0.0.0.0", 7860), H) as srv:
        srv.serve_forever()
except OSError:
    # port already taken (another instance or the real uvicorn grabbed it)
    sys.exit(0)
PYEOF
  # 8>&- closes the setup-lock FD so children don't inherit it (otherwise
  # the lock would persist for the lifetime of the daemon, blocking re-runs).
  setsid nohup python3 /tmp/ai-gen-api-v2-status.py </dev/null >/dev/null 2>&1 8>&- &
  echo $! > "$STATUS_PID_FILE"
  sleep 0.5
  if kill -0 "$(cat "$STATUS_PID_FILE")" 2>/dev/null; then
    log "Status server bound :7860 — /health returns 503 + install progress until API ready"
  else
    log "WARN: status server failed to start (port taken?) — continuing"
    rm -f "$STATUS_PID_FILE"
  fi
fi

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

# ─────────────────────────────────────────────
# 1. Pip dependencies + aria2 (for parallel model downloads)
# ─────────────────────────────────────────────
log "[1/4] Installing pip dependencies + aria2..."
$PIP install -q fastapi uvicorn httpx websockets python-multipart pillow 2>&1 | tail -1
# SageAttention package install only — DO NOT auto-enable via comfyui_args.txt.
# SageAttention 1.0.6 hangs the Gemma 12B prompt-enhancer's autoregressive
# token-generation path (the kernel is tuned for fixed-shape diffusion
# attention, not LLM decoding), which deadlocks every /ltx/* call before the
# KSampler ever runs. We leave the package installed so users who want sage
# for non-Gemma workflows can enable it manually in comfyui_args.txt.
$PIP install -q sageattention 2>&1 | tail -1 || log "  WARN: sageattention install failed — ComfyUI will fall back to PyTorch SDPA"

# Initialize the args file with just a header if it doesn't exist. We
# deliberately don't append --use-sage-attention or --fast here (see above).
ARGS_FILE="/workspace/runpod-slim/comfyui_args.txt"
[ -f "$ARGS_FILE" ] || echo "# Add your custom ComfyUI arguments here (one per line)" > "$ARGS_FILE"

if ! command -v aria2c >/dev/null 2>&1; then
  log "  Installing aria2..."
  apt-get update -qq 2>&1 | tail -1
  apt-get install -y -qq aria2 2>&1 | tail -1
fi

if ! command -v aria2c >/dev/null 2>&1; then
  log "  FATAL: aria2 install failed. Cannot do parallel downloads."
  exit 1
fi
log "  Done"

# ─────────────────────────────────────────────
# 2. Download all models in parallel via aria2
#    (Previously 6 serial wget phases → single parallel phase.
#     HF CDN routing + 16 connections/file yields 200–500 MB/s
#     vs ~1 MB/s serial wget. Full 72 GB in ~3–10 min, not hours.)
# ─────────────────────────────────────────────
log "[2/4] Downloading models (parallel, ~72 GB total)..."
mkdir -p "$MODELS/diffusion_models" "$MODELS/vae" "$MODELS/text_encoders" \
         "$MODELS/loras" "$MODELS/checkpoints" "$MODELS/latent_upscale_models"

ARIA2_INPUT="/tmp/ai-gen-api-v2-downloads.txt"
cat > "$ARIA2_INPUT" <<EOF
https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors
  dir=$MODELS/diffusion_models
  out=flux2-klein-9b.safetensors
https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors
  dir=$MODELS/vae
  out=flux2-vae.safetensors
https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors
  dir=$MODELS/text_encoders
  out=qwen_3_8b_fp8mixed.safetensors
https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors
  dir=$MODELS/loras
  out=bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors
https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors
  dir=$MODELS/checkpoints
  out=ltx-2.3-22b-dev-fp8.safetensors
https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors
  dir=$MODELS/loras
  out=ltx-2.3-22b-distilled-lora-384.safetensors
https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
  dir=$MODELS/loras
  out=gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
  dir=$MODELS/text_encoders
  out=gemma_3_12B_it_fp4_mixed.safetensors
https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors
  dir=$MODELS/latent_upscale_models
  out=ltx-2.3-spatial-upscaler-x2-1.0.safetensors
EOF

# HF token is passed as an Authorization header for all requests.
# Public repos ignore it; gated repos (FLUX.2, LTX-2.3-fp8) require it.
# --continue=true skips fully-downloaded files and resumes partials,
# so rerunning this script after a network blip is a no-op for done files.
run_aria2() {
  aria2c \
    --input-file="$ARIA2_INPUT" \
    --header="Authorization: Bearer $TOKEN" \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=10M \
    --max-concurrent-downloads=3 \
    --continue=true \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --file-allocation=none \
    --console-log-level=warn \
    --summary-interval=30 \
    2>&1 | tee -a "$LOG" | grep -E "Download complete|error|FAILED" || true
  return ${PIPESTATUS[0]}
}

run_aria2
ARIA2_EXIT=$?
if [ $ARIA2_EXIT -ne 0 ]; then
  log "  WARN: aria2 exited with code $ARIA2_EXIT — will verify and retry."
fi

# Per-file verification. aria2 has been observed to exit 0 while silently
# leaving individual files missing or short (e.g., a gated URL transiently
# 401s, or the parent shell gets SIGTERM mid-download). Trusting the exit
# code alone causes "All N models downloaded" to print over a broken set.
# So we compare each file's on-disk size to HF's x-linked-size header and
# resume any that don't match before declaring success.
#
# local_path|HF_URL — kept in lockstep with $ARIA2_INPUT above.
EXPECTED_FILES="\
diffusion_models/flux2-klein-9b.safetensors|https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors
vae/flux2-vae.safetensors|https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors
text_encoders/qwen_3_8b_fp8mixed.safetensors|https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors
loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors|https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors
checkpoints/ltx-2.3-22b-dev-fp8.safetensors|https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors
loras/ltx-2.3-22b-distilled-lora-384.safetensors|https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors
loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors|https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
text_encoders/gemma_3_12B_it_fp4_mixed.safetensors|https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors|https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

verify_models() {
  local bad=0
  local entry local_path url expected actual full
  while IFS='|' read -r local_path url; do
    [ -z "$local_path" ] && continue
    full="$MODELS/$local_path"
    expected=$(curl -sSI -H "Authorization: Bearer $TOKEN" "$url" 2>/dev/null \
      | tr -d '\r' | awk 'tolower($1)=="x-linked-size:"{print $2; exit}')
    actual=$(stat -c%s "$full" 2>/dev/null || echo 0)
    if [ -z "$expected" ]; then
      log "    WARN: could not fetch expected size for $local_path (HF unreachable?)"
      continue
    fi
    if [ "$expected" != "$actual" ]; then
      log "    INCOMPLETE: $local_path (have $actual, expected $expected)"
      bad=$((bad+1))
    fi
  done <<<"$EXPECTED_FILES"
  return $bad
}

log "  Verifying downloaded files against HF expected sizes..."
if ! verify_models; then
  log "  One or more files incomplete — resuming via aria2 (--continue)..."
  run_aria2 || true
  if ! verify_models; then
    log "  FATAL: model files still incomplete after retry. Check:"
    log "    - HF token has access to gated repos:"
    log "        black-forest-labs/FLUX.2-klein-9B"
    log "        Lightricks/LTX-2.3-fp8"
    log "    - Disk space ($(df -h "$MODELS" | awk 'NR==2{print $4 " free on " $6}'))"
    log "    - Network connectivity to huggingface.co"
    exit 1
  fi
fi

# Symlink so both filenames resolve (some workflows reference flux-2-klein-9b.safetensors)
ln -sf "$MODELS/diffusion_models/flux2-klein-9b.safetensors" \
       "$MODELS/diffusion_models/flux-2-klein-9b.safetensors"

log "  All 9 models verified at expected sizes"

# ─────────────────────────────────────────────
# 3. LanPaint custom node (required for FLUX face swap)
# ─────────────────────────────────────────────
mkdir -p "$NODES"
LANPAINT_FRESH=0
if [ ! -d "$NODES/LanPaint" ]; then
  log "[3/4] Installing LanPaint custom node..."
  (
    cd "$NODES"
    git clone -q https://github.com/scraed/LanPaint
    if [ -f "LanPaint/requirements.txt" ]; then
      $PIP install -q -r LanPaint/requirements.txt 2>&1 | tail -1
    fi
  )
  LANPAINT_FRESH=1
  log "  LanPaint installed"
else
  log "[3/4] LanPaint already installed"
fi

# (The conditional LanPaint-only ComfyUI relaunch that used to live here
# is now subsumed by the start_comfy.sh supervisor below: it unconditionally
# kills /start.sh's unsupervised ComfyUI and relaunches under flock'd
# auto-restart, so newly-installed custom nodes are always picked up.)

# ─────────────────────────────────────────────
# 4. Download API + create startup scripts
# ─────────────────────────────────────────────
log "[4/4] Setting up API..."
mkdir -p /workspace/api

# Always fetch latest main.py from repo
wget -q -O /workspace/api/main.py "${API_REPO}/main.py"
if [ ! -s "/workspace/api/main.py" ]; then
  log "  ERROR: Failed to download main.py"
  exit 1
fi
log "  main.py downloaded (latest)"

# Save detected paths for start_api.sh and start_comfy.sh
cat > /workspace/api/config.env << CONFEOF
COMFY_ROOT=$COMFY_ROOT
PYTHON=$PYTHON
PIP=$PIP
API_REPO=$API_REPO
CONFEOF

# ─────────────────────────────────────────────
# Create the ComfyUI supervisor (mirrors start_api.sh: flock'd, auto-restart
# loop, parsed CLI args from comfyui_args.txt). RunPod's /start.sh launches
# ComfyUI as an unsupervised child — if it segfaults, OOMs, or is killed,
# /start.sh's `wait` returns and ComfyUI stays down until pod restart. The
# supervisor relaunches it within 5 seconds on any exit code.
# ─────────────────────────────────────────────
cat > /workspace/start_comfy.sh << 'COMFYEOF'
#!/bin/bash
# =============================================================
# AI Gen API v2 — supervisor for ComfyUI on :8188
#
# Safe to invoke multiple times: a flock guards the while-loop so
# a second invocation (e.g. setup.sh re-running on pod restart)
# exits immediately instead of racing the first supervisor.
#
# Reads extra ComfyUI CLI flags one-per-line from
# /workspace/runpod-slim/comfyui_args.txt (comments + blanks ignored).
# =============================================================
LOG_SETUP="/workspace/comfy_setup.log"
LOG_OUT="/workspace/comfyui.log"
ARGS_FILE="/workspace/runpod-slim/comfyui_args.txt"
PORT=8188
FIXED_ARGS=(--listen 0.0.0.0 --port "$PORT" --enable-cors-header)

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_SETUP"; }

# ─── single-instance guard ───
exec 9>/var/lock/ai-gen-comfy.lock
if ! flock -n 9; then
  log "start_comfy.sh: another supervisor already running — exiting"
  exit 0
fi

# Truncate old logs on restart (cap at last 500 lines each)
tail -500 "$LOG_SETUP" > "${LOG_SETUP}.tmp" 2>/dev/null && mv "${LOG_SETUP}.tmp" "$LOG_SETUP"
tail -500 "$LOG_OUT"   > "${LOG_OUT}.tmp"   2>/dev/null && mv "${LOG_OUT}.tmp"   "$LOG_OUT"

if [ ! -f /workspace/api/config.env ]; then
  log "ERROR: /workspace/api/config.env missing — setup.sh did not complete"
  exit 1
fi
source /workspace/api/config.env

if [ -z "$PYTHON" ] || [ -z "$COMFY_ROOT" ]; then
  log "ERROR: PYTHON or COMFY_ROOT not set in config.env"
  exit 1
fi
if [ ! -x "$PYTHON" ]; then
  log "ERROR: PYTHON ($PYTHON) not executable"
  exit 1
fi
if [ ! -f "$COMFY_ROOT/main.py" ]; then
  log "ERROR: $COMFY_ROOT/main.py not found"
  exit 1
fi

# Free :PORT if a stale ComfyUI is holding it (e.g. the one /start.sh
# launched on pod boot). Target by socket owner — safer than pkill -f
# against an argv pattern, which can accidentally match caller shells.
STALE_PID=$(netstat -tlnp 2>/dev/null | awk -v p=":$PORT\$" '$4 ~ p {split($7, a, "/"); print a[1]; exit}')
if [ -n "$STALE_PID" ] && [ "$STALE_PID" != "-" ]; then
  log "Freeing :$PORT (stale owner PID=$STALE_PID)"
  kill "$STALE_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "$STALE_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$STALE_PID" 2>/dev/null || true
fi

# Parse extra args from comfyui_args.txt: strip comments + blank lines,
# word-split each remaining line into one or more args.
EXTRA_ARGS=()
if [ -f "$ARGS_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    trimmed="${line#"${line%%[![:space:]]*}"}"
    [ -z "$trimmed" ] && continue
    [[ "$trimmed" =~ ^# ]] && continue
    # shellcheck disable=SC2206
    args=($trimmed)
    EXTRA_ARGS+=("${args[@]}")
  done < "$ARGS_FILE"
fi
log "ComfyUI extra args: ${EXTRA_ARGS[*]:-(none)}"

cd "$COMFY_ROOT" || exit 1
log "Starting ComfyUI on port $PORT..."
while true; do
  "$PYTHON" main.py "${FIXED_ARGS[@]}" "${EXTRA_ARGS[@]}" >> "$LOG_OUT" 2>&1
  EXIT_CODE=$?
  log "ComfyUI exited with code $EXIT_CODE — restarting in 5s..."
  sleep 5
done
COMFYEOF

chmod +x /workspace/start_comfy.sh

# Launch the ComfyUI supervisor unless one's already running. The supervisor's
# STALE_PID step will adopt :8188 from /start.sh's unsupervised ComfyUI on the
# first launch; subsequent setup re-runs hit the flock and exit cleanly.
# 8>&- closes the setup-lock FD so the daemon doesn't inherit it.
if pgrep -xf "bash /workspace/start_comfy.sh" >/dev/null 2>&1; then
  log "  ComfyUI supervisor already running — leaving it alone"
else
  setsid nohup bash /workspace/start_comfy.sh </dev/null >>/workspace/api_setup.log 2>&1 8>&- &
  disown 2>/dev/null || true
  log "  ComfyUI supervisor launched (will adopt :8188 from /start.sh)"
fi

# Create startup script (runs on every pod start/restart)
cat > /workspace/start_api.sh << 'STARTEOF'
#!/bin/bash
# =============================================================
# AI Gen API v2 — supervisor for uvicorn on :7860
#
# Safe to invoke multiple times: a flock guards the while-loop so
# a second invocation (e.g. setup.sh re-running on pod restart)
# exits immediately instead of racing the first supervisor.
# =============================================================
LOG="/workspace/api_setup.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

# ─── single-instance guard ───
# Hold an exclusive lock for the lifetime of this process. If another
# supervisor is already running, exit 0 (not an error — it's a no-op).
exec 9>/var/lock/ai-gen-api-v2.lock
if ! flock -n 9; then
  log "start_api.sh: another supervisor already running — exiting"
  exit 0
fi

# Truncate old logs on restart (cap at last 500 lines each)
tail -500 "$LOG" > "${LOG}.tmp" 2>/dev/null && mv "${LOG}.tmp" "$LOG"
tail -500 /workspace/api.log > /workspace/api.log.tmp 2>/dev/null && mv /workspace/api.log.tmp /workspace/api.log

# Load detected Python/pip paths
if [ ! -f /workspace/api/config.env ]; then
  log "ERROR: /workspace/api/config.env missing — setup.sh did not complete"
  exit 1
fi
source /workspace/api/config.env

# Reinstall pip deps (can be lost on pod restart)
log "Installing pip deps..."
$PIP install -q fastapi uvicorn httpx websockets python-multipart pillow 2>&1 | tail -1

# Always fetch latest main.py from repo on restart
log "Fetching latest API code..."
wget -q -O /workspace/api/main.py.new "${API_REPO}/main.py"
if [ -s "/workspace/api/main.py.new" ]; then
  mv /workspace/api/main.py.new /workspace/api/main.py
else
  log "WARN: Failed to download main.py — using existing version"
  rm -f /workspace/api/main.py.new
fi

# Wait for ComfyUI to be ready
log "Waiting for ComfyUI..."
MAX_WAIT=600; WAITED=0
until curl -s http://localhost:8188/system_stats > /dev/null 2>&1; do
  sleep 3; WAITED=$((WAITED + 3))
  if [ $WAITED -ge $MAX_WAIT ]; then log "ERROR: ComfyUI did not start within 10 min"; exit 1; fi
done
log "ComfyUI ready after ${WAITED}s"

# Free :7860 if a stale uvicorn from a prior supervisor is holding it.
# Target by socket owner (netstat) — safer than pkill -f against an argv
# pattern, which can accidentally match caller shells whose cmdline
# happens to contain "uvicorn main:app" as a substring.
STALE_PID=$(netstat -tlnp 2>/dev/null | awk '$4 ~ /:7860$/ {split($7, a, "/"); print a[1]; exit}')
if [ -n "$STALE_PID" ]; then
  log "Freeing :7860 (stale owner PID=$STALE_PID)"
  kill "$STALE_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "$STALE_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$STALE_PID" 2>/dev/null || true
fi

# Start API with auto-restart on crash
cd /workspace/api || exit 1
log "Starting API on port 7860..."
while true; do
  $PYTHON -m uvicorn main:app --host 0.0.0.0 --port 7860 >> /workspace/api.log 2>&1
  EXIT_CODE=$?
  log "API exited with code $EXIT_CODE — restarting in 5s..."
  sleep 5
done
STARTEOF

chmod +x /workspace/start_api.sh

# ─────────────────────────────────────────────
# Patch /start.sh (idempotent) so pod RESTARTS also auto-launch the API.
#
# /start.sh is the image's CMD — RunPod runs it on every pod start.
# Without this hook, a restart would bring up ComfyUI but not the API,
# forcing manual `bash /workspace/start_api.sh` each time.
#
# The patch is injected on the container layer (/start.sh is not on the
# /workspace volume). It survives pod restarts but is lost on pod
# recreate/rebuild — setup.sh re-applies it on every run, so as long as
# setup.sh is in the template Start Command, the hook self-heals.
# ─────────────────────────────────────────────
patch_start_sh() {
  local f="/start.sh"
  [ -f "$f" ] || { log "  /start.sh not found — skipping restart hook"; return; }
  # The hook block is versioned (v2). If the file already contains the v2
  # marker, skip; if it contains the older v1 marker, we strip it and re-apply
  # v2 so we don't end up with stacked hooks.
  if grep -q "AI Gen API v2 auto-start hook v2" "$f"; then
    log "  /start.sh already has v2 restart hook"
    return
  fi
  # Insert hook right after ComfyUI is launched (line: `python main.py $FIXED_ARGS &`),
  # before the `wait $COMFY_PID` call. setsid + nohup + </dev/null fully detaches so
  # the supervisors survive SSH disconnect, shell exit, and /start.sh teardown.
  #
  # Use atomic rename (write-new-then-mv) so the currently-running /start.sh
  # (which may still be reading the script) isn't truncated mid-read. Processes
  # holding the old inode via their open fd continue reading the old content;
  # new invocations see the new file.
  python3 - "$f" <<'PYEOF'
import os, sys, re
p = sys.argv[1]
src = open(p).read()
# Strip any previously-injected hook block (v1 or v2) so re-runs don't stack.
src = re.sub(
    r'\n# === AI Gen API v2 auto-start hook[^\n]*===\n.*?# === end AI Gen API v2 auto-start hook ===\n',
    '\n',
    src,
    flags=re.S,
)
hook = '''
# === AI Gen API v2 auto-start hook v2 ===
# Launches the ComfyUI + API supervisors in detached sessions so they
# survive shell exit, SSH disconnect, and /start.sh teardown.
#   - start_comfy.sh adopts :8188 from /start.sh's unsupervised ComfyUI
#     and provides auto-restart on crash.
#   - start_api.sh waits for ComfyUI before binding :7860.
if [ -x /workspace/start_comfy.sh ]; then
    echo "AI Gen API v2: launching /workspace/start_comfy.sh"
    setsid nohup bash /workspace/start_comfy.sh </dev/null >>/workspace/api_setup.log 2>&1 &
fi
if [ -x /workspace/start_api.sh ]; then
    echo "AI Gen API v2: launching /workspace/start_api.sh"
    setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &
fi
# === end AI Gen API v2 auto-start hook ===
'''
m = re.search(r'^(python main\.py \$FIXED_ARGS &\s*\nCOMFY_PID=\$!\s*\n)', src, re.M)
if not m:
    sys.exit("could not locate ComfyUI launch block in /start.sh")
out = src[:m.end()] + hook + src[m.end():]
tmp = p + ".new"
with open(tmp, "w") as fh:
    fh.write(out)
os.chmod(tmp, os.stat(p).st_mode)
os.rename(tmp, p)
PYEOF
  log "  /start.sh patched with v2 restart hook (supervisors for ComfyUI + API)"
}
patch_start_sh

# ─────────────────────────────────────────────
# Start the API — but only if it isn't already healthy.
#
# On a pod restart, the /start.sh hook already launched start_api.sh in
# parallel with setup.sh. If that supervisor is up and the health probe
# passes, don't tear it down — a pointless relaunch would cause a ~10s
# outage where uvicorn isn't bound to :7860.
# Otherwise: kill any stale supervisor and start fresh.
# ─────────────────────────────────────────────
if pgrep -xf "bash /workspace/start_api.sh" >/dev/null 2>&1 && \
   curl -s -m 3 http://localhost:7860/health 2>/dev/null | grep -q '"status":"ok"'; then
  log "API supervisor already healthy — leaving it alone"
else
  # Use -xf (exact full-argv match) so we only kill processes whose argv
  # is literally "bash /workspace/start_api.sh" — never a caller shell
  # that merely mentions the string in its own command line.
  pkill -xf "bash /workspace/start_api.sh" 2>/dev/null || true
  sleep 1
  setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 8>&- &
  disown 2>/dev/null || true
  log "API supervisor launched"
fi

log "=========================================="
log "Setup Complete!"
log "  API docs: https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/docs"
log "  Swagger:  https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/docs"
log "  Health:   https://${RUNPOD_POD_ID}-7860.proxy.runpod.net/health"
log "  Setup log: tail -f /workspace/api_setup.log"
log "  API log:   tail -f /workspace/api.log"
log ""
log "  Pod restarts auto-launch the API via /start.sh hook."
log "  Manual relaunch (if needed):"
log "    setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &"
log "=========================================="
