# AI Gen API v2

API on RunPod for:
- **FLUX.2 Klein 9B** — text-to-image + AI head/face swap
- **LTX 2.3 22B** — image-to-video, text-to-video, face-animate pipeline

Full step-by-step setup: [SETUP.md](SETUP.md). Full API reference: [API.md](API.md).

---

## Prerequisites

You need both of these before the pod can install:

1. **HuggingFace token** with Read access — https://huggingface.co/settings/tokens
2. **Accept model licenses** (click "Agree and access repository" on each):
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
   - https://huggingface.co/Lightricks/LTX-2.3-fp8

---

## Quick Deploy (new pod)

### 1. Create Template
- **runpod.io** → **Templates** → **+ New Template**
- **Container Image:** `runpod/comfyui:latest`
- **Volume:** `200 GB` mounted at `/workspace`
- **Expose Ports:** `7860, 8188, 8888`

### 2. Set Environment Variables
In the template, add:

```
HF_TOKEN = hf_your_token_here
```

> Setting `SETUP_SCRIPT_URL` as an env var alone does **not** trigger the installer — nothing in the stock image reads it. You must set the Start Command (next step).

### 3. Set the Start Command
In the template, paste this as the **Container Start Command**:

```bash
bash -c "/start.sh & sleep 3 && wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh && bash /tmp/setup.sh; wait"
```

> **Why this exact form:** the `runpod/comfyui:latest` image ships `/start.sh` as its
> entrypoint (starts SSH, JupyterLab, FileBrowser, and ComfyUI). A Container Start
> Command replaces that entrypoint, so you must re-invoke `/start.sh` yourself in the
> background, otherwise SSH and ComfyUI never come up. The trailing `wait` keeps the
> container alive after `setup.sh` finishes so `/start.sh` (which ends in
> `sleep infinity`) continues supervising ComfyUI.

### 4. Deploy and Wait
Click Deploy. `setup.sh` runs 4 steps:

1. **Install pip deps + aria2** (~5 s)
2. **Download all 9 models in parallel via aria2** (~72 GB; resumable, skips anything already on the volume)
3. **Install LanPaint custom node** (skipped if already present)
4. **Fetch `main.py`, patch `/start.sh` with the restart hook, launch the API supervisor**

**First deploy: ~3–10 minutes** on a warm HuggingFace CDN (parallel aria2 at 200–500 MB/s beats serial wget by 30–50×).
**Subsequent deploys / pod restarts: ~10–20 seconds** — models are already on the volume, so aria2 just verifies and skips.

### 5. Monitor Progress
Open the Jupyter terminal (port 8888) and run:
```bash
tail -f /workspace/api_setup.log
```

If the file doesn't exist, the start command never ran — verify step 3 above.

### 6. Verify
```
https://YOUR_POD_ID-7860.proxy.runpod.net/health
https://YOUR_POD_ID-7860.proxy.runpod.net/docs
```

---

## Scaling (spin up more pods)

Just repeat steps 1-3. Each new pod auto-configures:
- Models download once per volume, skip on restart
- Pod ID is read from `$RUNPOD_POD_ID` at runtime — no manual config needed
- API starts automatically after ComfyUI is ready

---

## API Endpoints

### Health Check
```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/health
```

### Text to Image
```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/t2i \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains, photorealistic, 4K",
    "width": 1024,
    "height": 1024
  }'
```

**Parameters:**
| Param | Default | Description |
|-------|---------|-------------|
| `prompt` | required | What to generate |
| `width` | 1024 | Image width |
| `height` | 1024 | Image height |
| `seed` | -1 (random) | Reproducibility seed |
| `steps` | 4 | Inference steps (4 is good for Klein) |
| `cfg` | 1.0 | CFG scale |
| `guidance` | 4.0 | FLUX guidance strength (2.0-6.0) |

### Head Swap (FLUX)
```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_template.png" \
  -F "face_image=@my_face.png"
```

**Parameters:**
| Param | Default | Description |
|-------|---------|-------------|
| `target_image` | required | Body/template image (head gets replaced) |
| `face_image` | required | Face photo (identity to transfer) |
| `seed` | -1 (random) | Reproducibility seed |
| `megapixels` | 2.0 | Output resolution (1.0-2.0) |
| `steps` | 4 | Inference steps |
| `cfg` | 1.0 | CFG scale |
| `guidance` | 4.0 | FLUX guidance (2.0-6.0) |
| `lora_strength` | 1.0 | Head swap LoRA strength (0.0-1.5) |

### Check Job Status
```bash
# Poll until status is "completed"
curl https://YOUR_POD_ID-7860.proxy.runpod.net/status/JOB_ID
```

Response when completed:
```json
{
  "status": "completed",
  "url": "https://YOUR_POD_ID-7860.proxy.runpod.net/image/filename.png",
  "filename": "filename.png",
  "duration_seconds": 12.3
}
```

### List All Jobs
```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/jobs
```

### Delete Jobs
```bash
# Delete one job + its file
curl -X DELETE https://YOUR_POD_ID-7860.proxy.runpod.net/jobs/JOB_ID

# Delete all completed jobs
curl -X DELETE https://YOUR_POD_ID-7860.proxy.runpod.net/jobs

# Delete ALL jobs (including queued/processing)
curl -X DELETE "https://YOUR_POD_ID-7860.proxy.runpod.net/jobs?completed_only=false"
```

---

## Typical Workflow

```bash
POD="https://YOUR_POD_ID-7860.proxy.runpod.net"

# 1. Submit a head swap job
JOB=$(curl -s -X POST "$POD/flux/face-swap" \
  -F "target_image=@template.png" \
  -F "face_image=@face.png" | jq -r '.job_id')

echo "Job: $JOB"

# 2. Poll until done
while true; do
  STATUS=$(curl -s "$POD/status/$JOB" | jq -r '.status')
  echo "Status: $STATUS"
  [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] && break
  sleep 3
done

# 3. Download result
URL=$(curl -s "$POD/status/$JOB" | jq -r '.url')
curl -o result.png "$URL"
```

---

## Troubleshooting

### API not starting
Open Jupyter terminal (port 8888) and run:
```bash
tail -20 /workspace/api.log
```

### Manual start (if auto-start fails)
The supervisor script handles everything (deps, fetch `main.py`, wait for ComfyUI,
launch uvicorn with auto-restart on crash). Launch it fully detached:

```bash
setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &

# Verify
sleep 10 && curl http://localhost:7860/health
```

### API didn't come back after pod restart
Confirm the `/start.sh` hook is in place (`setup.sh` installs it on first run):
```bash
grep -c "AI Gen API v2 auto-start" /start.sh    # should print 2
```
If missing (pod was recreated/rebuilt, not just restarted), re-run setup:
```bash
wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh && bash /tmp/setup.sh
```

### main.py failed to download
```bash
wget -O /workspace/api/main.py \
  "https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/main.py"
```

### ComfyUI not starting
```bash
# Check GPU
nvidia-smi

# Restart ComfyUI (same command /start.sh uses)
cd /workspace/runpod-slim/ComfyUI && \
  nohup .venv-cu128/bin/python main.py --listen 0.0.0.0 --port 8188 \
    --enable-cors-header >> /workspace/comfyui.log 2>&1 & disown
```

---

## Models (~72 GB total)

| Model | Size | Purpose | HF token |
|-------|------|---------|----------|
| FLUX.2 Klein 9B UNET | 18 GB | Image gen + head swap | Yes (gated) |
| FLUX VAE | 321 MB | VAE decoder | No |
| Qwen 3 8B text encoder | 8.1 GB | FLUX text encoder | No |
| BFS Head Swap LoRA | 633 MB | Head swap LoRA | No |
| LTX-2.3 22B checkpoint | 27 GB | Video gen | Yes (gated) |
| LTX-2.3 distilled LoRA | 7.1 GB | LTX speed LoRA | No |
| Gemma abliterated LoRA | 599 MB | LTX prompt LoRA | No |
| Gemma 3 12B text encoder | 8.8 GB | LTX text encoder | No |
| LTX-2.3 spatial upscaler | 950 MB | 2× spatial upscaler | No |

## File Structure on Pod

```
/workspace/
  api/
    main.py              # FastAPI app
    config.env           # detected Python/pip/ComfyUI paths
  start_api.sh           # Supervisor: flock-guarded while-loop around uvicorn
  api.log                # API runtime log
  api_setup.log          # Setup + supervisor progress log
  runpod-slim/
    ComfyUI/             # the runpod/comfyui:latest image's ComfyUI tree
      .venv-cu128/       # Python venv (used by both ComfyUI and API)
      models/
        diffusion_models/  # FLUX UNET
        vae/               # FLUX VAE
        text_encoders/     # Qwen 3 8B, Gemma 12B
        loras/             # BFS, distilled LTX, Gemma abliterated
        checkpoints/       # LTX 2.3 22B
        latent_upscale_models/  # LTX spatial upscaler
      custom_nodes/
        LanPaint/          # FLUX head swap nodes

/start.sh                # container entrypoint; patched by setup.sh to
                         # auto-launch start_api.sh after ComfyUI boots
```
