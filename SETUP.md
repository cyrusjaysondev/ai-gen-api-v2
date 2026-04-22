# AI Gen API v2 — RunPod Setup Guide

FLUX.2 Klein 9B · Face Swap + Text-to-Image
LTX 2.3 22B · Image-to-Video + Text-to-Video + Face-Animate Pipeline

---

## Prerequisites

Before starting, you need:

1. **HuggingFace account** — https://huggingface.co
2. **HuggingFace token** — https://huggingface.co/settings/tokens (create a token with **Read** access)
3. **Accept model licenses:**
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B — click **Agree and access repository**
   - https://huggingface.co/Lightricks/LTX-2.3-fp8 — click **Agree and access repository**

---

## Step 1 — Create a RunPod Template

1. Go to **runpod.io** → **Templates** → **+ New Template**
2. Set the following:
   - **Template Name:** `AI Gen API v2`
   - **Container Image:** `runpod/comfyui:latest`
   - **Volume:** `200 GB` mounted at `/workspace`
   - **Expose Ports:** `7860, 8188, 8888`
3. Under **Environment Variables**, add:
   ```
   HF_TOKEN = hf_your_token_here
   ```
4. Under **Container Start Command**, paste:
   ```bash
   bash -c "/start.sh & sleep 3 && wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh && bash /tmp/setup.sh; wait"
   ```

   **Why this form:** the `runpod/comfyui:latest` image ships `/start.sh` as its entrypoint
   (starts SSH, JupyterLab, FileBrowser, and ComfyUI). Setting a **Container Start Command**
   in the template replaces that entrypoint, so you must re-invoke `/start.sh` yourself —
   otherwise only `setup.sh` runs and SSH/ComfyUI never come up. This command runs
   `/start.sh` in the background, waits briefly, then fetches and runs `setup.sh`. The
   trailing `wait` keeps the container alive after `setup.sh` returns so `/start.sh`
   (which ends in `sleep infinity`) continues supervising ComfyUI.
5. Save the template.

---

## Step 2 — Deploy a Pod

1. Go to **Pods** → **+ Deploy**
2. Select GPU: **RTX 5090** (32 GB VRAM) — recommended
3. Select your **AI Gen API v2** template
4. Click **Deploy**

---

## Step 3 — Monitor Setup Progress

The setup runs automatically as part of the Container Start Command. Open the Jupyter
terminal (port 8888) — or SSH in — and tail the log:

```bash
tail -f /workspace/api_setup.log
```

You should see output like:
```
[HH:MM:SS] ==========================================
[HH:MM:SS] AI Gen API v2 Setup Started
[HH:MM:SS] Pod ID: xxxxxxxxxxxxxxxx
[HH:MM:SS] ComfyUI: /workspace/runpod-slim/ComfyUI
[HH:MM:SS] Python: /workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python
[HH:MM:SS] ==========================================
[HH:MM:SS] [1/4] Installing pip dependencies + aria2...
[HH:MM:SS]   Done
[HH:MM:SS] [2/4] Downloading models (parallel, ~72 GB total)...
[HH:MM:SS]   Download complete: /workspace/runpod-slim/ComfyUI/models/...
[HH:MM:SS]   All 9 models downloaded
[HH:MM:SS] [3/4] Installing LanPaint custom node...
[HH:MM:SS] [4/4] Setting up API...
[HH:MM:SS]   main.py downloaded (latest)
[HH:MM:SS]   /start.sh patched with API restart hook
[HH:MM:SS] API supervisor launched
[HH:MM:SS] Setup Complete!
```

**First deploy: ~3–10 minutes** on a warm HuggingFace CDN. The 9 model files are
pulled in parallel via aria2 (16 connections/file), which typically saturates the
pod's network link.
**Subsequent deploys / pod restarts: ~10–20 seconds.** Models are cached on the
`/workspace` volume; aria2 just verifies sizes and skips. `setup.sh` is idempotent
and skips every step whose work is already done.

---

## Step 4 — Verify the API is Running

Once setup completes, check the health endpoint:

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/health
```

Expected response:
```json
{"status": "ok", "pod_id": "YOUR_POD_ID"}
```

Or open the interactive docs in your browser:
```
https://YOUR_POD_ID-7860.proxy.runpod.net/docs
```

---

## Step 5 — Test the Endpoints

### Text to Video (fast, no audio)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/t2v \
  -F "prompt=a sunset over the ocean, cinematic, slow motion" \
  -F "preset=fast"
```

### Image to Video (fast)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=camera slowly zooms in" \
  -F "preset=fast"
```

### Image to Video (quality, with audio)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=birds chirping, gentle breeze" \
  -F "preset=quality" \
  -F "audio=true"
```

### Text to Image

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/t2i \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains, photorealistic, 4K"}'
```

### Face Swap

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg"
```

### Face Swap + Animate (Pipeline)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/face-animate \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg" \
  -F "animate_prompt=person smiles and looks at the camera" \
  -F "preset=fast"
```

All endpoints return a `job_id`. Poll for results:

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/status/JOB_ID
```

---

## Speed & Quality Presets

All LTX video endpoints (`/ltx/i2v`, `/ltx/t2v`, `/face-animate`) support `preset` and `audio` parameters.

| Preset | Steps | Pipeline | Speed (5s 720p, warm) | Use case |
|--------|-------|----------|----------------------|----------|
| `fast` (default) | 5 | Single pass, no upscale, no audio | ~36s | Previews, rapid iteration |
| `quality` | 20+5 | Two-pass + spatial upscale | ~100s+ | Final renders, maximum detail |

### Speed tips

- **Lower resolution = faster.** 768x448 generates in ~20s vs ~36s at 1280x720
- **Shorter clips = faster.** `length=49` (2s) is much faster than `length=121` (5s)
- **audio=false (default) saves ~5-10s** by skipping the audio VAE entirely
- **First request after pod start is slow** (~3-5 min) because models load into VRAM. All subsequent requests use cached models
- **Warm model benchmarks (fast preset):**

| Resolution | 2s video | 5s video |
|------------|----------|----------|
| 768x448 | ~10s | ~20s |
| 1024x576 | ~15s | ~28s |
| 1280x704 | ~22s | ~36s |

### Audio control

| Parameter | Effect |
|-----------|--------|
| `audio=false` (default) | Video only — faster generation |
| `audio=true` | Generates audio track with the video |

---

## Pod Restart Behavior

**You don't have to do anything on pod restart** — the API comes back automatically.

`setup.sh` patches the image's `/start.sh` with an idempotent hook that launches
`/workspace/start_api.sh` (detached via `setsid nohup`) right after ComfyUI boots.
So the boot chain on every pod restart is:

1. RunPod runs the template Start Command → invokes `/start.sh`
2. `/start.sh` starts SSH, Jupyter, FileBrowser, and ComfyUI
3. Patched hook in `/start.sh` launches `start_api.sh` in a detached session
4. `start_api.sh` fetches the latest `main.py`, waits for ComfyUI, then starts uvicorn on :7860
5. `start_api.sh` auto-restarts uvicorn if it crashes; `flock` prevents duplicate supervisors

The patch lives on the container layer (not `/workspace`), so it survives pod
**restart** but is lost on pod **recreate/rebuild**. That's fine: the template Start
Command always re-runs `setup.sh`, which re-applies the patch. Self-healing.

### Deploying API updates

1. Push changes to `main.py` on GitHub
2. Restart the pod — `start_api.sh` fetches the latest `main.py` automatically
3. No SSH, no manual redeploy step

### Manually relaunching (only if you skipped the restart)

```bash
setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &
```

Plain `bash /workspace/start_api.sh &` also works from an interactive shell, but will
die if the shell exits. `setsid nohup … </dev/null &` fully detaches.

---

## Scaling to Multiple Pods

1. Create your template once (Step 1 above)
2. Deploy as many pods as you need — each runs the same setup
3. Push API updates to the repo — all pods pick up changes on restart
4. Models are cached on each pod's volume — only first deploy downloads ~72 GB

### Recommended scaling setup

| Component | Setting |
|-----------|---------|
| Template | `AI Gen API v2` with start command |
| GPU | RTX 5090 (32 GB) or A100 (80 GB) |
| Volume | 200 GB (persistent across restarts) |
| Ports | 7860 (API), 8188 (ComfyUI), 8888 (Jupyter) |
| Start command | `bash -c "/start.sh & sleep 3 && wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh && bash /tmp/setup.sh; wait"` |

---

## Logs

| Log file | Contents |
|----------|----------|
| `/workspace/api_setup.log` | Setup progress, model downloads, API start |
| `/workspace/api.log` | Live API request/response logs |

```bash
# Watch setup progress
tail -f /workspace/api_setup.log

# Watch API logs live
tail -f /workspace/api.log
```

Logs are automatically truncated to the last 500 lines on each restart to prevent disk bloat.

---

## Troubleshooting

**Setup stuck / no progress**
```bash
tail -50 /workspace/api_setup.log
```

**API not responding on port 7860**
```bash
# Check if the supervisor + uvicorn are alive
pgrep -xaf "bash /workspace/start_api.sh"
netstat -tlnp 2>/dev/null | grep :7860

# Check API logs
tail -50 /workspace/api.log

# Manually relaunch the supervisor (fully detached — survives shell exit)
setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &
```

**Job stuck in `processing`**
First job after pod start is slow — models load into VRAM (~3–5 min). Subsequent jobs are fast.

**`HF_TOKEN` error in setup log**
Set `HF_TOKEN` in your RunPod template environment variables and redeploy.
Make sure you have accepted both model licenses (see Prerequisites above).

**LTX checkpoint download failed**
Ensure you've accepted the license at https://huggingface.co/Lightricks/LTX-2.3-fp8

**API crashed and not restarting**
The `start_api.sh` has an auto-restart loop. Check if the supervisor is running:
```bash
ps aux | grep -v grep | grep start_api
# If not running:
setsid nohup bash /workspace/start_api.sh </dev/null >>/workspace/api_setup.log 2>&1 &
```

**API didn't come back after pod restart**
Confirm the `/start.sh` hook is in place (it should be, after `setup.sh` ran once):
```bash
grep -A2 "AI Gen API v2 auto-start" /start.sh
```
If the hook is missing, re-run setup to re-apply it:
```bash
wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh && bash /tmp/setup.sh
```

---

## Models Installed

| Model | Size | Source | Token required |
|-------|------|--------|----------------|
| FLUX.2 Klein 9B UNET | 18 GB | black-forest-labs/FLUX.2-klein-9B | Yes (gated) |
| FLUX VAE | 321 MB | Comfy-Org/flux2-klein-9B | No |
| Qwen 3 8B text encoder | 8.1 GB | Comfy-Org/flux2-klein-9B | No |
| BFS Head Swap LoRA | 633 MB | Alissonerdx/BFS-Best-Face-Swap | No |
| LTX-2.3 22B checkpoint | 27 GB | Lightricks/LTX-2.3-fp8 | Yes (gated) |
| LTX-2.3 distilled LoRA | 7.1 GB | Lightricks/LTX-2.3 | No |
| Gemma abliterated LoRA | 599 MB | Comfy-Org/ltx-2 | No |
| Gemma 3 12B text encoder | 8.8 GB | Comfy-Org/ltx-2 | No |
| LTX-2.3 spatial upscaler | 950 MB | Lightricks/LTX-2.3 | No |
| **Total** | **~72 GB** | | |

> Recommended volume size: **200 GB** to have room for generated outputs.

---

## Architecture

```
RunPod Pod
├── ComfyUI (port 8188) — model inference engine
│   ├── models/checkpoints/     — LTX 2.3, FLUX Klein
│   ├── models/loras/           — distilled LoRA, BFS, Gemma
│   ├── models/text_encoders/   — Gemma 12B, Qwen 8B
│   └── custom_nodes/LanPaint/  — face swap node
│
├── FastAPI (port 7860) — REST API layer
│   ├── /workspace/api/main.py  — all endpoints, workflow builder
│   └── /workspace/api/config.env
│
├── /workspace/start_api.sh     — auto-start + auto-restart script
└── /workspace/api_setup.log    — setup + runtime logs
```

Requests flow: **Client → FastAPI (7860) → ComfyUI (8188) → GPU → output video/image**
