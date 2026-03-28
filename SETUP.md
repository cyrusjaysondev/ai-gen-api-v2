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
4. Under **Start Command**, paste:
   ```bash
   bash -c "wget -qO /tmp/setup.sh https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh && bash /tmp/setup.sh &"
   ```
5. Save the template.

---

## Step 2 — Deploy a Pod

1. Go to **Pods** → **+ Deploy**
2. Select GPU: **RTX 5090** (32 GB VRAM) — recommended
3. Select your **AI Gen API v2** template
4. Click **Deploy**

---

## Step 3 — Monitor Setup Progress

The setup script runs automatically in the background and downloads ~55 GB of models. Track progress via the Jupyter terminal (port 8888):

```bash
tail -f /workspace/api_setup.log
```

You should see output like:
```
[HH:MM:SS] AI Gen API v2 Setup Started
[HH:MM:SS] [1/8] Installing pip dependencies...
[HH:MM:SS] [2/8] FLUX.2 Klein 9B UNET...
[HH:MM:SS]   Downloading FLUX Klein 9B UNET (18GB)...
[HH:MM:SS] [3/8] FLUX VAE + Qwen text encoder...
[HH:MM:SS] [4/8] BFS Head Swap LoRA...
[HH:MM:SS] [5/8] Installing LanPaint custom node...
[HH:MM:SS] [6/8] LTX-2.3 checkpoint...
[HH:MM:SS]   Downloading LTX-2.3 22B dev fp8 (27GB)...
[HH:MM:SS] [7/8] LTX-2.3 LoRAs + text encoder + upscaler...
[HH:MM:SS] [8/8] Setting up API...
[HH:MM:SS] Setup Complete!
```

**First deploy takes ~30–45 minutes** (downloading ~55 GB of models).
**Subsequent restarts take ~1–2 minutes** (models already on volume, skip download).

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

### Text to Image

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/t2i \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains, photorealistic, 4K", "width": 1024, "height": 1024}'
```

### Face Swap

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg"
```

### Image to Video (LTX 2.3)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=camera slowly zooms in" \
  -F "aspect_ratio=16:9"
```

### Text to Video (LTX 2.3)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/t2v \
  -F "prompt=a sunset over the ocean, cinematic, slow motion" \
  -F "aspect_ratio=16:9" \
  -F "length_seconds=5"
```

### Face Swap + Animate (Pipeline)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/face-animate \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg" \
  -F "animate_prompt=person smiles and looks at the camera" \
  -F "aspect_ratio=16:9" \
  -F "length_seconds=5"
```

All endpoints return a `job_id`. Poll for results:

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/status/JOB_ID
```

---

## Logs

| Log file | Contents |
|----------|----------|
| `/workspace/api_setup.log` | Setup progress, model downloads, API start |
| `/workspace/api.log` | Live API request/response logs |

```bash
# Watch API logs live
tail -f /workspace/api.log
```

---

## Troubleshooting

**Setup stuck / no progress**
```bash
tail -50 /workspace/api_setup.log
```

**API not responding on port 7860**
```bash
tail -50 /workspace/api.log
ps aux | grep uvicorn
```

**Job stuck in `processing`**
First job after pod start is slow — models load into VRAM (~3–5 min). Subsequent jobs are fast.

**`HF_TOKEN` error in setup log**
Set `HF_TOKEN` in your RunPod template environment variables and redeploy.
Make sure you have accepted both model licenses (see Prerequisites above).

**LTX checkpoint download failed**
Ensure you've accepted the license at https://huggingface.co/Lightricks/LTX-2.3-fp8

---

## Models Installed

| Model | Size | Source | Token required |
|-------|------|--------|----------------|
| FLUX.2 Klein 9B UNET | 18 GB | black-forest-labs/FLUX.2-klein-9B | ✅ Yes (gated) |
| FLUX VAE | 321 MB | Comfy-Org/flux2-klein-9B | No |
| Qwen 3 8B text encoder | 8.1 GB | Comfy-Org/flux2-klein-9B | No |
| BFS Head Swap LoRA | 633 MB | Alissonerdx/BFS-Best-Face-Swap | No |
| LTX-2.3 22B checkpoint | 27 GB | Lightricks/LTX-2.3-fp8 | ✅ Yes (gated) |
| LTX-2.3 distilled LoRA | 7.1 GB | Lightricks/LTX-2.3 | No |
| Gemma abliterated LoRA | 599 MB | Comfy-Org/ltx-2 | No |
| Gemma 3 12B text encoder | 8.8 GB | Comfy-Org/ltx-2 | No |
| LTX-2.3 spatial upscaler | 950 MB | Lightricks/LTX-2.3 | No |
| **Total** | **~72 GB** | | |

> Recommended volume size: **200 GB** to have room for generated outputs.
