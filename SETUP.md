# AI Gen API v2 — RunPod Setup Guide

FLUX.2 Klein 9B · Text-to-Image + Head/Face Swap

---

## Prerequisites

Before starting, you need:

1. **HuggingFace account** — https://huggingface.co
2. **HuggingFace token** — https://huggingface.co/settings/tokens (create a token with **Read** access)
3. **Accept the model license** — Visit https://huggingface.co/black-forest-labs/FLUX.2-klein-9B and click **Agree and access repository**

---

## Step 1 — Create a RunPod Template

1. Go to **runpod.io** → **Templates** → **+ New Template**
2. Set the following:
   - **Template Name:** `AI Gen API v2`
   - **Container Image:** `runpod/comfyui:latest`
   - **Volume:** `150 GB` mounted at `/workspace`
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

The setup script runs automatically in the background and downloads ~28 GB of models. Track progress via the Jupyter terminal (port 8888):

```bash
tail -f /workspace/api_setup.log
```

You should see output like:
```
[HH:MM:SS] AI Gen API v2 Setup Started
[HH:MM:SS] [1/6] Installing pip dependencies...
[HH:MM:SS] [2/6] FLUX.2 Klein 9B UNET...
[HH:MM:SS]   Downloading FLUX Klein 9B UNET (18GB)...
[HH:MM:SS] [3/6] FLUX VAE + Qwen text encoder...
[HH:MM:SS] [4/6] BFS Head Swap LoRA...
[HH:MM:SS] [5/6] Installing LanPaint custom node...
[HH:MM:SS] [6/6] Setting up API...
[HH:MM:SS] Setup Complete!
```

**First deploy takes ~15–20 minutes** (downloading models).
**Subsequent restarts take ~1 minute** (models already on volume, skip download).

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

## Step 5 — Generate Your First Image

### Text to Image

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/t2i \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains, photorealistic, 4K",
    "width": 1024,
    "height": 1024,
    "steps": 4
  }'
```

Response:
```json
{
  "job_id": "abc123",
  "status": "queued",
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/abc123"
}
```

Poll for the result:
```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/status/abc123
```

Completed response:
```json
{
  "status": "completed",
  "url": "https://YOUR_POD_ID-7860.proxy.runpod.net/image/t2i_12345_00001_.png",
  "filename": "t2i_12345_00001_.png",
  "duration_seconds": 12.4
}
```

---

## Step 6 — Face / Head Swap

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.png" \
  -F "face_image=@face_photo.png"
```

Response:
```json
{
  "job_id": "xyz789",
  "status": "queued",
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/xyz789"
}
```

Poll until `status: completed`, then open the `url` to download the result.

---

## API Parameters

### POST /t2i

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | required | What to generate |
| `width` | 1024 | Image width in pixels |
| `height` | 1024 | Image height in pixels |
| `seed` | -1 (random) | Set for reproducible results |
| `steps` | 4 | Inference steps (4 is ideal for FLUX Klein) |
| `cfg` | 1.0 | CFG scale |
| `guidance` | 4.0 | FLUX guidance strength (2.0–6.0) |

### POST /flux/face-swap

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_image` | required | Body/template photo — head gets replaced |
| `face_image` | required | Face photo — identity to transfer |
| `seed` | -1 (random) | Set for reproducible results |
| `megapixels` | 2.0 | Output resolution (1.0–2.0) |
| `steps` | 4 | Inference steps |
| `cfg` | 1.0 | CFG scale |
| `guidance` | 4.0 | FLUX guidance strength |
| `lora_strength` | 1.0 | BFS LoRA strength (0.5–1.0) |

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
# Check if uvicorn is running
ps aux | grep uvicorn
```

**Job stuck in `processing` for a long time**
First job after pod start is slow — models load into VRAM (~2–3 min). Subsequent jobs are fast.

**`HF_TOKEN` error in setup log**
Set `HF_TOKEN` in your RunPod template environment variables and redeploy.
Make sure you have accepted the license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B

---

## Models Used

| Model | Size | Source |
|-------|------|--------|
| FLUX.2 Klein 9B UNET | 17 GB | black-forest-labs/FLUX.2-klein-9B *(gated)* |
| FLUX VAE | 321 MB | Comfy-Org/flux2-klein-9B |
| Qwen 3 8B text encoder | 8.1 GB | Comfy-Org/flux2-klein-9B |
| BFS Head Swap LoRA | 633 MB | Alissonerdx/BFS-Best-Face-Swap |
