# AI Gen API v2

Minimal API for FLUX.2 Klein 9B on RunPod — text-to-image + AI head swap.

---

## Quick Deploy (new pod)

### 1. Create Pod
- Go to **runpod.io** > **Pods** > **+ Deploy**
- GPU: **RTX 5090** (32GB) or similar
- Template: Any **ComfyUI** template
- Volume: **150GB** at `/workspace`
- Ports: `7860`, `8188`, `8888`

### 2. Set Environment Variables
In the template settings, add:

```
SETUP_SCRIPT_URL = https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh
```

### 3. Deploy and Wait
Click Deploy. The setup script will automatically:
1. Install pip dependencies
2. Download FLUX Klein 9B UNET (18GB)
3. Download FLUX VAE + Qwen text encoder (9GB)
4. Download BFS head swap LoRA (663MB)
5. Install LanPaint custom node
6. Download `main.py` and start the API

**First deploy: ~10-15 min** (downloading ~28GB of models).
**Subsequent restarts: ~1 min** (models cached on volume).

### 4. Monitor Progress
Open the Jupyter terminal (port 8888) and run:
```bash
tail -f /workspace/api_setup.log
```

### 5. Verify
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
```bash
# Install deps
pip install -q fastapi uvicorn httpx websockets python-multipart

# Start API (from Jupyter terminal so it survives)
cd /workspace/api && nohup /opt/venv/bin/python -m uvicorn main:app \
  --host 0.0.0.0 --port 7860 >> /workspace/api.log 2>&1 & disown

# Verify
sleep 2 && curl http://localhost:7860/health
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

# Restart ComfyUI
cd /workspace/ComfyUI && nohup /opt/venv/bin/python main.py \
  --listen --port 8188 >> /workspace/comfyui.log 2>&1 & disown
```

---

## Models (~28GB total)

| Model | Size | Purpose |
|-------|------|---------|
| flux2-klein-9b | 18GB | FLUX UNET (image gen + head swap) |
| qwen_3_8b_fp8mixed | 8.7GB | Text encoder |
| bfs_head_v1 LoRA | 663MB | Head swap LoRA |
| flux2-vae | 336MB | VAE decoder |

## File Structure on Pod

```
/workspace/
  api/
    main.py              # FastAPI app
  start_api.sh           # Auto-start script
  api.log                # API runtime log
  api_setup.log          # Setup progress log
  ComfyUI/
    models/
      diffusion_models/  # FLUX UNET
      vae/               # FLUX VAE
      text_encoders/     # Qwen 3 8B
      loras/             # BFS head swap LoRA
    custom_nodes/
      LanPaint/          # FLUX head swap nodes
```
