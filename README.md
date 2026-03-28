# AI Gen API v2

Minimal API for FLUX.2 Klein 9B — head swap + text-to-image on RunPod.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/t2i` | POST | Text to image (FLUX Klein 9B) |
| `/flux/face-swap` | POST | AI head swap (FLUX Klein 9B) |
| `/status/{job_id}` | GET | Check job status |
| `/jobs` | GET | List all jobs |
| `/image/{filename}` | GET | Serve image |

## Deploy on RunPod

1. Create pod with any ComfyUI template (RTX 5090 recommended)
2. Set env var: `SETUP_SCRIPT_URL=https://raw.githubusercontent.com/cyrusjaysondev/ai-gen-api-v2/main/setup.sh`
3. Volume: 150GB at `/workspace`
4. Expose ports: 7860, 8188, 8888

## Models (~28GB)

- `flux2-klein-9b.safetensors` (18GB) — FLUX UNET
- `qwen_3_8b_fp8mixed.safetensors` (8.7GB) — text encoder
- `bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors` (663MB) — head swap LoRA
- `flux2-vae.safetensors` (336MB) — FLUX VAE

## Manual start

```bash
pip install fastapi uvicorn httpx websockets python-multipart
cd /workspace/api && uvicorn main:app --host 0.0.0.0 --port 7860
```
# ai-gen-api-v2
