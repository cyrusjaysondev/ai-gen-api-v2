import uuid, json, httpx, os, math, io
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import websockets

app = FastAPI(title="AI Gen API v2")

COMFYUI_URL = "http://127.0.0.1:8188"
POD_ID = os.environ.get("RUNPOD_POD_ID", "RUNPOD_POD_ID_PLACEHOLDER")
BASE_URL = f"https://{POD_ID}-7860.proxy.runpod.net"

# Auto-detect ComfyUI root
COMFY_ROOT = None
for _p in ["/workspace/runpod-slim/ComfyUI", "/workspace/ComfyUI"]:
    if Path(_p).exists():
        COMFY_ROOT = Path(_p)
        break
if not COMFY_ROOT:
    COMFY_ROOT = Path("/workspace/ComfyUI")

OUTPUT_DIR = COMFY_ROOT / "output"
INPUT_DIR = COMFY_ROOT / "input"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store
jobs = {}


# ─────────────────────────────────────────────
# Core job runner
# ─────────────────────────────────────────────

async def run_job(job_id: str, workflow: dict, cleanup_paths: list = None):
    jobs[job_id] = {**jobs.get(job_id, {}), "status": "processing", "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        client_id = str(uuid.uuid4())
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id})
            if resp.status_code != 200:
                jobs[job_id] = {**jobs[job_id], "status": "failed", "error": resp.text}
                return
            prompt_id = resp.json()["prompt_id"]

        ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
        async with websockets.connect(ws_url) as ws:
            while True:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    continue  # Skip binary preview frames
                msg = json.loads(raw)
                if msg.get("type") == "executing":
                    data = msg.get("data", {})
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        break

        async with httpx.AsyncClient() as client:
            history = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
            job_data = history.json().get(prompt_id, {})
            status = job_data.get("status", {}).get("status_str", "")
            if status == "error":
                messages = job_data.get("status", {}).get("messages", [])
                for m in messages:
                    if m[0] == "execution_error":
                        jobs[job_id] = {**jobs[job_id], "status": "failed", "error": m[1].get("exception_message")}
                        return
            outputs = job_data.get("outputs", {})

        for node_output in outputs.values():
            for key in ["videos", "gifs", "images"]:
                if key in node_output:
                    item = node_output[key][0]
                    filename = item["filename"]
                    subfolder = item.get("subfolder", "")
                    path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                    if path.exists():
                        ext = Path(filename).suffix.lower()
                        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
                            url = f"{BASE_URL}/image/{filename}"
                        else:
                            url = f"{BASE_URL}/video/{filename}"
                        completed_at = datetime.now(timezone.utc)
                        created_at_str = jobs[job_id].get("created_at")
                        duration_seconds = None
                        if created_at_str:
                            started = datetime.fromisoformat(created_at_str)
                            duration_seconds = round((completed_at - started).total_seconds(), 1)
                        jobs[job_id] = {"status": "completed", "url": url, "filename": filename, "completed_at": completed_at.isoformat(), "duration_seconds": duration_seconds}
                        return

        jobs[job_id] = {**jobs[job_id], "status": "failed", "error": "No output found"}
    except Exception as e:
        jobs[job_id] = {**jobs[job_id], "status": "failed", "error": str(e), "failed_at": datetime.now(timezone.utc).isoformat()}
    finally:
        if cleanup_paths:
            for p in cleanup_paths:
                Path(p).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# Health & job management
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "pod_id": POD_ID}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/jobs")
async def get_all_jobs():
    return {
        "total": len(jobs),
        "summary": {s: sum(1 for j in jobs.values() if j.get("status") == s) for s in ["queued", "processing", "completed", "failed"]},
        "jobs": [{"job_id": jid, **info} for jid, info in jobs.items()]
    }

@app.get("/queue")
async def get_queue():
    active = {jid: info for jid, info in jobs.items() if info.get("status") in ["queued", "processing"]}
    return {"count": len(active), "jobs": [{"job_id": jid, "status": info["status"]} for jid, info in active.items()]}

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    filename = job.get("filename")
    result = {"job_id": job_id, "deleted": True}
    if filename:
        for path in [OUTPUT_DIR / "video" / filename, OUTPUT_DIR / "images" / filename, OUTPUT_DIR / filename]:
            if path.exists():
                path.unlink()
                result["file_deleted"] = filename
    del jobs[job_id]
    return result

@app.delete("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.get("status") == "completed":
        raise HTTPException(400, "Job already completed")
    if job.get("status") == "failed":
        raise HTTPException(400, "Job already failed")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{COMFYUI_URL}/queue", json={"delete": [job_id]})
    except:
        pass
    jobs[job_id] = {"status": "cancelled"}
    return {"job_id": job_id, "status": "cancelled"}

@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.get("status") not in ["failed", "cancelled"]:
        raise HTTPException(400, f"Can only retry failed/cancelled jobs. Current: {job.get('status')}")
    if "workflow" not in job:
        raise HTTPException(400, "No workflow stored — submit a new request")
    new_job_id = str(uuid.uuid4())
    jobs[new_job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, new_job_id, job["workflow"])
    return {"new_job_id": new_job_id, "original_job_id": job_id, "status": "queued", "poll_url": f"{BASE_URL}/status/{new_job_id}"}

@app.delete("/jobs")
async def delete_all_jobs(completed_only: bool = True):
    deleted_jobs = deleted_files = 0
    for job_id in list(jobs.keys()):
        job = jobs[job_id]
        if completed_only and job.get("status") != "completed":
            continue
        filename = job.get("filename")
        if filename:
            for path in [OUTPUT_DIR / "video" / filename, OUTPUT_DIR / "images" / filename, OUTPUT_DIR / filename]:
                if path.exists():
                    path.unlink()
                    deleted_files += 1
        del jobs[job_id]
        deleted_jobs += 1
    return {"deleted_jobs": deleted_jobs, "deleted_files": deleted_files}


# ─────────────────────────────────────────────
# File serving
# ─────────────────────────────────────────────

@app.get("/image/{filename}")
async def serve_image(filename: str):
    for path in [OUTPUT_DIR / "images" / filename, OUTPUT_DIR / filename]:
        if path.exists():
            return FileResponse(str(path), media_type="image/png", filename=filename)
    raise HTTPException(404, f"Image not found: {filename}")

@app.get("/video/{filename}")
async def serve_video(filename: str):
    for path in [OUTPUT_DIR / "video" / filename, OUTPUT_DIR / filename]:
        if path.exists():
            return FileResponse(str(path), media_type="video/mp4", filename=filename)
    raise HTTPException(404, f"Not found: {filename}")

@app.delete("/video/{filename}")
async def delete_video(filename: str):
    for path in [OUTPUT_DIR / "video" / filename, OUTPUT_DIR / filename]:
        if path.exists():
            path.unlink()
            for job_id, info in list(jobs.items()):
                if info.get("filename") == filename:
                    del jobs[job_id]
            return {"status": "deleted", "filename": filename}
    raise HTTPException(404, f"File not found: {filename}")

@app.get("/videos")
async def list_videos():
    video_dir = OUTPUT_DIR / "video"
    if not video_dir.exists():
        return {"total": 0, "videos": []}
    videos = []
    for f in sorted(video_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        videos.append({"filename": f.name, "size_mb": round(stat.st_size / 1024 / 1024, 2), "url": f"{BASE_URL}/video/{f.name}", "created_at": stat.st_mtime})
    return {"total": len(videos), "videos": videos}


# ─────────────────────────────────────────────
# Text to Image (FLUX.2 Klein 9B)
# ─────────────────────────────────────────────

class T2IRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int = -1
    steps: int = 4
    cfg: float = 1.0
    guidance: float = 4.0

@app.post("/t2i")
async def text_to_image(req: T2IRequest, background_tasks: BackgroundTasks):
    seed = req.seed if req.seed != -1 else uuid.uuid4().int % 2**32
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-9b.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2", "device": "default"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": req.prompt, "clip": ["3", 0]}},
        "5": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": req.guidance}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "7": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": req.width, "height": req.height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "seed": seed, "steps": req.steps, "cfg": req.cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["2", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": f"images/t2i_{seed}"}},
    }
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow)
    return {"job_id": job_id, "status": "queued", "model": "flux2-klein-9b", "poll_url": f"{BASE_URL}/status/{job_id}"}


# ─────────────────────────────────────────────
# FLUX.2 Klein 9B Head/Face Swap
# ─────────────────────────────────────────────

ASPECT_RATIOS = {
    "1:1":  (1, 1),
    "4:3":  (4, 3),
    "3:4":  (3, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "3:2":  (3, 2),
    "2:3":  (2, 3),
    "21:9": (21, 9),
    "9:21": (9, 21),
}

def compute_dimensions(w_ratio: int, h_ratio: int, megapixels: float) -> tuple[int, int]:
    """Calculate width/height from aspect ratio and megapixels, snapped to multiples of 16."""
    total = megapixels * 1_000_000
    h = math.sqrt(total / (w_ratio / h_ratio))
    w = h * (w_ratio / h_ratio)
    # Snap to nearest multiple of 16 (VAE requirement)
    w = max(16, round(w / 16) * 16)
    h = max(16, round(h / 16) * 16)
    return int(w), int(h)

def crop_to_aspect(img_bytes: bytes, width: int, height: int) -> bytes:
    """Center-crop and resize image to exact width x height."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    src_w, src_h = img.size
    target_ratio = width / height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Source is wider — crop width
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    elif src_ratio < target_ratio:
        # Source is taller — crop height
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    img = img.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


DEFAULT_FLUX_PROMPT = """head_swap: Use image 1 as the base image, preserving its environment, background, camera perspective, framing, exposure, contrast, and lighting. Remove the head and hair from image 1 and seamlessly replace it with the head from image 2.
Match the original head size, face-to-body ratio, neck thickness, shoulder alignment, and camera distance so proportions remain natural and unchanged.
Adapt the inserted head to the lighting of image 1 by matching light direction, intensity, softness, color temperature, shadows, and highlights, with no independent relighting.
Preserve the identity of image 2, including hair texture, eye color, nose structure, facial proportions, and skin details.
Match the pose and expression from image 1, including head tilt, rotation, eye direction, gaze, micro-expressions, and lip position.
Ensure seamless neck and jaw blending, consistent skin tone, realistic shadow contact, natural skin texture, and uniform sharpness.
Photorealistic, high quality, sharp details, 4K."""

def get_flux_face_swap_workflow(target_filename, face_filename, seed, prompt=None, megapixels=2.0, steps=4, cfg=1.0, guidance=4.0, lora_strength=1.0):
    if not prompt:
        prompt = DEFAULT_FLUX_PROMPT
    return {
        "126": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-9b.safetensors", "weight_dtype": "default"}},
        "102": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "146": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2", "device": "default"}},
        "161": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["126", 0], "lora_name": "bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors", "strength_model": lora_strength}},
        "107": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["146", 0]}},
        "151": {"class_type": "LoadImage", "inputs": {"image": target_filename}},
        "121": {"class_type": "LoadImage", "inputs": {"image": face_filename}},
        "115": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["151", 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 1}},
        "125": {"class_type": "VAEEncode", "inputs": {"pixels": ["115", 0], "vae": ["102", 0]}},
        "147": {"class_type": "VAEDecode", "inputs": {"samples": ["125", 0], "vae": ["102", 0]}},
        "148": {"class_type": "GetImageSize", "inputs": {"image": ["147", 0]}},
        "149": {"class_type": "ImageScale", "inputs": {"image": ["151", 0], "upscale_method": "lanczos", "width": ["148", 0], "height": ["148", 1], "crop": "center"}},
        "150": {"class_type": "VAEEncode", "inputs": {"pixels": ["149", 0], "vae": ["102", 0]}},
        "120": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["121", 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 1}},
        "119": {"class_type": "VAEEncode", "inputs": {"pixels": ["120", 0], "vae": ["102", 0]}},
        "112": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["107", 0], "latent": ["150", 0]}},
        "118": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["112", 0], "latent": ["119", 0]}},
        "136": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["107", 0]}},
        "100": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["118", 0], "guidance": guidance}},
        "163": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["148", 0], "height": ["148", 1], "batch_size": 1}},
        "156": {"class_type": "LanPaint_KSampler", "inputs": {
            "model": ["161", 0], "positive": ["100", 0], "negative": ["136", 0],
            "latent_image": ["163", 0], "seed": seed,
            "control_after_generate": "randomize", "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "LanPaint_NumSteps": 2, "LanPaint_PromptMode": "Image First",
            "Inpainting_mode": "🖼️ Image Inpainting",
            "LanPaint_Info": "LanPaint KSampler"
        }},
        "104": {"class_type": "VAEDecode", "inputs": {"samples": ["156", 0], "vae": ["102", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["104", 0], "filename_prefix": f"images/flux_swap_{seed}"}}
    }

# ─────────────────────────────────────────────
# LTX-2.3 shared helpers
# ─────────────────────────────────────────────

LTX_ASPECT_RATIOS = {
    "1:1":  (1, 1),  "4:3":  (4, 3),  "3:4":  (3, 4),
    "16:9": (16, 9), "9:16": (9, 16), "3:2":  (3, 2),
    "2:3":  (2, 3),  "21:9": (21, 9), "9:21": (9, 21),
}

def compute_ltx_dimensions(width: int, height: int, aspect_ratio: str) -> tuple[int, int]:
    """Return (width, height) snapped to multiples of 32. If aspect_ratio given, derive height from width."""
    if aspect_ratio in LTX_ASPECT_RATIOS:
        w_r, h_r = LTX_ASPECT_RATIOS[aspect_ratio]
        height = round(width * h_r / w_r / 32) * 32
    width  = max(32, round(width  / 32) * 32)
    height = max(32, round(height / 32) * 32)
    return width, height

LTX_PRESETS = {
    "fast": {
        # Single pass at full resolution — no upscale second pass
        "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
        "lora_strength": 0.5,
        "two_pass": False,
    },
    "quality": {
        # Two-pass: low-res generate → upscale → high-res refine
        "low_res_sigmas": "1.0, 0.99688, 0.99375, 0.990625, 0.9875, 0.984375, 0.98125, 0.978125, 0.975, 0.96875, 0.9625, 0.95, 0.9375, 0.909375, 0.875, 0.84375, 0.78125, 0.725, 0.5625, 0.421875, 0.0",
        "high_res_sigmas": "0.85, 0.7875, 0.7250, 0.5734, 0.4219, 0.0",
        "lora_strength": 0.35,
        "two_pass": True,
    },
}

def _ltx_base_nodes(prompt, negative_prompt, width, height, length, fps, seed, low_res_video_src, high_res_video_src, prefix, preset="fast"):
    """Return the shared LTX workflow nodes.
    fast preset: single pass at full resolution — fast, no upscale overhead.
    quality preset: two-pass (half-res → upscale → refine at full-res) — slower, sharper.
    """
    p = LTX_PRESETS.get(preset, LTX_PRESETS["fast"])

    nodes = {
        # ── Model loaders ──
        "236": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}},
        "221": {"class_type": "LTXVAudioVAELoader",     "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}},
        "243": {"class_type": "LTXAVTextEncoderLoader", "inputs": {
            "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
            "ckpt_name":    "ltx-2.3-22b-dev-fp8.safetensors",
            "device": "default"
        }},
        "272": {"class_type": "LoraLoader", "inputs": {
            "model": ["236", 0], "clip": ["243", 0],
            "lora_name": "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
            "strength_model": 1.0, "strength_clip": 1.0
        }},
        "232": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["236", 0],
            "lora_name": "ltx-2.3-22b-distilled-lora-384.safetensors",
            "strength_model": p["lora_strength"]
        }},

        # ── Text conditioning ──
        "240": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": prompt}},
        "247": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["272", 1], "text": negative_prompt}},
        "239": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["240", 0], "negative": ["247", 0], "frame_rate": float(fps)
        }},

        # ── Audio latent ──
        "214": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": length, "frame_rate": fps, "batch_size": 1, "audio_vae": ["221", 0]
        }},
    }

    if p["two_pass"]:
        # ── QUALITY: two-pass pipeline ──
        half_w = max(32, (width // 2 // 32) * 32)
        half_h = max(32, (height // 2 // 32) * 32)

        nodes.update({
            "233": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"}},

            # Low-res latent
            "228": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
                "width": half_w, "height": half_h, "length": length, "batch_size": 1
            }},

            # Low-res sampling
            "222": {"class_type": "LTXVConcatAVLatent",    "inputs": {"video_latent": low_res_video_src, "audio_latent": ["214", 0]}},
            "231": {"class_type": "CFGGuider",              "inputs": {"model": ["232", 0], "positive": ["239", 0], "negative": ["239", 1], "cfg": 1.0}},
            "209": {"class_type": "KSamplerSelect",         "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
            "237": {"class_type": "RandomNoise",            "inputs": {"noise_seed": seed}},
            "252": {"class_type": "ManualSigmas",           "inputs": {"sigmas": p["low_res_sigmas"]}},
            "215": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["237", 0], "guider": ["231", 0], "sampler": ["209", 0],
                "sigmas": ["252", 0], "latent_image": ["222", 0]
            }},
            "217": {"class_type": "LTXVSeparateAVLatent",  "inputs": {"av_latent": ["215", 0]}},

            # Upscale 2×
            "253": {"class_type": "LTXVLatentUpsampler",   "inputs": {
                "samples": ["217", 0], "upscale_model": ["233", 0], "vae": ["236", 2]
            }},

            # High-res refinement
            "212": {"class_type": "LTXVCropGuides",         "inputs": {"positive": ["239", 0], "negative": ["239", 1], "latent": ["217", 0]}},
            "229": {"class_type": "LTXVConcatAVLatent",     "inputs": {"video_latent": high_res_video_src, "audio_latent": ["217", 1]}},
            "213": {"class_type": "CFGGuider",              "inputs": {"model": ["232", 0], "positive": ["212", 0], "negative": ["212", 1], "cfg": 1.0}},
            "246": {"class_type": "KSamplerSelect",         "inputs": {"sampler_name": "euler_cfg_pp"}},
            "216": {"class_type": "RandomNoise",            "inputs": {"noise_seed": (seed + 1) % 2**32}},
            "211": {"class_type": "ManualSigmas",           "inputs": {"sigmas": p["high_res_sigmas"]}},
            "219": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["216", 0], "guider": ["213", 0], "sampler": ["246", 0],
                "sigmas": ["211", 0], "latent_image": ["229", 0]
            }},
            "218": {"class_type": "LTXVSeparateAVLatent",  "inputs": {"av_latent": ["219", 0]}},

            # Decode from high-res output
            "220": {"class_type": "LTXVAudioVAEDecode",    "inputs": {"samples": ["218", 1], "audio_vae": ["221", 0]}},
            "251": {"class_type": "VAEDecodeTiled",         "inputs": {
                "samples": ["218", 0], "vae": ["236", 2],
                "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4
            }},
        })
    else:
        # ── FAST: single pass at full resolution — no upscale ──
        nodes.update({
            # Full-res latent directly
            "228": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
                "width": width, "height": height, "length": length, "batch_size": 1
            }},

            # Single sampling pass
            "222": {"class_type": "LTXVConcatAVLatent",    "inputs": {"video_latent": low_res_video_src, "audio_latent": ["214", 0]}},
            "231": {"class_type": "CFGGuider",              "inputs": {"model": ["232", 0], "positive": ["239", 0], "negative": ["239", 1], "cfg": 1.0}},
            "209": {"class_type": "KSamplerSelect",         "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
            "237": {"class_type": "RandomNoise",            "inputs": {"noise_seed": seed}},
            "252": {"class_type": "ManualSigmas",           "inputs": {"sigmas": p["sigmas"]}},
            "215": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["237", 0], "guider": ["231", 0], "sampler": ["209", 0],
                "sigmas": ["252", 0], "latent_image": ["222", 0]
            }},
            "217": {"class_type": "LTXVSeparateAVLatent",  "inputs": {"av_latent": ["215", 0]}},

            # Decode directly from single pass output
            "220": {"class_type": "LTXVAudioVAEDecode",    "inputs": {"samples": ["217", 1], "audio_vae": ["221", 0]}},
            "251": {"class_type": "VAEDecodeTiled",         "inputs": {
                "samples": ["217", 0], "vae": ["236", 2],
                "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4
            }},
        })

    # ── Output (shared) ──
    nodes.update({
        "242": {"class_type": "CreateVideo",  "inputs": {"images": ["251", 0], "audio": ["220", 0], "fps": float(fps)}},
        "75":  {"class_type": "SaveVideo",    "inputs": {
            "video": ["242", 0], "filename_prefix": f"video/{prefix}_{seed}", "format": "auto", "codec": "auto"
        }},
    })

    return nodes

LTX_DEFAULT_NEGATIVE = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"

@app.get("/ltx/presets")
async def get_ltx_presets():
    info = {}
    for k, v in LTX_PRESETS.items():
        if v["two_pass"]:
            info[k] = {"mode": "two_pass", "low_res_steps": v["low_res_sigmas"].count(","), "high_res_steps": v["high_res_sigmas"].count(","), "lora_strength": v["lora_strength"]}
        else:
            info[k] = {"mode": "single_pass", "steps": v["sigmas"].count(","), "lora_strength": v["lora_strength"]}
    return {"presets": info, "default": "fast", "endpoints": ["/ltx/i2v", "/ltx/t2v", "/face-animate"]}


# ─────────────────────────────────────────────
# LTX-2.3 Image to Video
# ─────────────────────────────────────────────

@app.post("/ltx/i2v")
async def ltx_image_to_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Input image to animate"),
    prompt: str = Form("", description="What should happen in the video"),
    negative_prompt: str = Form(LTX_DEFAULT_NEGATIVE),
    preset: str = Form("fast", description="Speed/quality preset: fast (~8 steps, <12s) or quality (~20 steps)"),
    aspect_ratio: str = Form("original", description="Output aspect ratio: original | 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21"),
    width: int = Form(1280, description="Output width in pixels (height auto-computed if aspect_ratio set)"),
    height: int = Form(720, description="Output height in pixels (ignored if aspect_ratio set)"),
    length: int = Form(121, description="Number of frames — 97 (~4s), 121 (~5s), 161 (~6.7s)"),
    fps: int = Form(24, description="Frames per second"),
    seed: int = Form(-1),
):
    if preset not in LTX_PRESETS:
        raise HTTPException(400, f"Invalid preset '{preset}'. Valid: {', '.join(LTX_PRESETS)}")
    if aspect_ratio != "original" and aspect_ratio not in LTX_ASPECT_RATIOS:
        raise HTTPException(400, f"Invalid aspect_ratio. Valid: original, {', '.join(LTX_ASPECT_RATIOS)}")

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    width, height = compute_ltx_dimensions(width, height, aspect_ratio)

    img_bytes = await image.read()
    img_filename = f"ltx_i2v_{uuid.uuid4().hex}.png"
    img_path = str(INPUT_DIR / img_filename)
    Path(img_path).write_bytes(img_bytes)

    two_pass = LTX_PRESETS[preset]["two_pass"]

    # Image-specific nodes: load → resize → preprocess → i2v inplace
    img_nodes = {
        "269": {"class_type": "LoadImage", "inputs": {"image": img_filename}},
        "238": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0], "resize_type": "scale dimensions",
            "resize_type.width": width, "resize_type.height": height,
            "resize_type.crop": "center", "scale_method": "lanczos"
        }},
        "235": {"class_type": "ResizeImagesByLongerEdge", "inputs": {"images": ["238", 0], "longer_edge": 1536}},
        "248": {"class_type": "LTXVPreprocess",           "inputs": {"image": ["235", 0], "img_compression": 18}},
        # Prompt enhancer uses the image
        "274": {"class_type": "TextGenerateLTX2Prompt", "inputs": {
            "clip": ["272", 1], "image": ["269", 0], "prompt": prompt,
            "max_length": 256, "sampling_mode": "on",
            "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
            "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
            "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": seed
        }},
        # I2V inplace for the generation pass (feeds into latent "228")
        "249": {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["236", 2], "image": ["248", 0], "latent": ["228", 0],
            "strength": 0.7 if two_pass else 1.0, "bypass": False
        }},
    }

    # High-res i2v inplace only needed for quality two-pass
    if two_pass:
        img_nodes["230"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["236", 2], "image": ["248", 0], "latent": ["253", 0], "strength": 1.0, "bypass": False
        }}
        high_res_src = ["230", 0]
    else:
        high_res_src = None  # not used in single-pass

    # Override CLIPTextEncode to use enhanced prompt
    img_nodes["240"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": ["274", 0]}}

    workflow = _ltx_base_nodes(
        prompt, negative_prompt, width, height, length, fps, seed,
        low_res_video_src=["249", 0], high_res_video_src=high_res_src, prefix="ltx_i2v", preset=preset
    )
    workflow.update(img_nodes)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, [img_path])
    return {"job_id": job_id, "status": "queued", "model": "ltx-2.3-22b", "poll_url": f"{BASE_URL}/status/{job_id}"}


# ─────────────────────────────────────────────
# LTX-2.3 Text to Video
# ─────────────────────────────────────────────

@app.post("/ltx/t2v")
async def ltx_text_to_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(..., description="What should appear/happen in the video"),
    negative_prompt: str = Form(LTX_DEFAULT_NEGATIVE),
    preset: str = Form("fast", description="Speed/quality preset: fast (~8 steps, <12s) or quality (~20 steps)"),
    aspect_ratio: str = Form("16:9", description="Output aspect ratio: 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21"),
    width: int = Form(1280, description="Output width in pixels (height auto-computed from aspect_ratio)"),
    height: int = Form(720, description="Output height in pixels (ignored if aspect_ratio set, default used for 'original')"),
    length: int = Form(121, description="Number of frames — 97 (~4s), 121 (~5s), 161 (~6.7s)"),
    fps: int = Form(24, description="Frames per second"),
    seed: int = Form(-1),
):
    if preset not in LTX_PRESETS:
        raise HTTPException(400, f"Invalid preset '{preset}'. Valid: {', '.join(LTX_PRESETS)}")
    if aspect_ratio not in LTX_ASPECT_RATIOS and aspect_ratio != "original":
        raise HTTPException(400, f"Invalid aspect_ratio. Valid: {', '.join(LTX_ASPECT_RATIOS)}")

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    width, height = compute_ltx_dimensions(width, height, aspect_ratio)

    # For t2v: empty latent feeds directly into concat (no image conditioning)
    workflow = _ltx_base_nodes(
        prompt, negative_prompt, width, height, length, fps, seed,
        low_res_video_src=["228", 0],   # empty latent straight to low-res concat
        high_res_video_src=["253", 0],  # upscaled latent straight to high-res concat
        prefix="ltx_t2v", preset=preset
    )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow)
    return {"job_id": job_id, "status": "queued", "model": "ltx-2.3-22b", "poll_url": f"{BASE_URL}/status/{job_id}"}


# ─────────────────────────────────────────────
# Face Swap + Animate Pipeline
# ─────────────────────────────────────────────

async def _submit_and_wait_comfyui(workflow: dict) -> tuple[str, str]:
    """Submit a workflow to ComfyUI, wait for completion, return (filename, full_path)."""
    client_id = str(uuid.uuid4())
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id})
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow: {resp.text}")
        prompt_id = resp.json()["prompt_id"]

    ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
    async with websockets.connect(ws_url) as ws:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "executing":
                data = msg.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break

    async with httpx.AsyncClient() as client:
        history = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
        job_data = history.json().get(prompt_id, {})

    status = job_data.get("status", {}).get("status_str", "")
    if status == "error":
        for m in job_data.get("status", {}).get("messages", []):
            if m[0] == "execution_error":
                raise RuntimeError(m[1].get("exception_message", "ComfyUI execution error"))
        raise RuntimeError("ComfyUI execution error")

    for node_output in job_data.get("outputs", {}).values():
        for key in ["images", "videos", "gifs"]:
            if key in node_output:
                item = node_output[key][0]
                filename = item["filename"]
                subfolder = item.get("subfolder", "")
                path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                if path.exists():
                    return filename, str(path)

    raise RuntimeError("No output file found in ComfyUI history")


async def run_face_animate_pipeline(
    job_id: str,
    face_swap_workflow: dict,
    animate_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    length: int,
    fps: int,
    seed: int,
    swap_cleanup_paths: list,
    preset: str = "fast",
):
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    ltx_img_path = None

    try:
        # ── Step 1: Face swap ──
        jobs[job_id]["step"] = "face_swap"
        swap_filename, swap_img_path = await _submit_and_wait_comfyui(face_swap_workflow)

        # ── Step 2: Animate the swapped image ──
        jobs[job_id]["step"] = "animating"

        img_bytes = Path(swap_img_path).read_bytes()
        ltx_input_filename = f"face_animate_{uuid.uuid4().hex}.png"
        ltx_img_path = str(INPUT_DIR / ltx_input_filename)
        Path(ltx_img_path).write_bytes(img_bytes)

        two_pass = LTX_PRESETS[preset]["two_pass"]

        img_nodes = {
            "269": {"class_type": "LoadImage", "inputs": {"image": ltx_input_filename}},
            "238": {"class_type": "ResizeImageMaskNode", "inputs": {
                "input": ["269", 0], "resize_type": "scale dimensions",
                "resize_type.width": width, "resize_type.height": height,
                "resize_type.crop": "center", "scale_method": "lanczos"
            }},
            "235": {"class_type": "ResizeImagesByLongerEdge", "inputs": {"images": ["238", 0], "longer_edge": 1536}},
            "248": {"class_type": "LTXVPreprocess",           "inputs": {"image": ["235", 0], "img_compression": 18}},
            "274": {"class_type": "TextGenerateLTX2Prompt", "inputs": {
                "clip": ["272", 1], "image": ["269", 0], "prompt": animate_prompt,
                "max_length": 256, "sampling_mode": "on",
                "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
                "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": seed
            }},
            "249": {"class_type": "LTXVImgToVideoInplace", "inputs": {
                "vae": ["236", 2], "image": ["248", 0], "latent": ["228", 0],
                "strength": 0.7 if two_pass else 1.0, "bypass": False
            }},
            "240": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": ["274", 0]}},
        }

        if two_pass:
            img_nodes["230"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
                "vae": ["236", 2], "image": ["248", 0], "latent": ["253", 0], "strength": 1.0, "bypass": False
            }}
            high_res_src = ["230", 0]
        else:
            high_res_src = None

        ltx_workflow = _ltx_base_nodes(
            animate_prompt, negative_prompt, width, height, length, fps, seed,
            low_res_video_src=["249", 0], high_res_video_src=high_res_src, prefix="face_animate", preset=preset
        )
        ltx_workflow.update(img_nodes)

        video_filename, _ = await _submit_and_wait_comfyui(ltx_workflow)

        ext = Path(video_filename).suffix.lower()
        url = f"{BASE_URL}/video/{video_filename}" if ext not in [".png", ".jpg", ".jpeg", ".webp"] else f"{BASE_URL}/image/{video_filename}"

        completed_at = datetime.now(timezone.utc)
        created_at_str = jobs[job_id].get("created_at")
        duration_seconds = round((completed_at - datetime.fromisoformat(created_at_str)).total_seconds(), 1) if created_at_str else None

        jobs[job_id] = {
            "status": "completed",
            "url": url,
            "filename": video_filename,
            "swap_filename": swap_filename,
            "completed_at": completed_at.isoformat(),
            "duration_seconds": duration_seconds,
        }

    except Exception as e:
        jobs[job_id] = {**jobs[job_id], "status": "failed", "error": str(e), "failed_at": datetime.now(timezone.utc).isoformat()}
    finally:
        for p in (swap_cleanup_paths or []):
            Path(p).unlink(missing_ok=True)
        if ltx_img_path:
            Path(ltx_img_path).unlink(missing_ok=True)


@app.post("/face-animate")
async def face_animate(
    background_tasks: BackgroundTasks,
    target_image: UploadFile = File(..., description="Template/body photo — head gets replaced"),
    face_image: UploadFile = File(..., description="User's face photo — identity to transfer"),
    animate_prompt: str = Form(..., description="Describes the motion/scene for the video"),
    swap_prompt: str = Form("", description="Prompt for the face swap step (uses smart default if empty)"),
    negative_prompt: str = Form(LTX_DEFAULT_NEGATIVE),
    preset: str = Form("fast", description="Speed/quality preset for video: fast (~8 steps, <12s) or quality (~20 steps)"),
    aspect_ratio: str = Form("16:9", description="Output video aspect ratio: 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21 | original"),
    width: int = Form(1280, description="Output width in pixels (height auto-derived from aspect_ratio)"),
    height: int = Form(720, description="Output height — used only when aspect_ratio=original"),
    length_seconds: float = Form(5.0, description="Video duration in seconds"),
    fps: int = Form(24, description="Frames per second"),
    seed: int = Form(-1),
    megapixels: float = Form(2.0, description="Face swap resolution in megapixels (0.5–4.0)"),
    lora_strength: float = Form(1.0, description="BFS LoRA strength for face swap (0.5–1.0)"),
    swap_steps: int = Form(4),
    swap_guidance: float = Form(4.0),
):
    if preset not in LTX_PRESETS:
        raise HTTPException(400, f"Invalid preset '{preset}'. Valid: {', '.join(LTX_PRESETS)}")
    if aspect_ratio != "original" and aspect_ratio not in LTX_ASPECT_RATIOS:
        raise HTTPException(400, f"Invalid aspect_ratio. Valid: original, {', '.join(LTX_ASPECT_RATIOS)}")

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    width, height = compute_ltx_dimensions(width, height, aspect_ratio)
    length = max(25, round(length_seconds * fps))

    target_bytes = await target_image.read()
    face_bytes = await face_image.read()

    # Pre-crop target image to match output aspect ratio for face swap
    if aspect_ratio != "original":
        w_r, h_r = LTX_ASPECT_RATIOS[aspect_ratio]
        swap_w, swap_h = compute_dimensions(w_r, h_r, megapixels)
        target_bytes = crop_to_aspect(target_bytes, swap_w, swap_h)

    target_filename = f"fa_target_{uuid.uuid4().hex}.png"
    face_filename = f"fa_face_{uuid.uuid4().hex}.png"
    target_path = str(INPUT_DIR / target_filename)
    face_path = str(INPUT_DIR / face_filename)
    Path(target_path).write_bytes(target_bytes)
    Path(face_path).write_bytes(face_bytes)

    face_swap_workflow = get_flux_face_swap_workflow(
        target_filename, face_filename, seed,
        prompt=swap_prompt or None,
        megapixels=megapixels, steps=swap_steps, cfg=1.0,
        guidance=swap_guidance, lora_strength=lora_strength,
    )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(
        run_face_animate_pipeline,
        job_id, face_swap_workflow, animate_prompt, negative_prompt,
        width, height, length, fps, seed,
        [target_path, face_path], preset,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "flux2-klein-9b + ltx-2.3-22b",
        "pipeline": ["face_swap", "image_to_video"],
        "poll_url": f"{BASE_URL}/status/{job_id}",
    }


@app.post("/flux/face-swap")
async def flux_face_swap(
    background_tasks: BackgroundTasks,
    target_image: UploadFile = File(..., description="Base/template image — body stays, head gets replaced"),
    face_image: UploadFile = File(..., description="Source face — identity to transfer"),
    seed: int = Form(-1),
    megapixels: float = Form(2.0, description="Total output resolution in megapixels (0.5–4.0)"),
    aspect_ratio: str = Form("original", description="Output aspect ratio: original | 1:1 | 16:9 | 9:16 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21"),
    steps: int = Form(4),
    cfg: float = Form(1.0),
    guidance: float = Form(4.0),
    lora_strength: float = Form(1.0),
):
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    # Validate aspect_ratio
    if aspect_ratio != "original" and aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(400, f"Invalid aspect_ratio '{aspect_ratio}'. Valid values: original, {', '.join(ASPECT_RATIOS)}")

    target_bytes = await target_image.read()
    face_bytes = await face_image.read()

    # If aspect ratio is specified, crop target image to that ratio before sending to ComfyUI.
    # The workflow's ImageScaleToTotalPixels + GetImageSize will then produce output at that AR.
    if aspect_ratio != "original":
        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
        target_w, target_h = compute_dimensions(w_ratio, h_ratio, megapixels)
        target_bytes = crop_to_aspect(target_bytes, target_w, target_h)

    target_filename = f"flux_target_{uuid.uuid4().hex}.png"
    face_filename = f"flux_face_{uuid.uuid4().hex}.png"
    target_path = str(INPUT_DIR / target_filename)
    face_path = str(INPUT_DIR / face_filename)
    Path(target_path).write_bytes(target_bytes)
    Path(face_path).write_bytes(face_bytes)

    workflow = get_flux_face_swap_workflow(target_filename, face_filename, seed, megapixels=megapixels, steps=steps, cfg=cfg, guidance=guidance, lora_strength=lora_strength)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, [target_path, face_path])
    return {"job_id": job_id, "status": "queued", "model": "flux2-klein-9b", "poll_url": f"{BASE_URL}/status/{job_id}"}
