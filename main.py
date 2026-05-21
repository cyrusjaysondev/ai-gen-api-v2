import re
import subprocess
import uuid, json, httpx, os
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import websockets

from workflows import (
    ASPECT_RATIOS,
    LTX_ASPECT_RATIOS,
    LTX_DEFAULT_NEGATIVE,
    LTX_PRESETS,
    build_flux_i2i_workflow,
    build_t2i_workflow,
    build_ltx_i2v_workflow,
    build_ltx_t2v_workflow,
    compute_dimensions,
    compute_ltx_dimensions,
    crop_to_aspect,
    get_flux_face_swap_workflow,
    ltx_base_nodes,
)

# Compliance face filter (loaded lazily on first face_filter=true request).
# Module exists even if insightface is uninstalled — it'll raise a clear
# RuntimeError when actually invoked, never at import time.
try:
    import safety as face_safety
except ImportError:
    face_safety = None

# Compliance logo/flag filter — CLIP-based, separate blocklist dir.
try:
    import logo_safety
except ImportError:
    logo_safety = None

# Optional output watermark — defaults to off; callers pass `watermark="AI"`
# (or any short string) to overlay it on the result.
try:
    import watermark
except ImportError:
    watermark = None

app = FastAPI(title="AI Gen API v2")

# Open CORS so browser-based admin UIs (super-cms-vn /ai-pods + /blocked-faces)
# can call /admin/blocklist directly across the multi-pod registry. We
# previously routed everything through the face-swap-proxy edge function,
# but the proxy was timing out for admin paths and routing through it adds
# a hop for what's already an admin-only operation. With CORS on the pod
# itself, the CMS can fan out to every registered pod URL in parallel.
#
# allow_origins=["*"] is acceptable because:
#   - /admin/* endpoints will require ADMIN_API_TOKEN when set (future)
#   - All other endpoints are already meant to be reachable from app browsers
#   - RunPod's pod hostnames aren't truly secret but aren't published either
# If we add auth later we can lock origins down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

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
# Helpers
# ─────────────────────────────────────────────

_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".gif"}


def _extract_video_thumbnail(video_path: Path) -> Path | None:
    """Grab the first frame of `video_path` as a JPG sibling under
    OUTPUT_DIR/images/. Returns the saved thumbnail path, or None on failure.

    Always uses the very first frame — predictable for /ltx/i2v (the input
    image) and fast (~50 ms with libx264-decoded mp4). If you need a
    cinematic mid-frame later, add an `-ss` offset.

    Saved name: `{video_stem}_thumb.jpg`. Lands under OUTPUT_DIR/images/ so
    the existing GET /image/{filename} handler picks it up without route
    changes.
    """
    if not video_path.exists():
        return None
    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    thumb = images_dir / f"{video_path.stem}_thumb.jpg"
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "3",          # 1=best, 31=worst — 3 ≈ visually lossless JPG
                "-update", "1",
                str(thumb),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and thumb.exists() and thumb.stat().st_size > 0:
            return thumb
        # ffmpeg succeeded but produced nothing — log enough to debug without
        # spamming on every "no video" job.
        print(
            f"[thumbnail] ffmpeg rc={proc.returncode} for {video_path.name}: "
            f"{(proc.stderr or '').strip()[-200:]}"
        )
    except Exception as exc:  # noqa: BLE001 — never crash a completed job
        print(f"[thumbnail] exception on {video_path.name}: {exc}")
    return None


# ─────────────────────────────────────────────
# Core job runner
# ─────────────────────────────────────────────

async def run_job(job_id: str, workflow: dict, cleanup_paths: list = None,
                  watermark_text: str | None = None,
                  watermark_image: bool = False):
    jobs[job_id] = {**jobs.get(job_id, {}), "status": "processing", "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        client_id = str(uuid.uuid4())
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id})
            if resp.status_code != 200:
                jobs[job_id] = {**jobs[job_id], "status": "failed", "error": resp.text}
                return
            prompt_id = resp.json()["prompt_id"]

        # Disable client-side pings + close timeout so very long generations
        # (LTX i2v at 1280×2272 = ~3-5 min on RTX 5090) don't get killed by
        # the websockets library's default 20 s ping timeout. ComfyUI sends
        # progress messages frequently enough that the connection stays
        # alive on its own.
        ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
        async with websockets.connect(
            ws_url, ping_interval=None, close_timeout=None, max_size=None
        ) as ws:
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
                        # Optional watermark — text and/or logo. Both run in
                        # place. Failures don't nuke the job; the
                        # unwatermarked file is still valid output.
                        wm_warnings: list[str] = []
                        if watermark_text and watermark is not None:
                            try:
                                watermark.apply(path, watermark_text)
                            except Exception as wm_err:
                                wm_warnings.append(f"text: {wm_err}")
                        if watermark_image and watermark is not None:
                            try:
                                watermark.apply_logo(path)
                            except Exception as wm_err:
                                wm_warnings.append(f"image: {wm_err}")
                        # For video outputs, snap a thumbnail (first frame).
                        # Runs AFTER watermarks so the thumbnail reflects the
                        # final, stamped video. Failure is non-fatal — the
                        # video itself is still returned.
                        thumbnail_url = None
                        if ext in _VIDEO_EXTS:
                            thumb = _extract_video_thumbnail(path)
                            if thumb is not None:
                                thumbnail_url = f"{BASE_URL}/image/{thumb.name}"
                        completed_at = datetime.now(timezone.utc)
                        created_at_str = jobs[job_id].get("created_at")
                        duration_seconds = None
                        if created_at_str:
                            started = datetime.fromisoformat(created_at_str)
                            duration_seconds = round((completed_at - started).total_seconds(), 1)
                        # IMPORTANT: build the completed dict last so we can
                        # fold the watermark warning into it. Replacing
                        # jobs[job_id] without this carry would silently
                        # swallow the warning.
                        completed = {
                            "status": "completed",
                            "url": url,
                            "filename": filename,
                            "completed_at": completed_at.isoformat(),
                            "duration_seconds": duration_seconds,
                        }
                        if thumbnail_url:
                            completed["thumbnail_url"] = thumbnail_url
                        if wm_warnings:
                            completed["watermark_warning"] = " | ".join(wm_warnings)
                        jobs[job_id] = completed
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

def _delete_output_files(filename: str) -> int:
    """Remove the primary output file plus its `_thumb.jpg` sibling, if any.
    Returns the number of files actually deleted (0–2)."""
    deleted = 0
    for path in [OUTPUT_DIR / "video" / filename, OUTPUT_DIR / "images" / filename, OUTPUT_DIR / filename]:
        if path.exists():
            path.unlink()
            deleted += 1
    thumb = OUTPUT_DIR / "images" / f"{Path(filename).stem}_thumb.jpg"
    if thumb.exists():
        thumb.unlink()
        deleted += 1
    return deleted


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    filename = job.get("filename")
    result = {"job_id": job_id, "deleted": True}
    if filename and _delete_output_files(filename):
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
            deleted_files += _delete_output_files(filename)
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
            # Pick a sensible content-type from the extension so .jpg
            # thumbnails don't get served as image/png and broken in some
            # clients (Safari is strict about this).
            ext = path.suffix.lower()
            mt = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(ext, "image/png")
            return FileResponse(str(path), media_type=mt, filename=filename)
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
            # Drop the thumbnail sibling too (best-effort).
            thumb = OUTPUT_DIR / "images" / f"{Path(filename).stem}_thumb.jpg"
            thumb.unlink(missing_ok=True)
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
    images_dir = OUTPUT_DIR / "images"
    for f in sorted(video_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        entry = {
            "filename": f.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "url": f"{BASE_URL}/video/{f.name}",
            "created_at": stat.st_mtime,
        }
        thumb = images_dir / f"{f.stem}_thumb.jpg"
        if thumb.exists():
            entry["thumbnail_url"] = f"{BASE_URL}/image/{thumb.name}"
        videos.append(entry)
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
    watermark: str | None = None  # e.g. "AI" — overlay at bottom-right; null/empty = off
    watermark_image: bool = False  # composite the GenReel logo at bottom-right

@app.post("/t2i")
async def text_to_image(req: T2IRequest, background_tasks: BackgroundTasks):
    seed = req.seed if req.seed != -1 else uuid.uuid4().int % 2**32
    workflow = build_t2i_workflow(
        prompt=req.prompt, width=req.width, height=req.height, seed=seed,
        steps=req.steps, cfg=req.cfg, guidance=req.guidance,
    )
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, None, req.watermark, req.watermark_image)
    return {"job_id": job_id, "status": "queued", "model": "flux2-klein-9b", "poll_url": f"{BASE_URL}/status/{job_id}"}


# ─────────────────────────────────────────────
# Compliance helper — checks N input images against the blocklist.
# Raises HTTPException(400) on the first blocked image. Returns silently
# if the filter is disabled or no images match.
#
# `face_filter=False` is recorded to /workspace/face_filter_bypass.log for
# audit purposes — anyone calling these endpoints with face_filter=false
# leaves a trail.
# ─────────────────────────────────────────────

def _apply_face_filter(endpoint: str, job_id: str, face_filter: bool,
                       images_with_names: list) -> None:
    """images_with_names: list of (bytes, label) pairs. label is used in the error."""
    if not face_filter:
        if face_safety is not None:
            face_safety.log_bypass(job_id, endpoint, note=f"face_filter=false, {len(images_with_names)} images")
        return
    if face_safety is None:
        raise HTTPException(503, "face filter requested but `safety` module unavailable (insightface not installed)")
    for idx, (img_bytes, label) in enumerate(images_with_names):
        try:
            result = face_safety.check_image(img_bytes)
        except RuntimeError as e:
            raise HTTPException(503, f"face filter unavailable: {e}")
        if result.blocked:
            raise HTTPException(400, {
                "error": "blocked",
                "filter": "face",
                "reason": f"{label} matches blocked face identity",
                "matched_identity": result.matched_identity,
                "score": round(result.score, 4),
                "image_index": idx,
            })


def _apply_logo_filter(endpoint: str, job_id: str, logo_filter: bool,
                       images_with_names: list) -> None:
    """Parallel to _apply_face_filter but for the logo/flag blocklist (CLIP-based)."""
    if not logo_filter:
        # Reuse the face-filter bypass log so admins have one audit trail
        if face_safety is not None:
            face_safety.log_bypass(job_id, endpoint, note=f"logo_filter=false, {len(images_with_names)} images")
        return
    if logo_safety is None:
        raise HTTPException(503, "logo filter requested but `logo_safety` module unavailable (open_clip_torch not installed)")
    for idx, (img_bytes, label) in enumerate(images_with_names):
        try:
            result = logo_safety.check_image(img_bytes)
        except RuntimeError as e:
            raise HTTPException(503, f"logo filter unavailable: {e}")
        if result.blocked:
            raise HTTPException(400, {
                "error": "blocked",
                "filter": "logo",
                "reason": f"{label} matches blocked logo/flag",
                "matched_logo": result.matched_logo,
                "score": round(result.score, 4),
                "image_index": idx,
            })


# ─────────────────────────────────────────────
# FLUX.2 Klein 9B Head/Face Swap — builders live in workflows.py
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# LTX-2.3 — presets & shared helpers live in workflows.py
# ─────────────────────────────────────────────


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
    preset: str = Form("fast", description="Speed/quality preset: fast (8 steps single-pass, ~10-15s @544×960) or quality (8+3 steps two-pass with 2× spatial upscale, ~40-60s)"),
    aspect_ratio: str = Form("9:16", description="Output aspect ratio: original | 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21. When set, height is derived from width — see below."),
    width: int = Form(544, description="Output width in pixels. When aspect_ratio is set, height is COMPUTED from this and `height` is ignored. For 9:16 use 544 (→544×960, fast) or 720 (→720×1280, quality)."),
    height: int = Form(960, description="Output height in pixels. IGNORED when aspect_ratio is set (only used with aspect_ratio=original)."),
    length: int = Form(121, description="Number of frames — 97 (~4s), 121 (~5s), 161 (~6.7s)"),
    fps: int = Form(24, description="Frames per second"),
    seed: int = Form(-1),
    audio: bool = Form(False, description="Generate audio track with the video (adds overhead)"),
    enhance_prompt: bool = Form(True, description="Rewrite prompt via Gemma 12B using the input image as context (adds 2-5s + VRAM). Recommended ON for short prompts (e.g. 'make her run'); OFF when you've already written a detailed scene description."),
    inplace_strength: float = Form(0.7, ge=0.3, le=1.0, description="How tightly each frame is pinned to the input image. 0.7 = reference distilled value (good identity, weak motion). Lower it for action prompts: 0.5 ≈ moderate motion, 0.4 ≈ strong motion (some identity drift), 0.3 ≈ near-t2v. Two-pass refine tracks this (= min(1.0, x+0.3))."),
    watermark: str | None = Form(None, description="Optional text to overlay at the bottom-right of the output (e.g. 'AI'). Null/empty = no watermark. Video re-encodes via ffmpeg (~1-3s for a 5s clip)."),
    watermark_image: bool = Form(False, description="Composite the GenReel logo (loaded once from /workspace/assets/genreel_logo.png) at the bottom-right. Stacks with `watermark` if both are set."),
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

    workflow = build_ltx_i2v_workflow(
        image_filename=img_filename, prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, length=length, fps=fps, seed=seed,
        preset=preset, audio=audio, enhance_prompt=enhance_prompt,
        inplace_strength=inplace_strength,
    )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, [img_path], watermark, watermark_image)
    return {"job_id": job_id, "status": "queued", "model": "ltx-2.3-22b", "poll_url": f"{BASE_URL}/status/{job_id}"}


# ─────────────────────────────────────────────
# LTX-2.3 Text to Video
# ─────────────────────────────────────────────

@app.post("/ltx/t2v")
async def ltx_text_to_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(..., description="What should appear/happen in the video"),
    negative_prompt: str = Form(LTX_DEFAULT_NEGATIVE),
    preset: str = Form("fast", description="Speed/quality preset: fast (8 steps single-pass, ~10-15s @544×960) or quality (8+3 steps two-pass with 2× spatial upscale, ~40-60s)"),
    aspect_ratio: str = Form("16:9", description="Output aspect ratio: 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 3:2 | 2:3 | 21:9 | 9:21"),
    width: int = Form(1280, description="Output width in pixels (height auto-computed from aspect_ratio)"),
    height: int = Form(720, description="Output height in pixels (ignored if aspect_ratio set, default used for 'original')"),
    length: int = Form(121, description="Number of frames — 97 (~4s), 121 (~5s), 161 (~6.7s)"),
    fps: int = Form(24, description="Frames per second"),
    seed: int = Form(-1),
    audio: bool = Form(False, description="Generate audio track with the video (adds overhead)"),
    watermark: str | None = Form(None, description="Optional text to overlay at the bottom-right of the output (e.g. 'AI'). Null/empty = no watermark. Video re-encodes via ffmpeg (~1-3s for a 5s clip)."),
    watermark_image: bool = Form(False, description="Composite the GenReel logo (loaded once from /workspace/assets/genreel_logo.png) at the bottom-right. Stacks with `watermark` if both are set."),
):
    if preset not in LTX_PRESETS:
        raise HTTPException(400, f"Invalid preset '{preset}'. Valid: {', '.join(LTX_PRESETS)}")
    if aspect_ratio not in LTX_ASPECT_RATIOS and aspect_ratio != "original":
        raise HTTPException(400, f"Invalid aspect_ratio. Valid: {', '.join(LTX_ASPECT_RATIOS)}")

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    width, height = compute_ltx_dimensions(width, height, aspect_ratio)

    workflow = build_ltx_t2v_workflow(
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, length=length, fps=fps, seed=seed,
        preset=preset, audio=audio,
    )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, None, watermark, watermark_image)
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

    # Same long-job-tolerant settings as run_job — see the comment there.
    ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
    async with websockets.connect(
        ws_url, ping_interval=None, close_timeout=None, max_size=None
    ) as ws:
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
    audio: bool = False,
    watermark_text: str | None = None,
    watermark_image: bool = False,
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

        ltx_workflow = ltx_base_nodes(
            animate_prompt, negative_prompt, width, height, length, fps, seed,
            low_res_video_src=["249", 0], high_res_video_src=high_res_src, prefix="face_animate", preset=preset, audio=audio
        )
        ltx_workflow.update(img_nodes)

        video_filename, video_full_path = await _submit_and_wait_comfyui(ltx_workflow)

        ext = Path(video_filename).suffix.lower()
        url = f"{BASE_URL}/video/{video_filename}" if ext not in [".png", ".jpg", ".jpeg", ".webp"] else f"{BASE_URL}/image/{video_filename}"

        # Optional watermarks — applied to the final video, not the
        # intermediate swap. Text + logo can both be set; they stack.
        watermark_warnings: list[str] = []
        if watermark_text and watermark is not None:
            try:
                watermark.apply(video_full_path, watermark_text)
            except Exception as wm_err:
                watermark_warnings.append(f"text: {wm_err}")
        if watermark_image and watermark is not None:
            try:
                watermark.apply_logo(video_full_path)
            except Exception as wm_err:
                watermark_warnings.append(f"image: {wm_err}")
        watermark_warning = " | ".join(watermark_warnings) if watermark_warnings else None

        # Thumbnail of the final (post-watermark) video. Skip for the rare
        # case where the workflow produced an image instead of a video.
        thumbnail_url = None
        if ext in _VIDEO_EXTS:
            thumb = _extract_video_thumbnail(Path(video_full_path))
            if thumb is not None:
                thumbnail_url = f"{BASE_URL}/image/{thumb.name}"

        completed_at = datetime.now(timezone.utc)
        created_at_str = jobs[job_id].get("created_at")
        duration_seconds = round((completed_at - datetime.fromisoformat(created_at_str)).total_seconds(), 1) if created_at_str else None

        result = {
            "status": "completed",
            "url": url,
            "filename": video_filename,
            "swap_filename": swap_filename,
            "completed_at": completed_at.isoformat(),
            "duration_seconds": duration_seconds,
        }
        if thumbnail_url:
            result["thumbnail_url"] = thumbnail_url
        if watermark_warning:
            result["watermark_warning"] = watermark_warning
        jobs[job_id] = result

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
    preset: str = Form("fast", description="Speed/quality preset for video: fast (8 steps single-pass) or quality (8+3 steps two-pass with 2× spatial upscale)"),
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
    audio: bool = Form(False, description="Generate audio track with the video (adds overhead)"),
    watermark: str | None = Form(None, description="Optional text to overlay at the bottom-right of the final video (e.g. 'AI'). Null/empty = no watermark. Re-encodes via ffmpeg (~1-3s for a 5s clip)."),
    watermark_image: bool = Form(False, description="Composite the GenReel logo (loaded once from /workspace/assets/genreel_logo.png) at the bottom-right of the final video. Stacks with `watermark` if both are set."),
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
        [target_path, face_path], preset, audio, watermark, watermark_image,
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
    face_filter: bool = Form(False, description="Reject the request if either input image matches a face in /workspace/blocklist/. Off by default."),
    logo_filter: bool = Form(False, description="Reject the request if either input image matches a logo/flag in /workspace/blocklist_logos/. Off by default."),
    watermark: str | None = Form(None, description="Optional text to overlay at the bottom-right of the output (e.g. 'AI'). Null/empty = no watermark."),
    watermark_image: bool = Form(False, description="Composite the GenReel logo (loaded once from /workspace/assets/genreel_logo.png) at the bottom-right. Stacks with `watermark` if both are set."),
):
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    # Validate aspect_ratio
    if aspect_ratio != "original" and aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(400, f"Invalid aspect_ratio '{aspect_ratio}'. Valid values: original, {', '.join(ASPECT_RATIOS)}")

    target_bytes = await target_image.read()
    face_bytes = await face_image.read()

    # Compliance filters — must run before any heavy work, before writing to disk
    job_id = str(uuid.uuid4())
    inputs = [(target_bytes, "target_image"), (face_bytes, "face_image")]
    _apply_face_filter("flux/face-swap", job_id, face_filter, inputs)
    _apply_logo_filter("flux/face-swap", job_id, logo_filter, inputs)

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

    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, [target_path, face_path], watermark, watermark_image)
    return {"job_id": job_id, "status": "queued", "model": "flux2-klein-9b", "poll_url": f"{BASE_URL}/status/{job_id}"}


# ─────────────────────────────────────────────
# FLUX.2 Klein 9B Image-to-Image (multi-reference editing)
# Up to 5 reference images — each one feeds a ReferenceLatent chained
# onto the prompt's conditioning. Output dimensions default to the first
# image's (rescaled) size, or override via width/height.
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# /flux/i2i composition modes (additive — see API.md "Composition modes")
#
# Each mode is a pre-baked prompt template + a recommended lora_strength
# tuned for that use case. When `composition_mode` is left at the default
# `"none"`, none of this fires — the existing /flux/i2i behavior is
# preserved bit-for-bit (caller's prompt is required, caller's
# lora_strength wins).
#
# When set, the mode supplies a template prompt + LoRA default the caller
# would have had to write themselves. The caller's `prompt` and
# `lora_strength`, if explicitly provided, always win — modes only fill
# in blanks.
# ─────────────────────────────────────────────

_I2I_MODE_PROMPTS: dict[str, str] = {
    "auto": (
        "high quality detailed composition of the reference images, "
        "photorealistic, sharp, natural lighting"
    ),
    "scene_blend": (
        "the subject(s) from the reference images placed naturally in the "
        "scene shown in the first image, matched lighting, integrated "
        "shadows, photorealistic, sharp focus, detailed environment"
    ),
    "outfit_swap": (
        "the person from the first image wearing the outfit shown in the "
        "second image, full body, photorealistic, natural lighting, "
        "detailed fabric texture"
    ),
    "style_transfer": (
        "the first image reimagined in the artistic style of the second "
        "image, preserving composition and subject"
    ),
}

# Default lora_strength per mode. Only applied when the caller didn't pass
# an explicit value (sentinel: lora_strength = -1).
_I2I_MODE_LORA: dict[str, float] = {
    "auto": 0.0,
    "scene_blend": 0.5,
    "outfit_swap": 0.7,
    "style_transfer": 0.0,
}

_I2I_QUALITY_PRESET_STEPS: dict[str, int] = {
    "fast": 4,
    "balanced": 8,
    "high": 12,
}


def _resolve_i2i_config(
    *,
    composition_mode: str,
    prompt: str,
    lora_strength: float,
    steps: int,
    quality_preset: str,
    scene_image_index: int,
    n_images: int,
) -> tuple[str, float, int, list[int]]:
    """Translate the public knobs into the final (prompt, lora, steps,
    image_order) tuple the workflow builder consumes.

    `composition_mode = "none"` (the default) means: no mode logic — return
    the caller's values verbatim, leave image order untouched. This is
    what lets us add the feature without changing existing callers.

    For any other mode:
    * If `prompt` is empty, substitute the mode's template prompt.
    * If `lora_strength < 0` (sentinel), substitute the mode's default.
    * If `quality_preset` is one of fast/balanced/high, use its step
      count, overriding the `steps` argument.
    * For `scene_blend` with 2+ images, reorder so the scene image (last
      by default, or the explicit `scene_image_index`) becomes the FLUX
      canvas (index 0).
    """
    # Auto mode if caller asked for no template but also didn't send a prompt.
    effective_mode = composition_mode
    if effective_mode == "none" and not prompt.strip():
        effective_mode = "auto"

    # Prompt: caller wins, then template, then auto fallback.
    if prompt.strip():
        final_prompt = prompt
    else:
        final_prompt = _I2I_MODE_PROMPTS.get(effective_mode) or _I2I_MODE_PROMPTS["auto"]

    # LoRA: explicit (>=0) wins, else mode default, else 0.
    if lora_strength >= 0:
        final_lora = lora_strength
    else:
        final_lora = _I2I_MODE_LORA.get(effective_mode, 0.0)

    # Steps: preset wins when one is selected, else caller's `steps`.
    if quality_preset in _I2I_QUALITY_PRESET_STEPS:
        final_steps = _I2I_QUALITY_PRESET_STEPS[quality_preset]
    else:
        final_steps = steps

    # Image order: only scene_blend reshuffles, and only when we have
    # something to reshuffle. The scene image becomes the canvas (index 0).
    indices = list(range(n_images))
    if effective_mode == "scene_blend" and n_images >= 2:
        # `-1` (the default) means "last image", which matches how the
        # frontend uploads user photos first and the library scene last.
        scene_idx = scene_image_index if scene_image_index >= 0 else n_images - 1
        scene_idx = max(0, min(scene_idx, n_images - 1))
        if scene_idx != 0:
            indices = [scene_idx] + [i for i in indices if i != scene_idx]

    return final_prompt, final_lora, final_steps, indices


@app.post("/flux/i2i")
async def flux_image_to_image(
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(..., description="1 to 5 reference images. The first one's dimensions (after rescale) are used as the output canvas unless width/height are set."),
    prompt: str = Form("", description="What to do — edit instructions. Optional when `composition_mode` is set (server fills in a mode-specific template)."),
    seed: int = Form(-1),
    megapixels: float = Form(2.0, description="Resolution per reference image in megapixels (0.5–4.0)"),
    width: int = Form(0, description="Output width — 0 (default) means: derive from the first image"),
    height: int = Form(0, description="Output height — 0 (default) means: derive from the first image"),
    steps: int = Form(4, description="Inference steps. 4 is fine for FLUX Klein. Overridden by `quality_preset` when set."),
    cfg: float = Form(1.0),
    guidance: float = Form(4.0),
    lora_strength: float = Form(-1.0, description="Apply the head-swap LoRA. 0 = off (general edits). 0.5–1.0 for face/head-focused edits. -1 (default) = use the mode's recommended value (0 for `none`/`auto`, 0.5 for `scene_blend`, 0.7 for `outfit_swap`).", ge=-1.0, le=1.5),
    composition_mode: str = Form("none", description="Pre-baked prompt + LoRA preset for prompt-less callers. `none` (default) = no template, behaves like before. `auto` | `scene_blend` | `outfit_swap` | `style_transfer` = use that mode's template. See API.md → Composition modes."),
    quality_preset: str = Form("none", description="`none` (default) = use `steps` directly. `fast` = 4 steps, `balanced` = 8 steps, `high` = 12 steps. Overrides `steps` when set."),
    scene_image_index: int = Form(-1, description="For `composition_mode=scene_blend` only: which input image is the scene/canvas. -1 (default) = last image, which matches the typical 'user uploads first, library scene last' UI flow. Ignored for other modes.", ge=-1, le=4),
    face_filter: bool = Form(False, description="Reject if any input image matches a face in /workspace/blocklist/. Off by default."),
    logo_filter: bool = Form(False, description="Reject if any input image matches a logo/flag in /workspace/blocklist_logos/. Off by default."),
    watermark: str | None = Form(None, description="Optional text to overlay at the bottom-right of the output (e.g. 'AI'). Null/empty = no watermark."),
    watermark_image: bool = Form(False, description="Composite the GenReel logo (loaded once from /workspace/assets/genreel_logo.png) at the bottom-right. Stacks with `watermark` if both are set."),
):
    if not 1 <= len(images) <= 5:
        raise HTTPException(400, f"images must be 1–5 files, got {len(images)}")

    # Validate mode-ish inputs early so a typo doesn't silently behave as
    # `none`/`steps` (which would mask the bug for the caller).
    valid_modes = {"none", *_I2I_MODE_PROMPTS}
    if composition_mode not in valid_modes:
        raise HTTPException(
            400,
            f"composition_mode must be one of {sorted(valid_modes)}, got {composition_mode!r}",
        )
    valid_presets = {"none", *_I2I_QUALITY_PRESET_STEPS}
    if quality_preset not in valid_presets:
        raise HTTPException(
            400,
            f"quality_preset must be one of {sorted(valid_presets)}, got {quality_preset!r}",
        )

    seed = seed if seed != -1 else uuid.uuid4().int % 2**32

    final_prompt, final_lora, final_steps, image_order = _resolve_i2i_config(
        composition_mode=composition_mode,
        prompt=prompt,
        lora_strength=lora_strength,
        steps=steps,
        quality_preset=quality_preset,
        scene_image_index=scene_image_index,
        n_images=len(images),
    )

    image_bytes_list: list[bytes] = []
    for up in images:
        image_bytes_list.append(await up.read())

    # Apply the mode-driven image reorder (scene_blend only; identity
    # otherwise). The blocklist filters and the workflow both see the
    # post-reorder list so logging / canvas selection stay consistent.
    image_bytes_list = [image_bytes_list[i] for i in image_order]

    job_id = str(uuid.uuid4())
    labeled = [(b, f"images[{i}]") for i, b in enumerate(image_bytes_list)]
    _apply_face_filter("flux/i2i", job_id, face_filter, labeled)
    _apply_logo_filter("flux/i2i", job_id, logo_filter, labeled)

    # Save uploads to ComfyUI's input dir (only after filter passes)
    input_filenames: list[str] = []
    cleanup_paths: list[str] = []
    for idx, img_bytes in enumerate(image_bytes_list):
        fn = f"flux_i2i_{uuid.uuid4().hex}_{idx}.png"
        p = str(INPUT_DIR / fn)
        Path(p).write_bytes(img_bytes)
        input_filenames.append(fn)
        cleanup_paths.append(p)

    workflow = build_flux_i2i_workflow(
        input_filenames, final_prompt, seed,
        megapixels=megapixels,
        output_width=width, output_height=height,
        steps=final_steps, cfg=cfg, guidance=guidance,
        lora_strength=final_lora,
    )

    jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    background_tasks.add_task(run_job, job_id, workflow, cleanup_paths, watermark, watermark_image)
    return {
        "job_id": job_id,
        "status": "queued",
        "model": "flux2-klein-9b",
        "ref_count": len(images),
        "composition_mode": composition_mode,
        "resolved": {
            # Surfaced so callers can confirm what the server decided when
            # they passed prompt-less / mode-only requests.
            "prompt_used": final_prompt[:120] + ("…" if len(final_prompt) > 120 else ""),
            "lora_strength": final_lora,
            "steps": final_steps,
        },
        "poll_url": f"{BASE_URL}/status/{job_id}",
    }


# ─────────────────────────────────────────────
# Admin API — manage the face-filter blocklist
#
# The blocklist lives on the network volume at /workspace/blocklist/ — one
# image per blocked identity, filename (minus extension) is the identity
# name returned in block responses. Hot-reloaded by safety.py on every
# face-filter check, so changes take effect immediately for the pod AND
# for any serverless workers mounted on the same volume.
#
# Auth: every admin endpoint requires `Authorization: Bearer <ADMIN_TOKEN>`.
# ADMIN_TOKEN is read from the env at request time, so rotating it doesn't
# require a restart. If ADMIN_TOKEN is unset, all admin endpoints return
# 503 — this is a feature (no accidental open admin).
# ─────────────────────────────────────────────

BLOCKLIST_DIR = Path(os.environ.get("BLOCKLIST_DIR", "/workspace/blocklist"))
ALLOWED_BLOCKLIST_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
IDENTITY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _require_admin(authorization):
    """Admin auth is optional. Behavior depends on the ADMIN_TOKEN env var:
      - ADMIN_TOKEN unset     → admin endpoints are OPEN (no auth required).
                                Convenient for dev / when the pod URL isn't shared.
      - ADMIN_TOKEN set       → caller MUST send `Authorization: Bearer <token>`.
                                401 without header, 403 with wrong token.
    Set the env var on the RunPod template when going to production."""
    token = os.environ.get("ADMIN_TOKEN")
    if not token:
        return  # open mode — no token configured
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing Authorization: Bearer <ADMIN_TOKEN> header")
    if authorization[len("Bearer "):] != token:
        raise HTTPException(403, "invalid admin token")


def _validate_identity(identity: str) -> None:
    if not IDENTITY_NAME_PATTERN.match(identity):
        raise HTTPException(400, "identity must match [A-Za-z0-9_-]{1,64} — no spaces, no path separators")


def _find_existing_blocklist_file(identity: str):
    for ext in ALLOWED_BLOCKLIST_EXTS:
        p = BLOCKLIST_DIR / f"{identity}{ext}"
        if p.exists():
            return p
    return None


def _blocklist_count() -> int:
    if not BLOCKLIST_DIR.is_dir():
        return 0
    return sum(1 for p in BLOCKLIST_DIR.iterdir()
               if p.is_file() and p.suffix.lower() in ALLOWED_BLOCKLIST_EXTS)


@app.get("/admin/blocklist")
async def admin_list_blocklist(authorization: str = Header(default=None)):
    """List all identities currently on the blocklist."""
    _require_admin(authorization)
    BLOCKLIST_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in sorted(BLOCKLIST_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_BLOCKLIST_EXTS:
            entries.append({
                "identity": p.stem,
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "added_at": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
    return {"count": len(entries), "blocklist": entries}


@app.post("/admin/blocklist")
async def admin_upload_blocklist(
    image: UploadFile = File(..., description="Face image of the identity to block. Must contain exactly one clearly-visible face."),
    identity: str = Form(..., description="Stable identifier (used to delete later, and returned in block responses). [A-Za-z0-9_-]{1,64}"),
    overwrite: bool = Form(False, description="If true, replace an existing entry with the same identity"),
    authorization: str = Header(default=None),
):
    """Add a face to the blocklist. Validates the image contains exactly one
    detectable face before accepting — if face detection fails the upload
    is rejected so admins know the entry would have been silently skipped."""
    _require_admin(authorization)
    _validate_identity(identity)
    BLOCKLIST_DIR.mkdir(parents=True, exist_ok=True)

    existing = _find_existing_blocklist_file(identity)
    if existing and not overwrite:
        raise HTTPException(409, f"identity '{identity}' already on blocklist as {existing.name}. Pass overwrite=true to replace.")

    raw_bytes = await image.read()

    if face_safety is None:
        raise HTTPException(503, "face filter module unavailable — cannot validate the uploaded image")

    # Normalize first: EXIF-rotate + downscale to BLOCKLIST_MAX_EDGE + re-encode
    # as PNG. Lets the caller upload phone photos / 4K crops / odd formats
    # without hitting body-size or storage issues, and gives detection a
    # consistent input.
    try:
        norm_bytes, ext = face_safety.normalize_blocklist_image(raw_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, f"image normalizer unavailable: {e}")

    # Validate the face is detectable BEFORE writing to disk. Use the
    # detect-only helper: check_image short-circuits to face_count=0 when
    # the blocklist is empty, which would block the very first upload.
    try:
        face_count = face_safety.detect_face_count(norm_bytes)
    except RuntimeError as e:
        raise HTTPException(503, f"face filter unavailable: {e}")
    if face_count == 0:
        raise HTTPException(400, "no face detected in the uploaded image — pick a clearer crop")
    if face_count > 1:
        raise HTTPException(400, f"detected {face_count} faces — please upload an image with exactly one clearly-visible face")

    if existing:
        existing.unlink()

    target = BLOCKLIST_DIR / f"{identity}{ext}"
    target.write_bytes(norm_bytes)

    return {
        "status": "replaced" if existing else "added",
        "identity": identity,
        "filename": target.name,
        "size_bytes": target.stat().st_size,
        "blocklist_count": _blocklist_count(),
    }


@app.delete("/admin/blocklist/{identity}")
async def admin_delete_blocklist(
    identity: str,
    authorization: str = Header(default=None),
):
    """Remove a face from the blocklist."""
    _require_admin(authorization)
    _validate_identity(identity)
    existing = _find_existing_blocklist_file(identity)
    if not existing:
        raise HTTPException(404, f"identity '{identity}' is not on the blocklist")
    existing.unlink()
    return {
        "status": "deleted",
        "identity": identity,
        "filename": existing.name,
        "blocklist_count": _blocklist_count(),
    }


@app.get("/admin/blocklist/{identity}/image")
async def admin_get_blocklist_image(
    identity: str,
    authorization: str = Header(default=None),
):
    """Download the stored face image for a blocked identity (for CMS preview)."""
    _require_admin(authorization)
    _validate_identity(identity)
    existing = _find_existing_blocklist_file(identity)
    if not existing:
        raise HTTPException(404, f"identity '{identity}' is not on the blocklist")
    return FileResponse(str(existing), filename=existing.name)


# ─────────────────────────────────────────────
# Admin API — manage the LOGO/FLAG blocklist (CLIP-based)
#
# Parallel to /admin/blocklist (faces). Stored at /workspace/blocklist_logos/.
# Hot-reloaded on every logo-filter check.
# ─────────────────────────────────────────────

BLOCKLIST_LOGOS_DIR = Path(os.environ.get("BLOCKLIST_LOGOS_DIR", "/workspace/blocklist_logos"))


def _find_existing_logo_file(identity: str):
    for ext in ALLOWED_BLOCKLIST_EXTS:
        p = BLOCKLIST_LOGOS_DIR / f"{identity}{ext}"
        if p.exists():
            return p
    return None


def _logo_blocklist_count() -> int:
    if not BLOCKLIST_LOGOS_DIR.is_dir():
        return 0
    return sum(1 for p in BLOCKLIST_LOGOS_DIR.iterdir()
               if p.is_file() and p.suffix.lower() in ALLOWED_BLOCKLIST_EXTS)


@app.get("/admin/blocklist-logos")
async def admin_list_logos(authorization: str = Header(default=None)):
    """List all logos/flags on the blocklist."""
    _require_admin(authorization)
    BLOCKLIST_LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in sorted(BLOCKLIST_LOGOS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_BLOCKLIST_EXTS:
            entries.append({
                "identity": p.stem,
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "added_at": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
    return {"count": len(entries), "blocklist": entries}


@app.post("/admin/blocklist-logos")
async def admin_upload_logo(
    image: UploadFile = File(..., description="Logo / flag / symbol image. Should be cropped tight on the subject for best CLIP discrimination."),
    identity: str = Form(..., description="Stable identifier — e.g. 'apple_logo' or 'flag_xx'. Returned in block responses."),
    overwrite: bool = Form(False, description="If true, replace an existing entry with the same identity"),
    authorization: str = Header(default=None),
):
    """Add a logo/flag to the blocklist. Unlike faces, no face-detection
    prerequisite — but the file must be a valid image."""
    _require_admin(authorization)
    _validate_identity(identity)
    BLOCKLIST_LOGOS_DIR.mkdir(parents=True, exist_ok=True)

    existing = _find_existing_logo_file(identity)
    if existing and not overwrite:
        raise HTTPException(409, f"logo '{identity}' already on blocklist as {existing.name}. Pass overwrite=true to replace.")

    img_bytes = await image.read()

    if logo_safety is None:
        raise HTTPException(503, "logo filter module unavailable — cannot validate the uploaded image")
    err = logo_safety.validate_uploadable(img_bytes)
    if err:
        raise HTTPException(400, err)

    ext_from_filename = Path(image.filename or "").suffix.lower()
    ext = ext_from_filename if ext_from_filename in ALLOWED_BLOCKLIST_EXTS else ".png"

    if existing:
        existing.unlink()

    target = BLOCKLIST_LOGOS_DIR / f"{identity}{ext}"
    target.write_bytes(img_bytes)

    return {
        "status": "replaced" if existing else "added",
        "identity": identity,
        "filename": target.name,
        "size_bytes": target.stat().st_size,
        "blocklist_count": _logo_blocklist_count(),
    }


@app.delete("/admin/blocklist-logos/{identity}")
async def admin_delete_logo(
    identity: str,
    authorization: str = Header(default=None),
):
    """Remove a logo/flag from the blocklist."""
    _require_admin(authorization)
    _validate_identity(identity)
    existing = _find_existing_logo_file(identity)
    if not existing:
        raise HTTPException(404, f"logo '{identity}' is not on the blocklist")
    existing.unlink()
    return {
        "status": "deleted",
        "identity": identity,
        "filename": existing.name,
        "blocklist_count": _logo_blocklist_count(),
    }


@app.get("/admin/blocklist-logos/{identity}/image")
async def admin_get_logo_image(
    identity: str,
    authorization: str = Header(default=None),
):
    """Download the stored logo image for CMS preview."""
    _require_admin(authorization)
    _validate_identity(identity)
    existing = _find_existing_logo_file(identity)
    if not existing:
        raise HTTPException(404, f"logo '{identity}' is not on the blocklist")
    return FileResponse(str(existing), filename=existing.name)
