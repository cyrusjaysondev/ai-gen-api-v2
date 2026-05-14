"""
RunPod serverless handler — Image worker (FLUX.2 Klein 9B).

Endpoints supported via `event["input"]["endpoint"]`:
  - "t2i"            text-to-image
  - "flux/face-swap" face / head swap

Models read from the network volume at /runpod-volume/runpod-slim/ComfyUI/models
(linked into ComfyUI's models dir by start.sh before the handler boots).
Generated images are staged to /runpod-volume/outputs/<job_id>/<filename> so
the reaper can clean them up after RETENTION_DAYS and so clients can fetch
them via the same RunPod S3 API used for videos.

Input (t2i):
  {
    "input": {
      "endpoint": "t2i",
      "prompt": "...",
      "width": 1024, "height": 1024,
      "seed": -1, "steps": 4, "cfg": 1.0, "guidance": 4.0
    }
  }

Input (face-swap — base64 with or without `data:image/...;base64,` prefix):
  {
    "input": {
      "endpoint": "flux/face-swap",
      "target_image_b64": "iVBOR...",
      "face_image_b64":   "iVBOR...",
      "aspect_ratio": "original",
      "megapixels": 2.0,
      "seed": -1, "steps": 4, "cfg": 1.0, "guidance": 4.0,
      "lora_strength": 1.0
    }
  }

Output (success):
  {
    "image_path": "/runpod-volume/outputs/<job_id>/t2i_42_00001_.png",
    "filename":   "t2i_42_00001_.png",
    "size_bytes": 423104,
    "seed": 42,
    "duration_seconds": 12.3
  }
"""

import asyncio
import base64
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import httpx
import runpod

# Make repo-root /app/workflows.py importable
sys.path.insert(0, "/app")
from workflows import (
    ASPECT_RATIOS,
    build_t2i_workflow,
    compute_dimensions,
    crop_to_aspect,
    get_flux_face_swap_workflow,
)


COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
OUTPUT_DIR = COMFY_ROOT / "output"
INPUT_DIR = COMFY_ROOT / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Staging dir on the network volume — clients fetch results from here via
# the RunPod S3 API or any pod with the same volume mounted. The reaper
# (cleanup.sh) deletes contents older than RETENTION_DAYS.
VOLUME_OUTPUTS = Path(os.environ.get("VOLUME_OUTPUTS", "/runpod-volume/outputs"))
VOLUME_OUTPUTS.mkdir(parents=True, exist_ok=True)


def _stage_output_to_volume(filename: str, src: Path, job_id: str) -> Path:
    dest_dir = VOLUME_OUTPUTS / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.copy2(src, dest)
    return dest


# ─────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────

def _decode_image_b64(b64: str, field_name: str) -> bytes:
    """Decode a base64 image, accepting both raw and data-URI (`data:image/png;base64,…`) forms.
    Raises ValueError with the field name if the input is malformed."""
    if not isinstance(b64, str):
        raise ValueError(f"'{field_name}' must be a base64 string, got {type(b64).__name__}")
    # Strip data URI prefix if present — common in browser clients
    # (canvas.toDataURL, FileReader.readAsDataURL).
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64, validate=False)
    except Exception as e:
        raise ValueError(f"'{field_name}' is not valid base64: {e}") from e


# ─────────────────────────────────────────────
# ComfyUI readiness — workers cold-start with ComfyUI booting in parallel
# (start.sh spawns it). Block on first invocation until it's serving.
# ─────────────────────────────────────────────

def wait_for_comfyui(timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{COMFYUI_URL}/system_stats", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"ComfyUI not ready after {timeout_s}s at {COMFYUI_URL}")


# ─────────────────────────────────────────────
# Submit a workflow to ComfyUI and block until /history reports completion.
#
# We poll /history instead of using the /ws stream because the WS pattern has
# a race: the prompt may finish executing before we manage to connect, and
# then `ws.recv()` hangs forever waiting for a message that already fired.
# Polling adds ~0.5–1s overhead, which is invisible next to model load +
# inference time.
# ─────────────────────────────────────────────

async def submit_and_wait(workflow: dict, max_wait_s: float = 600.0) -> tuple[str, Path]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow: {resp.text}")
        prompt_id = resp.json()["prompt_id"]

    deadline = time.time() + max_wait_s
    job_data: dict = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            if time.time() > deadline:
                raise RuntimeError(
                    f"ComfyUI did not finish prompt {prompt_id} within {max_wait_s:.0f}s"
                )
            history = (await client.get(f"{COMFYUI_URL}/history/{prompt_id}")).json()
            job_data = history.get(prompt_id, {})
            if job_data.get("status", {}).get("completed"):
                break
            await asyncio.sleep(1.0)

    status_str = job_data.get("status", {}).get("status_str", "")
    if status_str == "error":
        for m in job_data.get("status", {}).get("messages", []):
            if m[0] == "execution_error":
                raise RuntimeError(m[1].get("exception_message", "ComfyUI execution error"))
        raise RuntimeError("ComfyUI execution error (no detail in history)")

    for node_output in job_data.get("outputs", {}).values():
        for key in ("images", "videos", "gifs"):
            if key in node_output:
                item = node_output[key][0]
                filename = item["filename"]
                subfolder = item.get("subfolder", "")
                path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                if path.exists():
                    return filename, path

    raise RuntimeError("ComfyUI completed but no output file was found in history")


# ─────────────────────────────────────────────
# Endpoint dispatchers
# ─────────────────────────────────────────────

async def run_t2i(inp: dict, job_id: str) -> dict:
    prompt = inp.get("prompt")
    if not prompt:
        raise ValueError("'prompt' is required for t2i")
    seed = inp.get("seed", -1)
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    workflow = build_t2i_workflow(
        prompt=prompt,
        width=int(inp.get("width", 1024)),
        height=int(inp.get("height", 1024)),
        seed=seed,
        steps=int(inp.get("steps", 4)),
        cfg=float(inp.get("cfg", 1.0)),
        guidance=float(inp.get("guidance", 4.0)),
    )
    started = time.time()
    filename, src = await submit_and_wait(workflow)
    dest = _stage_output_to_volume(filename, src, job_id)
    return {
        "image_path": str(dest),
        "filename": filename,
        "size_bytes": dest.stat().st_size,
        "seed": seed,
        "duration_seconds": round(time.time() - started, 2),
    }


async def run_flux_face_swap(inp: dict, job_id: str) -> dict:
    if not inp.get("target_image_b64") or not inp.get("face_image_b64"):
        raise ValueError("'target_image_b64' and 'face_image_b64' are required for flux/face-swap")

    seed = inp.get("seed", -1)
    seed = seed if seed != -1 else uuid.uuid4().int % 2**32
    aspect_ratio = inp.get("aspect_ratio", "original")
    megapixels = float(inp.get("megapixels", 2.0))

    if aspect_ratio != "original" and aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"invalid aspect_ratio '{aspect_ratio}'; valid: original, {', '.join(ASPECT_RATIOS)}")

    target_bytes = _decode_image_b64(inp["target_image_b64"], "target_image_b64")
    face_bytes   = _decode_image_b64(inp["face_image_b64"],   "face_image_b64")

    if aspect_ratio != "original":
        w_r, h_r = ASPECT_RATIOS[aspect_ratio]
        target_w, target_h = compute_dimensions(w_r, h_r, megapixels)
        target_bytes = crop_to_aspect(target_bytes, target_w, target_h)

    target_filename = f"flux_target_{uuid.uuid4().hex}.png"
    face_filename = f"flux_face_{uuid.uuid4().hex}.png"
    (INPUT_DIR / target_filename).write_bytes(target_bytes)
    (INPUT_DIR / face_filename).write_bytes(face_bytes)

    workflow = get_flux_face_swap_workflow(
        target_filename, face_filename, seed,
        prompt=inp.get("prompt") or None,
        megapixels=megapixels,
        steps=int(inp.get("steps", 4)),
        cfg=float(inp.get("cfg", 1.0)),
        guidance=float(inp.get("guidance", 4.0)),
        lora_strength=float(inp.get("lora_strength", 1.0)),
    )

    started = time.time()
    try:
        filename, src = await submit_and_wait(workflow)
        dest = _stage_output_to_volume(filename, src, job_id)
        return {
            "image_path": str(dest),
            "filename": filename,
            "size_bytes": dest.stat().st_size,
            "seed": seed,
            "duration_seconds": round(time.time() - started, 2),
        }
    finally:
        (INPUT_DIR / target_filename).unlink(missing_ok=True)
        (INPUT_DIR / face_filename).unlink(missing_ok=True)


ENDPOINTS = {
    "t2i": run_t2i,
    "flux/face-swap": run_flux_face_swap,
}


# ─────────────────────────────────────────────
# RunPod entrypoint
# ─────────────────────────────────────────────

_comfyui_ready = False


async def handler(event):
    global _comfyui_ready
    if not _comfyui_ready:
        wait_for_comfyui()
        _comfyui_ready = True

    inp = event.get("input") or {}
    endpoint = inp.get("endpoint")
    if endpoint not in ENDPOINTS:
        return {"error": f"unknown endpoint '{endpoint}'; valid: {list(ENDPOINTS)}"}

    job_id = event.get("id") or str(uuid.uuid4())
    try:
        return await ENDPOINTS[endpoint](inp, job_id)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
