# AI Gen API v2 — API Reference

Base URL: `https://YOUR_POD_ID-7860.proxy.runpod.net`

Interactive docs (Swagger UI): `https://YOUR_POD_ID-7860.proxy.runpod.net/docs`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/t2i` | Text to image (FLUX.2 Klein 9B) |
| POST | `/flux/face-swap` | Head / face swap (FLUX.2 Klein 9B) |
| POST | `/ltx/i2v` | Image to video (LTX 2.3) |
| POST | `/ltx/t2v` | Text to video (LTX 2.3) |
| POST | `/face-animate` | Face swap + animate pipeline |
| GET | `/ltx/presets` | List available speed/quality presets |
| GET | `/status/{job_id}` | Poll job status |
| GET | `/jobs` | List all jobs |
| GET | `/queue` | Active queue |
| POST | `/jobs/{job_id}/retry` | Retry failed job |
| DELETE | `/jobs/{job_id}` | Delete job + file |
| DELETE | `/jobs` | Bulk delete jobs |
| GET | `/image/{filename}` | Download image |
| GET | `/videos` | List all videos |

---

## Health Check

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/health
```

**Response**
```json
{
  "status": "ok",
  "pod_id": "771ykso2hagd1l"
}
```

---

## POST /t2i — Text to Image

Generate an image from a text prompt using FLUX.2 Klein 9B.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | What to generate |
| `width` | int | `1024` | Output width in pixels |
| `height` | int | `1024` | Output height in pixels |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `steps` | int | `4` | Inference steps (4 is ideal for FLUX Klein) |
| `cfg` | float | `1.0` | CFG scale |
| `guidance` | float | `4.0` | FLUX guidance strength (2.0 – 6.0) |

### Example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/t2i \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a woman in a red dress standing in Times Square, photorealistic, 4K",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "guidance": 4.0
  }'
```

**Response**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "queued",
  "model": "flux2-klein-9b",
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/a1b2c3d4-..."
}
```

### Portrait example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/t2i \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "professional headshot of a man in a suit, studio lighting, sharp",
    "width": 768,
    "height": 1024,
    "seed": 42
  }'
```

---

## POST /flux/face-swap — Head / Face Swap

Replace the head in a target image with a face from a source image using FLUX.2 Klein 9B + BFS LoRA.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_image` | file | **required** | Body/template photo — head gets replaced |
| `face_image` | file | **required** | Face photo — identity to transfer |
| `aspect_ratio` | string | `original` | Output aspect ratio (see options below) |
| `megapixels` | float | `2.0` | Total output resolution in megapixels (0.5 – 4.0) |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `steps` | int | `4` | Inference steps |
| `cfg` | float | `1.0` | CFG scale |
| `guidance` | float | `4.0` | FLUX guidance strength (2.0 – 6.0) |
| `lora_strength` | float | `1.0` | BFS LoRA strength (0.5 – 1.0) |

### Aspect Ratio Options

| Value | Ratio | Use case |
|-------|-------|----------|
| `original` | Input image AR | Preserve source composition (default) |
| `1:1` | Square | Social media posts |
| `4:3` | Standard | Photos, presentations |
| `3:4` | Portrait standard | Profile photos |
| `16:9` | Widescreen | YouTube, banners |
| `9:16` | Vertical | Instagram / TikTok Stories |
| `3:2` | Classic photo | DSLR landscape |
| `2:3` | Classic portrait | DSLR portrait |
| `21:9` | Cinematic | Ultra-wide |
| `9:21` | Tall cinematic | Ultra-tall |

### Basic example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg"
```

### With aspect ratio + resolution

```bash
# Instagram Story (9:16) at 2 megapixels
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg" \
  -F "aspect_ratio=9:16" \
  -F "megapixels=2.0"
```

```bash
# YouTube thumbnail (16:9) at 1 megapixel
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg" \
  -F "aspect_ratio=16:9" \
  -F "megapixels=1.0"
```

**Response**
```json
{
  "job_id": "x9y8z7w6-...",
  "status": "queued",
  "model": "flux2-klein-9b",
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/x9y8z7w6-..."
}
```

---

## POST /ltx/i2v — Image to Video

Generate a video from an input image using LTX 2.3 (22B, two-pass latent upscale).

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | file | **required** | Source image — first frame of the video |
| `prompt` | string | `""` | Description of the desired motion/scene (auto-enhanced via Gemma) |
| `negative_prompt` | string | *see below* | What to avoid in the output |
| `preset` | string | `fast` | Speed/quality preset: `fast` (~8 steps, <12s) or `quality` (~20 steps) |
| `aspect_ratio` | string | `original` | Output aspect ratio (see options below) |
| `width` | int | `1280` | Output width in pixels (ignored if `aspect_ratio` != `original`) |
| `height` | int | `720` | Output height in pixels (ignored if `aspect_ratio` != `original`) |
| `length` | int | `121` | Number of frames (121 = ~5 sec at 24fps) |
| `fps` | int | `24` | Frames per second |
| `seed` | int | `-1` (random) | Set for reproducible results |

Default negative prompt: `"low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"`

### Aspect Ratio Options

All dimensions are snapped to multiples of 32.

| Value | Ratio | Approx resolution | Use case |
|-------|-------|-------------------|----------|
| `original` | Input image AR | Derived from input | Preserve source composition (default) |
| `16:9` | Widescreen | 1280×720 | YouTube, landscape video |
| `9:16` | Vertical | 720×1280 | Instagram / TikTok Reels |
| `1:1` | Square | 1024×1024 | Social media posts |
| `4:3` | Standard | 1024×768 | Presentations |
| `3:4` | Portrait standard | 768×1024 | Profile / portrait video |
| `3:2` | Classic photo | 1152×768 | Landscape photography |
| `2:3` | Classic portrait | 768×1152 | Portrait photography |
| `21:9` | Cinematic | 1280×544 | Ultra-wide cinematic |
| `9:21` | Tall cinematic | 544×1280 | Ultra-tall |

### Resolution / Length Guide

| `width` | `aspect_ratio` | `length` | `fps` | Duration | VRAM |
|---------|---------------|----------|-------|----------|------|
| 1280 | `16:9` | 121 | 24 | ~5 sec | ~28 GB |
| 1280 | `16:9` | 257 | 24 | ~10 sec | ~32 GB |
| 1280 | `9:16` | 121 | 24 | ~5 sec | ~28 GB |
| 1024 | `1:1` | 121 | 24 | ~5 sec | ~24 GB |

> Recommended GPU: RTX 5090 (32 GB VRAM) or A100 80 GB

### Basic example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=the person walks forward slowly"
```

### With aspect ratio

```bash
# 9:16 vertical video (Instagram Reels)
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=camera slowly zooms in" \
  -F "aspect_ratio=9:16"
```

### Full control

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=a woman smiles and turns her head slowly" \
  -F "aspect_ratio=16:9" \
  -F "width=1280" \
  -F "length=121" \
  -F "fps=24" \
  -F "seed=42"
```

**Response**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "queued",
  "model": "ltx-2.3",
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/a1b2c3d4-..."
}
```

When completed, poll `/status/{job_id}` — the `url` field points to the video file.

---

## POST /ltx/t2v — Text to Video

Generate a video from a text prompt using LTX 2.3 (22B, two-pass latent upscale). No input image required.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | Description of the video to generate |
| `negative_prompt` | string | *see below* | What to avoid in the output |
| `preset` | string | `fast` | Speed/quality preset: `fast` (~8 steps, <12s) or `quality` (~20 steps) |
| `aspect_ratio` | string | `16:9` | Output aspect ratio (see options below) |
| `width` | int | `1280` | Output width in pixels (used when `aspect_ratio` is `original`) |
| `height` | int | `720` | Output height in pixels (used when `aspect_ratio` is `original`) |
| `length` | int | `121` | Number of frames (121 = ~5 sec at 24fps) |
| `fps` | int | `24` | Frames per second |
| `seed` | int | `-1` (random) | Set for reproducible results |

Default negative prompt: `"low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"`

### Aspect Ratio Options

Same as `/ltx/i2v` — all values snapped to multiples of 32.

| Value | Ratio | Approx resolution | Use case |
|-------|-------|-------------------|----------|
| `16:9` | Widescreen | 1280×720 | YouTube, landscape video (default) |
| `9:16` | Vertical | 720×1280 | Instagram / TikTok Reels |
| `1:1` | Square | 1024×1024 | Social media posts |
| `4:3` | Standard | 1024×768 | Presentations |
| `3:4` | Portrait standard | 768×1024 | Profile / portrait video |
| `3:2` | Classic photo | 1152×768 | Landscape photography |
| `2:3` | Classic portrait | 768×1152 | Portrait photography |
| `21:9` | Cinematic | 1280×544 | Ultra-wide cinematic |
| `9:21` | Tall cinematic | 544×1280 | Ultra-tall |
| `original` | Custom | width × height | Manually set width/height |

### Basic example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/t2v \
  -F "prompt=a futuristic city at night with neon lights and flying cars"
```

### With aspect ratio + length

```bash
# 9:16 vertical video, 10 seconds
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/t2v \
  -F "prompt=waves crashing on a rocky shoreline, cinematic, slow motion" \
  -F "aspect_ratio=9:16" \
  -F "length=257"
```

### Cinematic widescreen

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/t2v \
  -F "prompt=a lone astronaut walks on the surface of Mars, cinematic lighting, dust storm in the background" \
  -F "aspect_ratio=21:9" \
  -F "length=121" \
  -F "seed=99"
```

**Response**
```json
{
  "job_id": "b2c3d4e5-...",
  "status": "queued",
  "model": "ltx-2.3",
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/b2c3d4e5-..."
}
```

---

## POST /face-animate — Face Swap + Animate (Pipeline)

Two-step pipeline: replaces the head/face in a template image with the user's face (FLUX.2 Klein 9B), then animates the result into a video (LTX 2.3).

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_image` | file | **required** | Template/body photo — head gets replaced |
| `face_image` | file | **required** | User's face photo — identity to transfer |
| `animate_prompt` | string | **required** | Describes the motion/scene for the video |
| `swap_prompt` | string | `""` | Prompt for the face swap step (uses smart default if empty) |
| `negative_prompt` | string | *see below* | What to avoid in the video |
| `preset` | string | `fast` | Speed/quality preset for video: `fast` (~8 steps, <12s) or `quality` (~20 steps) |
| `aspect_ratio` | string | `16:9` | Output video aspect ratio (see options below) |
| `width` | int | `1280` | Output width in pixels (height derived from aspect ratio) |
| `height` | int | `720` | Output height — used only when `aspect_ratio=original` |
| `length_seconds` | float | `5.0` | Video duration in seconds (3–10) |
| `fps` | int | `24` | Frames per second |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `megapixels` | float | `2.0` | Face swap resolution in megapixels (0.5–4.0) |
| `lora_strength` | float | `1.0` | BFS LoRA strength for face swap (0.5–1.0) |
| `swap_steps` | int | `4` | Face swap inference steps |
| `swap_guidance` | float | `4.0` | Face swap guidance strength |

Default negative prompt: `"low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"`

### Aspect Ratio Options

All LTX video dimensions are snapped to multiples of 32.

| Value | Ratio | Approx resolution | Use case |
|-------|-------|-------------------|----------|
| `16:9` | Widescreen | 1280×720 | YouTube, landscape video (default) |
| `9:16` | Vertical | 720×1280 | Instagram / TikTok Reels |
| `1:1` | Square | 1024×1024 | Social media posts |
| `4:3` | Standard | 1024×768 | Presentations |
| `3:4` | Portrait | 768×1024 | Profile / portrait |
| `3:2` | Classic photo | 1152×768 | Landscape |
| `2:3` | Classic portrait | 768×1152 | Portrait |
| `21:9` | Cinematic | 1280×544 | Ultra-wide |
| `9:21` | Tall cinematic | 544×1280 | Ultra-tall |
| `original` | Custom | width × height | Manual width/height |

### How It Works

```
face_image + target_image
        ↓
  [Step 1] FLUX.2 Klein 9B face swap
        ↓
  swapped image
        ↓
  [Step 2] LTX 2.3 image-to-video animation
        ↓
  output video
```

Poll `/status/{job_id}` — the `step` field shows current progress: `face_swap` or `animating`.

### Basic example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/face-animate \
  -F "target_image=@template_body.jpg" \
  -F "face_image=@user_face.jpg" \
  -F "animate_prompt=the person smiles and looks at the camera, gentle head movement"
```

### Instagram Reel (9:16, 8 seconds)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/face-animate \
  -F "target_image=@template_body.jpg" \
  -F "face_image=@user_face.jpg" \
  -F "animate_prompt=person walks confidently forward, cinematic lighting" \
  -F "aspect_ratio=9:16" \
  -F "length_seconds=8"
```

### YouTube thumbnail animation (16:9, 5 seconds)

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/face-animate \
  -F "target_image=@template_body.jpg" \
  -F "face_image=@user_face.jpg" \
  -F "animate_prompt=person turns head slowly and smiles" \
  -F "aspect_ratio=16:9" \
  -F "length_seconds=5" \
  -F "seed=42"
```

### Full control

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/face-animate \
  -F "target_image=@template_body.jpg" \
  -F "face_image=@user_face.jpg" \
  -F "swap_prompt=head_swap: seamlessly replace head from image 1 with face from image 2" \
  -F "animate_prompt=person waves hello and smiles warmly, natural lighting" \
  -F "aspect_ratio=9:16" \
  -F "width=1080" \
  -F "length_seconds=6" \
  -F "fps=24" \
  -F "megapixels=2.0" \
  -F "lora_strength=1.0" \
  -F "seed=99"
```

**Response**
```json
{
  "job_id": "c3d4e5f6-...",
  "status": "queued",
  "model": "flux2-klein-9b + ltx-2.3-22b",
  "pipeline": ["face_swap", "image_to_video"],
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/c3d4e5f6-..."
}
```

**Processing status** (poll `/status/{job_id}`)
```json
{ "status": "processing", "step": "face_swap", "started_at": "..." }
```
```json
{ "status": "processing", "step": "animating", "started_at": "..." }
```

**Completed**
```json
{
  "status": "completed",
  "url": "https://YOUR_POD_ID-7860.proxy.runpod.net/video/face_animate_42_00001_.mp4",
  "filename": "face_animate_42_00001_.mp4",
  "swap_filename": "flux_swap_42_00001_.png",
  "duration_seconds": 142.3
}
```

### Typical Times

| Operation | Cold start | Warm |
|-----------|-----------|------|
| Face swap (2MP) | ~3–5 min | ~20–30 sec |
| Animation (1280×720, 5s) | (models already loaded) | ~60–90 sec |
| **Total pipeline** | **~5–8 min** | **~90–120 sec** |

---

## GET /status/{job_id} — Poll Job Status

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/status/a1b2c3d4-...
```

### Possible responses

**Queued**
```json
{ "status": "queued", "created_at": "2026-03-28T06:00:00Z" }
```

**Processing**
```json
{ "status": "processing", "created_at": "2026-03-28T06:00:00Z", "started_at": "2026-03-28T06:00:01Z" }
```

**Completed**
```json
{
  "status": "completed",
  "url": "https://YOUR_POD_ID-7860.proxy.runpod.net/image/flux_swap_42_00001_.png",
  "filename": "flux_swap_42_00001_.png",
  "completed_at": "2026-03-28T06:00:20Z",
  "duration_seconds": 18.4
}
```

**Failed**
```json
{
  "status": "failed",
  "error": "error message here",
  "failed_at": "2026-03-28T06:00:05Z"
}
```

---

## GET /image/{filename} — Download Image or Video

```bash
curl -O https://YOUR_POD_ID-7860.proxy.runpod.net/image/flux_swap_42_00001_.png
```

Also works for video files returned by `/ltx/i2v` and `/ltx/t2v`. Or open directly in a browser.

---

## GET /jobs — List All Jobs

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/jobs
```

**Response**
```json
{
  "total": 5,
  "summary": {
    "queued": 0,
    "processing": 1,
    "completed": 3,
    "failed": 1
  },
  "jobs": [
    { "job_id": "a1b2c3...", "status": "completed", "url": "..." },
    ...
  ]
}
```

---

## GET /queue — Active Queue

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/queue
```

**Response**
```json
{
  "count": 1,
  "jobs": [
    { "job_id": "a1b2c3...", "status": "processing" }
  ]
}
```

---

## POST /jobs/{job_id}/retry — Retry a Failed Job

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/jobs/a1b2c3d4-.../retry
```

**Response**
```json
{
  "new_job_id": "e5f6g7h8-...",
  "original_job_id": "a1b2c3d4-...",
  "status": "queued",
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/e5f6g7h8-..."
}
```

---

## DELETE /jobs/{job_id} — Delete a Job

Deletes the job record and its output file.

```bash
curl -X DELETE https://YOUR_POD_ID-7860.proxy.runpod.net/jobs/a1b2c3d4-...
```

**Response**
```json
{
  "job_id": "a1b2c3d4-...",
  "deleted": true,
  "file_deleted": "flux_swap_42_00001_.png"
}
```

---

## DELETE /jobs — Bulk Delete Jobs

By default deletes only completed jobs. Pass `?completed_only=false` to delete all.

```bash
# Delete all completed jobs
curl -X DELETE https://YOUR_POD_ID-7860.proxy.runpod.net/jobs

# Delete all jobs (including queued/processing)
curl -X DELETE "https://YOUR_POD_ID-7860.proxy.runpod.net/jobs?completed_only=false"
```

**Response**
```json
{
  "deleted_jobs": 4,
  "deleted_files": 4
}
```

---

## Polling Pattern

Jobs are async. Submit → poll until `completed` or `failed`.

```bash
# 1. Submit (works for t2i, face-swap, ltx/i2v, ltx/t2v)
JOB_ID=$(curl -s -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/t2v \
  -F "prompt=a sunset over the ocean" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")

echo "Job submitted: $JOB_ID"

# 2. Poll every 10 seconds
while true; do
  STATUS=$(curl -s https://YOUR_POD_ID-7860.proxy.runpod.net/status/$JOB_ID)
  STATE=$(echo $STATUS | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATE"
  if [ "$STATE" = "completed" ] || [ "$STATE" = "failed" ]; then
    echo $STATUS | python3 -m json.tool
    break
  fi
  sleep 10
done
```

---

## Typical Generation Times

| Operation | Cold start (first job) | Warm (model cached) |
|-----------|----------------------|---------------------|
| Text to image (1024×1024, 4 steps) | ~3–5 min | ~10–15 sec |
| Face swap (2MP, 4 steps) | ~3–5 min | ~20–30 sec |
| LTX i2v / t2v (1280×720, 121 frames) | ~5–8 min | ~60–90 sec |
| LTX i2v / t2v (1280×720, 257 frames) | ~8–12 min | ~2–3 min |
| Face swap + animate (1280×720, 5s) | ~5–8 min | ~90–120 sec |

> Cold start loads ~50 GB of models into VRAM. All subsequent jobs are fast.

---

## Error Responses

All errors return standard HTTP status codes with a JSON body.

```json
{ "detail": "error message here" }
```

| Code | Meaning |
|------|---------|
| `400` | Bad request (missing field, invalid aspect_ratio, etc.) |
| `404` | Job or file not found |
| `422` | Validation error (wrong type for a parameter) |
| `500` | Internal server error |
