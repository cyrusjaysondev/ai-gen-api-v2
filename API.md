# AI Gen API v2 — API Reference

Base URL: `https://YOUR_POD_ID-7860.proxy.runpod.net`

Interactive docs (Swagger UI): `https://YOUR_POD_ID-7860.proxy.runpod.net/docs`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/t2i` | Text to image |
| POST | `/flux/face-swap` | Head / face swap |
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

```bash
# Square (1:1) with fixed seed for reproducibility
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg" \
  -F "aspect_ratio=1:1" \
  -F "megapixels=2.0" \
  -F "seed=12345"
```

```bash
# Full control over all parameters
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg" \
  -F "aspect_ratio=3:4" \
  -F "megapixels=2.0" \
  -F "seed=99" \
  -F "steps=4" \
  -F "guidance=4.0" \
  -F "lora_strength=1.0"
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

## GET /image/{filename} — Download Image

```bash
curl -O https://YOUR_POD_ID-7860.proxy.runpod.net/image/flux_swap_42_00001_.png
```

Or open directly in a browser.

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
# 1. Submit
JOB_ID=$(curl -s -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body.jpg" \
  -F "face_image=@face.jpg" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")

echo "Job submitted: $JOB_ID"

# 2. Poll every 5 seconds
while true; do
  STATUS=$(curl -s https://YOUR_POD_ID-7860.proxy.runpod.net/status/$JOB_ID)
  STATE=$(echo $STATUS | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATE"
  if [ "$STATE" = "completed" ] || [ "$STATE" = "failed" ]; then
    echo $STATUS | python3 -m json.tool
    break
  fi
  sleep 5
done
```

---

## Typical Generation Times

| Operation | Cold start (first job) | Warm (model cached) |
|-----------|----------------------|---------------------|
| Text to image (512×512, 4 steps) | ~3–5 min | ~10–15 sec |
| Face swap (2MP, 4 steps) | ~3–5 min | ~20–30 sec |

> Cold start loads ~25 GB of models into VRAM. All subsequent jobs are fast.

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
