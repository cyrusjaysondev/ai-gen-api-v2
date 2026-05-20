# AI Gen API v2 — API Reference

Base URL: `https://YOUR_POD_ID-7860.proxy.runpod.net`

Interactive docs (Swagger UI): `https://YOUR_POD_ID-7860.proxy.runpod.net/docs`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/t2i` | Text to image (FLUX.2 Klein 9B) |
| POST | `/sana/t2i` | Text to image (SANA-Sprint 1.6B — ~sub-second on RTX 5090) |
| POST | `/flux/face-swap` | Head / face swap (FLUX.2 Klein 9B) |
| POST | `/flux/i2i` | Multi-reference image editing — 1 to 5 input images (FLUX.2 Klein 9B) |
| GET | `/admin/blocklist` | List blocked face identities (admin auth) |
| POST | `/admin/blocklist` | Upload a face to block |
| DELETE | `/admin/blocklist/{identity}` | Remove a blocked face |
| GET | `/admin/blocklist/{identity}/image` | Preview a blocked face image |
| GET | `/admin/blocklist-logos` | List blocked logos/flags (admin auth) |
| POST | `/admin/blocklist-logos` | Upload a logo/flag to block |
| DELETE | `/admin/blocklist-logos/{identity}` | Remove a blocked logo/flag |
| GET | `/admin/blocklist-logos/{identity}/image` | Preview a blocked logo image |
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
| GET | `/video/{filename}` | Download video |
| GET | `/videos` | List all videos |

---

## Speed & Quality Presets

All LTX video endpoints (`/ltx/i2v`, `/ltx/t2v`, `/face-animate`) support a `preset` parameter. Both presets use the official LTX-2.3 distilled inference profile (sigmas + LoRA strength taken from Lightricks' reference workflows).

| Preset | Mode | Steps | LoRA Strength | Speed (4s @544×960, warm) | Best for |
|--------|------|-------|---------------|-----------------------------|----------|
| `fast` | Single pass at target resolution | 8 | 0.5 | **~12s** | Default. Quick iteration. |
| `quality` | Two-pass: 8 steps at half-res → 2× spatial upscale → 3 refine steps at full-res | 8 + 3 | 0.5 | **~12s** | Slightly sharper detail. Same wall-time as `fast` because the bulk of compute happens at half-res. |

### How fast preset works

- **Single pass** at full target resolution
- **8 denoising steps** using distilled LoRA (the 8-step warmup-cluster schedule the LoRA was trained against)
- **No audio** by default (skip audio VAE load/encode/decode entirely)

### How quality preset works

- **Two-pass pipeline**: 8 low-res steps → spatial upscale → 3 refine steps at full-res
- The 2× spatial upscale uses Lightricks' `ltx-2.3-spatial-upscaler-x2-1.0.safetensors`
- LoRA strength 0.5 across both passes — matches reference

### Speed by length (fast preset, 544×960, warm GPU)

| `length` | Duration | Approx wall time |
|----------|----------|------------------|
| 49 | ~2s | ~7s |
| 97 | ~4s | ~12s |
| 121 | ~5s | ~14s |
| 161 | ~6.7s | ~18s |

> **First request after pod start takes ~30-60s longer** as ComfyUI loads the 27 GB LTX checkpoint, 8.8 GB Gemma text encoder, and 7 GB distilled LoRA into VRAM. Subsequent requests reuse the cached models.

### Additional speed knobs

| Knob | Effect |
|------|--------|
| `length=49` | Roughly halves wall time (49-frame ≈ 2s clip) |
| `enhance_prompt=false` | Skips the Gemma prompt-rewrite pass, saves 2-5s. Use when you already wrote a detailed prompt. |
| Smaller `width`/`height` | LTX scales roughly with pixel count. `384×640` is ~2× faster than `544×960`. |
| `audio=false` | Default; skips audio encode/decode (saves ~5-10s) |

### Audio control

All LTX video endpoints accept `audio` (bool, default `false`).

- `audio=false` — video only, faster (skips audio VAE entirely)
- `audio=true` — generates audio track with the video (adds ~5-10s overhead)

```bash
# Without audio (default, faster)
curl -X POST .../ltx/t2v -F "prompt=cat on beach" -F "preset=fast"

# With audio
curl -X POST .../ltx/t2v -F "prompt=cat on beach" -F "preset=fast" -F "audio=true"
```

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

## GET /ltx/presets — List Presets

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/ltx/presets
```

**Response**
```json
{
  "presets": {
    "fast": { "mode": "single_pass", "steps": 8, "lora_strength": 0.5 },
    "quality": { "mode": "two_pass", "low_res_steps": 8, "high_res_steps": 3, "lora_strength": 0.5 }
  },
  "default": "fast",
  "endpoints": ["/ltx/i2v", "/ltx/t2v", "/face-animate"]
}
```

---

## Watermarks

Every generation endpoint accepts two optional watermark parameters. They
are independent and can stack — set both and you get a logo with text
beside it.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `watermark` | string \| null | `null` | Short text drawn at the bottom-right in bold white with a black outline. `null` / empty = off. Example: `"AI"`. |
| `watermark_image` | bool | `false` | Composite the **GenReel logo** at the bottom-right. The PNG lives on the network volume at `/workspace/assets/genreel_logo.png` (fetched by `setup.sh`). |

Images stamp in-place via Pillow. Videos re-encode through `libx264` /
`drawtext` / `overlay` filters with the audio stream-copied — typically
adds ~1–3 s for a 5 s clip. A missing logo file is logged and silently
skipped so a network blip during setup never fails a generation.

If a watermark step fails, the job still completes successfully — the
result will include a `watermark_warning` field on the `/status/{job_id}`
response.

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
| `watermark` | string \| null | `null` | See [Watermarks](#watermarks). |
| `watermark_image` | bool | `false` | See [Watermarks](#watermarks). |

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

---

## POST /sana/t2i — Text to Image (SANA-Sprint 1.6B)

NVIDIA's SANA-Sprint 1.6B does 1024px text-to-image in 2 SCM steps. On an RTX 5090 a warm
worker returns in **well under a second per image** — use this when you want previews or
high-volume generation. For best fidelity / large compositions, prefer `/t2i` (FLUX.2).

**Cold start:** ~30 s the first time after a pod starts — ComfyUI loads the Sana checkpoint
(6.5 GB), Gemma-2-2b text encoder (5 GB), and DC-AE VAE (1.25 GB) into VRAM. Subsequent
calls reuse the cached models. All three artifacts live in
`$HF_HOME=/workspace/.cache/huggingface` on the network volume, so they survive pod
restarts (no re-download).

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | What to generate |
| `negative_prompt` | string | `""` | Optional. Encoded by Gemma as a negative conditioning. |
| `width` | int | `1024` | Output width in pixels. 1024 is the trained resolution. |
| `height` | int | `1024` | Output height in pixels. |
| `seed` | int | `-1` (random) | Set for reproducible results. |
| `steps` | int | `2` | SCM steps. Sprint is tuned for 2; values up to ~4 yield marginal gains. |
| `cfg` | float | `1.0` | Sprint is a distilled / one-step-ish model — keep at 1.0. |
| `scm_cfg_scale` | float | `4.5` | SCM sampler's internal CFG scale (`ScmModelSampling.cfg_scale`). Defaults from NVlabs reference workflow. |
| `watermark` | string \| null | `null` | Stamp this text in the bottom-right corner. `null` / `""` = off. |
| `watermark_image` | bool | `false` | Composite the GenReel logo at the bottom-right. |

### Example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/sana/t2i \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a tiny astronaut hatching from an egg on the moon, photoreal, dramatic light",
    "width": 1024,
    "height": 1024
  }'
```

**Response**
```json
{
  "job_id": "b2c3d4e5-...",
  "status": "queued",
  "model": "sana-sprint-1.6b",
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/b2c3d4e5-..."
}
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
| `watermark` | string \| null | `null` | See [Watermarks](#watermarks). |
| `watermark_image` | bool | `false` | See [Watermarks](#watermarks). |

### Aspect Ratio Options

| Value | Ratio | Use case |
|-------|-------|----------|
| `original` | Input image AR | Preserve source composition (default) |
| `1:1` | Square | Social media posts |
| `4:3` / `3:4` | Standard | Photos, presentations |
| `16:9` / `9:16` | Wide / Vertical | YouTube / Instagram |
| `3:2` / `2:3` | Classic | DSLR landscape / portrait |
| `21:9` / `9:21` | Cinematic | Ultra-wide / ultra-tall |

### Example

```bash
curl -X POST https://YOUR_POD_ID-7860.proxy.runpod.net/flux/face-swap \
  -F "target_image=@body_photo.jpg" \
  -F "face_image=@face_photo.jpg" \
  -F "aspect_ratio=9:16" \
  -F "megapixels=2.0"
```

---

## POST /flux/i2i — Multi-reference Image Editing

Edit / compose with 1 to 5 reference images. All inputs are chained as FLUX.2
reference latents on top of the prompt's conditioning — the prompt drives
the edit, the images supply style, identity, objects, composition cues.

Output canvas dimensions default to the **first image's rescaled size**, so
you can use the first image as the "edit target" and the rest as references.
Override explicitly with `width` and `height` if you want a fixed canvas.

### Parameters (multipart/form-data)

| Param | Default | Description |
|-------|---------|-------------|
| `prompt` | required | The edit instruction |
| `images` | required | 1 to 5 image files. First image's dimensions are used as the canvas unless `width`/`height` are set |
| `seed` | -1 (random) | Reproducibility seed |
| `megapixels` | 2.0 | Resolution per reference image (0.5–4.0) |
| `width` | 0 | Output width — `0` means "derive from first image" |
| `height` | 0 | Output height — `0` means "derive from first image" |
| `steps` | 4 | Inference steps |
| `cfg` | 1.0 | CFG scale |
| `guidance` | 4.0 | FLUX guidance strength (2.0–6.0) |
| `lora_strength` | 0.0 | Apply head-swap LoRA. `0` = general edits; `0.5–1.0` = face/head-focused |
| `watermark` | null | See [Watermarks](#watermarks). |
| `watermark_image` | false | See [Watermarks](#watermarks). |

### Example

```bash
curl -X POST "$POD/flux/i2i" \
  -F "prompt=combine the subject from image 1 with the outfit from image 2 in the setting of image 3" \
  -F "images=@subject.png" \
  -F "images=@outfit.png" \
  -F "images=@setting.png" \
  -F "megapixels=2.0" \
  -F "seed=42"
```

Response shape is the standard job-queue response:
```json
{
  "job_id": "...",
  "status": "queued",
  "model": "flux2-klein-9b",
  "ref_count": 3,
  "poll_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/status/..."
}
```

Poll `/status/{job_id}` for the result URL, just like the other endpoints.

---

## POST /ltx/i2v — Image to Video

Generate a video from an input image using LTX 2.3 (22B).

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | file | **required** | Source image — first frame of the video |
| `prompt` | string | `""` | Description of the desired motion/scene |
| `negative_prompt` | string | *see below* | What to avoid in the output |
| `preset` | string | `fast` | `fast` (8 steps single-pass, ~10–15s @544×960) or `quality` (8+3 steps two-pass with 2× spatial upscale, ~40–60s) |
| `audio` | bool | `false` | Generate audio track (`true` adds ~5-10s overhead) |
| `aspect_ratio` | string | `9:16` | Output aspect ratio — see table below. **When set, `height` is ignored** and derived from `width`. Use `original` to honor the explicit `width`/`height`. |
| `width` | int | `544` | Output width in pixels. With `aspect_ratio=9:16`, `width=544` → 544×960 (fast preset spec); `width=720` → 720×1280 (quality preset spec). |
| `height` | int | `960` | Output height in pixels. **Ignored unless `aspect_ratio=original`.** |
| `length` | int | `121` | Number of frames (121 = ~5 sec at 24fps) |
| `fps` | int | `24` | Frames per second |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `enhance_prompt` | bool | `true` | Rewrite the prompt via Gemma 12B using the input image as context (adds 2-5s + VRAM). Recommended ON for short prompts (`"make her run"`); OFF when you've already written a detailed scene description. |
| `inplace_strength` | float | `0.7` | How tightly each frame's latent is pinned to the input image. `0.7` is the reference distilled value (best identity, weakest motion). **Lower it for action prompts:** `0.5` ≈ moderate motion, `0.4` ≈ strong motion (some identity drift), `0.3` ≈ near-t2v. Range `0.3`–`1.0`. Two-pass refine tracks this. |
| `watermark` | string \| null | `null` | See [Watermarks](#watermarks). Video re-encodes via ffmpeg (~1-3s for a 5s clip). |
| `watermark_image` | bool | `false` | See [Watermarks](#watermarks). |

Default negative prompt: `"low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"`

### Why your video isn't moving

If the subject barely moves despite a clear action prompt, two things are usually fighting you:

1. **Wrong resolution.** Setting `width=1280 aspect_ratio=9:16` renders **1280×2272** (height is recomputed from width). At that size the input image dominates every frame. Use `width=544` (fast) or `width=720` (quality) for 9:16.
2. **`inplace_strength` too high.** The default `0.7` matches Lightricks' reference profile and prioritizes identity. For motion-heavy prompts, lower it to `0.5` or `0.4`. The model has cfg=1.0 hardwired (mandatory for the distilled LoRA), so the prompt cannot push hard — `inplace_strength` is the real motion knob.

### Aspect Ratio Options

All dimensions are snapped to multiples of 32. When an aspect ratio is set, **height is computed from width** — pick `width` from the table to land on the spec'd resolution.

| Value | `width` for fast | `width` for quality | Use case |
|-------|------------------|---------------------|----------|
| `original` | n/a (uses input dims) | n/a | Preserve source composition |
| `9:16` | `544` → 544×960 | `720` → 720×1280 | **Instagram / TikTok Reels (default)** |
| `16:9` | `960` → 960×544 | `1280` → 1280×720 | YouTube, landscape video |
| `1:1` | `768` → 768×768 | `1024` → 1024×1024 | Social media posts |
| `4:3` / `3:4` | `768` / `576` | `1024` / `768` | Standard / portrait |
| `3:2` / `2:3` | `864` / `576` | `1152` / `768` | Classic landscape / portrait |
| `21:9` / `9:21` | `1120` / `480` | `1280` / `544` | Cinematic ultra-wide / tall |

### Frame length guide

| `length` | Duration (24fps) |
|----------|-----------------|
| 49 | ~2 sec |
| 73 | ~3 sec |
| 97 | ~4 sec |
| 121 | ~5 sec |
| 161 | ~6.7 sec |
| 257 | ~10 sec |

### Examples

```bash
# 9:16 reel — recommended fast preview (5s, ~10-15s on a warm GPU)
curl -X POST .../ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=she walks forward, hair moving in the wind" \
  -F "preset=fast" \
  -F "aspect_ratio=9:16" \
  -F "width=544" \
  -F "length=121" \
  -F "fps=24" \
  -F "seed=-1"

# 9:16 action prompt — needs lower inplace_strength so the subject can actually move
curl -X POST .../ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=she runs across the frame" \
  -F "preset=fast" \
  -F "aspect_ratio=9:16" \
  -F "width=544" \
  -F "length=121" \
  -F "inplace_strength=0.45" \
  -F "enhance_prompt=true"

# 9:16 final-quality (two-pass with upscale, ~40-60s)
curl -X POST .../ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=camera slowly orbits around her, cinematic lighting" \
  -F "preset=quality" \
  -F "aspect_ratio=9:16" \
  -F "width=720" \
  -F "length=121"

# Subtle motion (default inplace_strength is fine for this)
curl -X POST .../ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=subtle head turn, eyes blink" \
  -F "preset=fast" \
  -F "aspect_ratio=9:16" \
  -F "width=544" \
  -F "length=49" \
  -F "seed=42"

# 16:9 with audio
curl -X POST .../ltx/i2v \
  -F "image=@my_photo.jpg" \
  -F "prompt=camera slowly zooms in, birds chirping" \
  -F "preset=quality" \
  -F "audio=true" \
  -F "aspect_ratio=16:9" \
  -F "width=1280"
```

---

## POST /ltx/t2v — Text to Video

Generate a video from a text prompt using LTX 2.3. No input image required.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | Description of the video to generate |
| `negative_prompt` | string | *see below* | What to avoid in the output |
| `preset` | string | `fast` | `fast` (8 steps single-pass) or `quality` (8+3 steps two-pass with 2× spatial upscale). Both ~12s on warm GPU at 544×960. |
| `audio` | bool | `false` | Generate audio track (`true` adds ~5-10s overhead) |
| `aspect_ratio` | string | `16:9` | Output aspect ratio |
| `width` | int | `1280` | Output width in pixels |
| `height` | int | `720` | Output height in pixels |
| `length` | int | `121` | Number of frames (121 = ~5 sec at 24fps) |
| `fps` | int | `24` | Frames per second |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `watermark` | string \| null | `null` | See [Watermarks](#watermarks). Video re-encodes via ffmpeg (~1-3s for a 5s clip). |
| `watermark_image` | bool | `false` | See [Watermarks](#watermarks). |

### Examples

```bash
# Fast text-to-video
curl -X POST .../ltx/t2v \
  -F "prompt=a golden retriever running through a meadow"

# Quality with audio, cinematic widescreen
curl -X POST .../ltx/t2v \
  -F "prompt=waves crashing on rocks, cinematic slow motion" \
  -F "preset=quality" \
  -F "audio=true" \
  -F "aspect_ratio=21:9"

# Vertical for social media
curl -X POST .../ltx/t2v \
  -F "prompt=a person dancing in a neon-lit room" \
  -F "aspect_ratio=9:16" \
  -F "length=121"
```

---

## POST /face-animate — Face Swap + Animate (Pipeline)

Two-step pipeline: replaces the head/face in a template image (FLUX.2 Klein 9B), then animates the result into a video (LTX 2.3).

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_image` | file | **required** | Template/body photo — head gets replaced |
| `face_image` | file | **required** | User's face photo — identity to transfer |
| `animate_prompt` | string | **required** | Describes the motion/scene for the video |
| `swap_prompt` | string | `""` | Prompt for the face swap step (uses smart default if empty) |
| `negative_prompt` | string | *see below* | What to avoid in the video |
| `preset` | string | `fast` | `fast` (8 steps single-pass) or `quality` (8+3 steps two-pass with 2× spatial upscale) |
| `audio` | bool | `false` | Generate audio track with the video |
| `aspect_ratio` | string | `16:9` | Output video aspect ratio |
| `width` | int | `1280` | Output width in pixels |
| `height` | int | `720` | Output height |
| `length_seconds` | float | `5.0` | Video duration in seconds |
| `fps` | int | `24` | Frames per second |
| `seed` | int | `-1` (random) | Set for reproducible results |
| `megapixels` | float | `2.0` | Face swap resolution in megapixels (0.5–4.0) |
| `lora_strength` | float | `1.0` | BFS LoRA strength for face swap (0.5–1.0) |
| `swap_steps` | int | `4` | Face swap inference steps |
| `swap_guidance` | float | `4.0` | Face swap guidance strength |
| `watermark` | string \| null | `null` | See [Watermarks](#watermarks). Applied to the final video. |
| `watermark_image` | bool | `false` | See [Watermarks](#watermarks). Applied to the final video. |

### How It Works

```
face_image + target_image
        |
  [Step 1] FLUX.2 Klein 9B face swap
        |
  swapped image
        |
  [Step 2] LTX 2.3 image-to-video animation
        |
  output video (.mp4)
```

Poll `/status/{job_id}` — the `step` field shows current progress: `face_swap` or `animating`.

### Examples

```bash
# Fast face-animate
curl -X POST .../face-animate \
  -F "target_image=@template_body.jpg" \
  -F "face_image=@user_face.jpg" \
  -F "animate_prompt=the person smiles and looks at the camera"

# Quality with audio, Instagram Reel
curl -X POST .../face-animate \
  -F "target_image=@template_body.jpg" \
  -F "face_image=@user_face.jpg" \
  -F "animate_prompt=person walks confidently forward" \
  -F "preset=quality" \
  -F "audio=true" \
  -F "aspect_ratio=9:16" \
  -F "length_seconds=8"
```

---

## GET /status/{job_id} — Poll Job Status

```bash
curl https://YOUR_POD_ID-7860.proxy.runpod.net/status/{job_id}
```

### Possible responses

**Queued**
```json
{ "status": "queued", "created_at": "2026-03-28T06:00:00Z" }
```

**Processing**
```json
{ "status": "processing", "created_at": "...", "started_at": "..." }
```

**Completed (video)**
```json
{
  "status": "completed",
  "url": "https://YOUR_POD_ID-7860.proxy.runpod.net/video/ltx_t2v_42_00001_.mp4",
  "thumbnail_url": "https://YOUR_POD_ID-7860.proxy.runpod.net/image/ltx_t2v_42_00001__thumb.jpg",
  "filename": "ltx_t2v_42_00001_.mp4",
  "completed_at": "2026-03-28T06:00:40Z",
  "duration_seconds": 36.1
}
```

> `thumbnail_url` is present only on video outputs (the four endpoints
> `/ltx/i2v`, `/ltx/t2v`, `/face-animate`, plus any future video
> workflows). It's the first frame as a JPG, served from the same domain.
> If a watermark was applied, the thumbnail reflects it.

**Completed (image)**
```json
{
  "status": "completed",
  "url": "https://YOUR_POD_ID-7860.proxy.runpod.net/image/t2i_42_00001_.png",
  "filename": "t2i_42_00001_.png",
  "completed_at": "2026-03-28T06:00:40Z",
  "duration_seconds": 1.2
}
```

**Failed**
```json
{ "status": "failed", "error": "error message here", "failed_at": "..." }
```

---

## Job Management

### GET /jobs — List all jobs

```bash
curl .../jobs
```

### GET /queue — Active queue

```bash
curl .../queue
```

### POST /jobs/{job_id}/retry — Retry failed job

```bash
curl -X POST .../jobs/{job_id}/retry
```

### DELETE /jobs/{job_id} — Delete job + output file

```bash
curl -X DELETE .../jobs/{job_id}
```

### DELETE /jobs — Bulk delete completed jobs

```bash
curl -X DELETE .../jobs
# Delete all (including queued/processing):
curl -X DELETE ".../jobs?completed_only=false"
```

---

## File Access

### GET /image/{filename}

```bash
curl -O .../image/flux_swap_42_00001_.png
```

### GET /video/{filename}

```bash
curl -O .../video/ltx_t2v_42_00001_.mp4
```

### GET /videos — List all videos

```bash
curl .../videos
```

---

## Typical Generation Times (warm model)

| Operation | fast preset | quality preset |
|-----------|------------|----------------|
| Text to image (1024×1024) | ~10-15s | N/A |
| Face swap (2MP) | ~20-30s | N/A |
| T2V / I2V (544×960, 4s) | ~12s | ~12s |
| T2V / I2V (544×960, 5s) | ~14s | ~14s |
| T2V / I2V (768×1344, 5s) | ~22s | ~24s |
| Face animate (544×960, 4s) | ~35s (swap + video) | ~35s |

> **Both presets land at roughly the same wall time** because `quality` does most of its work at half-resolution (8 steps at ~128k pixels → 2× spatial upscale → 3 refine steps at full-res). Use `quality` when you want sharper detail at no real speed cost; use `fast` for the simpler single-pass pipeline.

> Cold start (first job after pod start) takes ~30-60s extra to load ~50 GB of models into VRAM. All subsequent jobs use cached models.

> Adding `audio=true` adds ~5-10s overhead to any video generation.

---

## Polling Pattern

```bash
# 1. Submit
JOB_ID=$(curl -s -X POST .../ltx/t2v \
  -F "prompt=a sunset over the ocean" \
  -F "preset=fast" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")

echo "Job: $JOB_ID"

# 2. Poll every 5 seconds
while true; do
  STATUS=$(curl -s .../status/$JOB_ID)
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

## Compliance Filters

`/flux/face-swap` and `/flux/i2i` accept two **independent** compliance toggles:

| Parameter | Default | Detector | Blocklist dir | What it catches |
|---|---|---|---|---|
| `face_filter` | `false` | InsightFace `buffalo_l` (face recognition) | `/workspace/blocklist/` | Specific human faces (politicians, celebrities, banned individuals) |
| `logo_filter` | `false` | CLIP ViT-B/32 (whole-image semantic) | `/workspace/blocklist_logos/` | Logos, flags, symbols, propaganda imagery — anything that's the **main subject** of the input |

Set either or both to `true` per request. They run in sequence; the first
blocked input fails the whole request with `400`.

### Limits

- **Face filter** is precise (~99% recall on clear faces above threshold).
- **Logo filter** is whole-image — it catches "this image is mostly the Apple logo"
  but **may miss tiny logos in corners** of larger photos. For tight detection
  of small logos, that's a v2 feature (SIFT keypoint matching).

### Face Filter

### Request

```bash
curl -X POST "$POD/flux/i2i" \
  -F "prompt=stylize as a watercolor" \
  -F "images=@person.png" \
  -F "face_filter=true"
```

### Block response

```json
{
  "detail": {
    "error": "blocked",
    "reason": "images[0] matches blocked identity",
    "matched_identity": "tom_hanks",
    "score": 0.87,
    "image_index": 0
  }
}
```

`score` is cosine similarity vs the closest blocklist entry; default threshold
is `0.6` (override via `FACE_FILTER_THRESHOLD` env var on the pod).

### Logo / flag filter

```bash
curl -X POST "$POD/flux/i2i" \
  -F "prompt=stylize" \
  -F "images=@input.png" \
  -F "logo_filter=true"
```

Block response:
```json
{
  "detail": {
    "error": "blocked",
    "filter": "logo",
    "reason": "images[0] matches blocked logo/flag",
    "matched_logo": "apple_logo",
    "score": 0.91,
    "image_index": 0
  }
}
```

Threshold defaults to `0.85` (override via `LOGO_FILTER_THRESHOLD` env var).

### Both filters at once

```bash
curl -X POST "$POD/flux/face-swap" \
  -F "target_image=@body.png" -F "face_image=@face.png" \
  -F "face_filter=true" \
  -F "logo_filter=true"
```

The response's `filter` field (`"face"` or `"logo"`) tells you which check
fired. Face filter runs first.

### Bypass audit

Every `face_filter=false` and `logo_filter=false` call is appended to
`/workspace/face_filter_bypass.log` with timestamp, endpoint, job_id, and
which filter was bypassed.

---

## Admin API (blocklist management)

Two parallel sets of admin endpoints — one for faces, one for logos/flags.
Same shape, same hot-reload, same optional auth.

### Auth

Auth is **optional and off by default** so the admin API is easy to access
during development. Behavior is controlled by the `ADMIN_TOKEN` env var:

- **`ADMIN_TOKEN` unset (default):** admin endpoints are open — no auth
  required. Anyone with the pod URL can manage the blocklist. Fine for
  dev / private pods.
- **`ADMIN_TOKEN` set:** every admin call must include
  `Authorization: Bearer <token>`. `401` without header, `403` with wrong
  token. Recommended before going to production / sharing the pod URL.

Switch between modes by setting/unsetting the env var on the RunPod template
and restarting the pod — no code change needed.

### Faces — `/admin/blocklist`

The blocklist is stored on the network volume at `/workspace/blocklist/`,
one image per identity. It's shared with serverless workers (mounted at
`/runpod-volume/blocklist/`) and hot-reloaded on every face-filter check —
uploads and deletes take effect on the next request.

> The examples below show the open-mode (no `ADMIN_TOKEN`). If you set
> `ADMIN_TOKEN` on the pod, add `-H "Authorization: Bearer $ADMIN_TOKEN"`
> to every call.

### POST /admin/blocklist — Upload a face

```bash
curl -X POST "$POD/admin/blocklist" \
  -F "identity=tom_hanks" \
  -F "image=@hanks.png" \
  -F "overwrite=false"
```

**Accepted input:** PNG / JPG / JPEG / WEBP (anything Pillow can decode).
EXIF orientation is honored, so phone photos rotate correctly.

**Auto-normalize on upload:** every accepted image is downscaled so the
longer edge is ≤ `BLOCKLIST_MAX_EDGE` (default `1024` px, env-overridable)
and re-encoded as PNG before storage. Callers do **not** need to resize or
re-format client-side — upload the raw photo. The stored filename is
always `<identity>.png` regardless of input format. Identity match accuracy
is unaffected: InsightFace recognizes at 112x112 internally, so anything
above ~256 px on the face is identical to the full-resolution input.

Validation runs on the normalized image: the upload must contain **exactly
one detectable face**. Identity must match `[A-Za-z0-9_-]{1,64}` — no
spaces or path separators.

Returns:
```json
{
  "status": "added",       // or "replaced" if overwrite=true and existed
  "identity": "tom_hanks",
  "filename": "tom_hanks.png",
  "size_bytes": 87012,      // post-normalize PNG size, not the upload size
  "blocklist_count": 12
}
```

Errors:
- `400` — undecodable image, no face detected, multiple faces, or invalid identity name
- `409` — identity already exists (use `overwrite=true` to replace)
- `503` — face filter / image normalizer unavailable (e.g. `safety` module or Pillow not installed)

### GET /admin/blocklist — List

```bash
curl "$POD/admin/blocklist"
```

```json
{
  "count": 2,
  "blocklist": [
    {"identity": "tom_hanks", "filename": "tom_hanks.png",
     "size_bytes": 87012, "added_at": "2026-05-14T07:30:00+00:00"},
    {"identity": "celebrity_42", "filename": "celebrity_42.png",
     "size_bytes": 102488, "added_at": "2026-05-14T07:35:00+00:00"}
  ]
}
```

> Existing entries uploaded before the auto-normalize change may still have
> `.jpg` / `.jpeg` / `.webp` extensions — those keep working, but any new
> upload (or `overwrite=true` replace) re-saves as `.png`.

### DELETE /admin/blocklist/{identity} — Remove

```bash
curl -X DELETE "$POD/admin/blocklist/tom_hanks" \
 
```

```json
{"status": "deleted", "identity": "tom_hanks",
 "filename": "tom_hanks.png", "blocklist_count": 1}
```

### GET /admin/blocklist/{identity}/image — Preview

Returns the stored face image as raw bytes (for CMS preview).

```bash
curl "$POD/admin/blocklist/tom_hanks/image" \
  -o tom_hanks.png
```

### Logos / flags — `/admin/blocklist-logos`

Same shape as the face endpoints, different storage (`/workspace/blocklist_logos/`)
and different validation (no face-detection prereq — any valid image is accepted).

```bash
# Upload
curl -X POST "$POD/admin/blocklist-logos" \
  -F "identity=apple_logo" \
  -F "image=@apple.png"

# List
curl "$POD/admin/blocklist-logos"

# Delete
curl -X DELETE "$POD/admin/blocklist-logos/apple_logo" \
 

# Preview
curl "$POD/admin/blocklist-logos/apple_logo/image" \
  -o apple.png
```

Response shape mirrors `/admin/blocklist`. The list response groups blocked
logos by their `identity` (filename stem) — the same name returned in
`matched_logo` on a block.

**Tip:** tight crops give best CLIP discrimination. A blocklist image that
fills the frame with the logo/flag scores ~0.9+ against itself; if the
logo is small in the corner of your blocklist image, CLIP will embed the
background's content instead and miss real-world matches.

---

## Error Responses

All errors return standard HTTP status codes with a JSON body.

```json
{ "detail": "error message here" }
```

| Code | Meaning |
|------|---------|
| `400` | Bad request (missing field, invalid preset/aspect_ratio, etc.) |
| `404` | Job or file not found |
| `422` | Validation error (wrong type for a parameter) |
| `500` | Internal server error |
