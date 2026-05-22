"""
Shared ComfyUI workflow builders for AI Gen API v2.

Imported by both:
  - main.py     (pod-mode FastAPI on :7860)
  - serverless/image/handler.py  (RunPod serverless worker)
  - serverless/video/handler.py  (RunPod serverless worker)

Keep this file dependency-light: stdlib + Pillow only. No FastAPI, no runpod SDK,
no httpx — those are imported by the callers.
"""

import io
import math
from PIL import Image


# ─────────────────────────────────────────────
# FLUX.2 Klein 9B — shared helpers
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
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    elif src_ratio < target_ratio:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    img = img.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────
# FLUX.2 Klein 9B — Text to Image
# ─────────────────────────────────────────────

def build_t2i_workflow(prompt: str, width: int, height: int, seed: int,
                       steps: int = 4, cfg: float = 1.0, guidance: float = 4.0) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-9b.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2", "device": "default"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": guidance}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "7": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["2", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": f"images/t2i_{seed}"}},
    }


# ─────────────────────────────────────────────
# FLUX.2 Klein 9B — Head/Face Swap
# ─────────────────────────────────────────────

DEFAULT_FLUX_PROMPT = """head_swap: Use image 1 as the base image, preserving its environment, background, camera perspective, framing, exposure, contrast, and lighting. Remove the head and hair from image 1 and seamlessly replace it with the head from image 2.
Match the original head size, face-to-body ratio, neck thickness, shoulder alignment, and camera distance so proportions remain natural and unchanged.
Adapt the inserted head to the lighting of image 1 by matching light direction, intensity, softness, color temperature, shadows, and highlights, with no independent relighting.
Preserve the identity of image 2, including hair texture, eye color, nose structure, facial proportions, and skin details.
Match the pose and expression from image 1, including head tilt, rotation, eye direction, gaze, micro-expressions, and lip position.
Ensure seamless neck and jaw blending, consistent skin tone, realistic shadow contact, natural skin texture, and uniform sharpness.
Photorealistic, high quality, sharp details, 4K."""


def get_flux_face_swap_workflow(target_filename: str, face_filename: str, seed: int,
                                prompt: str = None, megapixels: float = 2.0,
                                steps: int = 4, cfg: float = 1.0, guidance: float = 4.0,
                                lora_strength: float = 1.0) -> dict:
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
# FLUX.2 Klein 9B — Image to Image (multi-reference editing)
# ─────────────────────────────────────────────

DEFAULT_I2I_PROMPT = (
    "edit the image faithfully according to the instructions, preserving "
    "lighting, perspective, and identity where not explicitly changed; "
    "photorealistic, sharp details, 4K."
)


def build_flux_i2i_workflow(image_filenames: list, prompt: str, seed: int,
                             megapixels: float = 2.0,
                             output_width: int = 0, output_height: int = 0,
                             steps: int = 4, cfg: float = 1.0, guidance: float = 4.0,
                             lora_strength: float = 0.0) -> dict:
    """Build an N-image FLUX.2 reference workflow (1 <= N <= 5).

    All input images are encoded to latents and chained as ReferenceLatents
    on top of the prompt's conditioning. The prompt drives the edit; the
    images supply style, identity, objects, composition cues.

    Output dimensions:
      - If output_width AND output_height are both > 0, use those directly.
      - Otherwise, derive from the FIRST image (encode → decode → GetImageSize)
        after the megapixels rescale, so the result matches the canvas the
        caller probably has in mind.

    `lora_strength` activates the head-swap LoRA when > 0 (use for face-related
    edits). Set 0 (default) for general edits.
    """
    if not image_filenames or len(image_filenames) > 5:
        raise ValueError(f"build_flux_i2i_workflow needs 1-5 images, got {len(image_filenames)}")
    if not prompt:
        prompt = DEFAULT_I2I_PROMPT

    nodes = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-9b.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "VAELoader",  "inputs": {"vae_name":  "flux2-vae.safetensors"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2", "device": "default"}},
    }

    if lora_strength > 0:
        nodes["4"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0],
            "lora_name": "bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors",
            "strength_model": lora_strength,
        }}
        model_ref = ["4", 0]
    else:
        model_ref = ["1", 0]

    # Prompt conditioning
    nodes["10"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}}

    # Load + scale + encode each image; chain ReferenceLatents.
    cond_chain = ["10", 0]
    for i, fname in enumerate(image_filenames):
        load_id  = f"20{i}"
        scale_id = f"21{i}"
        enc_id   = f"22{i}"
        ref_id   = f"23{i}"
        nodes[load_id]  = {"class_type": "LoadImage",                "inputs": {"image": fname}}
        nodes[scale_id] = {"class_type": "ImageScaleToTotalPixels",  "inputs": {
            "image": [load_id, 0], "upscale_method": "lanczos",
            "megapixels": megapixels, "resolution_steps": 1
        }}
        nodes[enc_id]   = {"class_type": "VAEEncode", "inputs": {"pixels": [scale_id, 0], "vae": ["2", 0]}}
        nodes[ref_id]   = {"class_type": "ReferenceLatent", "inputs": {"conditioning": cond_chain, "latent": [enc_id, 0]}}
        cond_chain = [ref_id, 0]

    nodes["30"] = {"class_type": "FluxGuidance",        "inputs": {"conditioning": cond_chain, "guidance": guidance}}
    nodes["31"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["10", 0]}}

    # Pick canvas dimensions
    if output_width > 0 and output_height > 0:
        nodes["40"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {
            "width": int(output_width), "height": int(output_height), "batch_size": 1
        }}
    else:
        # Derive from the first (rescaled) image via decode+GetImageSize
        nodes["41"] = {"class_type": "VAEDecode",    "inputs": {"samples": ["220", 0], "vae": ["2", 0]}}
        nodes["42"] = {"class_type": "GetImageSize", "inputs": {"image": ["41", 0]}}
        nodes["40"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {
            "width": ["42", 0], "height": ["42", 1], "batch_size": 1
        }}

    nodes["50"] = {"class_type": "KSampler", "inputs": {
        "model": model_ref, "positive": ["30", 0], "negative": ["31", 0],
        "latent_image": ["40", 0], "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
    }}
    nodes["60"] = {"class_type": "VAEDecode", "inputs": {"samples": ["50", 0], "vae": ["2", 0]}}
    nodes["70"] = {"class_type": "SaveImage", "inputs": {
        "images": ["60", 0], "filename_prefix": f"images/flux_i2i_{seed}"
    }}

    return nodes


# ─────────────────────────────────────────────
# LTX-2.3 — shared helpers
# ─────────────────────────────────────────────

LTX_ASPECT_RATIOS = {
    "1:1":  (1, 1),  "4:3":  (4, 3),  "3:4":  (3, 4),
    "16:9": (16, 9), "9:16": (9, 16), "3:2":  (3, 2),
    "2:3":  (2, 3),  "21:9": (21, 9), "9:21": (9, 21),
}

LTX_DEFAULT_NEGATIVE = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"

_LTX_DISTILLED_LOW_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

LTX_PRESETS = {
    "fast": {
        "sigmas": _LTX_DISTILLED_LOW_SIGMAS,
        "lora_strength": 0.5,
        "two_pass": False,
    },
    "quality": {
        "low_res_sigmas": _LTX_DISTILLED_LOW_SIGMAS,
        "high_res_sigmas": "0.85, 0.7250, 0.4219, 0.0",
        "lora_strength": 0.5,
        "two_pass": True,
    },
}


def compute_ltx_dimensions(width: int, height: int, aspect_ratio: str) -> tuple[int, int]:
    """Return (width, height) snapped to multiples of 32. If aspect_ratio given, derive height from width."""
    if aspect_ratio in LTX_ASPECT_RATIOS:
        w_r, h_r = LTX_ASPECT_RATIOS[aspect_ratio]
        height = round(width * h_r / w_r / 32) * 32
    width  = max(32, round(width  / 32) * 32)
    height = max(32, round(height / 32) * 32)
    return width, height


def ltx_base_nodes(prompt, negative_prompt, width, height, length, fps, seed,
                   low_res_video_src, high_res_video_src, prefix,
                   preset: str = "fast", audio: bool = False) -> dict:
    """Return the shared LTX workflow nodes.

    fast preset: single pass at full resolution — fast, no upscale overhead.
    quality preset: two-pass (half-res → upscale → refine at full-res) — slower, sharper.
    audio: if True, generate audio track with the video (adds ~5s overhead).
    """
    p = LTX_PRESETS.get(preset, LTX_PRESETS["fast"])

    nodes = {
        "236": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}},
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
        "240": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": prompt}},
        "247": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["272", 1], "text": negative_prompt}},
        "239": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["240", 0], "negative": ["247", 0], "frame_rate": float(fps)
        }},
    }

    if audio:
        nodes["221"] = {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}}
        nodes["214"] = {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": length, "frame_rate": fps, "batch_size": 1, "audio_vae": ["221", 0]
        }}

    if p["two_pass"]:
        half_w = max(32, (width // 2 // 32) * 32)
        half_h = max(32, (height // 2 // 32) * 32)

        nodes["228"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": half_w, "height": half_h, "length": length, "batch_size": 1
        }}
        nodes["233"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"}}

        if audio:
            nodes["222"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": low_res_video_src, "audio_latent": ["214", 0]}}
            sample_input = ["222", 0]
        else:
            sample_input = low_res_video_src

        nodes.update({
            "231": {"class_type": "CFGGuider",              "inputs": {"model": ["232", 0], "positive": ["239", 0], "negative": ["239", 1], "cfg": 1.0}},
            "209": {"class_type": "KSamplerSelect",         "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
            "237": {"class_type": "RandomNoise",            "inputs": {"noise_seed": seed}},
            "252": {"class_type": "ManualSigmas",           "inputs": {"sigmas": p["low_res_sigmas"]}},
            "215": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["237", 0], "guider": ["231", 0], "sampler": ["209", 0],
                "sigmas": ["252", 0], "latent_image": sample_input
            }},
        })

        if audio:
            nodes["217"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["215", 0]}}
            low_video_out = ["217", 0]
            low_audio_out = ["217", 1]
        else:
            low_video_out = ["215", 0]

        nodes["253"] = {"class_type": "LTXVLatentUpsampler", "inputs": {
            "samples": low_video_out, "upscale_model": ["233", 0], "vae": ["236", 2]
        }}

        nodes["212"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": ["239", 0], "negative": ["239", 1], "latent": low_video_out}}

        if audio:
            nodes["229"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": high_res_video_src, "audio_latent": low_audio_out}}
            hi_sample_input = ["229", 0]
        else:
            hi_sample_input = high_res_video_src

        nodes.update({
            "213": {"class_type": "CFGGuider",              "inputs": {"model": ["232", 0], "positive": ["212", 0], "negative": ["212", 1], "cfg": 1.0}},
            "246": {"class_type": "KSamplerSelect",         "inputs": {"sampler_name": "euler_cfg_pp"}},
            "216": {"class_type": "RandomNoise",            "inputs": {"noise_seed": (seed + 1) % 2**32}},
            "211": {"class_type": "ManualSigmas",           "inputs": {"sigmas": p["high_res_sigmas"]}},
            "219": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["216", 0], "guider": ["213", 0], "sampler": ["246", 0],
                "sigmas": ["211", 0], "latent_image": hi_sample_input
            }},
        })

        if audio:
            nodes["218"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["219", 0]}}
            hi_video_out = ["218", 0]
            hi_audio_out = ["218", 1]
        else:
            hi_video_out = ["219", 0]

        nodes["251"] = {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": hi_video_out, "vae": ["236", 2],
            "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4
        }}
        if audio:
            nodes["220"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": hi_audio_out, "audio_vae": ["221", 0]}}

    else:
        nodes["228"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": width, "height": height, "length": length, "batch_size": 1
        }}

        if audio:
            nodes["222"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": low_res_video_src, "audio_latent": ["214", 0]}}
            sample_input = ["222", 0]
        else:
            sample_input = low_res_video_src

        nodes.update({
            "231": {"class_type": "CFGGuider",              "inputs": {"model": ["232", 0], "positive": ["239", 0], "negative": ["239", 1], "cfg": 1.0}},
            "209": {"class_type": "KSamplerSelect",         "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
            "237": {"class_type": "RandomNoise",            "inputs": {"noise_seed": seed}},
            "252": {"class_type": "ManualSigmas",           "inputs": {"sigmas": p["sigmas"]}},
            "215": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["237", 0], "guider": ["231", 0], "sampler": ["209", 0],
                "sigmas": ["252", 0], "latent_image": sample_input
            }},
        })

        if audio:
            nodes["217"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["215", 0]}}
            video_out = ["217", 0]
            audio_out = ["217", 1]
        else:
            video_out = ["215", 0]

        nodes["251"] = {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": video_out, "vae": ["236", 2],
            "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4
        }}
        if audio:
            nodes["220"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": audio_out, "audio_vae": ["221", 0]}}

    create_video_inputs = {"images": ["251", 0], "fps": float(fps)}
    if audio:
        create_video_inputs["audio"] = ["220", 0]
    nodes["242"] = {"class_type": "CreateVideo", "inputs": create_video_inputs}
    nodes["75"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["242", 0], "filename_prefix": f"video/{prefix}_{seed}", "format": "auto", "codec": "auto"
    }}

    return nodes


# ─────────────────────────────────────────────
# LTX-2.3 — Image to Video (workflow assembly)
# ─────────────────────────────────────────────

def build_ltx_i2v_workflow(image_filename: str, prompt: str, negative_prompt: str,
                            width: int, height: int, length: int, fps: int, seed: int,
                            preset: str = "fast", audio: bool = False,
                            enhance_prompt: bool = True,
                            inplace_strength: float = 0.7) -> dict:
    """Build an LTX 2.3 image-to-video workflow. `image_filename` must already exist in ComfyUI's input dir.

    `inplace_strength` controls how tightly each generated frame's latent is pinned to the input
    image. Reference distilled value is 0.7 (first pass) / 1.0 (two-pass refine), which preserves
    identity but suppresses motion. Lower it for action prompts where the subject must change pose:
    0.5 ≈ moderate motion, 0.4 ≈ strong motion (some identity drift), 0.3 ≈ near-t2v behavior.
    The two-pass refine strength tracks the first pass: refine = min(1.0, inplace_strength + 0.3).
    """
    two_pass = LTX_PRESETS[preset]["two_pass"]
    refine_strength = min(1.0, inplace_strength + 0.3)

    img_nodes = {
        "269": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "238": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0], "resize_type": "scale dimensions",
            "resize_type.width": width, "resize_type.height": height,
            "resize_type.crop": "center", "scale_method": "lanczos"
        }},
        "235": {"class_type": "ResizeImagesByLongerEdge", "inputs": {"images": ["238", 0], "longer_edge": 1536}},
        "248": {"class_type": "LTXVPreprocess",           "inputs": {"image": ["235", 0], "img_compression": 18}},
        "249": {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["236", 2], "image": ["248", 0], "latent": ["228", 0],
            "strength": inplace_strength, "bypass": False
        }},
    }

    if enhance_prompt:
        img_nodes["274"] = {"class_type": "TextGenerateLTX2Prompt", "inputs": {
            "clip": ["272", 1], "image": ["269", 0], "prompt": prompt,
            "max_length": 256, "sampling_mode": "on",
            "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
            "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
            "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": seed
        }}
        img_nodes["240"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": ["274", 0]}}

    if two_pass:
        img_nodes["230"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["236", 2], "image": ["248", 0], "latent": ["253", 0], "strength": refine_strength, "bypass": False
        }}
        high_res_src = ["230", 0]
    else:
        high_res_src = None

    workflow = ltx_base_nodes(
        prompt, negative_prompt, width, height, length, fps, seed,
        low_res_video_src=["249", 0], high_res_video_src=high_res_src,
        prefix="ltx_i2v", preset=preset, audio=audio
    )
    workflow.update(img_nodes)

    # Color-match each generated frame back to the input image. LTX 2.3 (esp. fp8)
    # has a warm/saturated drift through VAE encode→sample→decode that t2v doesn't
    # share, because t2v has no reference to drift away from. Measured on a typical
    # i2v: red channel +12-18 vs input, saturation +15% — visible as an orange tint.
    # MKL (Monge-Kantorovich linearization) matches channel covariance and closes
    # the gap to within ~2 units per channel at ~+0.7s cost. ColorMatch ships with
    # ComfyUI-KJNodes which setup.sh already installs.
    workflow["280"] = {"class_type": "ColorMatch", "inputs": {
        "image_ref":    ["269", 0],
        "image_target": ["251", 0],
        "method":       "mkl",
        "strength":     1.0,
    }}
    workflow["242"]["inputs"]["images"] = ["280", 0]
    return workflow


# ─────────────────────────────────────────────
# LTX-2.3 — Text to Video (workflow assembly)
# ─────────────────────────────────────────────

def build_ltx_t2v_workflow(prompt: str, negative_prompt: str,
                            width: int, height: int, length: int, fps: int, seed: int,
                            preset: str = "fast", audio: bool = False) -> dict:
    """Build an LTX 2.3 text-to-video workflow."""
    return ltx_base_nodes(
        prompt, negative_prompt, width, height, length, fps, seed,
        low_res_video_src=["228", 0],
        high_res_video_src=["253", 0],
        prefix="ltx_t2v", preset=preset, audio=audio
    )


# ─────────────────────────────────────────────
# LTX-2.3 — Motion Control (workflow assembly)
#
# Kling-style motion transfer: take an image (identity) + a reference
# video (motion source) and produce a new video where the image animates
# along the motion structure of the reference. Implementation strategy:
#
#   1. VHS_LoadVideo decodes the reference clip into a frame tensor.
#   2. The same ResizeImageMaskNode + LTXVPreprocess chain that i2v uses
#      preps the character image at the target resolution.
#   3. VAEEncode (with the LTX VAE) encodes the reference frame batch into
#      a motion latent — same shape as EmptyLTXVLatentVideo would produce,
#      but with the reference's motion baked into the noise space instead
#      of pure Gaussian. The sampler then preserves that motion structure
#      while denoising toward the conditioning prompt + image.
#   4. LTXVImgToVideoInplace mixes the character image latent in at the
#      configured `inplace_strength` — high values stick to the identity
#      hard (preserves face but flatter motion); low values let motion
#      dominate (better dance fidelity but identity drift).
#   5. Standard LTX two-pass / single-pass sampler from ltx_base_nodes.
#
# Reference video preprocessing happens server-side (in main.py) before
# we reach this builder — we trim + resample to fit LTX's `length` cap
# (typically 97 frames) and downscale to the target resolution.
# ─────────────────────────────────────────────

def build_ltx_motion_workflow_no_vhs(reference_frame_filenames: list[str],
                                     character_image_filename: str,
                                     prompt: str, negative_prompt: str,
                                     width: int, height: int, length: int, fps: int, seed: int,
                                     preset: str = "fast", audio: bool = False,
                                     enhance_prompt: bool = True,
                                     inplace_strength: float = 0.5,
                                     motion_strength: float = 1.0) -> dict:
    """Fallback motion-control workflow that doesn't need ComfyUI-VideoHelperSuite.

    Where the VHS variant uses one `VHS_LoadVideo` node to read the whole
    reference clip in one shot, this version takes a list of per-frame
    PNG filenames (already extracted by main.py via ffmpeg into ComfyUI's
    input dir) and stitches them into a single image tensor using a
    LoadImage + ImageBatch chain — both of which are stock ComfyUI core
    nodes, so no custom-node install is required.

    Graph shape for the frame loader:
        LoadImage[f01] ┐
                      ImageBatch ┐
        LoadImage[f02] ┘         │
                                 ImageBatch ┐
        LoadImage[f03] ───────────┘         │
                                            ImageBatch ─► VAEEncode ─►
        LoadImage[f04] ─────────────────────┘            (motion latent)

    Linear chain rather than balanced tree — ComfyUI executes nodes
    bottom-up so depth doesn't really matter, and a chain keeps node
    IDs easy to reason about. Each ImageBatch combines a running
    accumulator with the next single-frame load.

    Performance vs VHS:
      - Workflow JSON is bigger (≈2N nodes vs 1)
      - Each LoadImage is a separate file open — a bit slower than
        VHS's bulk read, but still <2s total for 121 frames on local SSD
      - VAE encode + sampler stages are identical to the VHS path
    """
    if len(reference_frame_filenames) < 2:
        raise ValueError(
            f"motion workflow needs at least 2 reference frames; got {len(reference_frame_filenames)}"
        )

    two_pass = LTX_PRESETS[preset]["two_pass"]
    refine_strength = min(1.0, inplace_strength + 0.3)

    # Node-ID allocation. We reserve the same IDs as the VHS variant for
    # the shared bits (character image chain at 269/238/235/248, the
    # motion-strength multiplier at 313, the image-into-video mixer at
    # 249) so the rest of ltx_base_nodes wiring lines up identically.
    # Per-frame loaders use IDs 1000..1000+N and batchers 2000..2000+N
    # to stay out of the way of ltx_base_nodes' allocations.
    LOAD_BASE = 1000
    BATCH_BASE = 2000

    img_nodes: dict = {
        # Character image — identical chain to i2v / VHS variant.
        "269": {"class_type": "LoadImage", "inputs": {"image": character_image_filename}},
        "238": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0], "resize_type": "scale dimensions",
            "resize_type.width": width, "resize_type.height": height,
            "resize_type.crop": "center", "scale_method": "lanczos"
        }},
        "235": {"class_type": "ResizeImagesByLongerEdge", "inputs": {"images": ["238", 0], "longer_edge": 1536}},
        "248": {"class_type": "LTXVPreprocess", "inputs": {"image": ["235", 0], "img_compression": 18}},
    }

    # Per-frame LoadImage nodes. ComfyUI's LoadImage takes a filename
    # that's already been written into its input dir.
    for i, fn in enumerate(reference_frame_filenames):
        img_nodes[str(LOAD_BASE + i)] = {"class_type": "LoadImage", "inputs": {"image": fn}}

    # Each frame needs to be resized to the LTX canvas before batching —
    # the VAE expects all batched frames at the same dimensions. We
    # reuse the same ResizeImageMaskNode helper the character image
    # uses, just per-frame.
    RESIZE_BASE = LOAD_BASE + len(reference_frame_filenames)  # avoid collisions
    for i in range(len(reference_frame_filenames)):
        img_nodes[str(RESIZE_BASE + i)] = {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": [str(LOAD_BASE + i), 0],
            "resize_type": "scale dimensions",
            "resize_type.width": width, "resize_type.height": height,
            "resize_type.crop": "center", "scale_method": "lanczos"
        }}

    # Linear ImageBatch chain — combine resized[0] + resized[1], then
    # accumulate one frame at a time.
    img_nodes[str(BATCH_BASE)] = {"class_type": "ImageBatch", "inputs": {
        "image1": [str(RESIZE_BASE + 0), 0],
        "image2": [str(RESIZE_BASE + 1), 0],
    }}
    for i in range(2, len(reference_frame_filenames)):
        img_nodes[str(BATCH_BASE + i - 1)] = {"class_type": "ImageBatch", "inputs": {
            "image1": [str(BATCH_BASE + i - 2), 0],
            "image2": [str(RESIZE_BASE + i), 0],
        }}
    final_batch_id = str(BATCH_BASE + len(reference_frame_filenames) - 2)

    # VAE-encode the batched frames into a motion latent. Same shape as
    # what VHS would produce if it had loaded the video.
    img_nodes["312"] = {"class_type": "VAEEncode", "inputs": {
        "pixels": [final_batch_id, 0], "vae": ["236", 2],
    }}
    # Motion-strength multiplier (same as VHS path).
    img_nodes["313"] = {"class_type": "LatentMultiply", "inputs": {
        "samples": ["312", 0], "multiplier": motion_strength,
    }}
    # Mix character identity into the motion latent.
    img_nodes["249"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
        "vae": ["236", 2], "image": ["248", 0], "latent": ["313", 0],
        "strength": inplace_strength, "bypass": False,
    }}

    if enhance_prompt:
        img_nodes["274"] = {"class_type": "TextGenerateLTX2Prompt", "inputs": {
            "clip": ["272", 1], "image": ["269", 0], "prompt": prompt,
            "max_length": 256, "sampling_mode": "on",
            "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
            "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
            "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": seed,
        }}
        img_nodes["240"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": ["274", 0]}}

    if two_pass:
        img_nodes["230"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["236", 2], "image": ["248", 0], "latent": ["253", 0],
            "strength": refine_strength, "bypass": False,
        }}
        high_res_src = ["230", 0]
    else:
        high_res_src = None

    workflow = ltx_base_nodes(
        prompt, negative_prompt, width, height, length, fps, seed,
        low_res_video_src=["249", 0], high_res_video_src=high_res_src,
        prefix="ltx_motion", preset=preset, audio=audio,
    )
    workflow.update(img_nodes)

    workflow["280"] = {"class_type": "ColorMatch", "inputs": {
        "image_ref": ["269", 0],
        "image_target": ["251", 0],
        "method": "mkl",
        "strength": 1.0,
    }}
    workflow["242"]["inputs"]["images"] = ["280", 0]
    return workflow


def build_ltx_motion_workflow(reference_video_filename: str,
                              character_image_filename: str,
                              prompt: str, negative_prompt: str,
                              width: int, height: int, length: int, fps: int, seed: int,
                              preset: str = "fast", audio: bool = False,
                              enhance_prompt: bool = True,
                              inplace_strength: float = 0.5,
                              motion_strength: float = 0.95) -> dict:
    """Build the LTX 2.3 motion-control workflow.

    Uses the LTXVAddGuide node — the proper motion-conditioning primitive
    from ComfyUI-LTXVideo's `comfy_extras.nodes_lt`. The previous version
    passed a VAE-encoded reference into LTXVImgToVideoInplace's `latent`
    input, but that node is i2v-only (it overwrites the latent with the
    image encoding), so the reference motion was being discarded — the
    output looked like i2v of the character image rather than motion
    transfer.

    LTXVAddGuide actually injects the reference video into the conditioning
    AND the latent at a specific frame_idx with controllable strength. Two
    guides are stacked:

      1. Character image at frame_idx=0, strength=inplace_strength  →
         anchors WHO the output looks like (identity at the start).
      2. Reference video starting at frame_idx=0, strength=motion_strength →
         drives WHAT the output does (the actual motion to copy).

    The reference video must satisfy LTX's `8*n + 1` frame constraint —
    97, 121, 161, 257 all qualify so the default `length` values are
    safe. main.py's ffmpeg pre-step already enforces this via -frames:v.

    Knob meanings (renamed roles vs the old version):
      inplace_strength  — identity guide strength. 0.4 = weak identity
                          (motion dominates, face may drift), 0.5–0.7 =
                          balanced, 0.9 = strong identity (face stays put
                          but motion looks subdued).
      motion_strength   — reference-video guide strength. 0.0–1.0 (LTX
                          caps it at 1). 0.95 = strong copy of reference
                          motion (default). Lower it to blend the
                          character's natural motion with the reference's.
    """
    two_pass = LTX_PRESETS[preset]["two_pass"]
    refine_strength = min(1.0, inplace_strength + 0.3)
    # LTXVAddGuide enforces strength in [0, 1] — clamp early so a
    # passed-in motion_strength > 1 (legacy callers tuned for the broken
    # LatentMultiply path) doesn't 400 the request.
    motion_strength = max(0.0, min(1.0, motion_strength))

    img_nodes = {
        # Character image chain — identical to i2v's first-frame setup.
        # Acts as the identity guide via the first LTXVAddGuide call below.
        "269": {"class_type": "LoadImage", "inputs": {"image": character_image_filename}},
        "238": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0], "resize_type": "scale dimensions",
            "resize_type.width": width, "resize_type.height": height,
            "resize_type.crop": "center", "scale_method": "lanczos",
        }},
        "235": {"class_type": "ResizeImagesByLongerEdge", "inputs": {"images": ["238", 0], "longer_edge": 1536}},
        "248": {"class_type": "LTXVPreprocess", "inputs": {"image": ["235", 0], "img_compression": 18}},

        # Reference video chain — load with VHS, resize to canvas. The
        # IMAGE tensor goes directly into LTXVAddGuide as a multi-frame
        # guide; we don't VAE-encode it ourselves (LTXVAddGuide does that
        # internally as part of its guide-injection logic, which gets the
        # encoding right for the conditioning slot).
        "310": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": reference_video_filename,
            "force_rate": float(fps),
            "force_size": "Disabled",
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": length,
            "skip_first_frames": 0,
            "select_every_nth": 1,
        }},
        "311": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["310", 0], "resize_type": "scale dimensions",
            "resize_type.width": width, "resize_type.height": height,
            "resize_type.crop": "center", "scale_method": "lanczos",
        }},
        "315": {"class_type": "LTXVPreprocess", "inputs": {"image": ["311", 0], "img_compression": 18}},
    }

    if enhance_prompt:
        # Gemma-driven prompt enhancement — uses the character image as
        # visual context. With proper motion guidance now in place, the
        # prompt can stay simple ("the woman dances") and the reference
        # video tells the model HOW to dance.
        img_nodes["274"] = {"class_type": "TextGenerateLTX2Prompt", "inputs": {
            "clip": ["272", 1], "image": ["269", 0], "prompt": prompt,
            "max_length": 256, "sampling_mode": "on",
            "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
            "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
            "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": seed,
        }}
        img_nodes["240"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": ["274", 0]}}

    workflow = ltx_base_nodes(
        prompt, negative_prompt, width, height, length, fps, seed,
        # Sentinel that gets rewritten below — see "high_res_src" + the
        # guide chain. We need ltx_base_nodes to wire its sampler chain
        # through OUR guide-conditioned positive/negative/latent rather
        # than through 239 (the un-guided LTXVConditioning output it
        # would normally pick). The simplest way is to let ltx_base_nodes
        # set up the empty latent at "228", then OVERWRITE the sampler's
        # inputs (CFGGuider 231, latent_image on 215) downstream of the
        # guide chain. We do that after `workflow.update`.
        low_res_video_src=["228", 0],
        high_res_video_src=None,
        prefix="ltx_motion", preset=preset, audio=audio,
    )
    workflow.update(img_nodes)

    # Stack the two LTXVAddGuide calls — ORDER MATTERS at overlapping
    # frame_idx values: LTXVAddGuide writes guide-encoded latent slices
    # INTO the input latent at frame_idx with the given strength. When two
    # guides share frame_idx=0, the second one stacked wins for that frame.
    # Run reference video FIRST (drives motion across frames 0..N), then
    # the character image SECOND (overwrites frame 0 with identity).
    # That gives the sampler: identity anchor at frame 0 + reference motion
    # propagating through frames 1..N as the diffusion process unfolds.
    workflow["330"] = {"class_type": "LTXVAddGuide", "inputs": {
        "positive": ["239", 0],
        "negative": ["239", 1],
        "vae": ["236", 2],
        "latent": ["228", 0],
        "image": ["315", 0],
        "frame_idx": 0,
        "strength": motion_strength,
    }}
    workflow["331"] = {"class_type": "LTXVAddGuide", "inputs": {
        "positive": ["330", 0],
        "negative": ["330", 1],
        "vae": ["236", 2],
        "latent": ["330", 2],
        "image": ["248", 0],
        "frame_idx": 0,
        "strength": inplace_strength,
    }}

    # Rewire the sampler chain that ltx_base_nodes built so it consumes
    # the guide outputs. CFGGuider (231) takes positive/negative from the
    # final guide; SamplerCustomAdvanced (215) takes the guide latent as
    # its starting point.
    if "231" in workflow:
        workflow["231"]["inputs"]["positive"] = ["331", 0]
        workflow["231"]["inputs"]["negative"] = ["331", 1]
    if "215" in workflow:
        workflow["215"]["inputs"]["latent_image"] = ["331", 2]

    if two_pass:
        # Two-pass refine — re-apply guides into the upsampled latent.
        # Refine pass uses slightly higher identity strength (matches i2v's
        # refine_strength), motion stays the same.
        workflow["340"] = {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["239", 0],
            "negative": ["239", 1],
            "vae": ["236", 2],
            "latent": ["253", 0],
            "image": ["248", 0],
            "frame_idx": 0,
            "strength": refine_strength,
        }}
        workflow["341"] = {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["340", 0],
            "negative": ["340", 1],
            "vae": ["236", 2],
            "latent": ["340", 2],
            "image": ["315", 0],
            "frame_idx": 0,
            "strength": motion_strength,
        }}
        # ltx_base_nodes' refine sampler nodes are conventionally at 218/220.
        # Wire them to use the second-stage guide outputs.
        if "220" in workflow:
            workflow["220"]["inputs"]["latent_image"] = ["341", 2]
        # The refine CFGGuider — if ltx_base_nodes named it 219 or wired
        # 218's guider in-line, patch defensively.
        for gid in ("219", "218"):
            if gid in workflow and isinstance(workflow[gid].get("inputs", {}), dict):
                if "positive" in workflow[gid]["inputs"]:
                    workflow[gid]["inputs"]["positive"] = ["341", 0]
                if "negative" in workflow[gid]["inputs"]:
                    workflow[gid]["inputs"]["negative"] = ["341", 1]

    # ColorMatch the output against the character image — same as i2v.
    # Reduces the warm/saturated drift that fp8 VAE round-trips produce.
    workflow["280"] = {"class_type": "ColorMatch", "inputs": {
        "image_ref": ["269", 0],
        "image_target": ["251", 0],
        "method": "mkl",
        "strength": 1.0,
    }}
    workflow["242"]["inputs"]["images"] = ["280", 0]
    return workflow
