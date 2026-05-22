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
                              inplace_strength: float = 1.0,
                              motion_strength: float = 1.0) -> dict:
    """Build the LTX 2.3 motion-control workflow — IC-LoRA Union-Control path.

    THIS IS THE THIRD REWRITE. The earlier two strategies both produced
    appearance-leaked output (character "wore" the reference person's
    clothes/face from frame 1 onward) because they both fed raw RGB
    pixels from the reference video into the conditioning latent:

      v1 — LTXVImgToVideoInplace with VAE-encoded ref latent.
           Identity-only node; motion silently discarded.
      v2 — LTXVAddGuide with raw ref RGB + character batch.
           Motion landed but appearance bled through. There's no
           strength balance that separates motion from appearance when
           the guide pixels ARE the reference person.

    The fix is structural, not a knob tweak: feed the model a CONTROL
    SIGNAL (pose skeleton on black background) instead of the raw RGB
    reference, and use Lightricks' Union-Control IC-LoRA — a LoRA
    adapter trained to interpret pose/depth/canny inputs and re-render
    them with the appearance from a separate character image. Result:
    motion comes from the skeleton, appearance comes from the character
    image, and the two never mix in the latent.

    Pipeline overview (mirrors Lightricks' official
    LTX-2.3_ICLoRA_Union_Control_Distilled.json with DWPose as the
    control signal — pose is the right choice for human dance/gesture
    transfer; depth/canny are better for whole-scene composition):

      reference_video → VHS_LoadVideo → resize (shorter=544)
                     → DWPreprocessor (yolox_l + dw-ll_ucoco_384)
                     → resize-to-multiple-of-32
                     → LTXAddVideoICLoRAGuide.image  (motion signal)

      character_image → LoadImage → resize
                     → LTXVImgToVideoConditionOnly (bypass=False)
                     → LTXAddVideoICLoRAGuide.latent  (identity)

      checkpoint → distilled-LoRA → Union-Control IC-LoRA
                → CFGGuider → SamplerCustomAdvanced

    Knob meanings under IC-LoRA — DIFFERENT from the LTXVAddGuide path:
      inplace_strength  — `strength` on LTXVImgToVideoConditionOnly.
                          1.0 = full character identity (recommended).
                          Lower → character less locked, more prompt-
                          driven appearance.
      motion_strength   — `strength` on LTXAddVideoICLoRAGuide. 1.0 =
                          full pose-driven motion (recommended). Lower
                          → looser interpretation of the skeleton.

    Notes vs. earlier callers:
      • `audio` is ignored (caller in main.py forces it False and muxes
        reference audio post-generation).
      • `preset` is ignored for now — IC-LoRA path is single-pass with
        the 8-step distilled sigmas. Two-pass refinement on top of
        IC-LoRA is non-trivial and not part of Lightricks' example.
      • `enhance_prompt` is ignored — Gemma rewriting the prompt based
        on character image alone tends to fight pose conditioning.
    """
    _ = preset, audio, enhance_prompt, motion_strength, inplace_strength  # acknowledged-but-restricted
    # Clamp strengths into [0,1] — the IC-LoRA guide enforces this and
    # so does LTXVImgToVideoConditionOnly.
    motion_strength = max(0.0, min(1.0, motion_strength))
    inplace_strength = max(0.0, min(1.0, inplace_strength))
    distilled_lora_strength = LTX_PRESETS["fast"]["lora_strength"]  # 0.5
    sigmas = _LTX_DISTILLED_LOW_SIGMAS

    # ─── Snap canvas dims to a multiple of 64 ─────────────────────
    # v32 reverted to the IC-LoRA Union-Control path which has
    # latent_downscale_factor=2.0 — meaning the latent spatial dims
    # must be divisible by 2, which means image dims must be divisible
    # by 32 (LTX latent stride) × 2 = 64. Without this we hit:
    # "Latent spatial size 17x30 must be divisible by
    # latent_downscale_factor 2.0" (17 = 544/32 is odd → fail).
    # Snap UP so the canvas never shrinks; 544×960 → 576×960
    # (still 9:16, latent 18×30 — both even).
    width = ((width + 63) // 64) * 64
    height = ((height + 63) // 64) * 64

    # ─── Length + fps: match the Lightricks Union-Control example ─
    # Reverted v22's "halve EmptyLTXVLatentVideo length" — that was
    # based on a misreading of `ref0.5` (the IC-LoRA name suffix
    # refers to SPATIAL reference scale only, not temporal; see
    # iclora.py in ComfyUI-LTXVideo where `latent_downscale_factor`
    # only affects the dilate width/height, not the time dim).
    #
    # The real cause of the mid-clip noise was an fps mismatch:
    # Lightricks' official Union-Control workflow uses fps=30
    # throughout (VHS source rate, LTXVConditioning frame_rate, and
    # CreateVideo output fps). Our v19-v24 used fps=24 from the user
    # request. LTXAddVideoICLoRAGuide maps pose frames to output
    # latent slots positionally — there is no fps-aware re-sampling.
    # With our 121-frame pose video at 24fps and the model's
    # "expected timeline" at 30fps, the pose conditioning landed in
    # only the first ~60% of output latent slots; the remaining
    # slots had no control signal → colored-noise free generation.
    #
    # Fix: force fps=30 for the motion workflow regardless of the
    # user's `fps` parameter. The output mp4 will be at 30fps —
    # slightly different feel from 24fps but coherent throughout.
    # ffmpeg in main.py is also overridden to extract at 30fps.
    pose_length = length
    # Force the IC-LoRA-required 30fps timeline. The user's `fps`
    # parameter is accepted but overridden here — the IC-LoRA was
    # trained at this rate and any deviation causes the conditioning
    # to land in wrong latent slots.
    fps = 30

    # ─── Resolve DWPose preprocessor input (resize) target ────────
    # DWPose works best around 512px. We resize the reference video so
    # the shorter dimension is the SHORTER of (canvas-width, canvas-
    # height) — that way pose tracking has resolution while staying
    # cheap. The output then gets resized to a multiple of 64 (same
    # grid as the canvas) so its encoded latent is even-divisible.
    dw_shorter = min(width, height)
    if dw_shorter < 384:
        dw_shorter = 384  # floor — below this DWPose loses confidence

    workflow: dict = {
        # ─── Checkpoint + LoRAs ────────────────────────────────────
        "236": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors",
        }},
        # Text-encoder loader (used for CLIPTextEncode below)
        "243": {"class_type": "LTXAVTextEncoderLoader", "inputs": {
            "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
            "ckpt_name":    "ltx-2.3-22b-dev-fp8.safetensors",
            "device": "default",
        }},
        # Distilled LoRA (matches base workflow).
        "232": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["236", 0],
            "lora_name": "ltx-2.3-22b-distilled-lora-384.safetensors",
            "strength_model": distilled_lora_strength,
        }},
        # IC-LoRA loader RESTORED for the model weights (v29 confirmed:
        # without it the model can only render whatever pixels are in
        # the conditioning image, so DWPose skeletons rendered as
        # skeletons). The IC-LoRA's training teaches the model to
        # TRANSFORM skeleton conditioning into a character render — we
        # need that knowledge baked into the unet weights. But we'll
        # NOT use the matching LTXAddVideoICLoRAGuide node downstream;
        # it's the one that halves temporal coverage. Standard
        # LTXVAddGuide (no halving) feeds the same DWPose skeleton,
        # and the IC-LoRA-trained model interprets it correctly.
        "262": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {
            "model": ["232", 0],
            "lora_name": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
            "strength_model": 1.0,
        }},

        # ─── Prompts ───────────────────────────────────────────────
        # Skipping Gemma TextGenerateLTX2Prompt — IC-LoRA pose driving
        # works best when the prompt is a literal description and the
        # control signal does the heavy lifting on motion.
        "240": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["243", 0], "text": prompt or "the subject performs the motion",
        }},
        "247": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["243", 0], "text": negative_prompt,
        }},
        "239": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["240", 0], "negative": ["247", 0], "frame_rate": float(fps),
        }},

        # ─── Character image (identity source) ────────────────────
        "269": {"class_type": "LoadImage", "inputs": {"image": character_image_filename}},
        "238": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0],
            "resize_type": "scale dimensions",
            "resize_type.width": width, "resize_type.height": height,
            "resize_type.crop": "center", "scale_method": "lanczos",
        }},
        # No RepeatImageBatch — v35 hit OOM trying to VAE-encode 257
        # character frames at once. v36 uses sparse single-frame
        # anchors (see nodes 331a-331e below) instead, each a single
        # character image at a different frame_idx. Tiny VAE memory
        # footprint, same identity-reinforcement effect.
        "228": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": width, "height": height, "length": length, "batch_size": 1,
        }},
        # LTXVImgToVideoConditionOnly — applies the character image as
        # the identity anchor at frame 0. bypass=False means we USE the
        # character (Lightricks' example has bypass=True; we always
        # want identity locked).
        "325": {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {
            "vae": ["236", 2],
            "image": ["238", 0],
            "latent": ["228", 0],
            "strength": inplace_strength,
            "bypass": False,
        }},

        # ─── Reference video → DWPose skeleton (motion control) ───
        # frame_load_cap uses pose_length (2*length+7 rounded to 8n+1)
        # so the pose conditioning covers LTX 2.3's 2× output length.
        # main.py's ffmpeg pre-step is updated to extract pose_length
        # frames (looping the source ref with -stream_loop -1 if it's
        # shorter than the target).
        "310": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": reference_video_filename,
            "force_rate": float(fps),
            "force_size": "Disabled",
            "custom_width": 0, "custom_height": 0,
            "frame_load_cap": pose_length,
            "skip_first_frames": 0,
            "select_every_nth": 1,
        }},
        "311": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["310", 0],
            "resize_type": "scale shorter dimension",
            "resize_type.shorter_size": dw_shorter,
            "scale_method": "lanczos",
        }},
        # DWPose preprocessor — produces a skeleton-on-black-background
        # video of the same length as the input. THIS is what kills the
        # appearance leak: the pixels going into the IC-LoRA guide have
        # zero reference-person appearance, only joint geometry.
        "320": {"class_type": "DWPreprocessor", "inputs": {
            "image": ["311", 0],
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": 512,
            "bbox_detector": "yolox_l.onnx",
            "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
            "scale_stick_for_xinsr_cn": "disable",
        }},
        # Resize skeleton to multiple of 64 (IC-LoRA Union-Control needs
        # latent dims divisible by latent_downscale_factor=2 → image
        # dims divisible by 32*2 = 64).
        "321": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["320", 0],
            "resize_type": "scale to multiple",
            "resize_type.multiple": 64,
            "scale_method": "lanczos",
        }},

        # ─── IC-LoRA guide (single, factor=2 from loader) ─────────
        # PRAGMATIC v32: revert workflow to v25 setup. After 30+
        # iterations we've shown the IC-LoRA guide consistently
        # produces clean Marco-dancing for the first ~50% of output
        # frames, then collapses to noise for the second half. None
        # of the workarounds (stacked guides → ghosting; standalone
        # LTXVAddGuide → literal skeleton; LTXVAddGuidesFromBatch →
        # 10-minute hang) deliver both temporal coverage AND
        # character rendering simultaneously.
        #
        # So accept the limitation and TRIM the output in main.py to
        # keep only the clean first half. User asking for length=121
        # (~5s) gets ~4s of cleanly-rendered Marco doing the motion.
        # That's a real, shippable result — better than chasing the
        # unbounded "fix the second half" spiral.
        "330": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": ["239", 0],
            "negative": ["239", 1],
            "vae": ["236", 2],
            "latent": ["325", 0],
            "image": ["321", 0],
            "frame_idx": 0,
            "strength": motion_strength,
            "latent_downscale_factor": ["262", 1],
            "crop": "disabled",
            "use_tiled_encode": False,
            "tile_size": 256,
            "tile_overlap": 64,
        }},

        # ─── Sparse character anchors for identity consistency ────
        # v34 shipped clean ~6s output but identity drifted — Marco at
        # frame 0, bearded stranger with headband at frame 5.
        # v35 tried a full-length RepeatImageBatch but OOM'd at 257
        # character frames VAE-encoded at once.
        # v36 stacks SINGLE-FRAME LTXVAddGuide calls at sparse frame_idx
        # positions across the clip. Each guide encodes just one image,
        # so VAE memory is trivial. The model sees the character image
        # re-asserted every ~32 latent frames, which fights the drift
        # without requiring multi-frame VAE encoding.
        #
        # frame_idx values: 1, 33, 65, 97 (all valid: 1 mod 8). 4
        # anchors cover up to latent slot ~12, which is the surviving
        # region after the 40% trim for length=121 (covers all of it)
        # and length=257 (covers 36% of raw = the trimmed region).
        #
        # strength=0.4 per anchor. Stacking is additive in conditioning
        # but last-wins for latent writes at the anchor frame, so the
        # character latent gets written at each anchor frame with full
        # strength 0.4 (replaces 40% of the IC-LoRA pose conditioning
        # at that one latent slot — enough to lock identity without
        # disrupting motion).
        "331a": {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["330", 0],
            "negative": ["330", 1],
            "vae": ["236", 2],
            "latent": ["330", 2],
            "image": ["238", 0],  # single character image
            "frame_idx": 1,
            "strength": 0.4,
        }},
        "331b": {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["331a", 0],
            "negative": ["331a", 1],
            "vae": ["236", 2],
            "latent": ["331a", 2],
            "image": ["238", 0],
            "frame_idx": 33,
            "strength": 0.4,
        }},
        "331c": {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["331b", 0],
            "negative": ["331b", 1],
            "vae": ["236", 2],
            "latent": ["331b", 2],
            "image": ["238", 0],
            "frame_idx": 65,
            "strength": 0.4,
        }},
        "331d": {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["331c", 0],
            "negative": ["331c", 1],
            "vae": ["236", 2],
            "latent": ["331c", 2],
            "image": ["238", 0],
            "frame_idx": 97,
            "strength": 0.4,
        }},

        # ─── Sampler chain ─────────────────────────────────────────
        # Model: ["262", 0] = distilled + IC-LoRA weights baked in. The
        # IC-LoRA's pose-to-character training is what enables the model
        # to interpret skeleton pixels as a character pose rather than
        # literally rendering colored bones. Positive/negative come from
        # the character-batch guide (node 331) which sits AFTER the
        # IC-LoRA's pose conditioning and adds identity reinforcement
        # across all output frames.
        "231": {"class_type": "CFGGuider", "inputs": {
            "model": ["262", 0],
            "positive": ["331d", 0],
            "negative": ["331d", 1],
            "cfg": 1.0,
        }},
        "209": {"class_type": "KSamplerSelect", "inputs": {
            "sampler_name": "euler_ancestral_cfg_pp",
        }},
        "237": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "252": {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas}},
        "215": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["237", 0],
            "guider": ["231", 0],
            "sampler": ["209", 0],
            "sigmas": ["252", 0],
            "latent_image": ["331d", 2],  # last identity anchor
        }},

        # ─── Decode + colour-match + output ───────────────────────
        "251": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["215", 0],
            "vae": ["236", 2],
            "tile_size": 768, "overlap": 64,
            "temporal_size": 4096, "temporal_overlap": 4,
        }},
        # ColorMatch the output against the character image, mirroring
        # the i2v workflow — reduces the warm/saturated drift the fp8
        # VAE roundtrip produces.
        "280": {"class_type": "ColorMatch", "inputs": {
            "image_ref": ["269", 0],
            "image_target": ["251", 0],
            "method": "mkl",
            "strength": 1.0,
        }},
        "242": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["280", 0],
            "frame_rate": fps,
            "loop_count": 0,
            "filename_prefix": "ltx_motion",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        }},
    }
    return workflow
