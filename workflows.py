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
# LTX-2.3 — shared helpers
# ─────────────────────────────────────────────

LTX_ASPECT_RATIOS = {
    "1:1":  (1, 1),  "4:3":  (4, 3),  "3:4":  (3, 4),
    "16:9": (16, 9), "9:16": (9, 16), "3:2":  (3, 2),
    "2:3":  (2, 3),  "21:9": (21, 9), "9:21": (9, 21),
}

LTX_DEFAULT_NEGATIVE = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"

LTX_PRESETS = {
    "fast": {
        "sigmas": "1.0, 0.975, 0.909375, 0.725, 0.421875, 0.0",
        "lora_strength": 0.5,
        "two_pass": False,
    },
    "quality": {
        "low_res_sigmas": "1.0, 0.99688, 0.99375, 0.990625, 0.9875, 0.984375, 0.98125, 0.978125, 0.975, 0.96875, 0.9625, 0.95, 0.9375, 0.909375, 0.875, 0.84375, 0.78125, 0.725, 0.5625, 0.421875, 0.0",
        "high_res_sigmas": "0.85, 0.7875, 0.7250, 0.5734, 0.4219, 0.0",
        "lora_strength": 0.35,
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
                            enhance_prompt: bool = True) -> dict:
    """Build an LTX 2.3 image-to-video workflow. `image_filename` must already exist in ComfyUI's input dir."""
    two_pass = LTX_PRESETS[preset]["two_pass"]

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
            "strength": 0.7 if two_pass else 1.0, "bypass": False
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
            "vae": ["236", 2], "image": ["248", 0], "latent": ["253", 0], "strength": 1.0, "bypass": False
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
