"""
watermark.py — overlay a short text label or the GenReel logo onto generated
images and videos.

Public API:
    apply(path, text="AI") -> None
        Text overlay. No-op when `text` is None or empty after stripping.

    apply_logo(path) -> None
        Composite the GenReel logo onto the file. No-op when the logo asset
        is missing (setup.sh fetches it on every pod boot — see LOGO_PATH).

Both modify the file at `path` in place. Callers can chain them to stack
text over the logo. Failures are kept tolerant — a missing logo file or
broken ffmpeg leaves the unwatermarked file intact so the API can still
return a usable result.

Style:
  * Text: bold white with a black outline at the bottom-right. Size scales
    to ~4% of image height (min 24 px), padded ~3% from the edges.
  * Logo: PNG with transparency, placed at the bottom-right. Width scales
    to ~14% of the image's shorter side (min 64 px) so it's visible without
    dominating the frame.

Video paths re-encode through libx264 (CRF 18, veryfast). Audio, if present,
is stream-copied so we don't degrade it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# The single fixed-asset brand mark. setup.sh fetches it to the network
# volume so every pod / serverless worker sees it; the URL is documented in
# setup.sh next to the download line.
LOGO_PATH = Path("/workspace/assets/genreel_logo.png")

# Fraction of the image's shorter side used as the logo's width.
_LOGO_SCALE = 0.14
# Padding from the edge, as a fraction of the shorter side.
_EDGE_PAD = 0.03


def apply(path, text: str = "AI") -> None:
    """Overlay `text` on the file at `path` in place. No-op on empty text."""
    if text is None:
        return
    text = text.strip()
    if not text:
        return

    p = Path(path)
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        _apply_image(p, text)
    elif ext in VIDEO_EXTS:
        _apply_video(p, text)
    # Unknown extensions (e.g. .gif from a future workflow): leave alone
    # rather than ship a broken file.


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _apply_image(p: Path, text: str) -> None:
    img = Image.open(p)
    original_mode = img.mode
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)

    font_size = max(24, int(img.height * 0.04))
    stroke = max(2, font_size // 12)
    font = _load_font(font_size)

    # textbbox returns ink bounds; we want to right-align by the ink box width.
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = max(8, int(min(img.width, img.height) * 0.03))
    # bbox[0]/bbox[1] are the bbox origin offset relative to draw origin —
    # subtract so the visual edge lands at (x, y) rather than the origin.
    x = img.width - text_w - pad - bbox[0]
    y = img.height - text_h - pad - bbox[1]

    draw.text(
        (x, y), text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 255),
    )

    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img.convert("RGB").save(p, quality=95)
    elif suffix == ".webp":
        img.save(p, quality=95)
    else:
        # Preserve original mode where possible (avoid bloating greyscale PNGs to RGBA)
        if original_mode != "RGBA":
            img.convert(original_mode if original_mode in ("RGB", "L", "P") else "RGB").save(p)
        else:
            img.save(p)


def _probe_height(p: Path) -> int:
    """Return the video's frame height, or 720 as a safe fallback."""
    try:
        out = subprocess.run(
            ["ffprobe", "-loglevel", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    except Exception:
        pass
    return 720


def _apply_video(p: Path, text: str) -> None:
    # Escape ffmpeg filtergraph metacharacters in `text`. Order matters:
    # backslash first, then the others.
    safe = (
        text.replace("\\", r"\\\\")
            .replace(":", r"\:")
            .replace("'", r"\'")
            .replace(",", r"\,")
    )
    # Keep the original extension at the end — ffmpeg picks its muxer from
    # the filename suffix (foo.mp4 → mp4 muxer). foo.mp4.wm.tmp would fail.
    tmp = p.with_name(f"{p.stem}.wm{p.suffix}")
    # ffmpeg drawtext's borderw / fontsize need integers, not expressions.
    # Probe once and compute proportionate values (~4.5% font height, ≥2px border).
    height = _probe_height(p)
    fontsize = max(18, height // 22)
    borderw = max(2, height // 300)
    drawtext = (
        f"drawtext=fontfile={_FONT_PATH}:"
        f"text='{safe}':"
        f"fontcolor=white:fontsize={fontsize}:"
        f"borderw={borderw}:bordercolor=black:"
        f"x=w-tw-h/30:y=h-th-h/30"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(p),
        "-vf", drawtext,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg drawtext failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-500:] or '(no stderr)'}"
        )
    shutil.move(str(tmp), str(p))


# ─── Logo overlay (image asset, not text) ─────────────────────────────────


def apply_logo(path) -> None:
    """Composite the GenReel logo onto the file at `path` in place.

    No-op (with a printed warning) when LOGO_PATH is missing — setup.sh is
    responsible for fetching it. We don't raise: an unwatermarked output is
    better than a failed job.
    """
    if not LOGO_PATH.exists():
        # Caller already swallows exceptions; print so it lands in the API
        # log without forcing a structured logger dependency.
        print(f"[watermark] logo asset missing at {LOGO_PATH} — skipping image overlay")
        return

    p = Path(path)
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        _apply_image_logo(p)
    elif ext in VIDEO_EXTS:
        _apply_video_logo(p)


def _apply_image_logo(p: Path) -> None:
    base = Image.open(p)
    original_mode = base.mode
    base = base.convert("RGBA")

    logo = Image.open(LOGO_PATH).convert("RGBA")
    shorter = min(base.width, base.height)
    target_w = max(64, int(shorter * _LOGO_SCALE))
    ratio = target_w / logo.width
    target_h = max(16, int(logo.height * ratio))
    logo = logo.resize((target_w, target_h), Image.LANCZOS)

    pad = max(8, int(shorter * _EDGE_PAD))
    x = base.width - target_w - pad
    y = base.height - target_h - pad
    # Use the logo's own alpha as the paste mask so PNG transparency is honoured.
    base.alpha_composite(logo, dest=(x, y))

    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        base.convert("RGB").save(p, quality=95)
    elif ext == ".webp":
        base.save(p, quality=95)
    elif original_mode != "RGBA":
        target_mode = original_mode if original_mode in ("RGB", "L", "P") else "RGB"
        base.convert(target_mode).save(p)
    else:
        base.save(p)


def _apply_video_logo(p: Path) -> None:
    """Overlay LOGO_PATH onto the video at `p` via ffmpeg `overlay`.

    The logo is scaled to ~14% of the shorter side at runtime via filter
    expressions, so a single asset works across portrait + landscape.
    """
    tmp = p.with_name(f"{p.stem}.wm{p.suffix}")
    pad_expr = f"max(8, min(main_w,main_h)*{_EDGE_PAD})"
    target_w_expr = f"max(64, min(main_w,main_h)*{_LOGO_SCALE})"
    filter_complex = (
        f"[1:v]scale={target_w_expr}:-1[wm];"
        f"[0:v][wm]overlay=x=main_w-overlay_w-{pad_expr}:"
        f"y=main_h-overlay_h-{pad_expr}:format=auto"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(p),
        "-i", str(LOGO_PATH),
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg overlay failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-500:] or '(no stderr)'}"
        )
    shutil.move(str(tmp), str(p))
