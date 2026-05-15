"""
watermark.py — overlay a short text label onto generated images and videos.

Public API:
    apply(path, text="AI") -> None
        Modifies the file at `path` in place. If `text` is None or empty
        (after stripping), this is a no-op so callers don't need a guard.

Style: bold white text with a black outline at the bottom-right.
  • Images (.png/.jpg/.jpeg/.webp): Pillow draws the text; size scales
    to ~4% of image height (min 24 px), padded ~3% from the edges.
  • Videos (.mp4/.mov/.webm/.mkv): ffmpeg's drawtext filter renders the
    same look. The video is re-encoded (libx264, CRF 18, veryfast).
    Audio, if present, is stream-copied — no quality loss.

This module is intentionally tolerant of failure. If ffmpeg is missing or
the font file isn't where we expect, callers get a clear RuntimeError but
the original file is left intact, so the API can return the unwatermarked
result rather than failing the whole job.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


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
