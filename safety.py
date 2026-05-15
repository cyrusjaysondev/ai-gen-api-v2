"""
Face blocklist filter — compliance/safety check for image-generation endpoints.

Loads a directory of face images (one per blocked identity), computes
InsightFace embeddings, and provides a single check_image(bytes) method
that returns whether any face in an input image matches the blocklist.

Layout on the network volume:
  /workspace/blocklist/<identity_name>.png       (one face image per identity)
  /workspace/insightface_models/buffalo_l/...    (model cache, downloaded once)

Pod sees these at /workspace/; serverless workers see the same at
/runpod-volume/. The BLOCKLIST_DIR / MODEL_ROOT env vars override defaults.

Threshold (default 0.6): InsightFace embeddings are L2-normalized, so the
similarity score is in [-1, 1]. Same-identity scores are typically > 0.5;
different identities < 0.4. 0.6 is conservative (false-negative biased) —
adjust via the FACE_FILTER_THRESHOLD env var if you tune later.

Imported lazily by the handlers so a worker that never runs face-filtered
requests doesn't pay the import + model-load cost.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional


class FilterResult(NamedTuple):
    blocked: bool
    matched_identity: Optional[str]
    score: float
    face_count: int


# ─────────────────────────────────────────────
# Lazy module-level singleton — built on first check_image() call.
# Hot-reloads when the blocklist directory's mtime changes so admin
# uploads/deletes take effect on the next request without restarting
# uvicorn or the serverless worker.
# ─────────────────────────────────────────────

_FILTER = None
_FILTER_BLOCKLIST_MTIME: float = 0.0
_FILTER_INIT_ERROR: Optional[str] = None


def _blocklist_dir() -> Path:
    return Path(os.environ.get("BLOCKLIST_DIR", "/workspace/blocklist"))


def _blocklist_mtime() -> float:
    """Return the most recent mtime across the blocklist dir and its
    contents. We need both because adding/removing files updates the
    directory's mtime; replacing an existing file with new content does
    not (the dir mtime stays the same), but the file's own mtime updates."""
    d = _blocklist_dir()
    if not d.is_dir():
        return 0.0
    latest = d.stat().st_mtime
    for f in d.iterdir():
        if f.is_file():
            latest = max(latest, f.stat().st_mtime)
    return latest


def _maybe_reload():
    """If the blocklist on disk has changed since we built _FILTER, rebuild it.
    Cheap: one stat() per directory entry — negligible vs face-detection cost."""
    global _FILTER, _FILTER_BLOCKLIST_MTIME
    current_mtime = _blocklist_mtime()
    if _FILTER is not None and current_mtime != _FILTER_BLOCKLIST_MTIME:
        # Drop the cached filter so the next check_image rebuilds it. We
        # keep the InsightFace model loaded — it's expensive to re-init.
        _FILTER = None
    _build_filter()
    _FILTER_BLOCKLIST_MTIME = current_mtime


_CACHED_APP = None  # InsightFace FaceAnalysis instance — survives blocklist reloads


def _build_filter():
    global _FILTER, _FILTER_INIT_ERROR, _CACHED_APP
    try:
        import numpy as np
        from PIL import Image
        from insightface.app import FaceAnalysis
    except ImportError as e:
        _FILTER_INIT_ERROR = f"face filter requested but dependencies missing: {e}. Install insightface + onnxruntime-gpu."
        return

    blocklist_dir = _blocklist_dir()
    model_root    = os.environ.get("INSIGHTFACE_MODEL_ROOT", "/workspace/insightface_models")
    threshold     = float(os.environ.get("FACE_FILTER_THRESHOLD", "0.6"))

    # Reuse the FaceAnalysis instance across reloads — the model load is
    # ~5s on first call. The blocklist itself is cheap to rescan.
    if _CACHED_APP is None:
        try:
            _CACHED_APP = FaceAnalysis(name="buffalo_l", root=model_root,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            _CACHED_APP.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            _FILTER_INIT_ERROR = f"failed to initialize InsightFace: {e}"
            return
    app = _CACHED_APP

    # Load blocklist
    blocklist: dict[str, "np.ndarray"] = {}
    if blocklist_dir.is_dir():
        for img_path in sorted(blocklist_dir.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            try:
                arr = np.array(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"[face-filter] WARN: cannot open {img_path.name}: {e}")
                continue
            faces = app.get(arr)
            if not faces:
                print(f"[face-filter] WARN: no face detected in blocklist image {img_path.name} — skipping")
                continue
            # Use the largest face if multiple are detected (assume the
            # blocklist image is well-cropped around one identity)
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            blocklist[img_path.stem] = face.normed_embedding

    _FILTER = {
        "app": app,
        "blocklist": blocklist,
        "threshold": threshold,
        "np": np,
        "Image": Image,
    }
    print(f"[face-filter] ready: {len(blocklist)} identities loaded from {blocklist_dir}, threshold={threshold}")


def detect_face_count(image_bytes: bytes) -> int:
    """Run face detection on an image and return the number of faces found.

    Used by the admin upload endpoint to validate a blocklist entry without
    requiring the blocklist to be non-empty — `check_image` short-circuits
    on empty blocklist for perf and would otherwise report face_count=0
    for the very first upload.
    """
    _maybe_reload()
    if _FILTER is None:
        raise RuntimeError(_FILTER_INIT_ERROR or "face filter unavailable")
    app   = _FILTER["app"]
    np    = _FILTER["np"]
    Image = _FILTER["Image"]
    try:
        arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    except Exception:
        return 0
    return len(app.get(arr))


# Max edge length we store for a blocklist entry. Embeddings are computed by
# InsightFace at 112x112 internally, so downscaling above ~1024 has no effect
# on matching accuracy — it only saves disk + reload time.
BLOCKLIST_MAX_EDGE = int(os.environ.get("BLOCKLIST_MAX_EDGE", "1024"))


def normalize_blocklist_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Decode, EXIF-rotate, downscale (if needed) and re-encode as PNG.

    Returns (png_bytes, ".png"). Raises ValueError on undecodable input so
    the caller can return a 400 instead of a 500. Lets the admin upload
    accept phone photos / 4K crops / odd formats without the caller worrying
    about request-body limits or storing 20 MB blobs on the network volume.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError as e:
        raise RuntimeError(f"Pillow missing: {e}")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # honor camera rotation
        img = img.convert("RGB")
    except Exception as e:
        raise ValueError(f"could not decode image: {e}")
    w, h = img.size
    longer = max(w, h)
    if longer > BLOCKLIST_MAX_EDGE:
        scale = BLOCKLIST_MAX_EDGE / longer
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), ".png"


def check_image(image_bytes: bytes) -> FilterResult:
    """Detect faces in an image and compare against the blocklist.

    Returns FilterResult(blocked, matched_identity, score, face_count).
    If the filter can't initialize (missing deps, etc.) this raises
    RuntimeError — the caller should decide whether to fail closed or open.
    """
    # Self-heal: pick up any blocklist additions/removals since last call.
    _maybe_reload()
    if _FILTER is None:
        raise RuntimeError(_FILTER_INIT_ERROR or "face filter unavailable")

    app       = _FILTER["app"]
    blocklist = _FILTER["blocklist"]
    threshold = _FILTER["threshold"]
    np        = _FILTER["np"]
    Image     = _FILTER["Image"]

    # An empty blocklist means nothing to compare against → never blocks.
    if not blocklist:
        return FilterResult(False, None, 0.0, 0)

    try:
        arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    except Exception as e:
        # Unparseable image → let the workflow caller surface its own error
        return FilterResult(False, None, 0.0, 0)

    faces = app.get(arr)
    if not faces:
        return FilterResult(False, None, 0.0, 0)

    best_score = -1.0
    best_id: Optional[str] = None
    for face in faces:
        emb = face.normed_embedding
        for identity, ref_emb in blocklist.items():
            # Both embeddings are L2-normalized → dot product = cosine similarity
            score = float(np.dot(emb, ref_emb))
            if score > best_score:
                best_score = score
                best_id = identity

    blocked = best_score > threshold
    return FilterResult(blocked, best_id, best_score, len(faces))


# ─────────────────────────────────────────────
# Bypass audit log — every face_filter=false call logs here for compliance
# ─────────────────────────────────────────────

BYPASS_LOG = Path(os.environ.get("BYPASS_LOG", "/workspace/face_filter_bypass.log"))


def log_bypass(job_id: str, endpoint: str, note: str = "") -> None:
    """Append a timestamped entry recording a face_filter=false bypass."""
    try:
        BYPASS_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        line = f"{ts}\t{endpoint}\t{job_id}\t{note}\n"
        with open(BYPASS_LOG, "a") as f:
            f.write(line)
    except Exception:
        # Audit log is best-effort — don't fail the request if disk is full.
        pass
