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
    # Cosine-similarity cutoff for "this detected face matches a blocked
    # identity". Was 0.6, bumped to 0.7 after a confirmed false-positive
    # (unrelated woman scoring 0.666 against a Hun Sen entry). 0.7 is the
    # practical sweet spot for buffalo_l/Arcface — genuine same-person
    # matches typically land at 0.75–0.9, so the threshold catches real
    # hits while shaking off coincidental similarity between two faces of
    # similar demographics. If a known real identity slips past 0.7, drop
    # FACE_FILTER_THRESHOLD via env to widen the net for that pod only.
    threshold     = float(os.environ.get("FACE_FILTER_THRESHOLD", "0.7"))
    # InsightFace's default detection threshold (0.5) — and even our earlier
    # bump to 0.3 — kept rejecting clearly-visible elderly faces. SCRFD-10G
    # under-trains on older subjects and on photos with washed-out colour,
    # so we set the floor much lower (0.1) and use a larger detection canvas
    # (1024² instead of 640²) which dramatically improves recall on small or
    # low-contrast faces. The MATCHING threshold (FACE_FILTER_THRESHOLD)
    # above is independent — it's a cosine-similarity cutoff on the embedding,
    # not on detection, so a more permissive detector doesn't loosen blocking.
    det_thresh    = float(os.environ.get("FACE_DETECTOR_THRESHOLD", "0.1"))
    det_size_edge = int(os.environ.get("FACE_DETECTOR_SIZE", "1024"))

    # Reuse the FaceAnalysis instance across reloads — the model load is
    # ~5s on first call. The blocklist itself is cheap to rescan.
    if _CACHED_APP is None:
        try:
            _CACHED_APP = FaceAnalysis(name="buffalo_l", root=model_root,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            _CACHED_APP.prepare(ctx_id=0, det_size=(det_size_edge, det_size_edge), det_thresh=det_thresh)
        except Exception as e:
            _FILTER_INIT_ERROR = f"failed to initialize InsightFace: {e}"
            return
    app = _CACHED_APP

    # Load blocklist. Each successful embedding is one identity the
    # filter can actually block at query time. If detection FAILS on a
    # blocklist image, that identity is unblockable until the file is
    # replaced — so we apply the same autocontrast + larger-canvas
    # fallbacks that detect_face_count uses on inbound images. Anything
    # we still can't extract gets logged as a hard miss so admins can
    # see the gap in /admin/comfy-status or a future stats endpoint.
    from PIL import ImageOps  # local import — ImageOps isn't bound at top
    blocklist: dict[str, "np.ndarray"] = {}
    skipped: list[str] = []
    for img_path in sorted(blocklist_dir.iterdir()) if blocklist_dir.is_dir() else []:
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[face-filter] WARN: cannot open {img_path.name}: {e}")
            continue

        arr = np.array(pil)
        faces = app.get(arr)

        # Fallback chain — most blocklist images that fail naive
        # SCRFD detection at default exposure are recoverable with a
        # bit of preprocessing. We try progressively stronger
        # interventions; first one that gets a face wins. Audit logs
        # tell admins WHICH fallback recovered the file (so old
        # archive photos can be flagged for replacement with better
        # crops if too many fall through to the late stages).

        # Fallback 1 — autocontrast (histogram stretch). Recovers
        # washed-out / older photos.
        if not faces:
            try:
                enhanced = ImageOps.autocontrast(pil, cutoff=2)
                arr = np.array(enhanced)
                faces = app.get(arr)
                if faces:
                    print(f"[face-filter] {img_path.name}: recovered via autocontrast")
            except Exception as e:
                print(f"[face-filter] {img_path.name}: autocontrast errored: {e}")

        # Fallback 2 — autocontrast + sharpen. B&W archive photos
        # often have soft focus / grain that confuses SCRFD's edge-
        # sensitive features. Sharpening at 130% strength tightens
        # those edges.
        if not faces:
            try:
                from PIL import ImageFilter as _IF
                sharp = ImageOps.autocontrast(pil, cutoff=2).filter(
                    _IF.UnsharpMask(radius=1.5, percent=130, threshold=2),
                )
                arr = np.array(sharp)
                faces = app.get(arr)
                if faces:
                    print(f"[face-filter] {img_path.name}: recovered via autocontrast+sharpen")
            except Exception as e:
                print(f"[face-filter] {img_path.name}: sharpen errored: {e}")

        # Fallback 3 — upscale to 2x then retry. Helps when the face
        # is small relative to the canvas; SCRFD at det_size=1024
        # still struggles when the face is < ~100px wide in the source.
        if not faces:
            try:
                w, h = pil.size
                up = pil.resize((w * 2, h * 2), Image.LANCZOS)
                arr = np.array(up)
                faces = app.get(arr)
                if faces:
                    print(f"[face-filter] {img_path.name}: recovered via 2× upscale")
            except Exception as e:
                print(f"[face-filter] {img_path.name}: 2× upscale errored: {e}")

        # Fallback 4 — 4× upscale + autocontrast. Last-resort upsample
        # for very low-res archival scans where 2× isn't enough.
        if not faces:
            try:
                w, h = pil.size
                up4 = ImageOps.autocontrast(pil, cutoff=2).resize(
                    (w * 4, h * 4), Image.LANCZOS,
                )
                arr = np.array(up4)
                faces = app.get(arr)
                if faces:
                    print(f"[face-filter] {img_path.name}: recovered via 4× upscale + autocontrast")
            except Exception as e:
                print(f"[face-filter] {img_path.name}: 4× upscale errored: {e}")

        # Fallback 5 — center-crop to inner 80% then 2× upscale. Some
        # blocklist photos have wide borders / frames / busy
        # backgrounds that confuse the detector; cropping forces
        # attention to the centered face. Combined with upscale so
        # the resulting face is large enough to detect.
        if not faces:
            try:
                w, h = pil.size
                margin_x, margin_y = int(w * 0.1), int(h * 0.1)
                cropped = pil.crop((margin_x, margin_y, w - margin_x, h - margin_y))
                cw, ch = cropped.size
                cropped_up = cropped.resize((cw * 2, ch * 2), Image.LANCZOS)
                arr = np.array(cropped_up)
                faces = app.get(arr)
                if faces:
                    print(f"[face-filter] {img_path.name}: recovered via center-crop + 2× upscale")
            except Exception as e:
                print(f"[face-filter] {img_path.name}: crop+upscale errored: {e}")

        if not faces:
            skipped.append(img_path.name)
            print(f"[face-filter] WARN: no face detected in blocklist image {img_path.name} — skipping (will not be blockable!)")
            continue

        # Use the largest face if multiple are detected (assume the
        # blocklist image is well-cropped around one identity)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        blocklist[img_path.stem] = face.normed_embedding
    if skipped:
        print(f"[face-filter] WARNING: {len(skipped)} blocklist images had no detectable face: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    _FILTER = {
        "app": app,
        "blocklist": blocklist,
        "threshold": threshold,
        "np": np,
        "Image": Image,
        "skipped_files": skipped,
    }
    print(f"[face-filter] ready: {len(blocklist)} identities loaded from {blocklist_dir}, threshold={threshold}, skipped={len(skipped)}")


# Smallest fraction of the image area a detected bbox must cover to count
# as a "real" face. Below this, we treat the detection as noise — at the
# permissive detector settings we run (det_thresh=0.1), SCRFD sometimes
# reports a tiny background artifact as a low-confidence face, which used
# to manifest as spurious "detected 2 faces" rejections on otherwise clean
# portraits. 3% of the image area corresponds to a ~177×177 face inside
# a 1024×1024 frame — well below any reasonable portrait crop.
MIN_FACE_AREA_RATIO = float(os.environ.get("FACE_MIN_AREA_RATIO", "0.03"))


def _count_significant_faces(faces, img_area: int) -> int:
    """Count faces whose bbox covers at least MIN_FACE_AREA_RATIO of the
    image. Filters out background noise that the permissive detector
    sometimes catches. Returns 0 if `faces` is empty.
    """
    n = 0
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if img_area > 0 and (area / img_area) >= MIN_FACE_AREA_RATIO:
            n += 1
    return n


def get_status() -> dict:
    """Return the current filter state WITHOUT rebuilding.

    Used by /admin/blocklist GET to annotate each entry with whether
    it actually got loaded (vs being silently skipped). Lazily triggers
    a build on first call if the filter hasn't been initialized yet,
    so a fresh pod that's never run a generation can still answer.
    """
    _maybe_reload()
    if _FILTER is None:
        return {"initialized": False, "error": _FILTER_INIT_ERROR}
    skipped = _FILTER.get("skipped_files", [])
    return {
        "initialized": True,
        "threshold": _FILTER["threshold"],
        "blocklist_count": len(_FILTER["blocklist"]),
        "skipped_count": len(skipped),
        "skipped_files": list(skipped),  # full list, not truncated
    }


def force_reload_filter() -> dict:
    """Wipe the cached `_FILTER` and re-run `_build_filter()` immediately.

    Lets `/admin/reload-filter` apply config changes (FACE_FILTER_THRESHOLD,
    FACE_DETECTOR_THRESHOLD, FACE_MIN_AREA_RATIO env vars, or new blocklist
    files dropped on disk out-of-band) without restarting uvicorn.

    The underlying FaceAnalysis model instance (`_CACHED_APP`) is preserved
    so we don't pay the ~5s model-load tax on every reload; we only
    re-read the env-tunable knobs and re-scan the blocklist directory for
    embedding extraction.

    Returns a dict the admin endpoint can echo back so the caller sees the
    new threshold + identity count and knows the reload actually took.
    """
    global _FILTER, _FILTER_BLOCKLIST_MTIME
    # Force the next _maybe_reload to fall through to _build_filter even if
    # the blocklist dir mtime hasn't moved.
    _FILTER = None
    _FILTER_BLOCKLIST_MTIME = 0.0
    _build_filter()
    if _FILTER is None:
        return {
            "ok": False,
            "error": _FILTER_INIT_ERROR or "filter rebuild produced no state",
        }
    skipped = _FILTER.get("skipped_files", [])
    return {
        "ok": True,
        "threshold": _FILTER["threshold"],
        "blocklist_count": len(_FILTER["blocklist"]),
        # Surface a small sample so admins eyeballing the response can
        # confirm the entries they expect are loaded.
        "sample_identities": sorted(_FILTER["blocklist"].keys())[:10],
        # Hard misses: images on disk that produced no embedding, even
        # after autocontrast + 2× upscale fallbacks. These identities
        # are UNBLOCKABLE — the admin should replace them with a clearer
        # photo.
        "skipped_count": len(skipped),
        "skipped_files": skipped[:30],
    }


def detect_face_count(image_bytes: bytes) -> int:
    """Count significant faces in the image.

    Used by the admin upload endpoint to validate a blocklist entry. We
    filter detections by bbox area so background noise (a tiny artifact
    that the detector picks up at low det_thresh) doesn't get counted
    as a second face and trigger a false "detected 2 faces" rejection.

    If the first pass returns 0 significant faces, we retry on a
    contrast-equalized copy of the image. Older / washed-out portraits
    often fall below SCRFD's detection floor at default exposure but
    pass after a histogram stretch.
    """
    _maybe_reload()
    if _FILTER is None:
        raise RuntimeError(_FILTER_INIT_ERROR or "face filter unavailable")
    app   = _FILTER["app"]
    np    = _FILTER["np"]
    Image = _FILTER["Image"]
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return 0

    arr = np.array(pil)
    img_h, img_w = arr.shape[:2]
    img_area = img_h * img_w

    n = _count_significant_faces(app.get(arr), img_area)
    if n > 0:
        return n

    # Fallback: enhance and retry. ImageOps.autocontrast stretches the
    # histogram per channel so washed-out faces look closer to standard
    # exposure; that alone catches a lot of the "elderly photo" misses.
    try:
        from PIL import ImageOps
        enhanced = ImageOps.autocontrast(pil, cutoff=2)
        arr2 = np.array(enhanced)
        n2 = _count_significant_faces(app.get(arr2), img_area)
        if n2 > 0:
            print(f"[face-filter] detection recovered via autocontrast fallback (found {n2})")
            return n2
    except Exception as e:
        print(f"[face-filter] autocontrast fallback errored: {e}")

    return 0


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

    img_h, img_w = arr.shape[:2]
    img_area = img_h * img_w

    raw_faces = app.get(arr)
    if not raw_faces:
        return FilterResult(False, None, 0.0, 0)

    # Filter detections by area before matching. At det_thresh=0.1 the
    # detector occasionally produces low-confidence pseudo-faces from
    # background texture; those have garbage embeddings that can match
    # blocklist entries by chance and produce false positives. The same
    # MIN_FACE_AREA_RATIO that gates uploads also applies here so the
    # detect/match policy stays consistent.
    faces = [
        f for f in raw_faces
        if (max(0.0, f.bbox[2] - f.bbox[0]) * max(0.0, f.bbox[3] - f.bbox[1]) / max(1, img_area)) >= MIN_FACE_AREA_RATIO
    ]
    if not faces:
        # Detector saw only noise — nothing significant to match.
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
