"""
palm.py — palm-line extraction for the feng-shui palm-reading feature.

Detects the four principal palmar lines (life, head, heart, fate) from a hand
photo and returns them as normalized polylines, matching feng-shui's
`PalmAnalysis` contract (src/services/palmVisionCore.ts) so the existing overlay
+ reading code consume the result unchanged.

Pipeline:
  1. MediaPipe Hands -> 21 landmarks. This locates the palm by HAND SHAPE, so it
     is immune to busy / skin-toned backgrounds and arbitrary hand rotation that
     break colour-based skin segmentation. Gives the palm box, orientation and
     thumb side directly.
  2. Rotate the hand upright (fingers up) for stable band geometry.
  3. CLAHE -> scikit-image ridge filter (dark creases) masked to the palm.
  4. Trace one continuous crease per line band, smooth + quadratic-fit, reject
     scattered traces; map points back to the original image.
  5. If MediaPipe finds no hand, fall back to YCrCb skin segmentation (the
     in-browser worker's method); on any failure, canonical fallback geometry.

CPU only — no GPU / diffusion models (MediaPipe runs its TFLite model on CPU via
XNNPACK). Always warm on the persistent pod, so there is no cold start. Always
returns a valid PalmAnalysis, mirroring the worker's "never throw" contract.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

try:
    from skimage.filters import sato
    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover - black-hat fallback below
    _HAVE_SKIMAGE = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mpp
    from mediapipe.tasks.python import vision as _mpv
    _HAVE_MP = True
except Exception:  # pragma: no cover
    _HAVE_MP = False

# ── Tuning ─────────────────────────────────────────────────────────────────────
# Per-line bands as fractions of the PALM box (finger-base line at the top, wrist
# at the bottom; fingers up, thumb on the LEFT). x is mirrored at runtime for
# left-thumb-on-the-right hands. axis 'h' = the line runs horizontally (fit y over
# x); 'v' = vertically (fit x over y).
_BANDS = {
    "heart": ("h", 0.10, 0.04, 0.94, 0.26),
    "head":  ("h", 0.06, 0.28, 0.84, 0.54),
    "life":  ("v", 0.00, 0.02, 0.40, 0.94),
    "fate":  ("v", 0.40, 0.30, 0.64, 1.00),
}
_LINE_ORDER = ("heart", "head", "life", "fate")
_PROC_MAX_EDGE = 760

_TASK_PATH = "/workspace/assets/mediapipe/hand_landmarker.task"

# MediaPipe Hands 21-landmark indices.
_WRIST, _THUMB_CMC, _THUMB_MCP, _THUMB_TIP = 0, 1, 2, 4
_INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP = 5, 9, 13, 17
_PALM_LMS = [_WRIST, _THUMB_CMC, _THUMB_MCP, _INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP]

# One warm HandLandmarker, serialized (the Tasks API object isn't thread-safe).
_landmarker = None
_mp_lock = threading.Lock()


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        _landmarker = _mpv.HandLandmarker.create_from_options(
            _mpv.HandLandmarkerOptions(
                base_options=_mpp.BaseOptions(model_asset_path=_TASK_PATH),
                num_hands=1,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
            )
        )
    return _landmarker


def _clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else float(v)


# ── Crease response ────────────────────────────────────────────────────────────
def _ridge_response(gray: np.ndarray) -> np.ndarray:
    """0..255 crease-strength map (high where dark thin creases are)."""
    eq = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    if _HAVE_SKIMAGE:
        r = sato(eq.astype(np.float32) / 255.0, sigmas=(1, 1.6, 2.4), black_ridges=True)
    else:
        ksz = max(7, (gray.shape[1] // 28) | 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        r = cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, k).astype(np.float32)
    r = np.nan_to_num(r)
    m = float(r.max())
    if m > 1e-6:
        r = r / m * 255.0
    r = cv2.GaussianBlur(r, (3, 3), 0)
    return r.astype(np.uint8)


# ── Trace / fit (numpy ports of palmVisionCore.ts) ─────────────────────────────
def _trace_crease(R: np.ndarray, band, axis: str, min_resp: int):
    H, W = R.shape
    x0, y0, x1, y1 = band
    x0 = max(0, int(np.floor(x0))); x1 = min(W - 1, int(np.ceil(x1)))
    y0 = max(0, int(np.floor(y0))); y1 = min(H - 1, int(np.ceil(y1)))
    a0, a1, p0, p1 = (x0, x1, y0, y1) if axis == "h" else (y0, y1, x0, x1)
    if a1 - a0 < 8 or p1 - p0 < 2:
        return None
    win = max(5, round((p1 - p0) * 0.16))

    def scan(a, lo, hi):
        col = R[lo:hi + 1, a] if axis == "h" else R[a, lo:hi + 1]
        if col.size == 0:
            return -1, -1
        j = int(np.argmax(col))
        return int(col[j]), lo + j

    mid = (a0 + a1) // 2
    sv, sp = scan(mid, p0, p1)
    if sp < 0:
        return None
    path = [(mid, sp, sv)]
    prev = sp
    for a in range(mid + 1, a1 + 1):
        v, p = scan(a, max(p0, prev - win), min(p1, prev + win)); path.append((a, p, v)); prev = p
    prev = sp
    for a in range(mid - 1, a0 - 1, -1):
        v, p = scan(a, max(p0, prev - win), min(p1, prev + win)); path.insert(0, (a, p, v)); prev = p

    s, e = 0, len(path) - 1
    while s < e and path[s][2] < min_resp:
        s += 1
    while e > s and path[e][2] < min_resp:
        e -= 1
    span = a1 - a0 + 1
    if e - s < span * 0.2:
        return None
    pts, ssum, weak = [], 0, 0
    for i in range(s, e + 1):
        a, p, v = path[i]
        pts.append((a, p) if axis == "h" else (p, a))
        ssum += v
        if v < min_resp:
            weak += 1
    n = len(pts)
    return pts, (e - s + 1) / span, (ssum / n / 255.0 if n else 0.0), weak


def _smooth(pts, window=7):
    arr = np.asarray(pts, dtype=np.float32)
    if len(arr) <= 2:
        return arr
    half = window // 2
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = arr[max(0, i - half):min(len(arr), i + half + 1)].mean(axis=0)
    return out


def _fit_curve(pts, axis: str, samples=14):
    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) < 4:
        return arr, 0.0
    u, v = (arr[:, 0], arr[:, 1]) if axis == "h" else (arr[:, 1], arr[:, 0])
    try:
        coeffs = np.polyfit(u, v, 2)
    except Exception:
        return arr, 1e9
    f = np.poly1d(coeffs)
    rms = float(np.sqrt(np.mean((f(u) - v) ** 2)))
    us = np.linspace(u.min(), u.max(), samples)
    vs = f(us)
    out = np.stack([us, vs], 1) if axis == "h" else np.stack([vs, us], 1)
    return out, rms


def _curvature(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    a, b, m = pts[0], pts[-1], pts[len(pts) // 2]
    chord = float(np.hypot(a[0] - b[0], a[1] - b[1])) or 1.0
    area = abs((b[0] - a[0]) * (a[1] - m[1]) - (a[0] - m[0]) * (b[1] - a[1]))
    return _clamp01(area / chord / (chord * 0.5))


def _density_in(R: np.ndarray, mask: np.ndarray, x, y, w, h, thr: int) -> float:
    H, W = R.shape
    x0 = max(0, int(x)); x1 = min(W, int(x + w))
    y0 = max(0, int(y)); y1 = min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub_m = mask[y0:y1, x0:x1] > 0
    skin = int(sub_m.sum())
    if not skin:
        return 0.0
    strong = int(((R[y0:y1, x0:x1] > thr) & sub_m).sum())
    return strong / skin


# ── Canonical fallback ─────────────────────────────────────────────────────────
def _canonical_lines():
    def mk(key, pts, rel, cur):
        return {"key": key, "points": [{"x": p[0], "y": p[1]} for p in pts],
                "relLength": rel, "curvature": cur, "breaks": 0, "detected": False}
    return [
        mk("heart", [(0.24, 0.30), (0.45, 0.25), (0.66, 0.25), (0.80, 0.28)], 0.62, 0.25),
        mk("head",  [(0.22, 0.46), (0.42, 0.49), (0.62, 0.52), (0.72, 0.53)], 0.55, 0.18),
        mk("life",  [(0.34, 0.28), (0.26, 0.45), (0.22, 0.64), (0.28, 0.84)], 0.70, 0.45),
        mk("fate",  [(0.52, 0.84), (0.51, 0.62), (0.50, 0.42)], 0.40, 0.10),
    ]


def fallback_analysis(extra_lines=None) -> dict:
    lines = list(extra_lines or [])
    have = {l["key"] for l in lines}
    lines += [l for l in _canonical_lines() if l["key"] not in have]
    order = {k: i for i, k in enumerate(_LINE_ORDER)}
    lines.sort(key=lambda l: order[l["key"]])
    return {
        "lines": lines,
        "metrics": {"creaseDensity": 0.5, "palmAspect": 0.8, "minorDensityPinky": 0.4, "brightness": 0.6},
        "palmBox": {"x": 0.12, "y": 0.08, "w": 0.76, "h": 0.86},
        "fallback": True,
    }


# ── Shared band-trace over a known palm box ────────────────────────────────────
def _trace_lines(R, mask, box, mirror, W, H, to_orig=None):
    """Trace the four lines in `box` (pixels, current frame). `to_orig` maps a
    point from the current frame back to the original image (for the rotated
    case); None = identity. Returns (detected_lines, thr)."""
    bx, by, bw, bh = box
    vals = R[mask > 0]
    thr = int(max(8, np.percentile(vals, 80))) if vals.size else 10
    detected = []
    for key in _LINE_ORDER:
        axis, fx0, fy0, fx1, fy1 = _BANDS[key]
        if mirror:
            fx0, fx1 = 1.0 - fx1, 1.0 - fx0
        band = (bx + fx0 * bw, by + fy0 * bh, bx + fx1 * bw, by + fy1 * bh)
        tr = _trace_crease(R, band, axis, thr)
        if tr is None:
            continue
        raw, coverage, strength, breaks = tr
        if coverage < 0.4 or len(raw) < 5:
            continue
        sm = _smooth(raw, 7)
        fit, rms = _fit_curve(sm, axis, 14)
        perp_extent = (band[3] - band[1]) if axis == "h" else (band[2] - band[0])
        if rms > perp_extent * 0.5:
            continue
        if to_orig is not None:
            fit = cv2.transform(fit.reshape(-1, 1, 2).astype(np.float32), to_orig).reshape(-1, 2)
        norm = [(_clamp01(p[0] / W), _clamp01(p[1] / H)) for p in fit]
        detected.append({
            "key": key,
            "points": [{"x": x, "y": y} for x, y in norm],
            "relLength": _clamp01(coverage * 0.85 + strength * 0.4),
            "curvature": _curvature(np.asarray(norm)),
            "breaks": int(breaks),
            "detected": True,
        })
    return detected, thr


def _metrics(R, mask, box, mirror, gray):
    bx, by, bw, bh = box
    thr = 10
    vals = R[mask > 0]
    if vals.size:
        thr = int(max(8, np.percentile(vals, 80)))
    crease_density = _clamp01(_density_in(R, mask, bx, by, bw, bh, thr) * 2.2)
    pinky_x = (bx + 0.05 * bw) if mirror else (bx + 0.55 * bw)
    minor_pinky = _clamp01(_density_in(R, mask, pinky_x, by + 0.30 * bh, 0.4 * bw, 0.26 * bh, thr) * 3)
    brightness = _clamp01(float(cv2.mean(gray, mask)[0]) / 255.0)
    return {"creaseDensity": crease_density, "palmAspect": _clamp01(bw / max(1.0, bh)),
            "minorDensityPinky": minor_pinky, "brightness": brightness}


# ── MediaPipe landmark path (primary) ──────────────────────────────────────────
def _mp_detect(rgb):
    if not _HAVE_MP:
        return None
    try:
        with _mp_lock:
            res = _get_landmarker().detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    except Exception:
        return None
    if not res.hand_landmarks:
        return None
    H, W = rgb.shape[:2]
    pts = np.array([[p.x * W, p.y * H] for p in res.hand_landmarks[0]], dtype=np.float32)
    return pts


def _extract_landmarks(proc):
    H, W = proc.shape[:2]
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    pts = _mp_detect(rgb)
    if pts is None:
        return None

    # Rotate so the hand points up (wrist -> middle MCP becomes screen-up).
    center = pts[_PALM_LMS].mean(axis=0)
    up = pts[_MIDDLE_MCP] - pts[_WRIST]
    rot_deg = np.degrees(np.arctan2(up[1], up[0])) - (-90.0)
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rot_deg, 1.0)
    rot = cv2.warpAffine(proc, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rpts = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)

    mcps = rpts[[_INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP]]
    top = float(mcps[:, 1].min())
    bottom = float(rpts[_WRIST][1])
    if bottom - top < 14:
        return None
    palm = rpts[_PALM_LMS]
    left, right = float(palm[:, 0].min()), float(palm[:, 0].max())
    padx = 0.05 * (right - left)
    box = (left - padx, top, (right - left) + 2 * padx, bottom - top)
    # Thumb side from geometry (robust to MediaPipe's mirror/handedness quirks):
    # if the thumb sits right of the palm centre, mirror the bands.
    mirror = bool(rpts[_THUMB_TIP][0] > rpts[_PALM_LMS].mean(axis=0)[0])

    gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
    R = _ridge_response(gray)
    hull = cv2.convexHull(palm.astype(np.int32))
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    grow = max(13, int(0.12 * box[2])) | 1
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    R = cv2.bitwise_and(R, R, mask=mask)

    Minv = cv2.invertAffineTransform(M)
    detected, _ = _trace_lines(R, mask, box, mirror, W, H, to_orig=Minv)
    if len(detected) < 2:
        return None

    metrics = _metrics(R, mask, box, mirror, gray)
    corners = np.array([[box[0], box[1]], [box[0] + box[2], box[1]],
                        [box[0] + box[2], box[1] + box[3]], [box[0], box[1] + box[3]]], dtype=np.float32)
    oc = cv2.transform(corners.reshape(-1, 1, 2), Minv).reshape(-1, 2)
    ox0, oy0, ox1, oy1 = oc[:, 0].min(), oc[:, 1].min(), oc[:, 0].max(), oc[:, 1].max()
    return {
        "lines": detected, "metrics": metrics,
        "palmBox": {"x": _clamp01(ox0 / W), "y": _clamp01(oy0 / H),
                    "w": _clamp01((ox1 - ox0) / W), "h": _clamp01((oy1 - oy0) / H)},
        "fallback": False,
    }


# ── Skin-segmentation path (fallback when MediaPipe finds no hand) ─────────────
def _skin_box(bgr):
    H, W = bgr.shape[:2]
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < W * H * 0.10:
        return None
    x, y, w, h = cv2.boundingRect(c)
    if w < W * 0.18 or h < H * 0.18:
        return None
    clean = np.zeros((H, W), np.uint8)
    cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)
    # narrow full-hand box to the palm (first row reaching ~75% of max width)
    widths = (clean[y:y + h, x:x + w] > 0).sum(axis=1).astype(np.float32)
    if widths.size and widths.max() > 0:
        kw = max(3, int(0.03 * len(widths)) | 1)
        sw = cv2.GaussianBlur(widths.reshape(-1, 1), (1, kw), 0).ravel()
        upper = int(0.6 * len(sw))
        cand = np.where(sw[:upper] >= 0.75 * float(sw.max()))[0]
        fb = int(cand[0]) if cand.size else int(0.35 * len(sw))
        fb = min(max(fb, int(0.10 * len(sw))), int(0.55 * len(sw)))
        ptop = y + fb
        cols = np.where((clean[ptop:y + h, :] > 0).any(axis=0))[0]
        if cols.size:
            x, w = int(cols.min()), int(cols.max()) - int(cols.min())
            y, h = ptop, (y + h) - ptop
    return (x, y, w, h), clean


def _extract_skinseg(proc):
    res = _skin_box(proc)
    if res is None:
        return None
    (bx, by, bw, bh), skin = res
    if bw < 8 or bh < 8:
        return None
    H, W = proc.shape[:2]
    y0, y1 = int(by + 0.32 * bh), int(by + 0.68 * bh)
    cols = np.where((skin[max(0, y0):max(1, y1), :] > 0).max(axis=0))[0]
    cx = bx + bw / 2.0
    mirror = bool(cols.size and (cols.max() - cx) > (cx - cols.min()) * 1.08)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    R = cv2.bitwise_and(_ridge_response(gray), _ridge_response(gray), mask=skin)
    detected, _ = _trace_lines(R, skin, (bx, by, bw, bh), mirror, W, H, to_orig=None)
    if len(detected) < 2:
        return None
    return {"lines": detected, "metrics": _metrics(R, skin, (bx, by, bw, bh), mirror, gray),
            "palmBox": {"x": bx / W, "y": by / H, "w": bw / W, "h": bh / H}, "fallback": False}


# ── Public entry point ─────────────────────────────────────────────────────────
def extract_palm(bgr: np.ndarray) -> dict:
    """Extract palm lines from a BGR image. Always returns a PalmAnalysis dict."""
    try:
        H0, W0 = bgr.shape[:2]
        scale = min(1.0, _PROC_MAX_EDGE / max(H0, W0))
        proc = cv2.resize(bgr, (round(W0 * scale), round(H0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        result = _extract_landmarks(proc)          # primary: MediaPipe landmarks
        if result is None:
            result = _extract_skinseg(proc)        # fallback: YCrCb skin seg
        return result if result is not None else fallback_analysis()
    except Exception:
        return fallback_analysis()


# ── CLI: visual validation ─────────────────────────────────────────────────────
_COLORS = {"life": (0, 220, 0), "head": (255, 180, 0), "heart": (0, 90, 255), "fate": (220, 0, 220)}


def draw_overlay(bgr: np.ndarray, analysis: dict) -> np.ndarray:
    out = bgr.copy()
    H, W = out.shape[:2]
    b = analysis["palmBox"]
    cv2.rectangle(out, (int(b["x"] * W), int(b["y"] * H)),
                  (int((b["x"] + b["w"]) * W), int((b["y"] + b["h"]) * H)), (120, 120, 120), 1)
    for line in analysis["lines"]:
        col = _COLORS[line["key"]]
        pts = [(int(p["x"] * W), int(p["y"] * H)) for p in line["points"]]
        for i in range(len(pts) - 1):
            cv2.line(out, pts[i], pts[i + 1], col, 3, cv2.LINE_AA)
        if pts:
            cv2.putText(out, line["key"] + ("" if line["detected"] else "?"),
                        pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    import json
    import sys
    img = cv2.imread(sys.argv[1])
    if img is None:
        print("could not read", sys.argv[1]); sys.exit(1)
    a = extract_palm(img)
    print(json.dumps({"fallback": a["fallback"], "metrics": {k: round(v, 2) for k, v in a["metrics"].items()},
                      "lines": [{"key": l["key"], "detected": l["detected"], "relLength": round(l["relLength"], 2),
                                 "curvature": round(l["curvature"], 2), "n": len(l["points"])} for l in a["lines"]]}, indent=2))
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/palm_overlay.png"
    cv2.imwrite(out, draw_overlay(img, a))
    print("overlay ->", out)
