"""
palm.py — palm-line extraction for the feng-shui palm-reading feature.

Detects the four principal palmar lines (life, head, heart, fate) from a hand
photo and returns them as normalized polylines, matching feng-shui's
`PalmAnalysis` contract (src/services/palmVisionCore.ts) so the existing overlay
+ reading code consume the result unchanged.

This is the server-side counterpart to the in-browser OpenCV.js worker
(palmVision.worker.ts). Same proven pipeline — YCrCb skin segmentation -> palm
box -> crease response -> trace one continuous crease per line band -> smooth +
quadratic fit -> reject scattered traces — but run full-resolution with a
scikit-image ridge filter (cleaner curves than the browser's black-hat) instead
of the phone's 480px / 10 MB-WASM path.

CPU only — no GPU / diffusion models. Always warm on the persistent pod, so there
is no cold start. Always returns a valid PalmAnalysis (canonical fallback on any
failure), mirroring the worker's "never throw" contract.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    from skimage.filters import sato
    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover - black-hat fallback below
    _HAVE_SKIMAGE = False

# ── Tuning (mirrors palmVision.worker.ts) ──────────────────────────────────────
# Per-line bands as fractions of the PALM box (below the finger bases, fingers
# up, thumb on the LEFT). x mirrored at runtime for left hands. axis 'h' = the
# line runs horizontally (fit y over x); 'v' = vertically (fit x over y).
_BANDS = {
    "heart": ("h", 0.10, 0.04, 0.94, 0.24),
    "head":  ("h", 0.06, 0.28, 0.84, 0.52),
    "life":  ("v", 0.00, 0.02, 0.38, 0.92),
    "fate":  ("v", 0.38, 0.30, 0.62, 1.00),
}
_LINE_ORDER = ("heart", "head", "life", "fate")
_PROC_MAX_EDGE = 760  # downscale long edge before processing (coords are normalized)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else float(v)


# ── Palm region (YCrCb skin segmentation, same as the worker) ──────────────────
def _skin_box(bgr: np.ndarray):
    """Return ((x, y, w, h), skin_mask) for the largest skin blob, or None."""
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
    # Keep only the largest blob in the mask (drop stray skin-colored noise).
    clean = np.zeros((H, W), np.uint8)
    cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)
    return (x, y, w, h), clean


def _palm_region(skin: np.ndarray, box):
    """Narrow the full-hand box down to the PALM (below the finger bases) so the
    line bands land correctly regardless of how much finger is in frame. The
    finger-base ('knuckle') line is the first row, scanning down, where the hand
    reaches near its full width — fingers are narrower than palm+thumb."""
    x, y, w, h = box
    x0, x1 = int(x), int(x + w)
    top, bot = int(y), int(y + h)
    widths = (skin[top:bot, x0:x1] > 0).sum(axis=1).astype(np.float32)
    if widths.size == 0 or widths.max() < 1:
        return box
    # Smooth the width profile, then find the first row reaching 75% of max width
    # within the upper 60% of the hand — that's the finger-base line.
    kw = max(3, int(0.03 * len(widths)) | 1)
    sw = cv2.GaussianBlur(widths.reshape(-1, 1), (1, kw), 0).ravel()
    maxw = float(sw.max())
    upper = int(0.6 * len(sw))
    cand = np.where(sw[:upper] >= 0.75 * maxw)[0]
    fb = int(cand[0]) if cand.size else int(0.35 * len(sw))
    fb = min(max(fb, int(0.10 * len(sw))), int(0.55 * len(sw)))
    palm_top = top + fb
    # Re-bound left/right over the palm rows only (keeps the thumb/thenar).
    sub = skin[palm_top:bot, :] > 0
    cols = np.where(sub.any(axis=0))[0]
    if cols.size == 0:
        return box
    px0, px1 = int(cols.min()), int(cols.max())
    return (px0, palm_top, px1 - px0, bot - palm_top)


def _thumb_on_right(skin: np.ndarray, box) -> bool:
    """Guess handedness from the thumb's sideways protrusion: in the box's middle
    rows, whichever side the skin reaches furthest from center is the thumb side."""
    x, y, w, h = box
    y0, y1 = int(y + 0.32 * h), int(y + 0.68 * h)
    cx = x + w / 2.0
    rows = skin[max(0, y0):max(1, y1), :]
    cols = np.where(rows.max(axis=0) > 0)[0]
    if cols.size == 0:
        return False
    left_reach = cx - cols.min()
    right_reach = cols.max() - cx
    return right_reach > left_reach * 1.08  # bias toward the worker's right-hand default


# ── Crease response ────────────────────────────────────────────────────────────
def _ridge_response(gray: np.ndarray) -> np.ndarray:
    """Return a 0..255 crease-strength map (high where dark thin creases are)."""
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
    """Follow one continuous crease through a band: at each step along the band
    axis, pick the strongest response within a window of the previous position."""
    H, W = R.shape
    x0, y0, x1, y1 = band
    x0 = max(0, int(np.floor(x0))); x1 = min(W - 1, int(np.ceil(x1)))
    y0 = max(0, int(np.floor(y0))); y1 = min(H - 1, int(np.ceil(y1)))
    if axis == "h":
        a0, a1, p0, p1 = x0, x1, y0, y1
    else:
        a0, a1, p0, p1 = y0, y1, x0, x1
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
        v, p = scan(a, max(p0, prev - win), min(p1, prev + win))
        path.append((a, p, v)); prev = p
    prev = sp
    for a in range(mid - 1, a0 - 1, -1):
        v, p = scan(a, max(p0, prev - win), min(p1, prev + win))
        path.insert(0, (a, p, v)); prev = p

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


# ── Canonical fallback (mirrors palmVisionCore.fallbackAnalysis) ───────────────
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


# ── Main extraction ────────────────────────────────────────────────────────────
def extract_palm(bgr: np.ndarray) -> dict:
    """Extract palm lines from a BGR image. Always returns a PalmAnalysis dict."""
    try:
        H0, W0 = bgr.shape[:2]
        scale = min(1.0, _PROC_MAX_EDGE / max(H0, W0))
        proc = cv2.resize(bgr, (round(W0 * scale), round(H0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        H, W = proc.shape[:2]

        res = _skin_box(proc)
        if res is None:
            return fallback_analysis()
        hand_box, skin = res
        mirror = _thumb_on_right(skin, hand_box)
        # Narrow to the palm (below the finger bases) — bands are palm-relative.
        bx, by, bw, bh = _palm_region(skin, hand_box)
        if bw < 8 or bh < 8:
            return fallback_analysis()

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        R = cv2.bitwise_and(_ridge_response(gray), _ridge_response(gray), mask=skin)
        vals = R[skin > 0]
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
            norm = [(_clamp01(p[0] / W), _clamp01(p[1] / H)) for p in fit]
            detected.append({
                "key": key,
                "points": [{"x": x, "y": y} for x, y in norm],
                "relLength": _clamp01(coverage * 0.85 + strength * 0.4),
                "curvature": _curvature(np.asarray(norm)),
                "breaks": int(breaks),
                "detected": True,
            })

        if len(detected) < 2:
            return fallback_analysis(detected)

        crease_density = _clamp01(_density_in(R, skin, bx, by, bw, bh, thr) * 2.2)
        pinky_x = (bx + 0.05 * bw) if mirror else (bx + 0.55 * bw)
        minor_pinky = _clamp01(_density_in(R, skin, pinky_x, by + 0.30 * bh, 0.4 * bw, 0.26 * bh, thr) * 3)
        brightness = _clamp01(float(cv2.mean(gray, skin)[0]) / 255.0)

        return {
            "lines": detected,
            "metrics": {
                "creaseDensity": crease_density,
                "palmAspect": _clamp01(bw / max(1.0, bh)),
                "minorDensityPinky": minor_pinky,
                "brightness": brightness,
            },
            "palmBox": {"x": bx / W, "y": by / H, "w": bw / W, "h": bh / H},
            "fallback": False,
        }
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
                      "lines": [{"key": l["key"], "detected": l["detected"],
                                 "relLength": round(l["relLength"], 2), "curvature": round(l["curvature"], 2),
                                 "n": len(l["points"])} for l in a["lines"]]}, indent=2))
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/palm_overlay.png"
    cv2.imwrite(out, draw_overlay(img, a))
    print("overlay ->", out)
