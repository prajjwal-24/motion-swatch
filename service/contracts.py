"""MotionLife shared contracts (Step 0 of docs/BUILD_PLAN.md).

Three hand-offs frozen here so every backend plugs into stable shapes:
  A  decomposition  — what the VLM Router returns (this file's focus for Step 1)
  B  swatch         — the unified motion swatch every extractor distills into
  C  judge          — the VLM judge's verdict on applied motion

Rules: add fields, never rename (version bump if a shape must break). stdlib-only
so any venv (incl. the py3.9 pose server) can import it.
"""
import math

SCHEMA_VERSION = 1

# ── Motion taxonomy ────────────────────────────────────────────────────────
# Six classes. Each maps to one backend (extractor) and one applicator. Adding a
# motion type = adding a row here + its extractor/applicator — router never changes.
MOTION_CLASSES = {
    "articulated": {
        "desc": "a body with limbs — human, animal, hand or face — that moves its parts",
        "examples": "a person walking, a duck waddling, a waving hand, a talking face",
        "backend": "pose_mediapipe",     # service/pose_server.py (:8770)
        "applicator": "skeletal",         # rig / puppet
    },
    "cloth": {
        "desc": "a soft sheet rippling while anchored — flag, banner, hair, sail, cloth",
        "examples": "a flag fluttering on a pole, hair blowing, a hanging banner",
        "backend": "flow_raft",           # service/server.py (:8765)
        "applicator": "wave",
    },
    "fluid": {
        "desc": "a continuous flowing medium — water, smoke, fire, steam, ripples",
        "examples": "a waterfall, rising smoke, a campfire, river ripples",
        "backend": "flow_raft",
        "applicator": "flow_field",
    },
    "flock": {
        "desc": "many small similar things drifting together — birds, leaves, a crowd, particles",
        "examples": "a flock of birds, falling leaves, a school of fish",
        "backend": "flow_raft",
        "applicator": "flock_drift",
    },
    "rigid_path": {
        "desc": "ONE discrete solid object travelling across the scene as a whole",
        "examples": "a boat drifting, a car passing, a ball rolling, a single plane",
        "backend": "track_bytetrack",     # planned (Step 5)
        "applicator": "path_travel",
    },
    "oscillation": {
        "desc": "something swaying, bobbing or rocking in place without travelling",
        "examples": "a tree swaying, a hanging sign, a buoy bobbing, a pendulum",
        "backend": "parametric",
        "applicator": "oscillate",
    },
}


# Region-preprocess encodings (Step 2). The mask payload is OPAQUE to this module —
# encode/decode lives in preprocess.py so contracts.py stays stdlib-only (the py3.9
# pose venv imports it and must not pull numpy/cv2).
MASK_ENCODINGS = ("rle", "png_b64", "polygon")
CAMERA_MODELS = ("none", "affine", "homography")


def backend_for(motion_class):
    """Route a class -> its extractor backend id (None if unknown)."""
    return (MOTION_CLASSES.get(motion_class) or {}).get("backend")


def applicator_for(motion_class):
    return (MOTION_CLASSES.get(motion_class) or {}).get("applicator")


# ── Contract A: decomposition (VLM Router output) ──────────────────────────
def empty_decomposition(clip=None):
    return {"version": SCHEMA_VERSION, "clip": clip or {}, "static": True, "motions": []}


def normalize_decomposition(raw, clip=None):
    """Coerce a raw VLM result into a valid Contract-A object.

    Drops motions with an unknown class, clamps bbox/confidence to sane ranges,
    fills the routed backend/applicator from the registry, and assigns stable ids.
    Returns (contract, warnings[]).
    """
    warnings = []
    motions_out = []
    for i, m in enumerate(raw.get("motions", []) if isinstance(raw, dict) else []):
        if not isinstance(m, dict):
            warnings.append(f"motion #{i} not an object; skipped")
            continue
        cls = str(m.get("class", "")).strip().lower()
        if cls not in MOTION_CLASSES:
            warnings.append(f"motion #{i} unknown class {cls!r}; skipped")
            continue
        bbox = m.get("bbox") or [0, 0, 1, 1]
        try:
            x, y, w, h = (float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            warnings.append(f"motion #{i} bad bbox {bbox!r}; defaulted to full frame")
            x, y, w, h = 0.0, 0.0, 1.0, 1.0
        x = _clamp01(x); y = _clamp01(y)
        w = _clamp01(w) or 1.0; h = _clamp01(h) or 1.0
        if x + w > 1: w = 1 - x
        if y + h > 1: h = 1 - y
        conf = m.get("confidence", 0.5)
        try:
            conf = _clamp01(float(conf))
        except (TypeError, ValueError):
            conf = 0.5
        motions_out.append({
            "id": f"m{len(motions_out) + 1}",
            "label": str(m.get("label", cls))[:80],
            "class": cls,
            "bbox": [round(x, 4), round(y, 4), round(w, 4), round(h, 4)],
            "confidence": round(conf, 3),
            "backend": backend_for(cls),
            "applicator": applicator_for(cls),
            "notes": str(m.get("notes", ""))[:240],
        })
    return {
        "version": SCHEMA_VERSION,
        "clip": clip or {},
        "static": len(motions_out) == 0,
        "motions": motions_out,
    }, warnings


def _clamp01(v):
    return 0.0 if v < 0 else 1.0 if v > 1 else v


# ── Contract: region_preprocess (Step 2 — mask + camera + optional depth) ───
# Produced by the preprocess service (:8772) for ONE Contract-A motion. Turns a
# rough seed bbox into a clean object mask, plus per-frame camera motion so the
# consumer can subtract it. The mask.data payload stays opaque here (see
# preprocess.encode_mask_rle / decode_mask_rle).
def empty_region_preprocess(clip=None, motion_id="", cls="", seed_bbox=None):
    return {
        "version": SCHEMA_VERSION,
        "kind": "region_preprocess",
        "clip": clip or {},
        "motion_id": motion_id,
        "class": cls,
        "seed_bbox": _clamp_bbox(seed_bbox or [0, 0, 1, 1]),
        "mask": None,
        "camera": {"is_static": True, "model": "none", "reference": "frame0",
                   "per_frame": [], "residual_px": 0.0},
        "depth": None,
        "engine": "",
        "warnings": [],
    }


def normalize_region_preprocess(raw, clip=None):
    """Validate/clamp a raw region_preprocess dict into the frozen shape.

    Only clamps seed_bbox and sanity-checks the (opaque) mask + camera metadata —
    it never touches mask.data pixel content. Returns (contract, warnings).
    """
    warnings = []
    if not isinstance(raw, dict):
        return empty_region_preprocess(clip), ["raw preprocess result was not an object"]

    out = empty_region_preprocess(clip, str(raw.get("motion_id", "")),
                                  str(raw.get("class", "")), raw.get("seed_bbox"))

    mask = raw.get("mask")
    if isinstance(mask, dict):
        enc = str(mask.get("encoding", "")).lower()
        if enc not in MASK_ENCODINGS:
            warnings.append(f"unknown mask encoding {enc!r}; mask dropped")
        elif not isinstance(mask.get("w"), int) or not isinstance(mask.get("h"), int):
            warnings.append("mask missing integer w/h; mask dropped")
        else:
            out["mask"] = {
                "encoding": enc,
                "w": int(mask["w"]), "h": int(mask["h"]),
                "data": mask.get("data"),               # opaque
                "bbox": _clamp_bbox(mask.get("bbox") or out["seed_bbox"]),
                "coverage": _clamp01(_as_float(mask.get("coverage"), 0.0)),
                "method": str(mask.get("method", "")),   # e.g. grabcut+motion / *_fallback
            }

    cam = raw.get("camera")
    if isinstance(cam, dict):
        model = str(cam.get("model", "none")).lower()
        if model not in CAMERA_MODELS:
            warnings.append(f"unknown camera model {model!r}; treated as none")
            model = "none"
        out["camera"] = {
            "is_static": bool(cam.get("is_static", True)),
            "model": model,
            "reference": str(cam.get("reference", "frame0")),
            "per_frame": cam.get("per_frame") if isinstance(cam.get("per_frame"), list) else [],
            "residual_px": round(_as_float(cam.get("residual_px"), 0.0), 3),
        }

    if raw.get("depth") is not None:
        out["depth"] = raw["depth"]
    out["engine"] = str(raw.get("engine", ""))
    if isinstance(raw.get("warnings"), list):
        warnings = list(raw["warnings"]) + warnings
    out["warnings"] = warnings
    return out, warnings


def _clamp_bbox(b):
    try:
        vals = [float(v) for v in list(b)[:4]]
    except (TypeError, ValueError):
        return [0.0, 0.0, 1.0, 1.0]
    if len(vals) < 4 or not all(math.isfinite(v) for v in vals):   # reject NaN/inf
        return [0.0, 0.0, 1.0, 1.0]
    x, y, w, h = vals
    x = _clamp01(x); y = _clamp01(y)
    w = _clamp01(w) or 1.0; h = _clamp01(h) or 1.0
    if x + w > 1: w = 1 - x
    if y + h > 1: h = 1 - y
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


def _as_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
