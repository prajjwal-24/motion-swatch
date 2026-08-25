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


# Every applicator the taxonomy names, derived from the table so the two can't drift.
APPLICATORS = tuple(dict.fromkeys(r["applicator"] for r in MOTION_CLASSES.values()))

# What each applicator READS (Step 8). An applicator can only run on a swatch whose
# `kind` supplies its payload — a rig needs joints, path_travel needs points, the field
# applicators need a flow field. This table is why routing takes (kind, class) and not
# class alone, and it is what validate_swatch() checks the two against.
APPLICATOR_NEEDS = {
    "skeletal": "skeleton",       # pose payload: named joints per frame
    "wave": "texture",            # flow field, applied as coherent cloth deformation
    "flow_field": "texture",      # flow field, applied as laminar surface flow
    "flock_drift": "texture",     # flow field, applied per child element
    "path_travel": "path",        # travel path: [frame, dx, dy] offsets
    "oscillate": "texture",       # params only — the in-place whole-object default
}


# ── Contract A: decomposition (VLM Router output) ──────────────────────────
def empty_decomposition(clip=None):
    return {"version": SCHEMA_VERSION, "clip": clip or {}, "static": True, "motions": []}


def _enum(v, allowed, default):
    v = str(v).lower() if v is not None else ""
    return v if v in allowed else default


def _count_bucket(v):
    """Bucket a count into 'one'|'many' — handles numerics (2, '5') and synonyms."""
    try:
        return "many" if float(v) > 1 else "one"
    except (TypeError, ValueError):
        s = str(v).lower().strip()
        return "many" if s in ("many", "few", "several", "multiple", "group", "flock", "crowd") else "one"


def normalize_decomposition(raw, clip=None, has_text_prompt=False):
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
            # sub-attributes that steer extractors.resolve_best() to the right model
            "subject_type": _enum(m.get("subject_type"), ("human", "animal", "object"), "object"),
            "count": _count_bucket(m.get("count")),
            "has_text_prompt": bool(has_text_prompt),
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


# ── Contract B: skeleton swatch (Step 3 — articulated bodies) ───────────────
# One shape for pose / hands / face skeletons. The pose form is a strict SUPERSET
# of today's :8770 /extract response (it adds schema_version/kind/subject/edges/
# viewpoint/confidence but keeps joints/fps/frames/detected/total byte-identical), so
# the existing character rig keeps working. Frame payloads differ per subject:
#   pose : [ [x,y,c] x 13 ]            (c = visibility)
#   hands: [ {label,score,pts:[[x,y,z] x 21]} ]  (0..2 hands; NO per-point visibility)
#   face : [ [x,y,z] x 468 ]          (NO per-point visibility)
SUBJECTS = ("pose", "hands", "face")
VIEWPOINTS = ("front", "side", "unknown")

POSE_JOINTS = ["nose", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
               "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank"]
POSE_EDGES = [[0, 1], [0, 2], [1, 2], [1, 3], [3, 5], [2, 4], [4, 6],
              [1, 7], [2, 8], [7, 8], [7, 9], [9, 11], [8, 10], [10, 12]]

HAND_JOINTS = ["wrist",
               "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
               "index_mcp", "index_pip", "index_dip", "index_tip",
               "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
               "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
               "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
              [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
              [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]]

SUBJECT_JOINTS = {"pose": POSE_JOINTS, "hands": HAND_JOINTS, "face": []}
SUBJECT_EDGES = {"pose": POSE_EDGES, "hands": HAND_EDGES, "face": []}


# What the single `confidence` scalar MEANS per subject — it is NOT comparable across
# subjects, so we label it rather than pretend one 0..1 quality applies to all.
CONFIDENCE_OF = {"pose": "mean_visibility", "hands": "handedness_score",
                 "face": "detection_ratio"}


def empty_skeleton_swatch(subject, engine=""):
    subject = subject if subject in SUBJECTS else "pose"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "skeleton",
        "subject": subject,
        "engine": engine,
        "joints": list(SUBJECT_JOINTS[subject]),
        "edges": [list(e) for e in SUBJECT_EDGES[subject]],
        "fps": 15,
        "viewpoint": "unknown",
        "detected": 0,          # frames with a REAL detection (never counts interpolated)
        "total": 0,
        "interpolated": 0,      # frames synthesised by short-gap interpolation
        "confidence": 0.0,
        "confidence_of": CONFIDENCE_OF[subject],   # meaning of `confidence` for this subject
        "flags": [],            # per-frame provenance: 'ok' | 'interp' | 'gap'
        "frames": [],
    }


def normalize_skeleton_swatch(raw):
    """Validate/clamp a raw skeleton swatch into the frozen shape. Frame payloads
    are kept as-is (numeric); only scalars/enums are validated. Returns (contract, warnings).

    This is the SKELETON payload. Step 7's unified swatch (see below) nests it under
    `pose` rather than replacing it, so :8770's `?fmt=b` response stays valid as-is."""
    warnings = []
    if not isinstance(raw, dict):
        return empty_skeleton_swatch("pose"), ["raw skeleton swatch was not an object"]
    subject = str(raw.get("subject", "pose"))
    if subject not in SUBJECTS:
        warnings.append(f"unknown subject {subject!r}; defaulted to pose")
        subject = "pose"
    out = empty_skeleton_swatch(subject, str(raw.get("engine", "")))
    if isinstance(raw.get("joints"), list) and raw["joints"]:
        out["joints"] = [str(j) for j in raw["joints"]]
    if isinstance(raw.get("edges"), list):
        out["edges"] = raw["edges"]
    vp = str(raw.get("viewpoint", "unknown"))
    if vp not in VIEWPOINTS:
        warnings.append(f"unknown viewpoint {vp!r}; defaulted to unknown")
        vp = "unknown"
    out["viewpoint"] = vp
    out["fps"] = int(_as_float(raw.get("fps"), 15)) or 15
    out["confidence"] = round(_clamp01(_as_float(raw.get("confidence"), 0.0)), 3)
    if raw.get("confidence_of") in CONFIDENCE_OF.values():
        out["confidence_of"] = raw["confidence_of"]
    frames = raw.get("frames") if isinstance(raw.get("frames"), list) else []
    out["frames"] = frames
    if isinstance(raw.get("flags"), list):
        out["flags"] = raw["flags"]
    out["total"] = int(_as_float(raw.get("total"), len(frames)))
    out["detected"] = int(_as_float(raw.get("detected"),
                                    sum(1 for f in frames if f)))
    out["interpolated"] = int(_as_float(raw.get("interpolated"),
                                         sum(1 for fl in out["flags"] if fl == "interp")))
    return out, warnings


# ── Contract B: the UNIFIED motion swatch (Step 7) ──────────────────────────
# Every backend's output collapses into this one shape. The common core is always
# present; exactly one payload field carries the kind-specific data:
#
#   texture   params + tracks   flow field (cloth / fluid / flock / oscillation)
#   skeleton  pose             a skeleton swatch (see normalize_skeleton_swatch)
#   path      path             one object's travel path (see service/objpath.py)
#
# An applicator reads the core plus only the payload its kind owns, so adding a
# backend never changes this shape — the whole point of Step 0's registry.
#
# `class` and `kind` are NOT the same axis, and routing branches on both — see
# swatch_applicator() below. One clip can yield two swatches with the SAME class and
# different kinds: a rigid_path boat gives a `path` swatch (where the boat goes) AND a
# `texture` swatch (its internal motion, from the flow field). Routing on class alone would
# send that texture swatch to the path_travel applicator, which has no path to follow.
# class = what the motion IS; kind = what shape of data this swatch carries.
SWATCH_KINDS = ("texture", "skeleton", "path")

# What the single `confidence` scalar MEANS per kind. As with skeletons, one 0..1
# number does NOT mean the same thing across kinds, so it is labelled rather than
# silently compared. Skeletons keep their own subject-specific label.
SWATCH_CONFIDENCE_OF = {
    "texture": "gated_motion_amplitude",   # distill's amplitude: static-gated, coverage-weighted
    "skeleton": "",                        # taken from the nested pose payload
    "path": "tracked_fraction",            # fraction of the clip the object was tracked for
}


def swatch_applicator(kind, cls):
    """Which applicator can actually DRIVE this swatch — a function of (kind, class).

    `class` is what the motion IS; `kind` is what shape of data this swatch carries, and
    an applicator can only run on the shape it reads (APPLICATOR_NEEDS). So kind decides
    what is POSSIBLE and class only picks among the applicators that read that kind.

    The case this exists for: one rigid_path clip emits two swatches. The `path` one
    carries the travel, the `texture` one the object's internal motion. `rigid_path` ->
    path_travel is right for the first and wrong for the second — that texture swatch has
    no path — so the texture falls back to the applicator that always reads a flow field.
    Same for an `articulated` texture swatch: no joints, so it cannot drive a rig.
    """
    if kind not in SWATCH_KINDS:
        return ""
    app = applicator_for(cls) or ""
    if APPLICATOR_NEEDS.get(app) == kind:
        return app
    # unclassified, or the class's applicator reads a payload this swatch doesn't carry
    return {"skeleton": "skeletal", "path": "path_travel", "texture": "oscillate"}[kind]

# The 6 renderer dials + the 2 drift terms distill() emits. Frozen: the applicator
# reads these names directly (js/animate.js), so they may be added to, never renamed.
PARAM_KEYS = ("frequency", "amplitude", "direction", "turbulence", "damping",
              "phaseSpread", "driftX", "driftY")

# Bloat control (Step 7's "dense tracks are big"): a 12x12 grid over a 200-frame clip is
# ~500KB of JSON. Cap the sampled frames the way pose caps at 160 and drop a decimal — a
# swatch is a motion SUMMARY, not a lossless recording. The stride is recorded and `fps` is
# divided by it, so playback timing is preserved exactly.
#
# Measured on birds.mp4 (144 tracks x 192 samples, 492 KB), error = the largest distance
# between a dropped raw sample and the line joining the two kept samples that bracket it:
#     cap 192 (stride 1, rounding only)  437 KB  89%   0.34 px on a 480px frame
#     cap 120 (stride 2)                 219 KB  44%   2.11 px          <- chosen
#     cap  60 (stride 4)                 110 KB  22%   5.56 px
#     cap  24 (stride 8)                  55 KB  11%  10.29 px
# 120 halves the payload for 0.4% of a frame; 60 would halve it again but 1.2% starts to
# show as a visible cut corner on a fast track, so this is where it stops.
TRACK_FRAME_CAP = 120
TRACK_COORD_DP = 3         # 3dp of a normalized coord ≈ 0.5px on a 480px analysis frame


def empty_swatch(kind="texture", cls="", engine=""):
    kind = kind if kind in SWATCH_KINDS else "texture"
    cls = cls if cls in MOTION_CLASSES else ""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "class": cls,
        # (Step 8) which applicator this swatch can drive. Carried IN the swatch so the
        # taxonomy stays in this one file — the renderer maps applicator -> method and
        # never has to know the class list, so adding a class stays a one-file change.
        "applicator": swatch_applicator(kind, cls),
        "engine": engine,
        "fps": 15.0,
        "frames": 0,                 # frames this swatch spans (after any stride)
        "confidence": 0.0,
        "confidence_of": SWATCH_CONFIDENCE_OF.get(kind, ""),
        "params": None,              # texture: the 8 distilled dials
        "bulk": None,                # texture: the steady travel params keeps in driftX/Y
        "tracks": None,              # texture: [[ [x,y], ... ] x N] normalized point tracks
        "track_stride": 1,           # frames skipped when sampling `tracks`
        "pose": None,                # skeleton: a normalized skeleton swatch
        "path": None,                # path: an objpath travel contract
        "warnings": [],
    }


def texture_swatch(params, tracks, fps, cls="", engine="", warnings=None):
    """Build a texture swatch from distill()'s params + grid_trajectories()'s tracks.

    Downsamples/rounds `tracks` per TRACK_FRAME_CAP (and divides fps by the stride
    so the motion plays at its real speed). `bulk` restates driftX/driftY as the
    steady travel component — distill already separates it out, so this is a
    relabelling, not a second estimate.
    """
    sw = empty_swatch("texture", cls, engine)
    p = params if isinstance(params, dict) else {}
    sw["params"] = {k: _as_float(p.get(k), 0.0) for k in PARAM_KEYS}
    sw["bulk"] = {"dx": sw["params"]["driftX"], "dy": sw["params"]["driftY"],
                  "units": "+/-1 = 0.5% of frame width per frame (distill's drift scale)"}
    kept, stride = _thin_tracks(tracks)
    sw["tracks"] = kept
    sw["track_stride"] = stride
    sw["frames"] = len(kept[0]) if kept else 0
    sw["fps"] = round(_as_float(fps, 15.0) / stride, 3)
    # a texture swatch is only as trustworthy as the motion distill actually found
    sw["confidence"] = round(_clamp01(sw["params"]["amplitude"]), 3)
    if warnings:
        sw["warnings"] = [str(w) for w in warnings]
    return sw


def _thin_tracks(tracks, cap=TRACK_FRAME_CAP, dp=TRACK_COORD_DP):
    """[[ [x,y], ... ] x N] -> (thinned tracks, stride). Even stride so timing stays linear."""
    if not isinstance(tracks, list) or not tracks:
        return [], 1
    n = max(len(t) for t in tracks if isinstance(t, list)) if tracks else 0
    stride = max(1, -(-n // cap)) if n else 1          # ceil(n / cap)
    out = []
    for tr in tracks:
        if not isinstance(tr, list):
            continue
        out.append([[round(_as_float(pt[0], 0.0), dp), round(_as_float(pt[1], 0.0), dp)]
                    for pt in tr[::stride] if isinstance(pt, (list, tuple)) and len(pt) >= 2])
    return out, stride


def path_swatch(path, cls="rigid_path", engine="", warnings=None):
    """Wrap an objpath.build_path() contract as a unified swatch."""
    sw = empty_swatch("path", cls, engine)
    p = path if isinstance(path, dict) else {}
    sw["path"] = p
    sw["fps"] = round(_as_float(p.get("fps"), 15.0), 3)
    sw["frames"] = int(_as_float(p.get("frames"), len(p.get("points") or [])))
    # objpath's confidence is None when the caller didn't pass a clip length
    sw["confidence"] = round(_clamp01(_as_float(p.get("confidence"), 0.0)), 3)
    if p.get("confidence") is None:
        sw["confidence_of"] = "unknown (clip length not supplied to build_path)"
    if warnings:
        sw["warnings"] = [str(w) for w in warnings]
    return sw


def skeleton_swatch(skel, cls="articulated", engine="", warnings=None):
    """Wrap a normalize_skeleton_swatch() payload as a unified swatch."""
    sw = empty_swatch("skeleton", cls, engine or str((skel or {}).get("engine", "")))
    norm, warn = normalize_skeleton_swatch(skel)
    sw["pose"] = norm
    sw["fps"] = round(_as_float(norm.get("fps"), 15.0), 3)
    sw["frames"] = int(_as_float(norm.get("total"), len(norm.get("frames") or [])))
    sw["confidence"] = round(_clamp01(_as_float(norm.get("confidence"), 0.0)), 3)
    # a skeleton's confidence means whatever its subject says it means
    sw["confidence_of"] = str(norm.get("confidence_of", "")) or "unknown"
    sw["warnings"] = [str(w) for w in (list(warnings or []) + list(warn))]
    return sw


def normalize_swatch(raw):
    """Coerce any raw dict into a valid unified swatch. TOLERANT — clamps and fills
    rather than rejecting; use validate_swatch() when you need a hard yes/no.
    Returns (contract, warnings)."""
    warnings = []
    if not isinstance(raw, dict):
        return empty_swatch(), ["raw swatch was not an object"]
    kind = str(raw.get("kind", "")).lower()
    if kind not in SWATCH_KINDS:
        warnings.append(f"unknown swatch kind {kind!r}; defaulted to texture")
        kind = "texture"
    cls = str(raw.get("class", ""))
    if cls and cls not in MOTION_CLASSES:
        warnings.append(f"unknown motion class {cls!r}; dropped")
        cls = ""
    out = empty_swatch(kind, cls, str(raw.get("engine", "")))
    out["fps"] = round(_as_float(raw.get("fps"), 15.0), 3) or 15.0
    out["frames"] = int(_as_float(raw.get("frames"), 0))
    out["confidence"] = round(_clamp01(_as_float(raw.get("confidence"), 0.0)), 3)
    out["confidence_of"] = str(raw.get("confidence_of", "")) or SWATCH_CONFIDENCE_OF[kind]
    if isinstance(raw.get("warnings"), list):
        out["warnings"] = [str(w) for w in raw["warnings"]]
    # an applicator that reads a payload this kind doesn't carry is dropped, not honoured:
    # obeying it would send the swatch to a method that has nothing to read (Step 8)
    ra = str(raw.get("applicator", ""))
    if ra and APPLICATOR_NEEDS.get(ra) != kind:
        warnings.append(f"applicator {ra!r} reads a {APPLICATOR_NEEDS.get(ra)!r} payload "
                        f"but this is a {kind} swatch; routed to {out['applicator']!r}")
    elif ra:
        out["applicator"] = ra

    if kind == "texture":
        p = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        missing = [k for k in PARAM_KEYS if k not in p]
        if missing:
            warnings.append(f"texture swatch missing params {missing}; defaulted to 0")
        out["params"] = {k: _as_float(p.get(k), 0.0) for k in PARAM_KEYS}
        b = raw.get("bulk") if isinstance(raw.get("bulk"), dict) else {}
        out["bulk"] = {"dx": _as_float(b.get("dx"), out["params"]["driftX"]),
                       "dy": _as_float(b.get("dy"), out["params"]["driftY"]),
                       "units": str(b.get("units", "")) or
                                "+/-1 = 0.5% of frame width per frame (distill's drift scale)"}
        out["tracks"] = raw.get("tracks") if isinstance(raw.get("tracks"), list) else []
        out["track_stride"] = max(1, int(_as_float(raw.get("track_stride"), 1)))
    elif kind == "skeleton":
        out["pose"], w = normalize_skeleton_swatch(raw.get("pose"))
        warnings.extend(w)
        # a skeleton's confidence means whatever its SUBJECT says it means, so the
        # kind-level default is empty and the nested payload supplies the label
        if not str(raw.get("confidence_of", "")):
            out["confidence_of"] = str(out["pose"].get("confidence_of", "")) or "unknown"
    else:
        out["path"] = raw.get("path") if isinstance(raw.get("path"), dict) else {}
    out["warnings"] = out["warnings"] + [w for w in warnings if w not in out["warnings"]]
    return out, warnings


def validate_swatch(sw):
    """STRICT check that `sw` is a usable unified swatch. Returns (ok, errors[]).

    This is the executable form of Step 7's done-when: a texture, a skeleton and a
    path swatch must all pass this one function. Unlike normalize_swatch it fixes
    nothing — an applicator can trust anything that passes.
    """
    e = []
    if not isinstance(sw, dict):
        return False, ["not an object"]
    if sw.get("schema_version") != SCHEMA_VERSION:
        e.append(f"schema_version must be {SCHEMA_VERSION}, got {sw.get('schema_version')!r}")
    kind = sw.get("kind")
    if kind not in SWATCH_KINDS:
        return False, e + [f"kind must be one of {SWATCH_KINDS}, got {kind!r}"]
    if sw.get("class") not in MOTION_CLASSES and sw.get("class") != "":
        e.append(f"class {sw.get('class')!r} is not a known motion class")
    # (Step 8) the applicator must be able to READ this swatch. Checked here, at the
    # contract boundary, so the class/kind mix-up can never reach the renderer.
    app = sw.get("applicator")
    if app not in APPLICATORS:
        e.append(f"applicator must be one of {APPLICATORS}, got {app!r}")
    elif APPLICATOR_NEEDS.get(app) != kind:
        e.append(f"applicator {app!r} reads a {APPLICATOR_NEEDS.get(app)!r} payload "
                 f"but this is a {kind!r} swatch")
    if not isinstance(sw.get("engine"), str):
        e.append("engine must be a string (name the extractor that produced this)")
    fps = sw.get("fps")
    if not isinstance(fps, (int, float)) or not (0 < fps <= 240):
        e.append(f"fps must be in (0, 240], got {fps!r}")
    frames = sw.get("frames")
    if not isinstance(frames, int) or frames < 0:
        e.append(f"frames must be a non-negative int, got {frames!r}")
    conf = sw.get("confidence")
    if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
        e.append(f"confidence must be in [0,1], got {conf!r}")
    if not sw.get("confidence_of"):
        e.append("confidence_of must say what `confidence` measures for this kind")
    if not isinstance(sw.get("warnings"), list):
        e.append("warnings must be a list (empty is fine)")

    # exactly ONE payload for the kind, and nothing from another kind
    payloads = {"texture": ("params", "tracks"), "skeleton": ("pose",), "path": ("path",)}
    for other, keys in payloads.items():
        if other == kind:
            continue
        for k in keys:
            if sw.get(k) not in (None, [], {}):
                e.append(f"{kind} swatch must not carry the {other} payload field {k!r}")

    if kind == "texture":
        p = sw.get("params")
        if not isinstance(p, dict):
            e.append("texture swatch needs a params object")
        else:
            for k in PARAM_KEYS:
                if not isinstance(p.get(k), (int, float)):
                    e.append(f"params.{k} must be numeric, got {p.get(k)!r}")
        b = sw.get("bulk")
        if not isinstance(b, dict) or not all(isinstance(b.get(k), (int, float)) for k in ("dx", "dy")):
            e.append("texture swatch needs bulk {dx, dy} (the steady travel component)")
        tr = sw.get("tracks")
        if not isinstance(tr, list) or not tr:
            e.append("texture swatch needs a non-empty tracks list")
        else:
            bad = next((i for i, t in enumerate(tr)
                        if not isinstance(t, list) or not t
                        or not all(isinstance(pt, list) and len(pt) == 2
                                   and all(isinstance(v, (int, float)) for v in pt) for pt in t)), None)
            if bad is not None:
                e.append(f"tracks[{bad}] is not a list of [x,y] pairs")
            elif len({len(t) for t in tr}) != 1:
                e.append("every track must have the same length (one point per sampled frame)")
            elif len(tr[0]) != frames:
                e.append(f"frames says {frames} but tracks carry {len(tr[0])} points")
        if not isinstance(sw.get("track_stride"), int) or sw["track_stride"] < 1:
            e.append("track_stride must be an int >= 1")
    elif kind == "skeleton":
        p = sw.get("pose")
        if not isinstance(p, dict):
            e.append("skeleton swatch needs a pose payload")
        else:
            if p.get("kind") != "skeleton":
                e.append("pose payload must be a skeleton swatch (kind == 'skeleton')")
            if p.get("subject") not in SUBJECTS:
                e.append(f"pose.subject must be one of {SUBJECTS}, got {p.get('subject')!r}")
            if not isinstance(p.get("frames"), list) or not p["frames"]:
                e.append("pose payload needs a non-empty frames list")
            if p.get("subject") != "face" and not p.get("joints"):
                e.append("pose payload needs named joints so an applicator can retarget")
    else:
        p = sw.get("path")
        if not isinstance(p, dict):
            e.append("path swatch needs a path payload")
        else:
            pts = p.get("points")
            if not isinstance(pts, list) or len(pts) < 2:
                e.append("path payload needs >= 2 points ([frame, dx, dy])")
            elif not all(isinstance(q, list) and len(q) == 3
                         and all(isinstance(v, (int, float)) for v in q) for q in pts):
                e.append("every path point must be [frame, dx, dy], all numeric")
            elif len(pts) != frames:
                e.append(f"frames says {frames} but path carries {len(pts)} points")
            if not isinstance(p.get("travel"), dict):
                e.append("path payload needs a travel summary")
            if not p.get("label"):
                e.append("path payload needs the tracked object's label")
    return not e, e
