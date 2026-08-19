"""MotionLife shared contracts (Step 0 of docs/BUILD_PLAN.md).

Three hand-offs frozen here so every backend plugs into stable shapes:
  A  decomposition  — what the VLM Router returns (this file's focus for Step 1)
  B  swatch         — the unified motion swatch every extractor distills into
  C  judge          — the VLM judge's verdict on applied motion

Rules: add fields, never rename (version bump if a shape must break). stdlib-only
so any venv (incl. the py3.9 pose server) can import it.
"""

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
