"""SAM 2 (2.1 hiera-tiny) box-prompted segmentation — the object-mask upgrade over GrabCut.

Step 2's mask term. preprocess.py calls box_mask() when this is importable and falls back
to GrabCut otherwise, so the OpenCV-only path (routervenv, py3.9) keeps working unchanged
and the mask's `method` label always says which one actually ran.

Why it's worth the dependency, measured across 4 clips with YOLO/hand seed boxes as the
FINISHED mask (seg ∩ motion ∩ box, cleaned) as a fraction of the seed box:
    clip       SAM 2   GrabCut
    flag       0.221   0.221     a wash — a high-contrast flag on plain sky suits GrabCut
    walk-man   0.124   0.074     SAM 2 better: a person against a dark textured scene
    boat       0.174   0.192     GrabCut slightly better
    smoke      0.196   0.164     SAM 2 better
So it is NOT a uniform win on coverage. It earns the default because it never degenerated
to an empty mask, while GrabCut did on ill-conditioned boxes and is non-deterministic
there (identical call -> 458, 458, 10 px). Do not read a single frame as proof either way:
an earlier one-frame comparison suggested a much larger SAM 2 win than 4 clips support.

Cost on this box (MPS): 0.8s cached model load (19.7s the first time, 150MB checkpoint),
0.53s per predict. We prompt ONE representative frame, not the whole clip, so this is a
sub-second addition — image mode, not video propagation. The temporal side is already
covered by preprocess's Farneback motion gate.
"""
import os

import numpy as np

MODEL_ID = os.environ.get("MS_SAM2_MODEL", "facebook/sam2.1-hiera-tiny")
ENABLED = os.environ.get("MS_SAM2", "1") not in ("0", "false", "")   # kill switch
ENGINE = "sam2.1_hiera_tiny"

_PRED = None


def available():
    """(ok, reason) — import-only probe; never loads weights, never downloads."""
    if not ENABLED:
        return False, "disabled by MS_SAM2=0"
    try:
        import torch                                        # noqa: F401
        import sam2                                         # noqa: F401
        from sam2.sam2_image_predictor import SAM2ImagePredictor   # noqa: F401
    except ImportError as ex:
        return False, (f"pip install 'git+https://github.com/facebookresearch/sam2.git' "
                       f"(missing: {ex.name})")
    return True, f"sam2 + {MODEL_ID}"


def _predictor():
    global _PRED
    if _PRED is None:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _PRED = SAM2ImagePredictor.from_pretrained(MODEL_ID, device=dev)
    return _PRED


def box_mask(frame_bgr, box_xywh):
    """Segment the object inside a seed box. Returns (uint8 {0,255} [H,W], score float).

    box_xywh: (x, y, w, h) in PIXELS of frame_bgr — the same tuple preprocess passes to
    _build_mask. `score` is SAM 2's predicted IoU for the returned mask; it is NOT a
    detection confidence, so callers should not gate on it (a waving flag scores ~0.30
    while producing a visibly correct mask).

    The mask may extend slightly OUTSIDE the seed box — SAM 2 follows the object, and the
    router's boxes are hints. preprocess intersects with the box afterwards, which keeps
    the seed authoritative; drop that intersect if you ever want the box to be advisory.
    """
    import cv2
    x, y, bw, bh = (int(v) for v in box_xywh)
    p = _predictor()
    p.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = p.predict(box=np.array([[x, y, x + bw, y + bh]], dtype=float),
                                multimask_output=False)
    m = (np.squeeze(np.asarray(masks)) > 0).astype(np.uint8) * 255
    return m, float(np.ravel(scores)[0]) if np.size(scores) else 0.0
