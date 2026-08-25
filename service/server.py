"""Motion Swatch analysis service — thin FastAPI layer.

Wiring only: decode (video_io) -> flow (flow/extractors) -> distill -> segment,
exposed via /analyze /health /engines /route. Heavy logic lives in the modules.

Run:  venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765
"""
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


def _log(msg):
    print(msg, file=sys.stderr, flush=True)   # flush so it shows live in the service log


from config import DEVICE
from video_io import read_frames, _crop_frames, align_mask
from flow import ENGINE, raft_flow_series
from distill import distill, grid_trajectories
from segment import segment_regions
import extractors                # pluggable extractor registry (Step 4)
_log("[engines] registered: " + ", ".join(f"{n}:{e.kind}" for n, e in extractors.REGISTRY.items()))

app = FastAPI(title="motion-swatch-service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # localhost demo service
    allow_methods=["*"],
    allow_headers=["*"],
)


def _preprocess_mask(clip_path, bbox_str, want_depth=False):
    """Step 2 integration: run the preprocess helper (object mask + camera motion).

    Returns (raw uint8 mask at preprocess's OWN resolution, region_preprocess contract).
    Deliberately does no resampling — video_io.align_mask owns that, so the mask goes
    through the identical geometry as the frames it selects (see the note there about
    480x270 vs 480x272). preprocess.py is cv2+numpy only, so this runs IN-PROCESS;
    no HTTP hop to :8772. Returns (None, contract) when there's no mask to use.
    """
    import numpy as np
    import preprocess as P
    contract, _warn, _viz = P.preprocess(clip_path, [float(v) for v in bbox_str.split(",")][:4]
                                         if bbox_str else [0, 0, 1, 1],
                                         want_depth=want_depth)
    m = contract.get("mask")
    if not m or not m.get("data"):
        return None, contract
    return P.decode_mask_rle(m["data"], m["w"], m["h"]).astype(np.uint8), contract


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"ok": True, "engine": f"{ENGINE} (torchvision pretrained)", "device": DEVICE}


@app.get("/engines")
def engines():
    """Which extractor backends are installed on this machine, and why gated ones aren't."""
    return {"engines": extractors.available_engines(), "device": DEVICE}


@app.get("/route")
def route(cls: str, subject_type: str = None, count: str = None, has_text_prompt: bool = False):
    """Best AVAILABLE extractor for a motion class + sub-attributes (resolve_best).
    The app calls this after the VLM classifies a clip, then invokes /analyze?engine=/tracker=
    (FLOW/TRAJECTORY) accordingly. Single source of truth for routing + live availability."""
    e, reason = extractors.resolve_best(
        cls, {"subject_type": subject_type, "count": count, "has_text_prompt": has_text_prompt})
    ok = False
    if e:
        try:
            ok = bool(e.probe()[0])
        except Exception:
            ok = False
    _log(f"[route] {cls}/{subject_type}/{count} -> {e.name if e else None} "
         f"({e.kind if e else '-'}, avail={ok}) · {reason}")
    return {"engine": e.name if e else None, "kind": e.kind if e else None,
            "available": ok, "reason": reason}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  engine: str = None, tracker: str = None, preproc: str = None,
                  bbox: str = None, preprocess: int = 0, depth: int = 0):
    # Optional query params select pluggable backends (Step 4). With NO params the
    # response is byte-identical to the pre-Step-4 default (raft_small + RAFT grid).
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    # delete=False: ?preprocess=1 needs the file again (preprocess re-reads the clip)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(await file.read())
        tmp.flush()
        tmp.close()
        frames, fps = read_frames(tmp.name)

        if frames is None or len(frames) < 13:
            return {"ok": False, "error": "could not decode enough frames (need ≥ 13)"}

        notes = []
        region = None          # region_preprocess contract when ?preprocess=1
        raw_mask = None        # uint8 mask at preprocess's own resolution
        obj_mask = None        # bool [H,W] aligned to the flow array (set after flow)
        src_hw = (frames.shape[1], frames.shape[2])   # frame size BEFORE any crop

        # (Step 2) object mask + camera motion, seeded by the bbox.
        if depth and not preprocess:
            notes.append("?depth=1 needs ?preprocess=1 (depth is computed by the "
                         "preprocess pass); no depth returned")
        if preprocess:
            try:
                raw_mask, region = _preprocess_mask(tmp.name, bbox, want_depth=bool(depth))
                if raw_mask is None:
                    notes.append("preprocess found no usable mask; extracted over the full frame")
            except Exception as ex:
                notes.append(f"preprocess failed, extracted over the full frame: {ex}")

        # crop to one detected motion's region. The mask COMPOSES with the crop rather
        # than replacing it (mask ∩ crop): the crop supplies resolution (upscale to >=128px
        # for RAFT) and re-centres the GRIDxGRID trajectory field on the object, while the
        # mask excludes in-rectangle background from distill's statistics. Measured on
        # flag.mp4: mask-only left the trajectory field byte-identical to whole-frame
        # (energy 4.457 both) while the crop gave 10.367 — dropping the crop silently
        # traded the animated motion field for better scalar params.
        cropped = None
        if bbox:
            cropped = _crop_frames(frames, bbox)
            if cropped is None or len(cropped) < 13:
                notes.append(f"bbox {bbox} too small; used full frame")
                cropped = None
            else:
                frames = cropped

        # (0) optional preproc (e.g. EVM motion magnification) — default: frames unchanged.
        # A failed magnification surfaces a note (never a silent no-op) per the honesty rule.
        if preproc:
            pe, pwhy = extractors.resolve(preproc, extractors.PREPROC)
            if pwhy:
                notes.append(pwhy)
            if pe:
                try:
                    frames = pe.load()(frames, fps)
                except Exception as ex:
                    notes.append(f"{preproc} preproc failed, used raw frames: {ex}")

        # (1) FLOW engine — default (no param, or resolves to raft_small) uses the server's
        #     own raft_flow_series so the default path stays byte-identical.
        fe, fwhy = extractors.resolve(engine, extractors.FLOW)
        if fwhy:
            notes.append(fwhy)
        if engine and fe and fe.name != "raft_small":
            try:
                flows = fe.load()(frames)
                used_engine = fe.name
            except Exception as ex:                   # any load/weights failure -> graceful fallback
                flows = raft_flow_series(frames)
                used_engine = ENGINE
                notes.append(f"{fe.name} failed, used {ENGINE}: {ex}")
        else:
            flows = raft_flow_series(frames)
            used_engine = ENGINE

        # align the mask onto the flow array LAST: out_hw absorbs the crop and any
        # shape change a preproc engine introduced, so mask and flow can't drift apart.
        if raw_mask is not None:
            try:
                obj_mask = align_mask(raw_mask, src_hw,
                                      bbox if cropped is not None else None,
                                      flows.shape[-2:])
                if int(obj_mask.sum()) < 4:
                    obj_mask = None
                    notes.append("object mask covers <4 px of the analyzed region; "
                                 "used the whole region")
            except Exception as ex:
                obj_mask = None
                notes.append(f"mask alignment failed, used the whole region: {ex}")

        # the object mask (if any) restricts distill's statistics to the moving object
        params = distill(flows, fps, mask=obj_mask)
        if params is None and obj_mask is not None:
            # a thin mask can starve distill (<4 px, or <12 usable frames) — never let
            # that turn a request that would have worked into an error
            params = distill(flows, fps)
            if params is not None:
                notes.append("masked statistics were too sparse; used the whole region")
                obj_mask = None
        if params is None:
            return {"ok": False, "error": "not enough motion data"}

        # (2) TRAJECTORY engine — default uses the RAFT-integrated grid
        if tracker:
            te, twhy = extractors.resolve(tracker, extractors.TRAJECTORY)
            if twhy:
                notes.append(twhy)
            if te and te.kind == extractors.TRAJECTORY and not twhy:
                try:
                    trajectories = te.load()(frames)
                    used_tracker = te.name
                except Exception as ex:               # graceful fallback to the RAFT grid
                    trajectories = grid_trajectories(flows)
                    used_tracker = "raft-grid"
                    notes.append(f"{te.name} failed, used raft-grid: {ex}")
            else:
                trajectories = grid_trajectories(flows)
                used_tracker = "raft-grid"
        else:
            trajectories = grid_trajectories(flows)
            used_tracker = "raft-grid"

        # NOTE: segment_regions still runs unmasked. It builds its own per-region pixel
        # masks from flow clustering, and with ?bbox= the crop has already localized the
        # field; feeding the object mask in here (cell-coverage gate + per-region
        # intersect) is a refinement, not part of the Step 2 contract.
        regions = segment_regions(flows, fps, filename=file.filename or "")

        resp = {
            "ok": True,
            "engine": f"{used_engine}@{DEVICE}",
            "fps": round(fps, 2),
            "frames_analyzed": int(len(frames)),
            "params": params,
            "trajectories": trajectories,
            "regions": regions,
        }
        # additive fields ONLY when a param was used — keeps the no-query response byte-identical
        if engine or tracker or preproc or bbox or preprocess or depth:
            resp["tracker"] = used_tracker
            if notes:
                resp["notes"] = notes
        if region is not None:                        # Step 2: mask + camera provenance
            m = region.get("mask") or {}
            resp["preprocess"] = {
                "masked": obj_mask is not None,
                # coverage WITHIN the analyzed region — the fraction of pixels that
                # actually shaped the numbers. mask_coverage_frame is the whole-frame
                # figure from the preprocess contract (the two differ once a bbox crops).
                "mask_coverage": (round(float(obj_mask.mean()), 4)
                                  if obj_mask is not None else None),
                "mask_coverage_frame": m.get("coverage"),
                "mask_method": m.get("method"),
                # camera SUMMARY only — the full per-frame transform list is 9 floats x
                # frames and nothing here consumes it; fetch the whole contract from :8772
                # (POST /preprocess) when you need it.
                "camera": {k: (region.get("camera") or {}).get(k)
                           for k in ("is_static", "model", "residual_px")},
                "depth": region.get("depth"),      # None unless ?depth=1 (Depth Anything V2)
                "engine": region.get("engine"),
            }
            for w in (region.get("warnings") or []):
                if w not in notes:
                    notes.append(w)
            if notes:
                resp["notes"] = notes
        _log(f"[analyze] FLOW={used_engine} TRAJ={used_tracker} preproc={preproc or '-'} "
             f"bbox={bbox or '-'} crop={'yes' if cropped is not None else 'no'} "
             f"mask={f'{obj_mask.mean() * 100:.0f}% of region' if obj_mask is not None else 'no'} "
             f"depth={((region or {}).get('depth') or {}).get('rank', '-')} "
             f"frames={len(frames)} regions={len(regions)}"
             + (f" | NOTES: {'; '.join(notes)}" if notes else ""))
        return resp
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
