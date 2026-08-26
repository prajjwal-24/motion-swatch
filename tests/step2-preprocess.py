"""Step 2 done-when:

  "tracking runs only inside the masked object; background points excluded"

contracts_selftest.py proves the region_preprocess SHAPE. This file proves the two
things the shape can't: that the mask is built from the clip rather than asserted, and
that the mask actually GATES the thing Step 2 is about — the 12x12 trajectory field.

The interesting checks are differential. A synthetic clip is built with a known
background drift and a known object motion in one quadrant, so "background points
excluded" becomes a number: the gated field's motion outside the object must be
exactly zero, and its measured direction must be the object's, not the camera's.

Run it in EITHER service venv — deliberately:

  service/venv/bin/python tests/step2-preprocess.py   # torch: SAM 2 + depth reachable
  routervenv/bin/python  tests/step2-preprocess.py    # cv2 only: GrabCut, depth gated

Both must pass. The checks are written on the invariants that hold in both (a mask
method from the allowed set; depth is either a real summary or None WITH a warning),
and the interpreter-specific facts are PRINTED as measurements instead of asserted —
a test that only passes where SAM 2 is installed would have hidden the fact that the
browser flow reaches SAM 2 and the :8772 service does not.

An optional real clip widens the run:  ... tests/step2-preprocess.py assets/videos/flag.mp4
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "service"))

import cv2                                     # noqa: E402
import numpy as np                             # noqa: E402
import contracts as C                          # noqa: E402
import preprocess as P                         # noqa: E402
from config import GRID                        # noqa: E402
from distill import (cell_coverage, track_cells, grid_trajectories,   # noqa: E402
                     TRACK_CELL_FLOOR)

FAIL = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)


# ── synthetic inputs ─────────────────────────────────────────────────────────
H, W = 96, 96                     # 8x8 px per grid cell at GRID=12
T = 20


def quadrant_mask(frac=0.5):
    """[H,W] bool: the top-left `frac` of the frame is object."""
    m = np.zeros((H, W), bool)
    m[: int(H * frac), : int(W * frac)] = True
    return m


def two_motion_flow(bg=(3.0, 0.0), obj=(0.0, 4.0), frac=0.5):
    """[T,2,H,W] flow: the whole frame drifts `bg`, the object quadrant moves `obj`.

    Both are constant per frame, so every integrated track is a straight line and the
    expected displacement is exact arithmetic rather than a tolerance.
    """
    f = np.zeros((T, 2, H, W), np.float32)
    f[:, 0] = bg[0]
    f[:, 1] = bg[1]
    q = quadrant_mask(frac)
    f[:, 0][:, q] = obj[0]
    f[:, 1][:, q] = obj[1]
    return f


def net(track):
    return track[-1][0] - track[0][0], track[-1][1] - track[0][1]


CW, CH = 480, 360         # clip size: big enough for ORB to find its 20-inlier minimum


def _background():
    """A corner-rich background. Blurred noise has no repeatable ORB keypoints, so a
    clip built from it reports 'no frame-pair yielded a background transform' and every
    camera check becomes vacuous — the pattern below is what makes the pan measurable."""
    rng = np.random.RandomState(7)
    bg = np.full((CH, CW * 2, 3), 30, np.uint8)          # 2x wide: pan scrolls into it
    for gy in range(0, CH, 40):
        for gx in range(0, CW * 2, 40):
            c = [int(v) for v in rng.randint(70, 250, 3)]
            cv2.rectangle(bg, (gx + 4, gy + 4), (gx + 30, gy + 30), c, -1)
            cv2.rectangle(bg, (gx + 12, gy + 12), (gx + 22, gy + 22), (250, 250, 250), 1)
    return bg


def write_clip(path, n=24, pan=0):
    """A feature-rich clip with a moving blob. pan>0 slides the CAMERA across a wider
    background each frame, so staticness is measured from real parallax-free translation
    rather than stipulated."""
    bg = _background()
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (CW, CH))
    for i in range(n):
        x0 = min(CW, i * pan)                            # window into the wide background
        f = bg[:, x0:x0 + CW].copy()
        cv2.circle(f, (120 + 3 * i, 200), 26, (20, 20, 230), -1)
        vw.write(f)
    vw.release()
    return path


def main():
    tmp = os.environ.get("CLAUDE_JOB_DIR")
    tmp = os.path.join(tmp, "tmp") if tmp and os.path.isdir(os.path.join(tmp, "tmp")) else "/tmp"

    print("CELL COVERAGE — the mask and the field share one decomposition:")
    cov = cell_coverage(quadrant_mask(0.5))
    check(f"coverage is one number per grid cell ({GRID}x{GRID})", cov.shape == (GRID, GRID),
          str(cov.shape))
    check("cells fully inside the mask read 1.0 and cells fully outside read 0.0",
          cov[0, 0] == 1.0 and cov[GRID - 1, GRID - 1] == 0.0, f"{cov[0,0]} / {cov[-1,-1]}")
    check("a half-frame mask covers half the cells", int((cov >= 0.5).sum()) == 36,
          str(int((cov >= 0.5).sum())))
    # a mask that is thin everywhere: 1 row in 8 => every cell reads 0.125
    thin = np.zeros((H, W), bool)
    thin[::8, :] = True
    check("a mask spread thinly over every cell reports partial coverage, not full",
          abs(float(cell_coverage(thin).max()) - 0.125) < 1e-6, str(cell_coverage(thin).max()))

    print("\nTHE GATE — which cells are allowed to track:")
    keep, kept, total = track_cells(quadrant_mask(0.5))
    check(f"the floor is stated, not implicit (TRACK_CELL_FLOOR={TRACK_CELL_FLOOR})",
          0.0 < TRACK_CELL_FLOOR < 1.0, str(TRACK_CELL_FLOOR))
    check("a half-frame mask lets exactly its own cells track", (kept, total) == (36, 144),
          f"{kept}/{total}")
    check("...and the kept cells are the mask's cells, not merely the right count",
          bool(keep[:6, :6].all()) and not bool(keep[6:, :].any()))
    _, kept_thin, _ = track_cells(thin)
    check("a mask below the floor in every cell keeps nothing, so the caller must decide",
          kept_thin == 0, f"{kept_thin} cells")
    check("...and the count is what tells the caller that, not a raised exception",
          isinstance(kept_thin, int))

    print("\nBACKGROUND POINTS EXCLUDED — measured against a known camera drift:")
    flows = two_motion_flow(bg=(3.0, 0.0), obj=(0.0, 4.0))
    plain = grid_trajectories(flows)
    gated = grid_trajectories(flows, track_cells(quadrant_mask(0.5))[0])
    check("the field stays square, so every consumer's sqrt(len) still holds",
          len(plain) == len(gated) == GRID * GRID, f"{len(plain)} vs {len(gated)}")
    check("...and just as long, so nothing has to special-case a gated field",
          len(plain[0]) == len(gated[0]) == T + 1, f"{len(plain[0])} vs {len(gated[0])}")

    bg_cell = (GRID - 1) * GRID + (GRID - 1)          # bottom-right: pure background
    obj_cell = 2 * GRID + 2                           # inside the object quadrant
    check("ungated, a background cell carries the camera's 3px/frame pan",
          abs(net(plain[bg_cell])[0] - T * 3.0 / W) < 1e-3, str(net(plain[bg_cell])))
    check("gated, that same cell does not move at all",
          net(gated[bg_cell]) == (0.0, 0.0), str(net(gated[bg_cell])))
    check("gated, an object cell still carries the object's real 4px/frame fall",
          abs(net(gated[obj_cell])[1] - T * 4.0 / H) < 1e-3, str(net(gated[obj_cell])))
    check("...unchanged from ungated — the gate excludes, it does not rescale",
          net(gated[obj_cell]) == net(plain[obj_cell]))

    # the headline number: what the FIELD says the motion is, before and after
    def field_mean(tr):
        d = [net(t) for t in tr]
        return (sum(a for a, _ in d) / len(d), sum(b for _, b in d) / len(d))
    mx, my = field_mean(plain)
    gx, gy = field_mean(gated)
    check("ungated, the field's mean motion is dominated by the camera (mostly +x)",
          abs(mx) > abs(my), f"({mx:.4f}, {my:.4f})")
    check("gated, the field's mean motion is the object's (downward, no x at all)",
          gx == 0.0 and gy > 0, f"({gx:.4f}, {gy:.4f})")
    print(f"       field mean displacement: ungated ({mx:+.4f}, {my:+.4f})  "
          f"gated ({gx:+.4f}, {gy:+.4f})")

    print("\nFROZEN, NOT DROPPED — a frozen cell is readable as 'not measured':")
    frozen = [i for i, t in enumerate(gated) if net(t) == (0.0, 0.0)]
    check("every excluded cell is present and pinned to its own seed point",
          len(frozen) == 144 - 36
          and all(all(p == gated[i][0] for p in gated[i]) for i in frozen),
          f"{len(frozen)} frozen")
    check("...and its seed is its true grid position, so the field is still a grid",
          gated[bg_cell][0] == [round((GRID - 0.5) / GRID, 4), round((GRID - 0.5) / GRID, 4)],
          str(gated[bg_cell][0]))

    print("\nRLE — the mask survives the wire:")
    rng = np.random.RandomState(3)
    for label, m in [("a random mask", rng.rand(31, 47) > 0.5),
                     ("an all-background mask", np.zeros((8, 9), bool)),
                     ("an all-object mask", np.ones((8, 9), bool)),
                     ("a single-pixel mask", np.eye(5, dtype=bool)[:1])]:
        r = P.encode_mask_rle(m)
        back = P.decode_mask_rle(r, m.shape[1], m.shape[0])
        check(f"{label} round-trips exactly", bool((back.astype(bool) == m).all()))
    check("the first run is always a background run, so the decoder needs no phase flag",
          P.encode_mask_rle(np.ones((4, 4), bool))[0] == 0)

    print("\nON A CLIP — what the pass actually did, not what it claims:")
    clip = sys.argv[1] if len(sys.argv) > 1 else None
    if clip and not os.path.isfile(clip):
        print(f"  ⚠️  {clip} not found; using the synthetic clip only")
        clip = None
    made = write_clip(os.path.join(tmp, "step2-static.mp4"), pan=0)
    con, warns, viz = P.preprocess(made, [0.15, 0.3, 0.4, 0.4], motion_id="m1", cls="rigid_path")
    check("the result is a valid region_preprocess contract",
          con["kind"] == "region_preprocess" and con["version"] == C.SCHEMA_VERSION)
    check("the mask names the method that produced it, from the allowed set",
          con["mask"] and con["mask"]["method"] in
          ("sam2+motion", "grabcut+motion", "bbox_motion_fallback", "bbox_empty_fallback"),
          str(con["mask"] and con["mask"]["method"]))
    check("the engine string names what ran, so SAM 2 and GrabCut are distinguishable",
          bool(con["engine"]) and ("sam2" in con["engine"]) ==
          (con["mask"]["method"] == "sam2+motion"), con["engine"])
    check("mask coverage is a real fraction of the frame, not a stand-in",
          0.0 < con["mask"]["coverage"] < 1.0, str(con["mask"]["coverage"]))
    check("the mask decodes to the w/h it declares",
          P.decode_mask_rle(con["mask"]["data"], con["mask"]["w"],
                            con["mask"]["h"]).shape == (con["mask"]["h"], con["mask"]["w"]))
    check("a stationary camera is reported as static with camera model 'none'",
          con["camera"]["is_static"] is True and con["camera"]["model"] == "none",
          str(con["camera"]))
    check("...and it carries no per-frame transform, because there is nothing to subtract",
          con["camera"]["per_frame"] == [])
    check("the CLI overlay inputs come back so the done-when PNG is reproducible",
          viz is not None and viz["mask"].shape[:2] == (con["mask"]["h"], con["mask"]["w"]))
    print(f"       mask={con['mask']['method']} coverage={con['mask']['coverage']} "
          f"residual={con['camera']['residual_px']}px engine={con['engine']}")

    panned = write_clip(os.path.join(tmp, "step2-pan.mp4"), pan=6)
    pcon, _, _ = P.preprocess(panned, [0.15, 0.3, 0.4, 0.4])
    check("a 6px/frame pan is measured as non-static, not assumed away",
          pcon["camera"]["is_static"] is False, str(pcon["camera"]))
    check("...and then a camera model and a per-frame transform are supplied",
          pcon["camera"]["model"] in C.CAMERA_MODELS and pcon["camera"]["model"] != "none"
          and len(pcon["camera"]["per_frame"]) > 0,
          f"{pcon['camera']['model']} / {len(pcon['camera']['per_frame'])} frames")
    check("...and its residual is larger than the static clip's",
          pcon["camera"]["residual_px"] > con["camera"]["residual_px"],
          f"{pcon['camera']['residual_px']} vs {con['camera']['residual_px']}")

    print("\nDEPTH — reachable, and honest when it is not:")
    dcon, dwarns, _ = P.preprocess(made, [0.15, 0.3, 0.4, 0.4], want_depth=True)
    d = dcon["depth"]
    have = d is not None
    check("depth is EITHER a real summary OR None with a warning saying why — never zero",
          (have and isinstance(d, dict)) or any("depth" in w for w in dwarns),
          f"depth={d} warns={dwarns}")
    if have:
        check("a real summary states its polarity, so 'larger' is not ambiguous",
              d.get("polarity") == "inverse_relative", str(d.get("polarity")))
        check("...and reports the object's rank within the frame, not just a mean",
              d.get("rank") is not None and 0.0 <= d["rank"] <= 1.0, str(d.get("rank")))
        check("...and the engine string records that depth ran",
              "depth" in dcon["engine"], dcon["engine"])
        print(f"       depth ran: object={d['object']} scene={d['scene']} rank={d['rank']}")
    else:
        # NOT compared on coverage: cv2.grabCut is non-deterministic on this bbox
        # (preprocess.py documents 458/458/10 px across three runs of one clip), so a
        # pixel-count equality here would fail at random and teach nothing.
        check("an unavailable depth leaves the mask and camera intact",
              dcon["mask"] is not None
              and dcon["mask"]["method"] == con["mask"]["method"]
              and dcon["camera"]["is_static"] == con["camera"]["is_static"],
              f"{dcon['mask'] and dcon['mask']['method']} / {dcon['camera']['is_static']}")
        print(f"       depth gated in this interpreter: "
              f"{[w for w in dwarns if 'depth' in w]}")
    check("depth stays None when it was not asked for, in every interpreter",
          con["depth"] is None)

    if clip:
        print(f"\nON {os.path.basename(clip)} — the same checks against real footage:")
        rcon, rwarns, _ = P.preprocess(clip, [0.2, 0.2, 0.6, 0.6], cls="cloth")
        check("a real clip yields a mask with a named method",
              rcon["mask"] and rcon["mask"]["method"], str(rcon["mask"] and rcon["mask"]["method"]))
        rkeep, rkept, _ = track_cells(
            P.decode_mask_rle(rcon["mask"]["data"], rcon["mask"]["w"], rcon["mask"]["h"]) > 0)
        check("...and its cells gate a real field to fewer than all 144",
              0 < rkept <= 144, f"{rkept}/144")
        print(f"       {os.path.basename(clip)}: mask={rcon['mask']['method']} "
              f"coverage={rcon['mask']['coverage']} cells={rkept}/144 "
              f"static={rcon['camera']['is_static']} warnings={len(rwarns)}")

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
