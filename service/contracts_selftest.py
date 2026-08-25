"""Step 7 done-when, as an executable check:

  "a texture, a skeleton, and a path swatch all validate against one schema"

Run it with ANY python3 — contracts.py is stdlib-only on purpose, so this works in the
RAFT venv, the py3.9 MediaPipe venv, or the system interpreter:

  python3 service/contracts_selftest.py

Two halves, both of which matter:
  POSITIVE  one swatch of each kind, built by its own builder from realistically-shaped
            backend output, must pass validate_swatch().
  NEGATIVE  a validator that accepts everything proves nothing. Each case below is a
            way a backend could actually get this wrong (a path swatch still carrying
            texture params, `frames` disagreeing with the payload length, an
            unlabelled confidence), and the validator must REJECT it.

`--json` dumps the three passing swatches so you can eyeball the shape.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contracts as C

FAIL = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)


def expect_valid(name, sw):
    ok, errs = C.validate_swatch(sw)
    check(name, ok, "; ".join(errs))
    return ok


def expect_invalid(name, sw, must_mention=""):
    ok, errs = C.validate_swatch(sw)
    blob = " | ".join(errs).lower()
    hit = (not ok) and (must_mention.lower() in blob if must_mention else True)
    check(name, hit, "accepted it" if ok else f"rejected but not for {must_mention!r}: {blob}")


# ── realistic backend output ────────────────────────────────────────────────
# distill() emits exactly these 8 keys (service/distill.py); grid_trajectories()
# emits GRID*GRID tracks of T+1 [x,y] pairs. 200 frames is over TRACK_FRAME_CAP,
# so this also exercises the downsample.
PARAMS = {"frequency": 1.42, "amplitude": 0.31, "direction": 274, "turbulence": 0.18,
          "damping": 0.04, "phaseSpread": 0.62, "driftX": -0.05, "driftY": 0.9}
NF = 200
TRACKS = [[[round(0.05 * g + 0.0004 * t, 4), round(0.07 * g + 0.0011 * t, 4)]
           for t in range(NF)] for g in range(12)]

# objpath.build_path()'s contract shape (service/objpath.py) — offsets from the
# object's own start, so points[0] is [t0, 0, 0].
PATH = {
    "label": "boat", "frames": 60, "fps": 29.97,
    "start": [0.31, 0.62], "size": [0.08, 0.05],
    "points": [[i, round(0.0008 * i, 5), round(-0.0003 * i, 5)] for i in range(60)],
    "travel": {"dx": 0.0472, "dy": -0.0177, "span_x": 0.0472, "span_y": 0.0177,
               "dist": 0.0925, "straightness": 0.5449},
    "confidence": 0.42,
}

# what pose_server's fmt=b hands back (13 joints, [x,y,visibility] per joint)
SKEL = C.empty_skeleton_swatch("pose", "mediapipe_blazepose")
SKEL.update({
    "fps": 15, "viewpoint": "side", "confidence": 0.87,
    "frames": [[[0.5, 0.2 + 0.001 * f, 0.9]] * 13 for f in range(40)],
    "flags": ["ok"] * 38 + ["interp", "ok"],
    "total": 40, "detected": 39, "interpolated": 1,
})


def main():
    print("POSITIVE — one swatch per kind, through its own builder:")
    tex = C.texture_swatch(PARAMS, TRACKS, 30.0, cls="flock", engine="raft_small+raft-grid")
    pth = C.path_swatch(PATH, cls="rigid_path", engine="yolo_bytetrack")
    skl = C.skeleton_swatch(SKEL, cls="articulated", engine="mediapipe_blazepose")
    expect_valid("texture swatch validates", tex)
    expect_valid("skeleton swatch validates", skl)
    expect_valid("path swatch validates", pth)
    check("all three share one core shape",
          {k for k in tex} == {k for k in skl} == {k for k in pth},
          f"{sorted(tex)} vs {sorted(skl)} vs {sorted(pth)}")
    check("one payload each",
          (tex["path"], tex["pose"]) == (None, None)
          and (skl["params"], skl["tracks"], skl["path"]) == (None, None, None)
          and (pth["params"], pth["tracks"], pth["pose"]) == (None, None, None))

    print("\nthe numbers survive the trip:")
    check("params pass through unchanged",
          all(tex["params"][k] == PARAMS[k] for k in PARAMS))
    check("bulk restates distill's drift, it does not re-estimate it",
          (tex["bulk"]["dx"], tex["bulk"]["dy"]) == (PARAMS["driftX"], PARAMS["driftY"]))
    stride = -(-NF // C.TRACK_FRAME_CAP)
    check(f"{NF} frames downsampled by {stride} to {len(tex['tracks'][0])} <= {C.TRACK_FRAME_CAP}",
          tex["track_stride"] == stride and len(tex["tracks"][0]) <= C.TRACK_FRAME_CAP
          and tex["frames"] == len(tex["tracks"][0]))
    check("fps divided by the stride, so playback speed is preserved",
          abs(tex["fps"] * stride - 30.0) < 1e-6, f"fps={tex['fps']} stride={stride}")
    check("track count preserved (only frames are thinned)", len(tex["tracks"]) == len(TRACKS))
    check("coords rounded to 3dp",
          all(len(str(v).split(".")[-1]) <= C.TRACK_COORD_DP
              for tr in tex["tracks"] for pt in tr for v in pt))
    check("path points and travel untouched", pth["path"]["points"] == PATH["points"]
          and pth["path"]["travel"] == PATH["travel"])
    check("skeleton frames untouched", skl["pose"]["frames"] == SKEL["frames"])

    print("\nconfidence is labelled, never silently compared:")
    check("texture -> gated_motion_amplitude", tex["confidence_of"] == "gated_motion_amplitude")
    check("path -> tracked_fraction", pth["confidence_of"] == "tracked_fraction")
    check("skeleton -> its subject's own meaning", skl["confidence_of"] == "mean_visibility")
    check("texture confidence is distill's amplitude", tex["confidence"] == PARAMS["amplitude"])
    check("path confidence is objpath's tracked fraction", pth["confidence"] == PATH["confidence"])
    nc = C.path_swatch(dict(PATH, confidence=None), engine="yolo_bytetrack")
    check("a path built without a clip length says its confidence is unknown",
          nc["confidence"] == 0.0 and "unknown" in nc["confidence_of"])
    expect_valid("...and still validates", nc)

    print("\nNEGATIVE — the validator has to reject these:")
    expect_invalid("no kind", dict(tex, kind="vibes"), "kind must be one of")
    expect_invalid("wrong schema_version", dict(tex, schema_version=99), "schema_version")
    expect_invalid("texture with no tracks", dict(tex, tracks=[]), "non-empty tracks")
    expect_invalid("texture with no bulk", dict(tex, bulk=None), "bulk")
    expect_invalid("texture missing a param",
                   dict(tex, params={k: v for k, v in tex["params"].items() if k != "driftY"}),
                   "params.drifty")
    expect_invalid("frames disagrees with the tracks", dict(tex, frames=999), "tracks carry")
    expect_invalid("ragged tracks", dict(tex, tracks=[tex["tracks"][0], tex["tracks"][1][:-3]]),
                   "same length")
    expect_invalid("texture carrying a path payload", dict(tex, path=PATH),
                   "must not carry the path payload")
    expect_invalid("path carrying texture params", dict(pth, params=tex["params"]),
                   "must not carry the texture payload")
    expect_invalid("path with one point", dict(pth, path=dict(PATH, points=PATH["points"][:1])),
                   ">= 2 points")
    expect_invalid("frames disagrees with the path", dict(pth, frames=7), "path carries")
    expect_invalid("path with no label", dict(pth, path={k: v for k, v in PATH.items()
                                                         if k != "label"}), "label")
    expect_invalid("skeleton with an empty pose", dict(skl, pose={}), "skeleton swatch")
    expect_invalid("skeleton with no frames",
                   dict(skl, pose=dict(skl["pose"], frames=[])), "non-empty frames")
    expect_invalid("unlabelled confidence", dict(tex, confidence_of=""), "confidence_of")
    expect_invalid("confidence out of range", dict(tex, confidence=1.4), "confidence must be in")
    expect_invalid("impossible fps", dict(tex, fps=0), "fps must be in")
    expect_invalid("not a swatch at all", "nope", "not an object")

    print("\nnormalize_swatch is TOLERANT where validate_swatch is strict:")
    n1, w1 = C.normalize_swatch({"kind": "texture", "class": "flock",
                                 "tracks": TRACKS[:2], "frames": NF, "fps": 30})
    check("fills missing params and says so", n1["params"]["driftX"] == 0.0
          and any("missing params" in w for w in w1), str(w1))
    expect_valid("...and the filled result validates", n1)
    check("the guess is recorded IN the swatch, not just returned alongside it",
          any("missing params" in w for w in n1["warnings"]), str(n1["warnings"]))
    # the line normalize does NOT cross: it fills in scalars, it never invents motion
    n0, _ = C.normalize_swatch({"kind": "texture", "class": "flock", "fps": 30})
    expect_invalid("a texture with no tracks stays invalid — normalize won't invent motion",
                   n0, "non-empty tracks")
    n2, w2 = C.normalize_swatch({"kind": "skeleton", "pose": SKEL})
    check("a normalized skeleton inherits its subject's confidence label",
          n2["confidence_of"] == "mean_visibility", n2["confidence_of"])
    n3, w3 = C.normalize_swatch({"kind": "interpretive_dance"})
    check("an unknown kind is reported, not guessed at",
          n3["kind"] == "texture" and any("unknown swatch kind" in w for w in w3), str(w3))
    n4, _ = C.normalize_swatch(C.path_swatch(PATH, engine="yolo_bytetrack"))
    expect_valid("normalize(build(x)) round-trips a real swatch", n4)
    check("...and is unchanged by the round-trip",
          n4 == C.path_swatch(PATH, engine="yolo_bytetrack"))

    if "--json" in sys.argv:
        print("\n" + json.dumps({"texture": dict(tex, tracks=[tex["tracks"][0][:4], "…"]),
                                 "skeleton": dict(skl, pose=dict(skl["pose"], frames=["…"])),
                                 "path": dict(pth, path=dict(pth["path"],
                                                             points=pth["path"]["points"][:3] + ["…"]))},
                                indent=2))

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
