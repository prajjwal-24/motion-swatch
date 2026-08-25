"""Steps 7 and 9 done-when, as an executable check:

  Step 7  "a texture, a skeleton, and a path swatch all validate against one schema"
  Step 9  "the judge's verdict is bounded, and the tune loop terminates" — Contract C,
          including the whole loop policy, exercised with no VLM anywhere near it.

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


def expect_j_valid(name, j):
    ok, errs = C.validate_judgement(j)
    check(name, ok, "; ".join(errs))
    return ok


def expect_j_invalid(name, j, must_mention=""):
    ok, errs = C.validate_judgement(j)
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

    print("\n(Step 8) routing takes BOTH axes — kind decides what is possible, class picks:")
    check("texture/flock -> flock_drift", tex["applicator"] == "flock_drift")
    check("path/rigid_path -> path_travel", pth["applicator"] == "path_travel")
    check("skeleton/articulated -> skeletal", skl["applicator"] == "skeletal")
    check("cloth texture -> wave",
          C.swatch_applicator("texture", "cloth") == "wave")
    check("fluid texture -> flow_field",
          C.swatch_applicator("texture", "fluid") == "flow_field")
    check("oscillation texture -> oscillate",
          C.swatch_applicator("texture", "oscillation") == "oscillate")
    # the whole reason routing can't use class alone: the SECOND swatch of a boat clip
    tex_rp = C.texture_swatch(PARAMS, TRACKS, 30.0, cls="rigid_path", engine="raft_small")
    check("a rigid_path TEXTURE swatch does NOT go to path_travel — it has no path",
          tex_rp["applicator"] == "oscillate" and tex_rp["class"] == "rigid_path",
          tex_rp["applicator"])
    expect_valid("...and it validates as a texture swatch", tex_rp)
    tex_ar = C.texture_swatch(PARAMS, TRACKS, 30.0, cls="articulated", engine="raft_small")
    check("an articulated TEXTURE swatch does NOT go to the rig — it has no joints",
          tex_ar["applicator"] == "oscillate", tex_ar["applicator"])
    check("an unclassified texture still routes somewhere honest",
          C.texture_swatch(PARAMS, TRACKS, 30.0)["applicator"] == "oscillate")
    check("every applicator in the taxonomy declares what payload it reads",
          set(C.APPLICATORS) <= set(C.APPLICATOR_NEEDS)
          and all(v in C.SWATCH_KINDS for v in C.APPLICATOR_NEEDS.values()),
          f"{C.APPLICATORS} vs {C.APPLICATOR_NEEDS}")
    check("every class routes to an applicator that can read the class's own backend kind",
          all(C.swatch_applicator(k, c) in C.APPLICATORS
              for c in C.MOTION_CLASSES for k in C.SWATCH_KINDS))

    print("\nNEGATIVE — the validator has to reject these:")
    expect_invalid("path_travel on a texture swatch (the class/kind mix-up)",
                   dict(tex, applicator="path_travel"), "reads a 'path' payload")
    expect_invalid("skeletal on a texture swatch", dict(tex, applicator="skeletal"),
                   "reads a 'skeleton' payload")
    expect_invalid("flock_drift on a path swatch", dict(pth, applicator="flock_drift"),
                   "reads a 'texture' payload")
    expect_invalid("an applicator that doesn't exist", dict(tex, applicator="interpretive"),
                   "applicator must be one of")
    expect_invalid("no applicator at all", dict(tex, applicator=None),
                   "applicator must be one of")
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
    n5, w5 = C.normalize_swatch(dict(tex, applicator="path_travel"))
    check("a claimed applicator this kind can't feed is dropped, not obeyed",
          n5["applicator"] == "flock_drift"
          and any("routed to" in w for w in n5["warnings"]), str(w5))
    n6, _ = C.normalize_swatch(dict(tex, applicator="wave"))
    check("...but a legitimate override for the same kind is honoured",
          n6["applicator"] == "wave")
    n4, _ = C.normalize_swatch(C.path_swatch(PATH, engine="yolo_bytetrack"))
    expect_valid("normalize(build(x)) round-trips a real swatch", n4)
    check("...and is unchanged by the round-trip",
          n4 == C.path_swatch(PATH, engine="yolo_bytetrack"))

    # ── Contract C — the judge verdict, and the loop policy around it ────────
    print("\nCONTRACT C — judge verdict:")
    good = C.normalize_judgement(
        {"verdict": "tune", "score": 0.4, "axes": {"speed": 0.2},
         "deltas": {"frequency": 0.3}, "critique": "reads too slow",
         "observations": ["the flag ripples about once a second"]},
        "match_to_reference", 8)[0]
    expect_j_valid("a well-formed verdict validates", good)

    # NEGATIVE — each of these is a way a VLM actually goes wrong.
    hallucinated, w = C.normalize_judgement({"verdict": "tune", "score": 0.4,
                                             "deltas": {"wobbliness": 0.5, "amplitude": 0.1}})
    check("a delta for a param the renderer doesn't have is dropped",
          "wobbliness" not in hallucinated["deltas"]
          and any("unknown param" in x for x in w), str(w))
    runaway, w = C.normalize_judgement({"verdict": "tune", "score": 0.1,
                                        "deltas": {"amplitude": 5.0}})
    check("a runaway delta is capped at the per-iteration limit, not rejected",
          runaway["deltas"]["amplitude"] == C.PARAM_DELTA_MAX["amplitude"]
          and any("cap" in x for x in w), str(runaway["deltas"]))
    expect_j_valid("...and the capped verdict is still actionable", runaway)
    expect_j_invalid("a score outside [0,1] is rejected",
                     dict(good, score=1.4), "score")
    expect_j_invalid("an unlabelled score is rejected — a bare number is not a claim",
                     dict(good, score_of="vibes"), "score_of")
    expect_j_invalid("a 'tune' verdict with nothing to tune is not actionable",
                     dict(good, deltas={}), "actionable")
    expect_j_invalid("a delta smuggled past the cap is rejected by the validator",
                     dict(good, deltas={"amplitude": 0.9}), "cap")
    expect_j_invalid("an invented axis is rejected", dict(good, axes={"grace": 0.5}), "axis")
    expect_j_invalid("a stale schema_version is rejected",
                     dict(good, schema_version=0), "schema_version")
    silent, w = C.normalize_judgement({"verdict": "good", "score": 0.9,
                                       "deltas": {"amplitude": 0.1}})
    check("'good' plus deltas is contradictory — the deltas go and it's recorded",
          not silent["deltas"] and any("ignoring the deltas" in x for x in w), str(w))
    check("a non-object verdict degrades instead of throwing",
          C.normalize_judgement("looks fine to me")[0]["score"] == 0.0)

    print("\nCONTRACT C — apply_deltas:")
    base = {"frequency": 1.0, "amplitude": 0.95, "direction": 350.0,
            "turbulence": 0.5, "damping": 0.1, "phaseSpread": 0.3,
            "driftX": 0.0, "driftY": 0.0, "leafFall": True}
    out, notes = C.apply_deltas(base, {"amplitude": 0.2, "direction": 45.0, "frequency": 0.5})
    check("an offset can lift a param off zero (a multiplier could not)",
          C.apply_deltas({"amplitude": 0.0}, {"amplitude": 0.2})[0]["amplitude"] == 0.2)
    check("direction wraps rather than clamping", out["direction"] == 35.0, out["direction"])
    check("a param pushed past its range is clamped and reported",
          out["amplitude"] == 1.0 and any("clamped" in n for n in notes), str(notes))
    check("non-param keys survive untouched", out["leafFall"] is True)
    check("apply_deltas does not mutate its input", base["frequency"] == 1.0)
    check("every renderer param has a range and a delta cap",
          set(C.PARAM_RANGES) == set(C.PARAM_KEYS) == set(C.PARAM_DELTA_MAX))
    check("no delta cap exceeds its param's own range",
          all(C.PARAM_DELTA_MAX[k] <= C.PARAM_RANGES[k][1] - C.PARAM_RANGES[k][0]
              for k in C.PARAM_KEYS))
    check("three capped iterations cannot walk a dial across its whole range",
          all(C.JUDGE_MAX_ITERS * C.PARAM_DELTA_MAX[k]
              < C.PARAM_RANGES[k][1] - C.PARAM_RANGES[k][0]
              for k in C.PARAM_KEYS if k not in C.PARAM_CIRCULAR))

    # ── the loop, with no VLM anywhere near it ───────────────────────────────
    print("\nCONTRACT C — loop policy (no VLM required):")
    def v(score, verdict="tune", deltas=None):
        return C.normalize_judgement({"verdict": verdict, "score": score,
                                      "deltas": deltas if deltas is not None
                                      else {"amplitude": 0.1}})[0]

    check("an empty history means 'go judge it'", C.judge_should_continue([])[0] is True)
    check("a good-enough score stops the loop",
          C.judge_should_continue([v(0.85)])[0] is False)
    check("a wrong applicator stops the loop instead of nudging dials",
          C.judge_should_continue([v(0.2, "wrong_class", {})])[0] is False)
    check("...and says why", "applicator is wrong"
          in C.judge_should_continue([v(0.2, "wrong_class", {})])[1])
    check("a low score with deltas keeps going",
          C.judge_should_continue([v(0.3)])[0] is True)
    check("a verdict with no deltas stops the loop",
          C.judge_should_continue([v(0.3, "tune", {})])[0] is False)
    check("improvement below the threshold stops the loop",
          C.judge_should_continue([v(0.30), v(0.31)])[0] is False)
    check("clear improvement continues it",
          C.judge_should_continue([v(0.30), v(0.50)])[0] is True)
    check("a regression stops the loop",
          C.judge_should_continue([v(0.50), v(0.20)])[0] is False)
    check(f"the cap holds even while improving ({C.JUDGE_MAX_ITERS} iterations)",
          C.judge_should_continue([v(0.1), v(0.3), v(0.5)])[0] is False)
    check("...and the reason names the cap",
          "cap" in C.judge_should_continue([v(0.1), v(0.3), v(0.5)])[1])
    check("every stop carries a reason worth showing",
          all(C.judge_should_continue(h)[1]
              for h in ([], [v(0.9)], [v(0.5), v(0.2)], [v(0.2, "wrong_class", {})])))
    # the loop stops when a pass FAILS to improve, so the last state is the bad one
    check("the best iteration is identified, not the last",
          C.judge_best([v(0.3), v(0.7), v(0.4)]) == 1)
    check("ties keep the earlier, cheaper iteration",
          C.judge_best([v(0.5), v(0.5)]) == 0)
    check("no history has no best", C.judge_best([]) == -1)
    # a full run, driven only by the policy
    hist, params = [], {"amplitude": 0.1, "frequency": 1.0}
    for score in (0.30, 0.55, 0.80):
        go, _ = C.judge_should_continue(hist)
        if not go:
            break
        hist.append(v(score))
        params, _ = C.apply_deltas(params, hist[-1]["deltas"])
    check("a converging run terminates at the cap having applied every delta",
          len(hist) == 3 and abs(params["amplitude"] - 0.4) < 1e-9,
          f"{len(hist)} iters, amplitude {params.get('amplitude')}")

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
