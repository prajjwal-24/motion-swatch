#!/usr/bin/env python3
"""Step 6 — the text2motion GATE and the SMPL-22 -> Contract-B converter.

Run with any interpreter in the repo; needs nothing installed and nothing running:

    routervenv/bin/python tests/step6-text2motion.py
    service/venv/bin/python tests/step6-text2motion.py

This suite is unusual in that MOST OF IT ASSERTS AN ABSENCE. Step 6's done-when («"a
person waves" -> a usable skeletal swatch drives a rigged character») is not met in this
checkout: MoMask's checkpoints are not vendored and CLIP is not installed. So the thing
worth pinning is that the absence behaves — the engine probes False, its reason names the
specific missing file or module, calling it RAISES instead of returning a plausible
skeleton, and routing walks past it to a real pose extractor. A regression here would not
break the app; it would make the app quietly lie, which is worse.

The converter half is a real test of real code: MoMask's (T, 22, 3) output shape is fixed
and public, so the index mapping, the Y flip and the per-frame bbox normalisation can all
be checked against hand-computed values without any weights.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "service"))

import contracts                                        # noqa: E402
import extractors as X                                  # noqa: E402
import t2m                                              # noqa: E402

CHECKS = [0]
FAILS = []


def ok(cond, what):
    CHECKS[0] += 1
    if not cond:
        FAILS.append(what)
        print(f"  FAIL {what}")


def eq(a, b, what):
    ok(a == b, f"{what}: expected {b!r}, got {a!r}")


def close(a, b, what, tol=1e-4):
    ok(abs(a - b) <= tol, f"{what}: expected ~{b}, got {a}")


# ── 1. The engine is registered, is a text2motion engine, and is HONEST ──────
print("1. registry + gate")
eng = X.REGISTRY.get("momask")
ok(eng is not None, "momask is registered")
eq(eng.kind, X.TEXT2MOTION, "momask.kind")
ok("momask-codes" in eng.describe, "describe names the repo actually vendored")
ok("MDM" not in eng.describe and "MotionGPT" not in eng.describe,
   "describe no longer advertises MDM/MotionGPT (the old row's lie)")
ok(not eng.default, "a text2motion engine is never a default (nothing may route to it blind)")

avail, why = eng.probe()
ok(avail is False, "momask probes False in this checkout (no checkpoints, no CLIP)")
# The whole point of the retarget: a reason that names the missing THING. "not set up"
# would be indistinguishable from a probe that never looked.
named = any(s in why for s in ("clip", "torch", "numpy", "einops", "tqdm",
                               ".tar", ".npy", "momask-codes"))
ok(named, f"probe reason names a specific missing module or file (got {why!r})")
ok(why != t2m.SETUP_HINT, "the reason is the specific gap, not the generic setup hint")

# probe() must stay dependency-free — it runs on every GET /engines.
src = open(os.path.join(HERE, "..", "service", "t2m.py")).read()
head = src.split("def available", 1)[0]
for heavy in ("import torch", "import numpy", "import cv2", "import clip"):
    ok(heavy not in head, f"t2m.py does not {heavy} at module level (probe stays cheap)")

# The gated engine RAISES. It does not return a hand-authored wave.
gen = eng.load()
raised = None
try:
    gen("a person waves")
except RuntimeError as ex:
    raised = str(ex)
except Exception as ex:                                  # noqa: BLE001
    raised = f"WRONG EXCEPTION TYPE: {type(ex).__name__}: {ex}"
ok(raised is not None, "generate() raises when the engine is not set up")
ok(raised is not None and raised.startswith("momask text2motion is not set up"),
   f"the raise says what is wrong (got {raised!r})")

# ── 2. The `mdm` alias still resolves ───────────────────────────────────────
print("2. the mdm alias")
alias = X.REGISTRY.get("mdm")
ok(alias is not None, "the old `mdm` name is still registered")
eq(alias.kind, X.TEXT2MOTION, "mdm.kind")
ok("alias" in alias.describe.lower(), "mdm's describe says it is an alias")
eq(alias.probe(), eng.probe(), "the alias probes identically to momask")

e_a, why_a = X.resolve("mdm", X.TEXT2MOTION)
ok(e_a is None, "resolve('mdm') has no default to fall back to — text2motion has none")
ok(why_a is not None and "not a registered" not in why_a,
   f"?engine=mdm is still RECOGNISED, just unavailable (got {why_a!r})")
ok(why_a is not None and "mdm unavailable" in why_a, "the reason names mdm")

# ── 3. Routing FALLS BACK rather than crashing ──────────────────────────────
print("3. fallback, not crash")
best, reason = X.resolve_best("articulated", {"subject_type": "human",
                                              "has_text_prompt": True})
ok(best is not None, "a text-prompted human still resolves to SOME engine")
ok(best.name != "momask", "it is not momask (which is unavailable)")
ok(best.name in ("wham", "pose_mediapipe", "keypointrcnn"),
   f"it fell back down the preference list (got {best.name!r})")
ok("momask" in reason, "the reason names momask as skipped")
ok(any(s in reason for s in ("clip", "torch", ".tar", "momask-codes")),
   f"the reason carries momask's specific missing piece (got {reason!r})")

# MoMask is HumanML3D — human motion only. An animal text prompt must NOT route to it.
_names = [n for w, names in X.ROUTING_TABLE["articulated"]
          for n in names if w.get("subject_type") == "animal"]
ok("momask" not in _names and "mdm" not in _names,
   "no text2motion engine on the animal rows (MoMask has no animal motions)")

# ── 4. The converter: SMPL-22 -> Contract B's 13 joints ─────────────────────
print("4. converter shape + mapping")
eq(len(t2m.SMPL22_TO_POSE13), 13, "13 joints mapped")
eq([n for _i, n, _w in t2m.SMPL22_TO_POSE13], contracts.POSE_JOINTS,
   "the mapping emits Contract B's POSE_JOINTS in order")
eq(sorted([i for i, _n, _w in t2m.SMPL22_TO_POSE13] + t2m.DROPPED_SMPL),
   list(range(22)), "the 13 kept + 9 dropped indices are exactly SMPL-22")
eq(t2m.GEN_FPS, 20, "MoMask samples at 20 fps (HumanML3D), not the pose service's 15")

# A synthetic (T, 22, 3) array: joint j of frame t sits at (x=j, y=j, z=0), so every
# converted value is hand-checkable. This tests the CONVERTER — it is test input, and it
# never becomes a swatch anyone applies.
T = 4
arr = [[[float(j), float(j), 0.0] for j in range(22)] for _t in range(T)]
pose, warns = t2m.momask22_to_pose13(arr)

eq(pose["kind"], "skeleton", "payload kind")
eq(pose["subject"], "pose", "payload subject")
eq(pose["joints"], contracts.POSE_JOINTS, "payload joint names")
eq(pose["fps"], 20, "payload fps")
eq(pose["total"], T, "total frames")
eq(pose["detected"], T, "all frames converted")
eq(pose["flags"], ["ok"] * T, "per-frame flags")
eq(len(pose["frames"][0]), 13, "13 joints per frame")
eq(len(pose["frames"][0][0]), 3, "each joint is [x, y, vis]")

# Index mapping: nose <- SMPL 15, l_ank <- SMPL 7. With x = j, min index kept is
# l_hip = 1 and max is r_wri = 21, so the frame bbox spans 1..21 -> width 20.
xs = [f[0] for f in pose["frames"][0]]
close(min(xs), 0.0, "the smallest kept joint normalises to x=0")
close(max(xs), 1.0, "the largest kept joint normalises to x=1")
jn = {n: pose["frames"][0][i] for i, n in enumerate(pose["joints"])}
close(jn["l_hip"][0], 0.0, "l_hip is SMPL joint 1 (the x-minimum here)")
close(jn["r_wri"][0], 1.0, "r_wri is SMPL joint 21 (the x-maximum here)")
close(jn["nose"][0], (15 - 1) / 20.0, "nose is SMPL joint 15")
close(jn["l_ank"][0], (7 - 1) / 20.0, "l_ank is SMPL joint 7")
close(jn["r_knee"][0], (5 - 1) / 20.0, "r_knee is SMPL joint 5")

# Y FLIP: MoMask is Y-UP, Contract B / the rig is Y-DOWN. With y = j, the joint with the
# LARGEST SMPL y (r_wri, 21) must land at the SMALLEST screen y.
close(jn["r_wri"][1], 0.0, "Y is flipped: the highest SMPL joint gets the smallest screen y")
close(jn["l_hip"][1], 1.0, "Y is flipped: the lowest SMPL joint gets the largest screen y")
close(jn["nose"][1], 1.0 - (15 - 1) / 20.0, "nose's flipped y")
ok(all(0.0 <= v <= 1.0 for f in pose["frames"] for j in f for v in j[:2]),
   "every converted coordinate is inside 0..1")
ok(all(j[2] == 1.0 for f in pose["frames"] for j in f),
   "vis is 1.0 (generated motion is not observed)")

# PER-FRAME normalisation, exactly as pose_server.py does it: give frame 1 a translated,
# scaled copy of frame 0 and the normalised output must be IDENTICAL — that is what
# discards root travel, and it is why the rig animates in place.
arr2 = [[[float(j), float(j), 0.0] for j in range(22)],
        [[3.0 * j + 100.0, 3.0 * j - 50.0, 0.0] for j in range(22)]]
p2, _ = t2m.momask22_to_pose13(arr2)
eq(p2["frames"][1], p2["frames"][0],
   "a translated+scaled frame normalises to the same numbers (root travel is discarded)")

# A degenerate frame (every joint at one point) must divide by the 1e-3 floor, not by 0.
p3, _ = t2m.momask22_to_pose13([[[1.0, 1.0, 0.0]] * 22])
ok(all(v == 0.0 for j in p3["frames"][0] for v in j[:2]),
   "a collapsed frame yields zeros, not a division by zero")

# A short frame is a GAP, never a guessed pose.
p4, w4 = t2m.momask22_to_pose13([[[0.0, 0.0, 0.0]] * 10,
                                 [[float(j), float(j), 0.0] for j in range(22)]])
eq(p4["frames"][0], None, "a frame that is not 22 joints becomes null")
eq(p4["flags"], ["gap", "ok"], "and is flagged as a gap")
eq(p4["detected"], 1, "detected counts only real frames")
ok(any("became gaps" in w for w in w4), "the gap is reported in warnings")

# ── 5. Every fabrication is DISCLOSED ───────────────────────────────────────
print("5. disclosure")
blob = " ".join(warns).lower()
ok("substitution" in blob and "nose" in blob,
   "the warnings say nose is a SUBSTITUTION (SMPL-22 has no nose)")
ok("not observed" in blob or "not measured" in blob,
   "the warnings say vis is not a measurement")
ok("root travel is discarded" in blob, "the warnings say root travel is discarded")
ok("no contract-b slot" in blob or "dropped" in blob,
   "the warnings say which SMPL joints were dropped")
for i in t2m.DROPPED_SMPL:
    ok(str(i) in blob, f"dropped SMPL joint {i} is named in the warnings")

# ── 6. The produced swatch is a VALID unified swatch ────────────────────────
print("6. Contract B")
eq(contracts.GENERATED_CONFIDENCE, "generation_only", "the generated-confidence label")
ok(contracts.GENERATED_CONFIDENCE not in contracts.CONFIDENCE_OF,
   "it is NOT a fourth subject in CONFIDENCE_OF (that table is per subject)")
ok(contracts.GENERATED_CONFIDENCE in contracts.CONFIDENCE_MEANINGS,
   "but it IS an accepted confidence meaning")
for subj, meaning in contracts.CONFIDENCE_OF.items():
    ok(meaning in contracts.CONFIDENCE_MEANINGS,
       f"{subj}'s existing meaning still accepted")

sw = t2m.swatch_from_joints(arr, prompt="a person waves")
valid, errs = contracts.validate_swatch(sw)
ok(valid, f"the generated swatch passes validate_swatch (errors: {errs})")
eq(sw["kind"], "skeleton", "swatch kind")
eq(sw["class"], "articulated", "swatch class")
eq(sw["applicator"], contracts.swatch_applicator("skeleton", "articulated"),
   "swatch applicator (the one that reads a skeleton payload)")
eq(sw["fps"], 20.0, "swatch fps is MoMask's 20, not resampled to 15")
eq(sw["frames"], T, "swatch frame count")
eq(sw["confidence"], 0.0, "a generated swatch claims NO confidence")
eq(sw["confidence_of"], "generation_only",
   "and says why: there is no observation to be confident about")
eq(sw["pose"]["confidence_of"], "generation_only",
   "the nested pose payload survives normalize_skeleton_swatch")
ok(any("a person waves" in w for w in sw["warnings"]),
   "the prompt travels as provenance in warnings")
ok(any("no video was observed" in w for w in sw["warnings"]),
   "and says plainly that no video was observed")

# The rig reads joints BY NAME (js/animate.js _applyCharacter: jn.l_hip, jn.nose), so the
# names and the frame shape are the whole compatibility contract with the character rig.
eq(sw["pose"]["joints"], contracts.POSE_JOINTS, "the rig's joint names, in the rig's order")
for need in ("nose", "l_hip", "r_hip", "l_sho", "r_sho"):
    ok(need in sw["pose"]["joints"], f"js/animate.js reads {need} by name")

# ── 7. Empty / junk input degrades, never invents ───────────────────────────
print("7. junk input")
p5, w5 = t2m.momask22_to_pose13([])
eq(p5["frames"], [], "no frames in, no frames out")
eq(p5["total"], 0, "total 0")
ok(any("no frame converted" in w for w in w5), "and it says so")
p6, w6 = t2m.momask22_to_pose13(None)
eq(p6["frames"], [], "None in, no frames out")
ok(any("not a sequence" in w for w in w6), "and it says so")
p7, w7 = t2m.momask22_to_pose13([["junk"] * 22])
eq(p7["frames"], [None], "unparseable joints become a gap, not zeros-as-a-pose")

print(f"\n{CHECKS[0]} checks, {len(FAILS)} failed")
if FAILS:
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
