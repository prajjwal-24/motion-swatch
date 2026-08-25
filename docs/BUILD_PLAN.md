# MotionLife — Build Plan of Action (pluggable backend registry)

> Step-wise plan to build the HLD in `hld.html`. Each step has an **objective**,
> **how to build it**, the **challenges we'll hit → the solution**, and a **done-when**.
> Ordered so each step is testable on its own and nothing blocks on a later step.
> Companion: `hld.html` (diagram + per-step descriptions), `docs/ARCHITECTURE_FLOW.md`.

**Guiding principle:** the registry is a *dispatch table*. Adding a motion type = adding
one entry (a classifier label + an extractor + an applicator). The router, swatch schema,
and judge never change. Build the skeleton first, then plug backends in one at a time.

---

## Step 0 — Contracts & registry skeleton
**Objective:** freeze the three hand-offs (A: decomposition, B: swatch, C: judge) and a
registry interface so every later step plugs into stable shapes.

**How:** one shared schema module (imported by service + app). A `BACKENDS` registry:
`{ class → { extractor(fn), applicator(fn) } }`. Stub `/decompose`, `/extract`, `/judge`
to return canned contract JSON.

**Challenges → solutions**
- *Schema churn breaks everyone.* → Version the schema; add fields, never rename; one file is the source of truth.
- *Backends live in different Python envs (RAFT=py3.13, MediaPipe=py3.9).* → Run each backend as its own small HTTP service on its own port; the router calls them over HTTP, so env conflicts never touch each other.

**Done-when:** `curl /decompose` and `/judge` return valid A/C JSON; the app renders against stub swatches.

---

## Step 1 — VLM Router (classify + multi-motion decompose)
**Objective:** given a clip, return every distinct motion with its class, box, and confidence.

**How:** sample ~8 frames, send to Claude (vision) with a fixed rubric that outputs strict
JSON (the 6 classes + bbox + confidence). Map each class → a registry entry.

**Challenges → solutions**
- *VLM returns prose / invalid JSON.* → Constrain with a JSON schema + few-shot examples; validate and retry on parse failure.
- *Boxes are approximate.* → Treat them as *hints*; Step 3 (SAM 2) refines them — never track on the raw box.
- *Hallucinated motions on static clips.* → Require a confidence field; drop below threshold; return empty rather than invent (honesty rule).
- *Overlapping motions (smoke over a flag).* → Ask for instance separation; SAM 2 masks resolve the overlap downstream.

**Done-when:** a flag+smoke+birds clip → 3 correctly-classed entries, no filename hints.

---

## Step 2 — Pre-step helpers (SAM 2 / Depth / camera)
**Objective:** turn a rough box into a clean mask; optionally add depth and camera motion.

**How:** SAM 2 video segmentation seeded by the router's box → per-object mask. Depth
Anything V2 for a depth map. Homography/DROID-SLAM to estimate camera motion.

**Challenges → solutions**
- *SAM 2 is heavy / slow.* → Downscale to the analysis width (we already use 480px); mask once, reuse for the whole clip; cache per region.
- *Mask flickers between frames.* → Use SAM 2's *video* mode (temporal propagation), not per-frame image mode.
- *Camera vs object motion is ambiguous.* → Estimate global homography first, subtract it, so what's left is true object motion (fixes the "train-window scenery scrolls" case).

**Done-when:** tracking runs only inside the masked object; background points excluded (verify by overlay).

---

## Step 3 — Backend A: articulated bodies (extend what we have)
**Objective:** robust human + hands + face + animal skeletons.

**How:** MediaPipe Pose ✅ (have it) + add MediaPipe Hands & FaceLandmarker (same stack,
cheap). Animals → MMPose animal model. Optional 3D + world travel → WHAM.

**Challenges → solutions**
- *MediaPipe wheel needs Python ≤3.12.* → Already isolated in its own py3.9 service (`pose_server.py`); keep it separate.
- *Occlusion / fast motion loses joints.* → Use MediaPipe's tracking mode + confidence; interpolate short gaps; drop long ones.
- *Front-facing vs side-view mismatch.* → Record the capture viewpoint in the swatch; the applicator picks marching-in-place vs profile stride accordingly.
- *Animal skeleton ≠ human skeleton.* → Separate joint set per class; the applicator retargets by relative offsets (as our duck already does).

**Done-when:** upload a walk → a clean full-clip skeleton swatch; a hand clip → 21-landmark swatch.

---

## Step 4 — Backend B: texture / fluid / flock (upgrade)
**Objective:** cleaner, longer-range, larger motion for flag/water/clouds/smoke/flock.

**How:** swap `raft_small` → **SEA-RAFT** (drop-in); add **CoTracker3** for long-range
tracks; **keep bulk + residual** (don't delete travel); optional Eulerian magnification.

**Challenges → solutions**
- *Motion looks small/in-place today.* → Stop subtracting the mean unconditionally; split bulk (travel) + residual (texture) and expose a "travel" dial.
- *CoTracker is compute-heavy.* → Track a sparse-but-dense grid (e.g. 32×32) inside the mask only; downscale; cache.
- *RAFT can't see thin rain / smooth slow-mo water.* → Detect near-zero amplitude and report "no usable motion" instead of faking a region.

**Done-when:** applying a captured flow makes the object travel a visibly larger, natural distance without tearing.

**Status (shipped):** the pluggable **extractor registry** (`service/extractors.py` + `/engines`
+ `?engine=/?tracker=/?preproc=`) is built and verified — `raft_small`/`raft_large`,
**CoTracker3** (long-range tracks) and **Eulerian magnification** all run locally; **SEA-RAFT**
is honestly gated (needs a clone + weights, falls back to `raft_small`). The default response
is byte-identical. **Not shipped:** the "travel dial" — folding bulk back into the residual does
NOT increase travel, because the amplitude persistence-gate cancels coherent drift and steady
travel already lives in `driftX/driftY`. Making objects visibly travel farther is a
**renderer-side** change (the bounded `DRIFT_PX` sawtooth in `js/animate.js`), deferred.

---

## Step 5 — Backend C: discrete objects on a path (new)
**Objective:** a boat/car/single bird that *travels* across the scene.

**How:** object detector (YOLO/RT-DETR) + multi-object tracker (ByteTrack) → per-object
centroid path over time. FoundationPose/Objectron for rigid 3D rotation.

**Challenges → solutions**
- *ID switches when objects cross.* → ByteTrack's two-stage association; smooth/interpolate the path; pick the longest stable track.
- *Path is in video pixels, artwork is different size.* → Normalize the path to the scene and to the object's on-canvas travel room (bound it so it can't leave the frame).
- *A single moving object vs. a texture flock.* → The router decides: "one boat" → C; "a flock drifting" → B.

**Done-when:** a boat clip yields a smooth path the artwork's boat follows within the scene bounds.

**Status (shipped):** end to end. `yolo_bytetrack` is the registered **OBJECT_PATH default**;
`service/objpath.py` turns its longest track into a path contract (gap interpolation,
endpoint-pinned smoothing, offsets from the object's OWN start, `travel` + `straightness`,
`confidence` = tracked fraction of the clip); `/analyze?path=1` returns it, computed on the
**full frames** (cropping to where the object starts would clip the travel this backend exists
to measure). Client side, `js/animate.js:_applyPathTravel` follows it ahead of the name-keyed
curated behaviours, with ONE uniform scale to the object's on-canvas room (so a diagonal drift
can't flatten into a vertical one) plus a hard per-frame clamp — the Intensity slider goes to
2×, so the fit alone isn't enough. Loops **ping-pong**: every position shown is a real sample,
only the return leg's time order is reversed (snapping back reads as a teleport; easing back
would be motion that isn't in the video). Verified against the real contract: seam == last
sample, loop returns to the origin, legs mirror, max step 0.15 px/frame, on-canvas at 2× even
when jammed 12 px from the edge. Measured: `boat.mp4` → label boat, 114 frames, dist 0.097,
straightness 0.51, conf 0.42; `birds.mp4` → dist 0.676; `walk-man.mp4` → conf 1.0;
`boat-night.mp4` correctly **rejected as scene-sized** (0.94×0.18 box). `YOLO_CONF` defaults to
**0.15**, not ultralytics' 0.25: at 0.25 a distant boat drops out for a few frames and ByteTrack
issues a new id, fragmenting the "longest track" (4 tracks, longest 10% of the clip → 1 track,
44%).

**Not shipped:** Objectron/FoundationPose rigid **3D rotation** — `objectron` is still an
honestly-gated stub, so a path is 2D translation plus the flow field's residual bob, with no
out-of-plane turn. The path is also **whole-frame**: it takes no bbox, so the multi-motion branch
deliberately does not request one (it could return a different object's travel than the region
being extracted). Same-label track *merging* was not added — the plan's own remedy ("pick the
longest stable track") is what runs, and the confidence gate fixed the fragmentation that
motivated it; `boat.mp4` still only covers 42% of its clip and says so in `notes`.

---

## Step 6 — Backend E: text → motion (new, optional)
**Objective:** generate motion with no source video (the NLP theme).

**How:** MDM / MotionGPT text-to-motion → a skeletal sequence → the same swatch schema.

**Challenges → solutions**
- *Model is large / GPU-hungry.* → Run as an optional service; cache generated clips by prompt; ship a small pre-generated library for the demo.
- *Generated motion may be jittery.* → Smooth + retime; loop-optimize.

**Done-when:** "a person waves" → a usable skeletal swatch drives a rigged character.

---

## Step 7 — Distill → unified swatch schema
**Objective:** every backend's output collapses into one Contract-B swatch.

**How:** per-class distillers write `{class, params, tracks?, pose?, bulk, fps, confidence}`.
The app only ever reads this shape.

**Challenges → solutions**
- *Different backends produce very different data (field vs skeleton vs path).* → Optional fields per class; the applicator reads only what its class needs.
- *Swatch bloat (dense tracks are big).* → Downsample tracks, round coordinates, cap frame count (as we already do for pose).

**Done-when:** a texture, a skeleton, and a path swatch all validate against one schema and appear in the library identically.

**Status (shipped):** one Contract B in `service/contracts.py` covering all three payload
shapes — `SWATCH_KINDS = (texture, skeleton, path)`, a common core
(`schema_version/kind/class/engine/fps/frames/confidence/confidence_of/warnings`) and exactly
ONE payload per swatch (`params`+`tracks` / `pose` / `path`). Builders
`texture_swatch()`/`skeleton_swatch()`/`path_swatch()`, a tolerant `normalize_swatch()`, and a
**strict `validate_swatch()`** so the done-when is executable rather than asserted. Emitted by
both backends behind additive query params — `/analyze?swatch=1[&cls=…]` (:8765) and
`/extract?fmt=swatch` (:8770) — and each service **validates its own output** and reports a
failure in `notes`/`warnings` instead of shipping a bad shape quietly.

`class` and `kind` are separate axes and Step 8 must branch on **both**: one `rigid_path`
clip yields a `path` swatch (where the boat goes) *and* a `texture` swatch (its internal
motion), so routing on class alone would hand the texture swatch to `path_travel`, which has
no path. `/analyze` therefore returns a swatch **list**, primary first — two backends ran, so
saying "two swatches" is honest where folding a path into a texture swatch would not be.
Warnings are attributed per swatch: general notes (engine fallback, mask problems) go on
both, path-tracking caveats only on the path, so the flow field is never blamed for a gap the
object tracker had.

**Verified**, not asserted. `service/contracts_selftest.py` (stdlib-only, runs in the RAFT
venv *and* the MediaPipe venv) is 46 checks: one swatch per kind validates; all three share
one core; one payload each; distill's params and objpath's points survive unchanged; and 18
**negative** cases the validator must reject (a path swatch still carrying texture params,
`frames` disagreeing with the payload length, ragged tracks, an unlabelled confidence, …) —
a validator that accepts everything proves nothing. Against the **live services**: `birds.mp4`
→ texture/flock/96 frames/conf 0.105 *(gated_motion_amplitude)*; `boat.mp4` → path/rigid_path/
114 frames/conf 0.420 *(tracked_fraction)* **plus** texture; `walk-man.mp4` → skeleton/
articulated/155 frames/conf 0.866 *(mean_visibility)*, and hands/face swatches that honestly
report conf 0.0 on a clip with neither. The library line is generated by one
`swatchSummary()` over the core alone, so all three read identically (checked by driving the
real function from `js/main.js` over the real service output). The **no-query `/analyze`
response is byte-identical** (454,931 B on boat.mp4, before and after), and `?fmt=legacy`
on :8770 is untouched — `fmt=swatch` was measured frame-for-frame identical to it on
walk-man.mp4, differing only by gap-filling frames the detector missed.

Bloat control is measured, not guessed. `TRACK_FRAME_CAP=120` + 3dp coords, with the stride
recorded and `fps` divided by it so playback speed is exact. On birds.mp4 (144 tracks × 192
samples, 492 KB), max deviation of a dropped sample from the line joining its neighbours:
stride 1 (rounding only) 437 KB / 0.34 px · **stride 2 → 219 KB / 2.11 px** · stride 4 →
110 KB / 5.56 px · stride 8 → 55 KB / 10.29 px. 120 halves the payload for 0.4% of a 480 px
frame; 60 would halve it again but 1.2% starts to show as a cut corner, so that is where it
stops.

**Not shipped:** the client still sends BOTH shapes — `params`/`trajectories`/`path` (what
`js/animate.js` reads today) *and* `swatches` — so a `?swatch=1` response is ~40% larger than
it needs to be. Step 8 made the applicator read
`swatch.class` + `swatch.kind` (and all three payloads) instead of the raw fields, so the
duplication is now *removable* — but the fields are still on the wire because the extraction
overlay, multi-motion picking and library summaries read them (see Step 8's "Not shipped"). The in-browser Lucas–Kanade fallback returns `swatches: []` rather than a
client-built swatch: the builders live in `contracts.py` as the single source of truth, and
that fallback runs exactly when that service is unreachable. `normalize_swatch` also fills
missing scalars with 0 and records the guess in `warnings` — it will not invent motion (a
texture with no tracks stays invalid), but a caller that ignores `warnings` can mistake a
defaulted param for a measured one.

---

## Step 8 — Application (class-keyed) + mesh-warp
**Objective:** the applicator branches on `swatch.class`, not layer name.

**How:** registry's `applicator` per class — texture→**MLS/ARAP mesh-warp**, human→skeletal
rig, flat character→puppet (have it), object→path travel. Retire the `/\bflag\b/`-style name regexes.

**Challenges → solutions**
- *Flat artwork has no separable limbs (the bear).* → Two modes: rigged parts → skeletal; single-path → puppet or mesh-warp skinning.
- *Point-resample tears geometry / mangles fine detail (the chakra).* → Mesh-warp instead of 48-point resample; keep detail paths crisp via the "ride, don't deform" rule (already implemented).
- *Object leaves the canvas (river slide, cloud drift).* → Bound travel to the object's on-canvas room (already implemented for clouds/birds).

**Done-when:** the correct behavior fires from the class even if the layer is renamed; large deformation doesn't tear.

**Status (shipped):** dispatch is keyed on the swatch, in two places that must agree.
`service/contracts.py` gains `APPLICATOR_NEEDS` (which payload each applicator can read) and
`swatch_applicator(kind, cls)`, so every swatch carries the applicator the *service* resolved;
`js/animate.js` gains `APPLICATOR_BY_CLASS` + `_applicatorFor()`, which prefers that field and
falls back to the class. The routing takes **both axes** — one `rigid_path` clip emits a `path`
swatch *and* a `texture` swatch, and only the first can drive `path_travel`, so class alone
would hand the texture to an applicator with no path. Verified on the live services:
`boat.mp4?path=1&swatch=1` returns `path/rigid_path → path_travel` **and**
`texture/rigid_path → oscillate` from the same clip.

The four name regexes (`birds` / `clouds` / `river|ripples` / `boat|rowboat|canoe|ferry|ship`)
and the cloth `/flag|banner|pennant|ensign|standard/` regex are now **preset-only fallback**,
gated behind `if (!app)`: a classified swatch never reaches them, so real extracted motion
always wins even where the curated version looks nicer. A swatch with an **empty** class is
skipped deliberately — `swatch_applicator` still fills a payload-appropriate default
(`oscillate` for a texture), but that is a shape default, not a classification, and treating
it as one would retire the curated behaviour on the strength of a guess. Driving the real
`_applyAll` over real `?swatch=1` payloads with the layer deliberately misnamed:
flag.mp4→`cloth` on layers *Flag/Birds/River/Boat/Layer 7* all route to `_applyCloth`;
birds.mp4→`flock` to `_applyFlock` on all five; clouds/smoke→`fluid` to `_applyFluid`.

Deformation is a **rigid Moving Least Squares mesh warp** (Schaefer, McPhail & Warren 2006,
§rigid) over a 5×4 lattice sampled from the captured flow field, replacing the per-point
`fieldD` resample that made cloth opt out of real motion. `anchor:'x0'` pins the minX column
and ramps by `pow(u, 1.15)` (cloth on a pole); `anchor:'none'` pins nothing (a river surface
is held by nothing). Measured on flag.mp4's real field, local shape distortion as a % of a
4 px patch (Procrustes residual after best-fit rotation): per-point `fieldD` p50 27.3 / p95
86.8 / max 153; **rigid MLS p50 9.4 / p95 33.1 / max 67** — 2.6× better at p95. 1 px-apart
neighbours stay 1.76 px apart (vs 2.73 px), the lattice never folds (worst 0.0% of a cell),
a zero field leaves the artwork within 2.5e-14 px, and detail paths get a real 7.06° rotation
to ride rather than a slide.

Flock drift now reads the captured params instead of bird constants: `direction` 0 → +x
(52.1 px), 90 → −y (52.1 px), members never reverse into the flock (min dx 0.000), and
`turbulence` measurably loosens the formation (spread 25.9 → 35.7). Travel stays bounded by
each member's own on-canvas room.

`js/animate.js` also reads all three payloads **from the swatch alone** now
(`buildTrajField`'s texture fallback, `_poseFor`, `_pathFor`), which is what Step 7's
"Not shipped" note was waiting for. Confirmed against real service output with the legacy
top-level fields stripped: cloth→`_applyCloth`, flock→`_applyFlock`, path→`_applyPathTravel`,
and the real MediaPipe skeleton swatch (155/155 frames) drives the rig on rigged artwork and
declines on unrigged artwork.

**Verified**, not asserted: `tests/step8-applicators.js` (node only, no browser and no
service) drives the **real** `Animator._applyAll` with the applicator methods replaced by
spies and a stubbed DOM, so a check passes only if the dispatch genuinely routes on class.
47 checks across routing-by-class-not-name, two-axis `(kind, class)` routing, preset-only
fallback, honest capability fall-through, JS-table-mirrors-`contracts.py` (parsed out of the
`.py`), the mesh-warp properties above, and the flock params. Pass an `/analyze?swatch=1`
response as `argv[2]` to run the mesh measurements on a real field instead of a synthetic one.
`service/contracts_selftest.py` is now 65 checks, including the negative cases (*a
`rigid_path` texture swatch does NOT get `path_travel`*, *an articulated texture swatch does
NOT get the rig*). The **no-query `/analyze` response and `?fmt=legacy` on :8770 are still
byte-identical** — `applicator` appears only inside a `?swatch=1` swatch.

**Not shipped:** the wire payload is still duplicated. The applicator no longer *needs* the
legacy `trajectories`/`path`/`pose` fields, but `js/extractviz.js`, `js/multipick.js` and
`js/main.js` still read them (extraction overlay, multi-motion picking, library summaries), so
dropping them from the response is a separate change with its own blast radius. `skeletal`
inside `_applyByClass` is a `return false`: the rig branch runs earlier in `_applyAll` because
it needs the pose payload as well as the class, so reaching the switch means the artwork has
no rig. The mesh warp is **as-rigid-as-possible, not rigid** — 33% p95 residual is a
reduction, not an elimination, and at amplitudes far above what distill reports (≈0.9
normalized, ~1.8× the lattice spacing) the lattice folds and the warp tears. Real captured
amplitudes are 0.02–0.1, which is the range the numbers above were measured in.

---

## Step 9 — VLM Judge + auto-tune loop
**Objective:** the system scores its own output and self-corrects.

**How:** capture N frames → send frames + reference + class to Claude → get score + deltas →
apply → re-render → re-judge (2–3×). Show the critique in the UI.

**Challenges → solutions**
- *VLMs judge motion poorly from single stills.* → Send a frame *sequence* / GIF / motion-trail composite, or a video-capable VLM; judge the trend, not one frame.
- *Loop could oscillate / never converge.* → Cap iterations (≤3); require monotonic score improvement or stop; clamp delta magnitude.
- *Cost / latency of repeated VLM calls.* → Only auto-tune on demand; cache the reference embedding; small frame count.

**Done-when:** a deliberately weak motion improves over ≤3 iterations and the loop stops on success.

**Status (shipped):** end to end, on demand, behind one button. `service/contracts.py` gains
**Contract C** — the third contract its header comment has promised since Step 1 — and
`service/vlm_router.py` gains `POST /judge` + `/judge/reset` alongside `/decompose`: same model,
same forced-tool-call plumbing, opposite direction. `/decompose` looks at a video and proposes
motion; `/judge` looks at motion and proposes corrections.

Three decisions in Contract C are worth naming. **Deltas are signed additive offsets, never
multipliers** — amplitude 0.0 times any multiplier is still 0.0, so a multiplier can never
rescue a dial that is stuck at zero while an offset can. Measured: on a preset with no `driftY`
at all, the judge added downward travel to a waterfall from nothing (`driftY` 0 → 0.2 → 0.4).
**`wrong_class` is a first-class verdict, not a low score**, because no amount of dial-tuning
turns a flock drift into cloth, and it stops the loop instead of paying for two more passes.
And **the score says what it measured**: `match_to_reference` when the source clip was sent,
`class_plausibility` when it wasn't, mirroring a swatch's `confidence_of`. The panel prints
"vs. source clip" or "plausibility only (no source clip)" so the user sees which claim they got.

Clamping is two-stage and the order matters: the per-param cap in `PARAM_DELTA_MAX` is applied
to the **offset first**, then the sum is clamped to `PARAM_RANGES`. Capping before adding means
a wild delta still lands near the current value rather than at a range edge — a `direction`
delta of 400° becomes +45°, not a jump to 360. `direction` wraps mod 360 instead of clamping,
or every over-rotation would pin at due east. Every range but `frequency` is taken from
`distill.py`'s own `_clamp01`/±1 clamps rather than invented; `frequency`'s 6.0 ceiling is
labelled in-code as a **renderer sanity limit chosen here**, since distill computes `k·fps/T`
with no upper bound of its own.

**The server owns the loop policy.** `js/judge.js` sends frames and obeys `next_params` and
`continue`; it does not decide when to stop, compute the next params, or clamp anything. Caps
that stop one confident-wrong verdict are worth nothing if the caller can route around them,
and a browser is the easiest thing in the world to route around. Sessions are an in-memory
dict, so a caller inventing a fresh session id gets a fresh budget — documented in-code as a
cost guard, not a security boundary. Three stop rules: `JUDGE_MAX_ITERS` 3, `JUDGE_GOOD` 0.8,
and `JUDGE_MIN_GAIN` 0.03. The over-budget call is refused **before** the request is built, so
it costs nothing.

The loop stops when a pass *fails* to improve, which means the state it stops in is by
definition the one that didn't help — so `judge_best` finds the highest-scoring iteration and
`tune()` puts those params back (ties keep the earlier, cheaper one). Live run: 0.72 then 0.55,
the −0.170 tripped the no-gain rule, and the params returned to the pass-1 set exactly.

Frames are sampled by calling `animator._applyAll(t)` at chosen times rather than by recording
playback: the animation is a pure function of `t`, so this gives exactly one cycle, evenly
spaced, reproducibly, where a real-time capture gives whatever 8 moments the event loop
allowed. The cycle end is excluded so frame 8 isn't a duplicate of frame 1. **What one cycle
means depends on the payload**: a captured pose or path repeats over its own clip duration
(`frames/fps`), not `1/frequency` — sampling a 4-second walk at 1/1.4 s shows the same third of
a stride eight times, the judge honestly reports "I see almost no motion", and the loop cranks
the amplitude on a motion that was fine.

Two fixes came out of measuring rather than asserting. A canvas `toDataURL()` hands over JPEG
bytes that were being declared `image/png`, which the API rejects with a 400 instead of just
reading the file — `_sniff_media()` now reads magic bytes and ignores the declared header
entirely. And the **intent** sent to the judge was the *layer* name, so a waterfall applied to
a layer called "title" was graded as a title reveal — a fair verdict on the wrong question.
`label` is now the motion's name and `element` is the layer, separately; a motion with no class
says so rather than printing `class ''`. After the fix the same run produced *"For 'Waterfall
Flow' I'd expect a clearer, sustained downward-biased travel"* and steered `direction` 90→45→0
and `turbulence` 0.55→0.25.

**Verified**, not asserted:
- `tests/step9-judge-loop.py` — 46 checks driving the **real** `judge()` with a scripted fake
  VLM. The model's taste is not what's under test; the point is that no verdict, however wrong
  or greedy, can push a param past its cap or the loop past its budget. Covers the four stop
  rules, that the refused call spends no VLM request, session independence, end-to-end
  clamping, what actually gets sent (forced `tool_choice`, every frame, frames labelled in
  order, SOURCE before RENDER, 7 images for 3+4, caps generated *from* contracts so the prompt
  can't drift from what `normalize` accepts), and the media-sniffing regressions.
- `tests/step9-sampling.js` — 25 checks, dependency-free, on `cycleSeconds` precedence
  (pose > path > frequency, nested Step-7 swatches included), the [0.4, 4] s bounds, speed
  scaling, and that a model-supplied critique cannot inject markup into the panel.
- `service/contracts_selftest.py` is now **101 checks** (was 65), including the negative cases:
  hallucinated param names, runaway deltas, a cap smuggled past `normalize`, a `tune` verdict
  with no deltas (not actionable), a `good` verdict *with* deltas (a contradiction), and the
  arithmetic one — three capped iterations cannot cross a whole range.
- Live discrimination against the real model, same strip rendered three ways: amplitude 0.00 →
  **0.05**, 0.06 → **0.55**, 0.55 → **0.82** (spread 0.77, monotonic). Frozen frames drew
  `amplitude: 0.0` and *"flag edges identical in every frame"* — it did not invent motion it
  could not see, which is the property the whole loop rests on.
- Headless-Chrome end-to-end through the real `#btn-judge`: 8 frames, **8 unique**, mean pixel
  difference 4.5 (the serialized SVG's transforms genuinely moved between samples), verdict
  rendered into `#judge-out`, revert offered, and the status line quoting the server's own stop
  reason verbatim. The panel reveals itself from the real flow (layer click → motion chip) and
  hides again with the inspector.

**Not shipped:** no reference-frame caching — `referenceFrames()` re-seeks the source clip on
every press (it's once per run, not once per pass, so it isn't the cost driver; the 8 render
frames are). Sessions are in-memory, so restarting the router forgets a run in progress. The
judge sees the whole canvas and is told in the prompt which element to watch rather than being
handed a crop, so a busy scene can still draw its attention elsewhere. And the loop tunes
**dials only**: a `wrong_class` verdict is surfaced honestly and stops the loop, but nothing
re-routes the applicator or re-extracts — that would be a Step 10 orchestration change.

---

## Step 10 — Integration, orchestration & polish
**Objective:** the router drives the whole chain per motion; multi-motion clips fan out.

**How:** router → per-region {SAM2 → backend → distill} → multi-pick modal → per-object
apply → judge. Auto-label layers with the VLM. Remove the synthetic `Autumn Fall` shortcut.

**Challenges → solutions**
- *Many services to run (RAFT, MediaPipe, SAM2, etc.).* → A single launcher script + health checks; the app degrades gracefully if a backend is down (shows which to start).
- *Browser can't run heavy models.* → Keep heavy models server-side; the browser only uploads media and renders swatches.
- *Reproducibility across machines.* → Per-service `requirements-*.txt` + a README with the exact env/versions (MediaPipe needs py≤3.12).

**Done-when:** upload one multi-motion clip → named swatches auto-applied to the right objects → judged & tuned — end to end, no filename hints, minimal hardcoding.

---

## Suggested order & effort
1. **Step 0** (contracts) — required first, small.
2. **Step 8 quick win** (un-cap + bulk, mesh-warp) — visible motion improvement immediately.
3. **Step 9** (judge loop) — highest "wow", self-contained.
4. **Step 1 + Step 10** (router + multi-motion wiring) — retires the biggest hardcoding.
5. **Step 2 (SAM 2)** + **Step 4 (SEA-RAFT/CoTracker)** — the real-motion quality lift.
6. **Steps 3/5/6** (Hands/Face, object-path, text-to-motion) — plug in as needed.

**Highest-leverage first, any team size:** Step 0 → Step 8 quick win → Step 9 judge loop →
Step 1 router. Everything after is "add a registry entry."
