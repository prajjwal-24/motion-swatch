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
