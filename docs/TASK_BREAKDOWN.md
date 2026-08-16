# MotionLife 2.0 — Task Breakdown

> Module-based so you can assign to any headcount. Each task has a **specific
> description** and a **✅ Done when** (the concrete, testable thing that should work
> after it's implemented). Companion: `docs/ARCHITECTURE_FLOW.md` + `architecture.html`.

**North star:** upload any video → the system understands what's moving (even multiple
motions), extracts each as real long-range motion, applies it realistically, and judges
its own output and self-tunes. Minimal hardcoding.

---

## Module 0 · Foundation & Contracts  *(do first — unblocks everything)*

**T0.1 — Freeze the JSON contracts (A/B/C).**
Define the decomposition list, motion swatch, and judge report as one shared schema both
the service and app import.
✅ **Done when:** a single schema file exists; service and app both import it; changing a
field name breaks both in one place (no drift).

**T0.2 — Stub the API routes.**
Add `/decompose` and `/judge` to FastAPI returning canned JSON matching the contracts.
✅ **Done when:** `curl /decompose` and `curl /judge` return valid Contract-A / Contract-C
JSON, so the app and engine can be built before the real logic exists.

**T0.3 — Frame-capture helper.**
A function that grabs N frames (or a GIF / motion-trail composite) of the running
animation for the judge to see.
✅ **Done when:** calling it on a playing animation returns an ordered image sequence with
timestamps, ready to POST to `/judge`.

---

## Module 1 · VLM Router (multi-motion perception)

**T1.1 — Video → VLM decomposition.**
Send sampled frames of the clip to the VLM; return the list of distinct motions with
object label, bbox, and confidence.
✅ **Done when:** a clip containing a flag + smoke + birds returns **3 entries** with
sensible labels and boxes — **with no filename hints** — matching Contract A.

**T1.2 — Motion-class taxonomy.**
Constrain the VLM output to 6 classes (`cloth_wave`, `fluid_flow`, `particle_fall`,
`rise_plume`, `rigid_drift`, `sway_oscillate`) plus direction + speed hint.
✅ **Done when:** each detected motion carries exactly one valid class + direction, and a
flag clip classifies as `cloth_wave`, a river as `fluid_flow`, etc.

**T1.3 — Confidence gating.**
Drop low-confidence detections instead of inventing motion.
✅ **Done when:** a clip with one real motion + ambiguous background returns only the real
motion; a static clip returns an empty list (never a fabricated region).

**T1.4 — SVG layer auto-labeler.**
Feed the artwork render to the VLM → auto-name each object → write `data-name`.
✅ **Done when:** uploading an unlabeled SVG produces named, selectable layers
("flag", "smoke", "river"…) with no manual `data-name` editing.

---

## Module 2 · VLM Judge + Auto-tune

**T2.1 — Judge endpoint.**
Send applied-motion frames + reference clip + object class to the VLM; return score,
issues, and suggested parameter deltas.
✅ **Done when:** POSTing frames of a barely-moving flag returns `looks_real:false`, a low
score, and a concrete suggestion like `intensity:"+40%"` (Contract C).

**T2.2 — Per-class scoring rubric.**
Class-specific criteria (flag: "waves as one sheet, emblem crisp"; water: "flows, doesn't
slide"; etc.).
✅ **Done when:** the judge's issues reference the right failure modes per class (it flags
a torn flag or a sliding river specifically, not generic complaints).

**T2.3 — Auto-tune loop.**
Apply deltas → re-render → re-judge, 2–3 iterations or until a score threshold.
✅ **Done when:** a deliberately mis-tuned motion (too weak) visibly improves over ≤3
iterations and the loop stops once the score passes.

**T2.4 — Judge-loop UI.**
Show the critique + before/after in the app.
✅ **Done when:** during a demo, the app displays the judge's verdict ("clouds look rigid
→ increasing billow") and the motion updates on screen.

---

## Module 3 · Extraction backends (ML)

**T3.1 — SEA-RAFT swap.**
Replace `raft_small` with SEA-RAFT for cleaner, faster flow.
✅ **Done when:** the same clip extracts with equal-or-lower jitter and equal-or-faster
runtime; existing extraction still produces valid swatches.

**T3.2 — SAM 2 masking.**
Turn the VLM's approximate bbox into a precise per-object mask.
✅ **Done when:** tracking runs only inside the masked object; background points are
excluded (verified by overlaying the mask on the clip).

**T3.3 — CoTracker3 tracks.**
Dense long-range point tracks inside each mask across the full clip (replaces the 12×12
grid).
✅ **Done when:** a swatch carries per-point trajectories spanning the whole clip length
with occlusion flags — not a short, coarse, per-frame grid.

**T3.4 — Spectral bulk/residual split.**
Decompose each track into low-frequency bulk (travel) + high-frequency texture; keep both.
✅ **Done when:** the swatch exposes a `bulk` vector **and** residual detail, and a
"travel amount" dial visibly changes how far the object moves.

**T3.5 — Per-class distillers.**
Emit the Contract-B swatch tuned per motion class.
✅ **Done when:** each class produces a swatch whose params/tracks are sensible for that
medium (a flag swatch differs measurably from a smoke swatch).

---

## Module 4 · Application / rendering realism

**T4.1 — Un-cap displacement + preserve bulk.**  *(fastest visible win — ships today)*
Remove the hard displacement `CAP`; keep a tunable amount of bulk motion in
`buildTrajField`.
✅ **Done when:** applying a motion makes the object travel a visibly larger, natural
distance (not an in-place jiggle), controllable by a dial, without leaving the canvas.

**T4.2 — MLS / ARAP mesh warp.**
Deform a mesh over the object driven by the tracks — smooth, large, no tearing (replaces
the 48-point resample).
✅ **Done when:** a large deformation (e.g. a strong flag wave) bends the geometry smoothly
with no gaps/tears, and fine detail stays intact.

**T4.3 — Class-keyed applicators.**
Route by `swatch.class` instead of layer-name regex; one applicator per class.
✅ **Done when:** the correct behavior fires from the swatch's class even if the layer is
renamed — the `/\bflag\b/` etc. name matching is gone.

**T4.4 — Full-clip seamless looping.**
Replay the whole trajectory with loop optimization (not the short ping-pong).
✅ **Done when:** long motions play back over the full clip and loop with no visible
reset/snap.

**T4.5 — (stretch) Eulerian/spectral realism.**
A learned oscillatory motion field for water/trees/flags.
✅ **Done when:** at least one natural object (e.g. water) animates with large, physically
plausible, seamlessly looping motion beyond what the tracks alone give.

---

## Module 5 · App / UX integration

**T5.1 — Wire `/decompose` → multi-pick modal.**
Show the VLM's motion list as separately-applicable, auto-named swatches.
✅ **Done when:** uploading a multi-motion clip populates the picker with ≥2 named swatches
you can drop on different objects — **no `DEMO_PROFILES` filename match involved.**

**T5.2 — Auto-suggest swatch → object.**
Use the VLM object labels to pre-match each swatch to a layer.
✅ **Done when:** after decomposition, each swatch is pre-highlighted against its likely
target layer (flag swatch → flag layer) and one click confirms.

**T5.3 — Remove the synthetic `Autumn Fall` shortcut.**
Route leaf/autumn uploads through the real pipeline instead of the hardcoded synth.
✅ **Done when:** a falling-leaves clip is analyzed by the real extractor (no
filename-triggered synthetic trajectories); the biggest frontend hardcoding is gone.

**T5.4 — (stretch) Prompt-to-motion.**
Natural language → parameter/behavior change ("wave gently", "faster").
✅ **Done when:** typing an instruction adjusts the applied motion's params accordingly on
screen.

---

## Module 6 · Depth & polish  *(optional / later)*

**T6.1 — Depth Anything V2** per-frame depth for parallax/foreshortening.
✅ **Done when:** near objects visibly move more than far ones in a scene with depth.

**T6.2 — SpatialTracker** 3D-aware tracks.
✅ **Done when:** a turning/foreshortening motion (e.g. a flag rotating toward camera)
reads with correct perspective.

---

## Milestones (integration checkpoints)

- **M1 — Skeleton on fakes (Week 1):** contracts frozen (T0.x) · **T4.1 shipped** (motion
  already travels) · each stream runs standalone against stubs.
- **M2 — Real per-piece (Weeks 2–3):** T1.1 + T2.1 live on Claude · T3.1/T3.3 single-region
  · T4.2/T4.3 + T5.1 wired → one multi-motion clip → real swatches, real deformation.
- **M3 — Full loop + polish (Week 4):** T3.2 masks · T2.3 auto-tune closes · T2.4 + T1.4 +
  T5.3 → complete "understand → extract → apply → judge → self-correct" demo.

## Suggested slicing by headcount

- **1 person:** T0.x → T4.1 → Module 2 (judge loop) → Module 1 → the rest.
- **2 people:** P1 = Modules 1+2 (VLM, Python). P2 = Modules 4+5 (application + app, JS).
  Module 3 shared/deferred.
- **3 people:** P1 = Modules 1+2 (Perception). P2 = Module 3 (Extraction ML). P3 =
  Modules 4+5 (Application + App). Module 0 together on day 1.

**Highest-leverage first, any headcount:** **T4.1** (motion travels — visible today) →
**T2.1–T2.4** (the judge loop) → **T1.1 + T5.1** (real multi-motion, retires the biggest
hardcoding).
