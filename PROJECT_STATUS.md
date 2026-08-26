# MotionLife (Motion Swatch) — Project Status

> Master document, current as of **Step 10** of `docs/BUILD_PLAN.md`. Covers **(1)** what the
> project is and everything built, **(2)** exactly where and why we hardcoded, and **(3)** what is
> not working / known limitations. Written to be spoken to honestly in a review.
>
> Companion docs: `docs/BUILD_PLAN.md` (the roadmap, with the measurements for each step — **the
> authoritative record where anything disagrees**), `HARDCODING.md` (hardcoding detail),
> `docs/ARCHITECTURE_FLOW.md` and `docs/TASK_BREAKDOWN.md` (module/flow detail),
> `DEMO_SCRIPT.md` (video script), `README.md` (usage).

---

## 1. What it is

**MotionLife** captures real-world motion from a short video, distills it into a reusable,
editable **motion swatch**, and applies it to individual objects in a static vector illustration —
no keyframes, no rigging. The pitch: *motion becomes an asset you paint on, like a colour.*

Two pieces:
- **Browser app** — vanilla JS + inline SVG + Canvas 2D. No framework, no bundler. Served with any
  static server (`python3 -m http.server 8000`).
- **Analysis services** — four Python processes, each in its own venv because their dependencies
  genuinely conflict. `sh start-all.sh` brings them up; `sh start-all.sh status` prints UP/DOWN per
  port.

| Port | Service | Venv | What it does |
|---|---|---|---|
| 8000 | static file server | — | serves the app |
| 8765 | FastAPI extraction + registry | `service/venv` (3.13) | RAFT / SEA-RAFT / CoTracker3, SAM 2 masking, depth, YOLO+ByteTrack paths, distill, `/engines` |
| 8770 | MediaPipe Pose | `mpvenv` (3.11) | body motion (MediaPipe needs py ≤ 3.12) |
| 8771 | VLM router | `routervenv` (3.9) | `/decompose`, `/label`, `/judge` |
| 8772 | preprocess | `routervenv` | standalone SAM2/depth/camera helper (not called from the browser — see §5) |

Every service is **optional at runtime**: the app degrades and says which one to start. With
`:8765` down, extraction falls back to an in-browser Lucas–Kanade analyzer; with `:8771` down,
auto-label and auto-apply report that they did nothing rather than guessing.

---

## 2. Everything built

### App / UX
- **Upload artwork** — SVG (inline, per-object selectable) or raster (rectangle select).
- **Layer / object selection** — objects marked `<g class="layer" data-name="…">` become
  selectable; a Layers panel lists them. Uploaded SVGs with no usable names get a leaf-element
  fallback so they are still clickable.
- **Motion Library** — **9** built-in presets (Waterfall Flow, Cloud Drift, Flag Flutter, Gentle
  Sway, Water Ripple, Sun Pulse, Falling Leaves, Autumn Fall, Rising Smoke).
- **Capture from video** — upload a clip → the VLM reads it → the router picks an extractor →
  new swatch(es), shown with an animated "extraction" overlay (flow vectors + distilled params).
- **Auto-label the artwork** (`#btn-autolabel`) — one `/label` pass says what each layer depicts.
- **Auto-apply** — a multi-motion clip's swatches are placed on the objects whose class matches,
  with the placement and any refusals named in the status line.
- **Judge & tune** (`#btn-judge`) — renders one cycle, asks the VLM to grade it, applies bounded
  param deltas, and reverts to the best-scoring pass.
- **Per-object controls** — Speed, Intensity, Remove motion, Delete region.
- **Cloth / deform vs. rigid** — decided by evidence, with the winner recorded in `waveModeFrom`
  (see §3d).
- **Play / Pause**, **Preview** (Before | MotionLife | source videos), **Export** (SVG / video).

### Analysis / engine
- **Extractor registry** (`service/extractors.py` + `/engines`) — every engine declares what it
  needs and **probes** for it, so an uninstalled engine reports `False` with a setup hint and the
  router falls back instead of crashing.
- **Optical flow / tracking** — `raft_small`, SEA-RAFT, CoTracker3 → 8 distilled parameters
  (frequency, amplitude, direction, turbulence, damping, phaseSpread, driftX, driftY) + a **12×12
  trajectory grid** (144 real point-tracks).
- **Object masking + depth (Step 2)** — SAM 2 seeded by the VLM's bbox replaces the rectangular
  crop; Depth-Anything-V2-Small gives a depth *rank* over the mask. Both are reported per swatch
  (`masked 20% (sam2+motion, 38/144 cells tracked) + depth rank 0.4147`).
- **Body motion** — MediaPipe Pose → Contract B's 13 joints, driving a rigged character.
- **Object travel paths (Step 5)** — YOLO + ByteTrack tracks one discrete object across the clip.
- **Region segmentation** (`segment_regions`) for multi-motion clips.
- **One swatch contract (Step 7)** — `service/contracts.py` covers texture / skeleton / path
  payloads in a single schema, shared by every service and self-tested in all four interpreters.
- **Animation engine** (`js/animate.js`) — class-keyed dispatch, the rigid **MLS mesh warp**,
  trajectory-field replay, **deform-in-place** (subtract mean so flowing objects ripple without
  sliding away), per-glyph text motion, **detail-ride** (fine detail like a flag's chakra stays
  crisp while the cloth ripples), and `_applyPathTravel` for measured travel.

### The rule that governs the whole app
**Real extracted motion wins over curation, everywhere the two could disagree.** A measured travel
path beats the name-keyed behaviours; a classified swatch beats them too; a measured displacement
field beats the VLM's still-image reading of a layer; the VLM's reading beats the layer-name regex.
Each of those precedences is enforced in code and pinned by a test.

### Assets
- **Artwork** (`assets/Artwork/`): `poster.svg` (Independence Day flag), `Autumn.svg` /
  `AutumnPoster.svg`, `boat-river_layered.svg` (riverside scene, fully layer-labelled),
  `riverside-camp.svg`, `Scene.svg`, `train-window-adobe.svg`, `train-window.svg`,
  `independence-logo.svg`.
- **Scenes** (`assets/scenes/`): `character-bear.svg`, `character-duck.svg`, `motion-lab.svg`,
  `test-scene.svg`. The **Poster** and **Scenery** tabs are generated in JS
  (`createPosterSVG` / `createScenerySVG`), not files.
- **Source videos** (`assets/videos/`, 9 committed): `Autumn`, `birds`, `boat`, `boat-night`,
  `clouds`, `flag`, `smoke`, `walk-grid`, `walk-man`. `CompleteDemo.mp4` (734 MB) is deliberately
  **gitignored** and local-only.

### Tests
| Suite | Checks | Needs |
|---|---|---|
| `service/contracts_selftest.py` | 153 | nothing but the stdlib — runs in all four venvs |
| `tests/step6-text2motion.py` | 100 | nothing — runs in all four venvs |
| `tests/step10-orchestration.js` | 65 | node only |
| `tests/step10-label.py` | 68 | `routervenv` (no credentials) |
| `tests/step9-judge-loop.py` | 46 | `routervenv` (no credentials) |
| `tests/step8-applicators.js` | 47 | node only |
| `tests/step2-preprocess.py` | 38 | `routervenv` |
| `tests/step9-sampling.js` | 25 | node only |
| `tests/step2-field.js` | 14 | node only |
| `tests/step10-e2e.js` | 20 | **live**: every service up + real VLM calls |

### Pitch materials
- **`intro-deck.html`** — animated 9-slide pitch deck plus a dense one-frame thumbnail slide.
- **`architecture.html`**, **`hld.html`** — diagrams. Voiceover scripts in `DEMO_SCRIPT.md`.

### GitHub
- Repo: **github.com/prajjwal-24/motion-swatch** (public).

---

## 3. Where we hardcoded (honest accounting)

**The extraction, the animation, and the choice of what to animate are all real.** What remains is
curation of *which cells of a demo clip form a region and what to call it*, some tuned thresholds,
and two name-based fallbacks that only fire when nothing was measured. Ranked by how much it
matters if a reviewer asks. Full detail in `HARDCODING.md`.

### 3a. `DEMO_PROFILES` — filename-keyed region layouts (`service/segment.py:296`)
**The biggest one, and the only filename test left in the repo.** If an uploaded filename contains
a known keyword (`cherry`, `silk`, `flag`, `wheat`, `ink`, `ocean`, `two_flags`, `bosphorus`, …),
`segment_regions` uses a **hand-tuned region layout + names** instead of automatic segmentation. It
still runs the **real** extraction and distillation on those regions — only the region rectangles
and the labels are curated, and it decides nothing about which object a swatch is applied to.
Longest matching key wins. Rename the file and it goes through the fully automatic path.

### 3b. Name-keyed scenery behaviours — **preset-only fallback** (`js/animate.js:103-115`)
Four regexes select bespoke synthetic behaviours from the object's **name**. Since Step 8 they are
gated behind `if (!app)` — **a classified swatch never reaches them.** They run only for the
built-in presets and the in-browser Lucas–Kanade fallback, which carry no class and no captured
field, so there is nothing real to prefer over them.

| Object name matches | Fallback behaviour | Uses captured params? |
|---|---|---|
| `birds` | Wing-flap (vertical scale) + split left/right gentle drift | speed/intensity only |
| `clouds` | Slow one-directional wind drift + subtle edge billow | speed/intensity only |
| `river` / `ripples` | Smooth laminar travelling-wave flow | frequency, amplitude |
| `boat` / `rowboat` / `canoe` / `ferry` / `ship` | Gentle rigid bob + rock | frequency, amplitude |

Two more are role- or param-keyed rather than name-keyed, and are not gated because they *are* the
declared behaviour: `[data-motion-role="tree-canopy"]` sways its selectable overlay so dense canopy
artwork stays intact, and `params.leafFall` (the `autumn-fall` **preset**) runs the per-leaf
fall/tumble/wrap.

> **Bird wing-flap is synthetic — do NOT advertise it as extracted from the video.**

The old cloth `/flag|banner|pennant|ensign|standard/` regex is **gone**: the rigid MLS mesh warp
cut local shape distortion ~3× (27% → 9% median on flag.mp4), which is what made captured motion
usable on a flag without a synthetic stand-in.

### 3c. Preprocessed artwork
- **`train-window-adobe.svg`** — Illustrator flattened all layer names on export; a one-time script
  identified each object by geometry/colour and wrapped it into named `.layer` groups. Rendering is
  byte-identical; only structure changed. Its scene **tab** was removed from the UI.
- **`boat-river_layered.svg`** — every group hand-labelled with `data-name` + `data-motion-mode`,
  plus occlusion clip-groups (smoke behind foliage, etc.).

Since Step 10 the app no longer *acts* on these names: `/label` reads the pixels, and placement is
class equality. The names are a convenience for a human reading the file.

### 3d. The cloth-mode name hint — **gated last resort** (`js/regions.js:54`)
`/flag|banner|cloth|pennant|curtain|sail/i` decides whether a region's geometry is bent or moved
rigidly, from the layer name. It is now the last thing consulted, and every selection records what
decided it in `waveModeFrom`: `preset_leaffall` → `artwork_rigid` → `motion_field` → `vlm:<class>` →
`name_hint` → `default`. `motion_field` outranks `vlm:<class>` deliberately — a measured
displacement field beats a still-image reading. Kept rather than deleted because with the router
offline it is the only answer available, and a flag that does not ripple is a worse failure than an
honest guess.

### 3e. Tuning constants
Segmenter thresholds (`ANALYSIS_WIDTH` 480, `REGION_KEEP_FRAC` 0.25, `ABS_FLOOR`, `PEAK_FRAC`,
`DRIFT_DOM`, …), the wave constants (`WAVE_AMP_PX` 12, `WAVE_CYCLES`, …), the cost caps
(`LABEL_MAX_LAYERS` 12, `LABEL_CROP_W` 256), the evidence floor (`CONF_MIN` / `LABEL_CONF_MIN`
0.35), and the judge's stop rules (`JUDGE_MAX_ITERS` 3, `JUDGE_GOOD` 0.8, `JUDGE_MIN_GAIN` 0.03)
are hand-picked thresholds, not faked outputs. Each is cited with its reasoning where it is
defined, and every cap is *reported* rather than applied silently.

### 3f. Removed since the last revision of this document
- **The synthetic `Autumn Fall` capture** (was in `js/upload.js`): a `/leaf|leaves|falling|autumn/i`
  filename test that skipped extraction, played the "extraction" overlay over the real video, and
  returned a hand-written spiral plus eight hand-tuned dials captioned *"Captured from
  falling-leaves video"*. **Deleted at Step 10**; the same clip now routes like anything else. The
  hand-tuned leaf look survives only as the labelled `autumn-fall` **preset**.
- **The cloth applicator name regex** (§3b).
- **Layer-name matching for placement** (§3c).

---

## 4. What IS genuinely real
- Optical-flow / point-tracking extraction on **any** uploaded video that reaches the service —
  there is no longer a clip that bypasses it.
- Distillation to the 8 parameters + the 12×12 trajectory grid; SAM 2 masking and depth rank.
- The VLM's three jobs: reading the clip (`/decompose`), reading the artwork (`/label`), grading the
  result (`/judge`) — all real model calls on real images, with forced tool calls so there is no
  prose to parse.
- Class-keyed **application** (which animator runs) and class-keyed **placement** (which object it
  lands on).
- The animation primitives: the MLS mesh warp, trajectory-field replay, deform-in-place, per-glyph
  text, detail-ride, measured path travel.
- Object selection / the `.layer` contract; Speed / Intensity; Preview; Export.

---

## 5. What is NOT working / known limitations

1. **Bird wing-flap and the `autumn-fall` preset are synthetic**, not extracted. They now only run
   for presets and unclassified motions (§3b), but present them as "the object animates naturally,"
   never "extracted from the clip."
2. **Presets ignore the captured motion's character** — by definition; that is what a preset is.
   For a *captured* swatch the claim "apply the exact motion you captured" is literally true: the
   field drives the mesh warp.
3. **Optical flow can't see thin or smooth motion** — light rain and smooth slow-mo water extract
   amplitude ≈ 0. We drop them rather than invent a region (honest, but some clips "don't work").
4. **Auto-segmentation of complex 3–4-motion clips is unreliable** at 480 px; `DEMO_PROFILES`
   compensates for the curated demo clips only.
5. **`/label` is capped at 12 layers** and sees each layer as an isolated crop with no
   relationships between them, so "the flag *on* the pole" is two independent readings. A busier
   illustration gets the first 12 **plus a warning**, not a silent partial answer. The model's class
   for a thin cloud is genuinely unstable across runs (`cloth` 55% one run, `fluid` 55% the next).
6. **The judge is not auto-chained.** "Judged & tuned" is a separate deliberate button press,
   because a 3-iteration VLM loop firing on every upload is a cost the user should choose. A
   `wrong_class` verdict stops the loop honestly but nothing re-routes or re-extracts.
7. **`:8772` (preprocess) is not called from the browser.** It works and is tested, but the browser
   reaches masking through `preprocess=1` on `:8765` instead — so the standalone service is
   currently only useful from the CLI.
8. **Step 6 (text → motion): no prompt in this repo produces motion.** MoMask's six checkpoints
   are not vendored and CLIP is not installed, so the done-when *"a person waves" → a rigged
   character moves* is **not met**. What is shipped is the seam and the format bridge:
   `service/t2m.py` probes for the specific missing module/file, the registry row now names the
   repo actually vendored (**EricGuo5513/momask-codes**, not MDM/MotionGPT) with `mdm` kept as an
   alias, and `resolve_best` walks past it to MediaPipe. `momask22_to_pose13()` converts MoMask's
   `(T, 22, 3)` SMPL-22 output into Contract B's 13 joints with the same normalisation
   `pose_server.py` uses — so `load_npy()` will play a `.npy` generated on a machine that *does*
   have the weights, with no torch and no CLIP. Two disclosed fabrications in that conversion:
   `nose` is SMPL's **head** joint (SMPL-22 has no nose) and `vis` is 1.0 because generated motion
   is not observed; `confidence` is **0.0 / `generation_only`**. Root travel is discarded, so the
   rig animates in place. `generate()` is transcribed from the vendored `gen_t2m.py` and **has
   never run here**. No text box, no prompt cache, no pre-generated library, no smoothing/retime.
   `tests/step6-text2motion.py` (100 checks) pins the gate so the gap cannot quietly become a lie.
9. **The mesh warp is as-rigid-as-possible, not rigid.** 33% p95 residual is a reduction, not an
   elimination; above ~1.8× the lattice spacing the lattice folds and the warp tears. Real captured
   amplitudes are 0.02–0.1, the range it was measured in.
10. **Judge sessions are in-memory**, so restarting the router forgets a tuning run in progress.
11. **`train-window` backward-parallax is not built.** It needs bulk directional travel with
    looping/oversized content; deform-in-place intentionally suppresses bulk drift.
12. **Browser caching** — JS changes require bumping the `?v=NN` query string in `index.html`
    (19 hand-maintained refs) or a hard-refresh, or the browser serves stale scripts.
13. **Large source video in repo** — `birds.mp4` is 82 MiB (over GitHub's 50 MB soft warning; under
    the 100 MB hard limit). `CompleteDemo.mp4` (734 MB) is gitignored and local-only; its real
    content ends ~2:15 of a 4:24 file, so trim before combining.

---

## 6. One-line answer for reviewers

> "The extraction, the animation, and the *choice of what to animate* are all real. It runs optical
> flow or point tracking on your clip, masks it to the object SAM 2 found, and applies the measured
> field; a VLM looks at your artwork and says what each layer depicts, and swatches are placed by
> matching motion **class** — not by layer name. We proved that by relabelling every layer to
> `Layer 7` / `path2854` and re-running against the live model: 11 of 12 objects were still
> identified from the pixels alone. What's still curated is which cells of certain demo clips form a
> region and what those regions are called, a handful of tuned thresholds, and two name-based
> fallbacks that only fire when nothing was measured."
