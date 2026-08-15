# MotionLife (Motion Swatch) — Project Status

> Round-2 master document. Covers **(1)** what the project is and everything built,
> **(2)** exactly where and why we hardcoded, and **(3)** what is not working / known
> limitations. Written to be spoken to honestly in a review. Companion docs:
> `HARDCODING.md` (hardcoding detail), `IMPLEMENTATION.md` (module spec),
> `DEMO_SCRIPT.md` (video script), `README.md` (usage).

---

## 1. What it is

**MotionLife** captures real-world motion from a short video, distills it into a
reusable, editable **motion swatch**, and applies it to individual objects in a
static vector illustration — no keyframes, no rigging. The pitch: *motion becomes
an asset you paint on, like a color.*

Two pieces:
- **Browser app** — vanilla JS + inline SVG + Canvas 2D. No framework, no bundler.
  Served with any static server (`python3 -m http.server 8000`).
- **Analysis service** — FastAPI + **RAFT** deep optical flow (`raft_small`, torchvision).
  Optional at runtime; the app falls back to an in-browser Lucas–Kanade analyzer.

---

## 2. Everything built

### App / UX
- **Upload artwork** — SVG (inline, per-object selectable) or raster (rectangle select).
- **Layer / object selection** — objects marked `<g class="layer" data-name="…">`
  become selectable; a Layers panel lists them.
- **Motion Library** — 8 built-in presets (Waterfall Flow, Cloud Drift, Flag Flutter,
  Gentle Sway, Water Ripple, Sun Pulse, Falling Leaves, Rising Smoke).
- **Capture from video** — upload a clip → RAFT extraction → new swatch, shown with an
  animated "extraction" overlay (flow vectors + the distilled parameters).
- **Per-object controls** — Speed, Intensity, Remove motion, Delete region.
- **Cloth / deform vs. rigid** — driven by the artwork's `data-motion-mode` attribute
  and object name.
- **Play / Pause**, **Preview** (Before | MotionLife | source videos), **Export** (SVG / video).

### Analysis / engine
- **RAFT optical flow** → distilled to **8 parameters** (frequency, amplitude,
  direction, turbulence, damping, phase-spread, driftX, driftY) + a **12×12 trajectory
  grid** (144 real point-tracks).
- **Region segmentation** (`segment_regions`) for multi-motion clips.
- **Animation engine** (`js/animate.js`): `computeMotion` (rigid), wave/cloth
  geometry deformation, trajectory-field replay, **deform-in-place** (subtract mean so
  flowing objects ripple without sliding away), per-glyph text motion, and
  **detail-ride** (keep fine detail like a flag's chakra crisp while the cloth ripples).

### Name-keyed scenery behaviors (see §3)
Birds, clouds, river/ripples, boat, tree-canopy, falling-leaves, and flag each have a
bespoke, hand-tuned behavior.

### Assets
- **Scenes / posters**: `poster.svg` (Independence Day flag), `Autumn*.svg`,
  `boat-river_layered.svg` (riverside scene, fully layer-labeled), plus
  `train-window-adobe.svg`, `Scene.svg`, and earlier hand-built scenes.
- **Source videos** (`assets/videos/`): `flag`, `birds`, `clouds`, `smoke`, `boat`,
  `boat-night`, `Autumn`.

### Pitch materials
- **`intro-deck.html`** — animated 9-slide pitch deck (problem → reveal → how-it-works →
  live-app button) plus a dense **one-frame thumbnail** slide. Live particle-flow
  background; every slide has its own animation.
- Voiceover scripts and demo transcripts (in chat / `DEMO_SCRIPT.md`).

### GitHub
- Repo: **github.com/prajjwal-24/motion-swatch** (public). All code, assets, and the
  7 source videos are committed.

---

## 3. Where we hardcoded (honest accounting)

**The motion extraction and animation engine are real.** The hardcoding is about
*which behavior an object gets*, *what a demo clip is called*, and *one fully-synthetic
capture*. Ranked by how much it matters if a reviewer asks:

### 3a. `Autumn Fall` capture is fully synthetic — FRONTEND (`js/main.js`)
**The biggest one.** If an uploaded video's **filename** matches
`/leaf|leaves|falling|autumn/i`, the app **skips RAFT entirely**, generates synthetic
falling-leaf trajectories, and hands back hand-tuned parameters — while still showing
the animated "extraction" overlay over the video. So for the leaves demo, the
"extraction" is a visual; no real analysis happens. Rename the file and it goes through
the real pipeline.

### 3b. Name-keyed scenery behaviors — FRONTEND (`js/animate.js`)
Whenever a motion is applied to an object whose **name** matches a keyword, the animator
runs a **bespoke synthetic behavior** instead of replaying the captured motion:

| Object name matches | Hardcoded behavior | Uses captured params? |
|---|---|---|
| `birds` | Wing-flap (vertical scale) + split left/right gentle drift | speed/intensity only |
| `clouds` | Slow one-directional wind drift + subtle edge billow | speed/intensity only |
| `river` / `ripples` | Smooth laminar traveling-wave flow | frequency, amplitude |
| `boat` / `rowboat` / … | Gentle rigid bob + rock (floats on ripples) | frequency, amplitude |
| `flag` / `banner` / … | Coherent pole-anchored multi-fold wave (ignores raw field) | frequency, amplitude |
| `tree-canopy` (role) | Gentle sway | preset |
| `leafFall` param | Per-leaf independent fall/tumble/wrap | synthetic |

**Implication to state plainly:** for these scenery objects, the captured video mostly
triggers a *preset-style* behavior — the object animates its hardcoded way regardless of
which swatch is dropped on it. "Apply the *exact* motion you captured" is only literally
true for generic objects that go through the real wave / trajectory-field path.

> **Bird wing-flap is synthetic — do NOT advertise it as extracted from the video.**

### 3c. `DEMO_PROFILES` — filename-keyed region layouts (`service/server.py`)
If an uploaded filename contains a known keyword (`cherry`, `silk`, `flag`, `wheat`,
`ink`, `ocean`, `two_flags`, `bosphorus`, …), the service uses a **hand-tuned region
layout + names** instead of automatic segmentation. It still runs **real RAFT
extraction** on those regions — only the region rectangles and labels are curated.

### 3d. Preprocessed artwork
- **`train-window-adobe.svg`** — Illustrator flattened all layer names on export; we
  ran a one-time script to identify each object by geometry/color and wrap it into named
  `.layer` groups. Rendering is byte-identical; only structure changed.
- **`boat-river_layered.svg`** — every group hand-labeled with `data-name` +
  `data-motion-mode`, plus occlusion clip-groups (smoke behind foliage, etc.).

### 3e. Tuning constants
Segmenter thresholds (`ANALYSIS_WIDTH`, `REGION_KEEP_FRAC`, `ABS_FLOOR`, …) and the
wave constants (`WAVE_CYCLES`, `WAVE_AMP_PX`, `DETAIL_FRAC`, …) are hand-picked, not
faked outputs.

---

## 4. What IS genuinely real
- RAFT optical-flow extraction on any uploaded video **that isn't caught by 3a** and
  reaches the service.
- Distillation to the 8 parameters + 12×12 trajectory grid.
- The animation primitives: wave/cloth deformation, trajectory-field replay,
  deform-in-place, per-glyph text, detail-ride (crisp chakra).
- Object selection / the `.layer` contract; Speed / Intensity; Preview; Export.
- The coherent flag wave *is* driven by the captured clip's frequency & amplitude.

---

## 5. What is NOT working / known limitations

1. **Bird wing-flap and Autumn Fall are synthetic**, not extracted (see 3a, 3b).
   Present them as "the object animates naturally," never "extracted from the clip."
2. **Scenery objects ignore the captured motion's character.** The swatch triggers a
   name-keyed preset. The demo's "apply the *exact* motion" line is only fully honest
   for the flag (freq/amp-driven) and generic wave objects.
3. **RAFT can't see thin or smooth motion** — light rain and smooth slow-mo water
   extract amplitude ≈ 0. We drop them rather than invent a region (honest, but it
   means some clips "don't work").
4. **Auto-segmentation of complex 3–4-motion clips is unreliable** at demo resolution
   (480 px) — `DEMO_PROFILES` (3c) compensates for the curated demo clips only.
5. **Arbitrary uploaded SVGs need pre-labeled layers.** A raw Illustrator export with
   flattened names only yields coarse (whole-group) selection until preprocessed.
6. **Train-window "scenery moves backward" parallax is not built.** It needs bulk
   directional travel with looping/oversized content; the current deform-in-place logic
   intentionally suppresses bulk drift.
7. **`CompleteDemo.mp4` has a long black tail** — real content ends ~2:15 of a 4:24 file;
   trim before combining.
8. **Browser caching** — JS changes require bumping the `?v=NN` query string in
   `index.html` or a hard-refresh, or the browser serves stale scripts.
9. **Large source video in repo** — `birds.mp4` is 82 MB (over GitHub's 50 MB soft
   warning; under the 100 MB hard limit). Fine, but noted.

---

## 6. One-line answer for reviewers

> "The optical-flow extraction and the animation engine are real. What's curated is
> which behavior each scenery object gets, what we name certain demo clips, one artwork
> whose layers we re-labeled after Illustrator stripped them, and the falling-leaves
> capture, which is fully synthetic. Rename a clip or use a generic object and it runs
> through the real pipeline."
