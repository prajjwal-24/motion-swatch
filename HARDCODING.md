# What's real vs. hardcoded in Motion Swatch

**Last accounted for: Step 10** (VLM layer auto-label + auto-apply). If this file and
`docs/BUILD_PLAN.md` ever disagree, BUILD_PLAN is the one with the measurements attached and
wins; this file exists to be the *short* answer to "is this real or hardcoded?".

This POC extracts motion from real video, distills it into a reusable swatch, and applies it to
objects in artwork. Some parts run genuine algorithms; a few are curated so a specific demo clip
lands right. This is the honest, complete accounting — nothing here is hidden in the code, and
every surviving piece of curation is named below with the gate that keeps it from overriding a
real measurement.

**Short version:** the motion math is real end to end, and so is the *decision of what to apply
it to*. What remains curated is **which cells of a demo clip form a region and what to call it**,
a handful of tuned thresholds, and two name-based fallbacks that only run when nothing was
measured. The governing rule across the whole app is that **real extracted motion wins over
curation**, in every place the two could disagree.

---

## 1. Real (not hardcoded)

- **Optical-flow extraction.** RAFT (`raft_small`) and SEA-RAFT compute genuine dense flow from
  whatever you upload. Real inference, not a lookup.
- **Point tracking.** CoTracker3 tracks a real 12×12 grid of points through the clip
  (144 tracks × N frames).
- **Object masking (Step 2).** SAM 2, seeded by the VLM's bbox, replaces the rectangular crop so
  the background stops diluting the swatch. The mask coverage is reported in the swatch
  description (`masked 20% (sam2+motion, 38/144 cells tracked)`), so you can see how much of the
  frame the numbers came from.
- **Relative depth (Step 2).** Depth Anything V2 over the mask gives a depth *rank* — the
  fraction of the frame behind the object — sampled on 3 frames, reported per swatch.
- **Distillation to 8 parameters.** `service/distill.py` measures `frequency`, `amplitude`,
  `direction`, `turbulence`, `damping`, `phaseSpread`, `driftX`, `driftY` from the actual field.
  Real math on real pixels. (`direction` is an unsigned dominant-**axis** angle; `driftX/driftY`
  carry the sign.)
- **Body motion.** MediaPipe Pose extracts a real 33-joint skeleton, reduced to Contract B's 13
  joints, and drives a rigged character.
- **Object travel paths (Step 5).** YOLO + ByteTrack tracks one discrete object across the clip;
  the path is what moves the object, while the flow field supplies its internal motion.
- **The VLM router.** `/decompose` reads the clip and says what moves in it (Contract A);
  `/label` reads the *artwork* and says what each layer depicts (Contract D); `/judge` grades a
  rendered animation and proposes bounded param deltas (Contract C). All three are real model
  calls against real images, with forced tool calls so there is no prose to parse.
- **Class-keyed application (Step 8).** Which animator runs is decided by what the motion **is**
  (`swatch.applicator`, then class), never by what the layer is called. Rename "Birds" to
  "Layer 7" and a flock swatch still drifts it as a flock.
- **Class-keyed placement (Step 10).** Which *object* a swatch lands on is decided by matching
  the swatch's class against the VLM's reading of each layer
  (`contracts.match_swatches_to_layers`) — one-to-one, best-confidence first, no name matching.
- **The animation engine** (`js/animate.js`) — the rigid MLS mesh warp, trajectory-field replay,
  per-glyph text motion, deform-in-place (subtracting per-frame mean displacement so flowing
  objects ripple without sliding away) — general code that works on any object.
- **Object selection / the layer contract** (`js/regions.js`) — general; works on any SVG or
  raster.
- **The 9 built-in presets** (`js/motions.js`) — hand-authored parameter sets, like font weights.
  Not "hardcoded results": they are labelled presets in the UI and never described as extracted.

---

## 2. Hardcoded / curated

### a) `DEMO_PROFILES` — per-filename region layouts (`service/segment.py:296`)

The largest remaining piece of curation, and **the only filename test left in the repo.** If an
uploaded file's name contains a known substring (`cherry`, `silk`, `flag`, `wheat`, `ink`,
`ocean`, `two_flags`, `bosphorus`, …), `segment_regions` uses a hand-tuned region layout instead
of the automatic segmenter.

- It still runs the **real** RAFT extraction and distillation on those regions.
- What's curated is **which cells of the 12×12 grid form each region, and the region's name**
  (e.g. "California Flag" vs "US Flag", "Seagull Flight" + "Flag Flutter").
- It decides **nothing** about the extracted motion, and nothing about which object a swatch is
  applied to.

**Why:** 4K stock footage rarely auto-segments into clean 3–4 regions at demo resolution.
Longest matching key wins, so `waterfall` doesn't collide with a generic `wave`. Rename the file
(remove the keyword) and it falls back to the fully automatic path.

### b) Tuning thresholds (`service/config.py`, `service/segment.py`)

`ANALYSIS_WIDTH` (480), `TARGET_FPS`, `MAX_SECONDS`, `MIN_CELLS`, `REGION_KEEP_FRAC` (0.25),
`ABS_FLOOR`, `PEAK_FRAC`, `DRIFT_DOM`, `STATIC_PERSISTENCE_CUTOFF` are **tuned thresholds**, not
hardcoded outputs. They shape how the automatic path clusters motion. Reasonable but hand-picked
against the test clips.

Same category, elsewhere: `LABEL_MAX_LAYERS` 12 and `LABEL_CROP_W` 256 (cost caps on `/label`),
`LABEL_CONF_MIN` / `conf_min` 0.35 (the bar below which a layer label is not evidence enough to
animate anything), and the Step 9 loop's `JUDGE_MAX_ITERS` 3 / `JUDGE_GOOD` 0.8 /
`JUDGE_MIN_GAIN` 0.03. All are cited with their reasoning where they are defined, and the caps
are *reported* rather than applied silently.

### c) Name-keyed scenery behaviours — **preset-only fallback** (`js/animate.js:103-115`)

Four regexes (`birds` / `clouds` / `river|ripples` / `boat|rowboat|canoe|ferry|ship`) select
bespoke synthetic animations from the object's name. As of Step 8 they are gated behind
`if (!app)`: **a classified swatch never reaches them.** They run only for the built-in presets
and the in-browser Lucas–Kanade fallback, which carry no class and no captured field, so there is
nothing real to prefer over them. Real extracted motion always wins, even where the curated
version looks nicer.

The bird wing-flap in particular is synthetic — **do not advertise it as extracted.**

### d) The cloth-mode name hint — **gated last resort** (`js/regions.js:54`)

`CLOTH_NAME_HINT = /flag|banner|cloth|pennant|curtain|sail/i` decides whether a region's geometry
is bent or moved rigidly, *from the layer name* — a claim about a string an illustrator typed.
Since Step 10 it is the last thing consulted, and every selection records which evidence decided
it in `waveModeFrom`. Strongest first:

| `waveModeFrom` | what decided it |
|---|---|
| `preset_leaffall` | the `autumn-fall` **preset** is rigid by construction |
| `artwork_rigid` | the artwork marks the object `data-motion-mode="rigid"` |
| `motion_field` | a captured motion with a real trajectory field arrived (a measurement) |
| `vlm:<class>` | the VLM looked at the layer (Contract D) |
| `name_hint` | **this regex matched — a guess** |
| `default` | nothing matched; rigid |

`motion_field` outranks `vlm:<class>` deliberately: a measured per-point displacement field beats
a still-image reading. It is kept rather than deleted because with the router offline it is the
only answer available, and a flag that does not ripple is a worse failure than an honest guess.

### e) The `autumn-fall` preset (`js/motions.js:58`, `js/animate.js:_applyLeafFall`)

A hand-authored falling-leaf look with its own applicator. It is a **preset** in the presets
list, it carries `params.leafFall` so the code path is visible, and no extraction can produce it.

### f) Preprocessed artwork (`assets/Artwork/`)

Two scenes had their structure curated by hand so their objects are individually selectable:

- **`train-window-adobe.svg`** — Illustrator flattened all layer names on export (only `View` and
  `Train` survived), so a one-time script identified each object by geometry + colour and wrapped
  it into a named `<g class="layer" data-name="…">`. Rendering is byte-identical; only structure
  changed. Its scene **tab** was removed from the UI; the file is still loadable via *Upload
  artwork* or `loadScene('train')`.
- **`boat-river_layered.svg`** — every group hand-labelled with `data-name` +
  `data-motion-mode`, plus occlusion clip-groups (smoke behind foliage, etc.).

Once wrapped, selection and animation are fully general — and since Step 10 the labels the app
*acts* on come from `/label` looking at the pixels, not from these names.

---

## 3. Removed (was hardcoded, no longer is)

- **The synthetic `Autumn Fall` capture** (was `js/upload.js`). Uploads whose filename matched
  `/leaf|leaves|falling|autumn/i` used to skip extraction entirely, play the "extraction" overlay
  over the real video, and return a hand-written spiral from `synthFallTrajectories()` plus eight
  hand-tuned dials captioned *"Captured from falling-leaves video"*. **Deleted at Step 10.** The
  same clip now routes like anything else; `js/upload.js:106-115` is a comment recording what
  stood there, so a reader who finds the leaf look in `js/motions.js` knows it is a preset.
- **The cloth `/flag|banner|pennant|ensign|standard/` applicator regex** (was `js/animate.js`).
  It existed because the old per-point resample mangled clean stripes (27% median local shape
  distortion on flag.mp4, 153% worst case), so flag-like names had to opt out of real motion and
  use a synthetic sine. The rigid MLS mesh warp cuts that ~3× (9% median, 68% worst), which is
  what let the regex go — no name needed, and no synthetic stand-in.
- **Layer-name matching for placement.** Which object a swatch lands on is now the VLM's reading
  of the artwork plus class equality. Measured against the live model with **every** layer renamed
  `Layer N` / `pathNNNN`: 11/12 objects still identified from the pixels alone (12/12 with names),
  motion classes acceptable 12/12 in every run — including the flag, which came back
  `"flag"` / `cloth` / 85% while its layer was called `Layer 24`.
- **Trusting hand-labelled layer names.** The preprocessed scenes above still *carry* curated
  names, but nothing acts on them any more: placement comes from `/label` reading the crops, and
  the region's display name comes from the label too. The names are now a convenience for a human
  reading the file, not an input to a decision. (The curation itself is still disclosed — §2f.)

---

## 4. Honest limitations (not worked around)

- **Optical flow genuinely cannot see very thin motion** (e.g. light rain) at demo resolution —
  those clips extract near-zero amplitude, and we do **not** fake a region.
- **Smooth slow-motion water** reads as static texture (amplitude ≈ 0) and is dropped rather than
  invented. When the judge is shown frozen frames it reports `amplitude: 0.0` and *"flag edges
  identical in every frame"* rather than inventing motion it cannot see.
- **The mesh warp is as-rigid-as-possible, not rigid.** 33% p95 residual is a reduction, not an
  elimination; above ~1.8× the lattice spacing the lattice folds and the warp tears. Real captured
  amplitudes are 0.02–0.1, which is the range it was measured in.
- **`/label` sees each layer as an isolated crop**, with no relationships between them — "the flag
  *on* the pole" is two independent readings. The model's class for a thin cloud is genuinely
  unstable across runs (`cloth` 55% one run, `fluid` 55% the next); the 0.35 confidence floor
  mitigates this but does not remove it.
- **A gated engine that isn't installed probes `False` with a setup hint and falls back**, rather
  than pretending or crashing.

---

## Bottom line for a demo

If someone asks *"is this real or hardcoded?"*, the honest answer is:

> "The extraction, the animation, and the *choice of what to animate* are all real. It runs
> optical flow or point tracking on your video, masks it to the object SAM 2 found, and applies
> the measured field; a VLM looks at your artwork and says what each layer depicts, and swatches
> are placed by matching motion class — not by layer name. What's still curated is which cells of
> certain demo clips form a region and what those regions are called, plus a couple of tuned
> thresholds and two name-based fallbacks that only fire when nothing was measured. Rename a demo
> clip or upload a new one and it runs fully automatically."
