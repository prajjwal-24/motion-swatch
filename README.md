# Motion Swatch

Capture motion from a real video, extract it into a reusable **motion swatch**, and
apply it to any object in a piece of vector artwork. A flag clip becomes a "Flag
Flutter" swatch you can drop onto a banner; a smoke clip becomes rising steam over
a coffee cup. Motion is stored as **8 editable numbers + a 12×12 grid of real point
trajectories**, not baked video — so it re-times, re-scales, and re-applies to any
shape.

![Motion Swatch](docs/screenshot.png)

---

## What it does

1. **Load artwork** — a built-in Poster or Scenery scene, or upload your own
   SVG / PNG / JPEG.
2. **Select an object** — click any SVG object (or draw a rectangle on a raster
   image). Each object becomes an independently animatable region.
3. **Get a motion** — pick one of 9 built-in presets, or click **"Capture motion
   from video"** and upload a short clip. A VLM reads the clip, the router picks an
   extractor, and the measured flow field is distilled into a swatch. One clip
   containing several motions yields several swatches.
4. **Let it place itself** *(optional)* — **Auto-label** asks the VLM what each artwork
   layer depicts, and each swatch lands on the object whose **motion class** matches.
   No layer names are consulted.
5. **Apply & tune** — assign a swatch by hand if you'd rather, adjust Speed /
   Intensity, toggle **Cloth mode** (the geometry itself ripples, in place, instead
   of moving rigidly). **Judge** grades the result and proposes bounded tweaks.
6. **Play** — every object animates at once with its own motion.
7. **Export** — animated SVG or a video of the result.

---

## Quick start

The browser app is **vanilla JS + SVG + Canvas** — no build step, no npm install.
The Python services are **optional** (the app falls back to a lighter in-browser
analyzer, and says which service to start when a feature needs one), but recommended.

**All of it at once:**

```bash
sh start-all.sh          # brings up 8000, 8765, 8770, 8771, 8772
sh start-all.sh status   # UP/DOWN per port
sh start-all.sh stop
```

The rest of this section is the per-service detail if you want to run them individually.

### 1. Serve the web app

```bash
cd motion-swatch-poc
python3 -m http.server 8000
# open http://localhost:8000
```

### 2. (Optional) Start the RAFT analysis service

```bash
cd service
./run.sh        # first run creates a venv and downloads the ~4 MB raft_small checkpoint
```

This serves the analyzer at `http://127.0.0.1:8765`. The app auto-detects it; if it
is down, capture still works using the in-browser analyzer.

**Requirements** (installed automatically by `run.sh`): Python 3.9+, `torch`,
`torchvision`, `numpy`, `opencv-python-headless`, `fastapi`, `uvicorn`. Runs on
Apple-silicon MPS, CUDA, or CPU.

**Pluggable extractor registry (Step 4).** The service is no longer hardwired to one
model — `service/extractors.py` registers multiple backends and `/analyze` selects one:

```bash
curl 127.0.0.1:8765/engines                                   # which backends are installed
curl -X POST -F file=@clip.mp4 127.0.0.1:8765/analyze         # default: raft_small (byte-identical)
curl -X POST -F file=@clip.mp4 "127.0.0.1:8765/analyze?engine=raft_large"    # sharper flow
curl -X POST -F file=@clip.mp4 "127.0.0.1:8765/analyze?tracker=cotracker3"   # long-range point tracks
curl -X POST -F file=@clip.mp4 "127.0.0.1:8765/analyze?preproc=evm"          # magnify subtle motion
```

| Engine | Kind | Status |
|---|---|---|
| `raft_small` (default) / `raft_large` | dense flow | ✅ built-in (torchvision, cached) |
| `cotracker3` | long-range trajectories | ✅ works when weights are cached (~97MB via torch.hub) |
| `evm` (Eulerian magnification) | pre-process | ✅ pure numpy+cv2, no download |
| `searaft` | dense flow | ✅ opt-in — run **`service/setup-searaft.sh`** (clones the repo + installs deps; first call downloads ~79MB weights). Heavier/slower than raft_small and not auto-benchmarked-better, so it stays opt-in and **falls back to raft_small with a note** if not set up. |

The **no-query `/analyze` and `/health` responses are byte-identical** to before, so the app
is unaffected. An unavailable/unknown engine transparently **falls back** to `raft_small`.

### 3. (Optional) Start the character-motion (pose) service

This powers the **Character** tab and the `duck-walk.html` demo. It extracts a real
walk/dance from a video with **MediaPipe Pose** (no hardcoding) and retargets it onto
a rigged character.

```bash
./service/run-pose.sh   # first run creates ./mpvenv and installs MediaPipe, then serves :8770
```

⚠️ **MediaPipe has no wheel for Python 3.13** — it needs Python **3.9–3.12**. The
script auto-picks a compatible interpreter and builds a *separate* venv (`./mpvenv`),
so it never clashes with the RAFT service's 3.13 venv. Manual equivalent:

```bash
python3.12 -m venv mpvenv
mpvenv/bin/pip install -r service/requirements-pose.txt   # mediapipe==0.10.14, opencv-python
mpvenv/bin/python service/pose_server.py                  # POST /extract on :8770
```

Then open `http://localhost:8000/duck-walk.html` (standalone duck demo) or use the
**Character** tab in the main app: pick the character → **upload a walking/dancing
clip** → it animates. `assets/motion/walk-pose.json` ships a pre-captured walk so the
duck demo runs even without a fresh upload.

**Backends A — bodies + hands + face (Step 3).** The same service also extracts hands
and faces (same MediaPipe wheel, no extra install), via a `kind` param:

```bash
curl -X POST --data-binary @clip.mp4 "http://127.0.0.1:8770/extract"              # pose (default, byte-frozen)
curl -X POST --data-binary @clip.mp4 "http://127.0.0.1:8770/extract?kind=pose&fmt=b"  # + viewpoint, gap-filled
curl -X POST --data-binary @hand.mp4 "http://127.0.0.1:8770/extract?kind=hands"   # 21 landmarks x up to 2 hands
curl -X POST --data-binary @face.mp4 "http://127.0.0.1:8770/extract?kind=face"    # 468 face landmarks
```

The **default `/extract` response is unchanged** (the character rig depends on it).
`kind=hands|face` return a Contract-B *skeleton swatch* (`service/contracts.py`); they're
backend-ready but **not yet wired into the character UI** (no hands/face applicator
exists — a hands swatch would not fit the body rig). Pose `fmt=b` adds a front/side
**viewpoint** estimate and short-gap interpolation over dropped frames.

### 4. (Optional) Start the VLM Router — *motion decomposition* (Step 1)

The router looks at a clip with **Claude vision** and returns every distinct motion
in it (class + bounding box + confidence) — replacing filename/layer-name guessing
with real perception. See `docs/BUILD_PLAN.md` Step 1.

```bash
cp .env.example .env        # then put your key in .env (ANTHROPIC_API_KEY=sk-ant-...)
./service/run-router.sh     # first run creates ./routervenv, serves :8771
```

**Credentials.** Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` — the router
loads it automatically and uses the direct Anthropic API. If no real key is set it
falls back to **Amazon Bedrock** (`CLAUDE_CODE_USE_BEDROCK=1`, model
`global.anthropic.claude-opus-4-8`, region from `AWS_REGION`) via the standard AWS
credential chain. `.env` is gitignored.

Verify it end-to-end from the CLI (no server needed):

```bash
routervenv/bin/python service/decompose_cli.py assets/videos/flag.mp4
```

You get **Contract-A JSON**:

```json
{ "version": 1, "static": false, "motions": [
  { "id": "m1", "label": "waving flag", "class": "cloth",
    "bbox": [0.42, 0.10, 0.35, 0.30], "confidence": 0.94,
    "backend": "flow_raft", "applicator": "wave", "notes": "flutters left" } ] }
```

The six motion classes and their routing live in `service/contracts.py`. The app
calls it via `MotionCapture.decomposeMotion(file)` (in `js/capture.js`).

### 5. (Optional) Start the preprocess service — *mask + camera motion* (Step 2)

Turns a router bbox into a clean object **mask** (so extraction runs only inside the
moving object) and estimates **camera motion** (so it can be subtracted — fixing the
"scenery scrolls with the camera" case). Pure OpenCV, **no downloads, no API key** —
reuses `./routervenv`. See `docs/BUILD_PLAN.md` Step 2.

```bash
./service/run-preprocess.sh             # serves :8772
```

Verify from the CLI — writes a mask overlay you can eyeball (the Step-2 done-when):

```bash
routervenv/bin/python service/preprocess_cli.py assets/videos/flag.mp4 0.35,0.1,0.62,0.78 cloth
# -> /tmp/ms-flag-overlay.png : mask (red) hugs the flag, blue sky excluded; static camera
```

Returns a `region_preprocess` contract: `mask` (RLE), `camera` (`is_static`, per-frame
transforms, `residual_px`), optional `depth`. **SAM 2** (cleaner masks) and **Depth
Anything V2** are gated upgrades behind checkpoints; the download-free default is
OpenCV Farneback motion-gating. The app calls it via `MotionCapture.preprocessRegion(file, motion)`.

---

## How to use it (demo walkthrough)

1. Open `http://localhost:8000`.
2. Choose a scene at the top (**Poster** / **Scenery**), or click **Upload artwork**
   to load your own SVG.
3. Click an object in the artwork — it highlights and appears as a chip at the
   bottom. The **Region Inspector** opens on the right.
4. Either:
   - Click a preset in the **Motion Library** (Flag Flutter, Rising Smoke, Cloud
     Drift, …), **or**
   - Click **+ Capture motion from video**, choose a short clip (a flag, smoke,
     swaying trees, flowing water…). It extracts a new swatch into the library.
5. With the object selected, click the swatch to assign it. For cloth-like objects
   (flags, hair, water, foliage) turn on **Cloth mode** so the shape ripples in
   place rather than sliding.
6. Adjust **Speed** and **Intensity**.
7. Repeat for other objects, then press **▶ Play**.
8. **Export SVG** or **Export video** to save the result.

**The hands-off version** (needs `:8771`): load a scene, click **Auto-label layers** — the VLM
reads each layer and says what it depicts — then upload a clip with several motions in it. Each
extracted swatch is placed on the object whose motion class matches, and the status line names
every placement *and* every refusal (`not placed: "x" (no free layer labelled cloth)`). Then press
**Judge & tune** to have the result graded and the dials nudged. Nothing in that path looks at a
filename or a layer name.

### Source videos

Nine sample clips **are** committed in `assets/videos/`: `flag`, `smoke`, `clouds`, `birds`,
`boat`, `boat-night`, `Autumn`, `walk-man`, `walk-grid`. (`birds.mp4` is 82 MiB — over GitHub's
50 MB soft warning, under the 100 MB hard limit.) Any short clip works too: point the "Capture
motion from video" picker at your own footage. A clip with **several** distinct motions is
handled — each region is extracted into its own swatch with its own engine.

`.gitignore` excludes `*.mp4` everywhere **except** `assets/videos/`, so scratch recordings stay
out; the 734 MB `CompleteDemo.mp4` is excluded by name as well.

---

## Project layout

```
motion-swatch-poc/
├── index.html            # app shell + panel layout
├── css/style.css
├── start-all.sh          # bring up / stop / status all four services
├── js/
│   ├── motions.js        # 9 built-in motion presets
│   ├── capture.js        # video → analysis service (or in-browser fallback)
│   ├── flow.js           # in-browser Lucas–Kanade fallback analyzer
│   ├── distill.js        # flow field → 8 swatch parameters
│   ├── regions.js        # object selection + the .layer/.ms-wrap contract
│   ├── upload.js         # the upload pipeline: decompose → route → extract → apply
│   ├── autolabel.js      # Contract D: ask the VLM what each artwork layer depicts
│   ├── animate.js        # class-keyed application, mesh warp, trajectory-field replay
│   ├── motionfields.js   # the rigid MLS mesh warp + wave data
│   ├── judge.js          # Step 9: render one cycle, grade it, apply bounded deltas
│   ├── scenery.js        # built-in Poster / Scenery SVG generators
│   ├── poster.js, multipick.js, extractviz.js, export.js, videoexport.js
│   └── main.js           # app controller / wiring
├── service/
│   ├── server.py         # FastAPI extraction service + extractor registry (8765)
│   ├── contracts.py      # the shared swatch/decompose/label/judge schemas
│   ├── extractors.py     # engine registry — each engine probes for what it needs
│   ├── distill.py, flow.py, segment.py, preprocess.py, sam2_seg.py, depth.py, objpath.py
│   ├── pose_server.py    # MediaPipe pose (8770)
│   ├── vlm_router.py     # /decompose, /label, /judge (8771)
│   ├── preprocess_server.py  # standalone mask/depth/camera helper (8772)
│   ├── run.sh            # venv bootstrap + uvicorn launch (port 8765)
│   └── requirements*.txt # one per service — their deps genuinely conflict
├── assets/
│   ├── Artwork/          # poster / autumn / riverside / train-window SVGs
│   ├── scenes/           # ready-made + rigged scene SVGs (motion-lab, character-duck/bear…)
│   └── videos/           # sample source clips for extraction
├── docs/BUILD_PLAN.md    # the roadmap + the measurements behind each shipped step
└── tests/                # see "Tests" below
```

### The layer contract

An uploaded SVG becomes selectable when objects are marked
`<g class="layer" data-name="...">`. Elements with only an `id` are treated as
static backdrop. If neither is present (e.g. a flattened Illustrator export), the
app falls back to wrapping top-level drawable clusters automatically.

---

## Technology

- **Motion extraction:** RAFT (`torchvision` `raft_small`, C_T_V2 weights) deep optical flow,
  optionally SEA-RAFT or CoTracker3, distilled to 8 parameters + a 12×12 trajectory grid.
- **Localization:** SAM 2 masks the object (seeded by the VLM's bbox) so the background stops
  diluting the swatch; Depth-Anything-V2-Small gives a relative depth rank over the mask.
- **Body / path:** MediaPipe Pose (13 joints in the swatch), YOLO + ByteTrack for object travel.
- **Vision-language:** Claude via `service/vlm_router.py` — reads the clip (`/decompose`), reads the
  artwork (`/label`), grades the render (`/judge`). Forced tool calls, so there is no prose to parse.
- **In-browser fallback:** Lucas–Kanade flow when the service is offline.
- **Rendering:** vanilla JS, inline SVG, Canvas 2D. No framework, no bundler. Deformation is a
  rigid Moving Least Squares mesh warp (Schaefer, McPhail & Warren 2006) over the captured field.

See `docs/BUILD_PLAN.md` for what each step ships and the measurements behind it,
`docs/ARCHITECTURE_FLOW.md` for the module/flow detail, `HARDCODING.md` for what is curated, and
`DEMO_SCRIPT.md` for the demo narrative.

---

## Tests

Most of the suite needs **no browser, no service, and no credentials** — that is deliberate, so a
contract change breaks loudly on a laptop with nothing running.

```bash
# pure contracts — runs in every interpreter in the repo (153 checks each)
service/venv/bin/python  service/contracts_selftest.py
mpvenv/bin/python        service/contracts_selftest.py
routervenv/bin/python    service/contracts_selftest.py
/usr/bin/python3         service/contracts_selftest.py

# node only, no browser and no service
node tests/step2-field.js            # 14  masked-field distillation
node tests/step8-applicators.js      # 47  dispatch routes on class, never on the layer name
node tests/step9-sampling.js         # 25  one-cycle frame sampling for the judge
node tests/step10-orchestration.js   # 65  class-keyed placement + the evidence precedence chain

# python services driven with a scripted fake VLM — no credentials, no model calls
routervenv/bin/python tests/step2-preprocess.py   # 38  mask / depth / camera wiring
routervenv/bin/python tests/step9-judge-loop.py   # 46  no verdict can escape the param caps
routervenv/bin/python tests/step10-label.py       # 68  no wrong answer can move the wrong object
```

### The live end-to-end test

`tests/step10-e2e.js` is the one suite that costs real VLM calls. It drives the real UI in headless
Chrome and is the only thing that can answer "does the model recognise a river when the layer is
called `path2854`?" — so it labels the Scenery scene twice, once with real names and once with
**every** layer renamed `Layer N` / `pathNNNN`, then runs `assets/videos/Autumn.mp4` through the
real file picker onto the artwork it just labelled blind.

```bash
sh start-all.sh                 # every service must be up (sh start-all.sh status to check)
npm install --registry=https://registry.npmjs.org/ puppeteer-core   # once, into tests/node_modules
node tests/step10-e2e.js        # 20 live checks
```

`tests/node_modules` is gitignored. Install it from inside `tests/` (or with `--prefix tests`); the
explicit `--registry` matters if your npm defaults to a private registry.

---

## Honest note on hardcoding

This is a **proof-of-concept built to demo well**. Some parts are genuine algorithms
and some are curated for reliability. `HARDCODING.md` documents exactly which is
which — read it before presenting this as fully automatic.
