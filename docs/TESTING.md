# How to test Motion Swatch, and what to expect

Every number in this file was measured on this machine on **2026-08-26** against the current
`main`. Where a thing is broken or gated, that is stated instead of hidden — the whole point of the
tiers below is that **most of the suite needs no credentials**, so a real regression breaks loudly
without spending money.

Tiers, cheapest first:

| Tier | Needs | Cost | What it proves |
|---|---|---|---|
| **0** | nothing but the repo | free, ~40 s | the contracts and every routing/placement decision |
| **1** | `sh start-all.sh` | free, minutes | real extraction on real pixels |
| **2** | a working VLM credential | **real money** | perception: reading clips and artwork |
| **3** | a browser | free | the UX, the animation, export |

---

## Tier 0 — no services, no credentials, no browser

Run all twelve. Every one exits `0`. **Check the exit code, not the string** — ten print
`all checks passed`, and `step8-applicators` / `step10-orchestration` print
`all step N checks passed`, so grepping for one exact phrase twelve times will report a false
failure.

```bash
# the shared schemas, in all four interpreters (they must agree)   — 153 checks each
service/venv/bin/python  service/contracts_selftest.py
mpvenv/bin/python        service/contracts_selftest.py
routervenv/bin/python    service/contracts_selftest.py
/usr/bin/python3         service/contracts_selftest.py

# node only — these need nothing from tests/node_modules (fs, path, child_process only).
# step10-orchestration shells out to a bare `python3`, so python3 must be on PATH.
node tests/step2-field.js            #  14  masked-field distillation
node tests/step8-applicators.js      #  47  dispatch routes on class, never on the layer name
node tests/step9-sampling.js         #  25  one-cycle frame sampling for the judge
node tests/step10-orchestration.js   #  65  class-keyed placement + the evidence precedence chain

# stdlib only — needs no venv and no credential (imports os, sys, contracts, extractors, t2m)
routervenv/bin/python tests/step6-text2motion.py  # 100  the text2motion GATE (asserts an absence behaves)

# needs cv2 + numpy; portable across the two heavy venvs, NOT the system interpreter
routervenv/bin/python tests/step2-preprocess.py   #  38  mask / depth / camera wiring  (see note)

# hard-locked to routervenv: vlm_router.py imports cv2 + numpy + anthropic at module level
routervenv/bin/python tests/step9-judge-loop.py   #  46  no verdict can escape the param caps
routervenv/bin/python tests/step10-label.py       #  68  no wrong answer can move the wrong object
```

**Measured 2026-08-26: 12/12 green.**

Why four interpreters for one file: the services live in three venvs with genuinely conflicting
dependencies (py 3.13 / 3.11 / 3.9). If `contracts.py` ever picks up a version-specific
dependency, one of those four fails and you find out on a laptop instead of in a demo.

### Three counts that are environment-dependent — don't treat them as fixed

1. **`tests/step2-preprocess.py` is a floor of 38, not a constant.** Its own docstring says to run
   it in *either* heavy venv and both must pass. Measured:

   | interpreter | checks | exit |
   |---|---|---|
   | `service/venv/bin/python` (3.13) | **40** | 0 |
   | `mpvenv/bin/python` (3.11) | 38 | 0 |
   | `routervenv/bin/python` (3.9) | 38 | 0 |
   | `/usr/bin/python3` | — | **1** — `No module named 'cv2'` |

   The extra two in `service/venv` are the SAM 2 / depth checks that only run where those packages
   are importable. Passing a real clip raises it further (40 → 42). So a count *above* 38 is
   coverage, not corruption; a count *below* 38 is the regression.
2. **`tests/step6-text2motion.py` prints no per-check `ok` lines** — `grep -c '  ok '` returns `0`
   for it. Read its self-reported `100 checks, 0 failed` instead.
3. **`step9-judge-loop` and `step10-label` fail in `service/venv`** (exit 1) — that is expected, not
   a bug. A CI job that picks the wrong interpreter reports zero checks and a failure rather than a
   pass, which is the safe direction.

> Any change to these counts must be propagated to **three** places, which are easy to drift apart:
> `README.md` (Tests section), `PROJECT_STATUS.md` (§Tests table), and this file.

### The one Tier-0 check worth reading the output of

```bash
service/venv/bin/python -c "import sys; sys.path.insert(0,'service'); import t2m; print(t2m.available())"
```

**Expect a `False` that names *one specific missing thing*.** The specific thing **depends on the
interpreter**, because the probe reports the first gap it hits in *that* environment — this is the
gate working, not drift:

| interpreter | measured output |
|---|---|
| `service/venv/bin/python`, `/usr/bin/python3` | `(False, "momask needs the 'clip' module: pip install 'git+https://github.com/openai/CLIP.git'")` |
| `mpvenv/bin/python`, `routervenv/bin/python` | `(False, "momask needs the 'torch' module: pip install torch")` |

`tests/step6-text2motion.py:64` is what pins this: it accepts any reason naming
`clip` / `torch` / `numpy` / `einops` / `tqdm` / a `.tar` / a `.npy` / `momask-codes`, and line 67
separately *forbids* the generic `SETUP_HINT`. So the contract is "name a real gap", not "name this
exact gap."

If it ever returns `True` without you having installed the weights, or `False` with a vague
message, `tests/step6-text2motion.py` should have caught it. **Text → motion is not built** — there
is no text box in the UI and no prompt produces motion. What exists is the gate and a tested
SMPL-22 → Contract-B converter (`docs/BUILD_PLAN.md` Step 6).

---

## Tier 1 — services up, still no credentials

```bash
sh start-all.sh
sh start-all.sh status
```

**Expect:** `:8000 UP  :8765 UP  :8770 UP  :8771 UP  :8772 UP`.

> **`:8771 UP` does not mean the VLM works.** `/health` and `/` never touch the credential. See
> Tier 2.

### 1a. Which engines are actually installed

```bash
curl -s 127.0.0.1:8765/engines | python3 -m json.tool | head -40
```

**Measured: 17 registered, 10 available.**

| Available | Gated (probes `False` with a setup hint) |
|---|---|
| `raft_small`, `raft_large`, `searaft`, `cotracker3`, `evm`, `yolo_bytetrack`, `keypointrcnn`, `pose_mediapipe`, `sam2`, `depth` | `wham`, `mmpose_animal`, `tapir`, `objectron`, `droid_slam`, `momask`, `mdm` |

A gated engine must give a *reason*, e.g. `wham` → `clone yzhu.io/WHAM + weights (CUDA-leaning;
heavy)`. A gated engine with no reason, or one that crashes the endpoint, is a bug.

### 1b. Routing — free, instant, and the thing most worth checking

`/route` is the single source of truth for "which engine for which class", with live availability.
No video, no credential.

```bash
for c in articulated cloth fluid flock rigid_path oscillation; do
  echo -n "$c  "; curl -s "127.0.0.1:8765/route?cls=$c"; echo
done
```

**Measured — every class resolves to an *available* engine:**

| class | engine | kind |
|---|---|---|
| `articulated` | `pose_mediapipe` | skeleton |
| `cloth` | `searaft` | flow |
| `fluid` | `searaft` | flow |
| `flock` | `cotracker3` | trajectory |
| `rigid_path` | `yolo_bytetrack` | object_path |
| `oscillation` | `cotracker3` | trajectory |

> **Read this table before concluding "it fell back to RAFT".** `cloth` routes
> `searaft → raft_large → raft_small` and `fluid` routes `searaft → raft_small` — **every option is
> a RAFT variant**, because dense optical flow *is* the correct backend for cloth and fluid.
> `searaft` on a flag clip is the success case. The only classes that reach a non-RAFT engine are
> `flock` / `oscillation` (`cotracker3`) and `rigid_path` (`yolo_bytetrack`).

And the gated row, which must skip and say why:

```bash
curl -s "127.0.0.1:8765/route?cls=articulated&subject_type=human&has_text_prompt=true"
```

**Expect** `pose_mediapipe`, `available: true`, and a reason containing
`skipped momask: momask needs the 'clip' module…`. That is Step 6 falling back rather than
crashing — the whole done-when of the gate.

### 1c. Extraction on real pixels

```bash
curl -s -X POST -F file=@assets/videos/flag.mp4 127.0.0.1:8765/analyze | python3 -m json.tool | head -30
```

**Measured on `flag.mp4`, 191 frames:**

| request | `engine` | amplitude | frequency | tracks | `note` |
|---|---|---|---|---|---|
| `/analyze` (default) | `raft_small@mps` | 0.338 | 0.188 | 144 | — |
| `?engine=searaft` | `searaft@mps` | 0.35 | 0.188 | 144 | — |
| `?engine=raft_large` | `raft_large@mps` | 0.44 | 0.187 | 144 | — |

What to expect in general: **144 trajectories** (a 12×12 grid), amplitude in **0.02–0.1** for
typical scenery and up to ~0.4 for a hard-flapping flag, and **no `note`**. A `note` mentioning a
fallback is the honest signal that you did *not* get the engine you asked for — that is when
"falling back" is a real complaint.

Three gotchas that look like bugs and aren't:

1. **`?tracker=cotracker3` still reports `engine: raft_small@mps`.** `engine` is the *flow* engine;
   the tracker is a separate slot. Confirm it in `logs/raft.log` — the line reads
   `FLOW=raft_small TRAJ=cotracker3`. (Verified.)
2. **`?depth=1` alone returns no depth.** It needs `?preprocess=1`, and says so in `notes`:
   `?depth=1 needs ?preprocess=1 (depth is computed by the preprocess pass)`.
3. **An unknown or unavailable engine transparently falls back** to the kind's default *with* a
   note. That's by design (`service/extractors.py` `resolve`).

Full query surface: `engine`, `tracker`, `preproc`, `bbox`, `preprocess`, `depth`, `path`,
`swatch`, `cls`.

### 1d. Object travel paths (Step 5)

```bash
curl -s -X POST -F file=@assets/videos/boat.mp4 "127.0.0.1:8765/analyze?path=1" | python3 -m json.tool | head -40
tail -1 logs/raft.log
```

**Measured on `boat.mp4`, 200 frames** — a path *and* two honest warnings:

```
path=boat dist=0.097 … | NOTES: interpolated 30 frame(s) the detector missed across the 'boat'
track; 'boat' was tracked for only 42% of the clip; the path may be a fragment of its full travel
```

That 42% disclosure is the expected behaviour, not a failure: the path is real but partial, and it
says so rather than silently extrapolating.

### 1e. Body motion (Step 3)

```bash
curl -s -X POST --data-binary @assets/videos/walk-man.mp4 127.0.0.1:8770/extract | python3 -m json.tool | head -20
```

**Measured:** `engine: mediapipe_blazepose`, `fps: 15`, **detected 155 / 155 frames**, and exactly
13 joints per frame:

```
nose l_sho r_sho l_elb r_elb l_wri r_wri l_hip r_hip l_knee r_knee l_ank r_ank
```

`detected` well below `total` means the clip is too dark / the body too small / too cropped — that
is a clip problem, not a service problem. `?kind=hands` (21 landmarks) and `?kind=face` (468) also
work but **are not wired into any UI** — there is no hands applicator.

### 1f. Mask + camera motion (Step 2) — the one test you eyeball

This is the Step-2 done-when: *tracking runs only inside the masked object.*

```bash
routervenv/bin/python service/preprocess_cli.py assets/videos/flag.mp4 0.35,0.1,0.62,0.78 cloth
open /tmp/ms-flag-overlay.png
```

**Measured:** `engine: farneback+affine`, mask `coverage: 0.1014`, `is_static: false`,
`residual_px: 3.951`, and the warning `camera: 18/47 frame-pair(s) lacked reliable background
features`.

**In the PNG, expect red to hug the flag and the blue sky to be excluded.** That eyeball check is
the test — a rectangle of red, or red spread across the sky, means the mask failed and the swatch
is being diluted by background.

`coverage ≈ 0.10` is right for a flag occupying a tenth of the frame. Coverage near 1.0 means the
mask degenerated to the whole frame.

---

## Tier 2 — the VLM path (costs real money)

### First: is the credential actually usable?

```bash
curl -s 127.0.0.1:8771/ | python3 -m json.tool
```

This tells you the `backend` (`anthropic` or `bedrock`) and the model. **It does not tell you the
credential works** — `/` and `/health` never call the model. The only proof is one real call:

```bash
routervenv/bin/python service/decompose_cli.py assets/videos/flag.mp4
```

**Expect Contract-A JSON**, roughly:

```json
{ "version": 1, "static": false, "motions": [
  { "id": "m1", "label": "waving flag", "class": "cloth",
    "bbox": [0.42, 0.10, 0.35, 0.30], "confidence": 0.94,
    "backend": "flow_raft", "applicator": "wave", "notes": "flutters left" } ] }
```

Checks worth making on that output: `class` is one of the six; `confidence ≥ 0.35` (the evidence
floor — below it nothing is animated); `bbox` is fractional `[x, y, w, h]` in 0–1; a multi-motion
clip yields **several** entries.

**⚠️ Status on this machine right now: BLOCKED.** `logs/router.log` shows

```
403 … not authorized to perform: bedrock:InvokeModel on
arn:aws:bedrock:us-west-2:…:inference-profile/global.anthropic.claude-opus-4-8
because no session policy allows the bedrock:InvokeModel action
```

The Bedrock token *authenticates* and is then *refused* — the role's session policy doesn't grant
`bedrock:InvokeModel`. A fresh Bedrock token produces the same 403. Fix either way:

- **Direct Anthropic key (fastest).** Put an `sk-ant-…` key on the clipboard and run
  `sh setkey.sh` (a local helper, gitignored — a fresh clone won't have it; edit `.env` by hand
  instead). It **merges** into `.env` instead of overwriting it, backs the old file up to
  `.env.bak`, and then reports which credential the router will actually use.
- **⚠️ A real Anthropic key does not automatically win.** `_use_bedrock()`
  (`service/vlm_router.py:86-94`) returns `True` when `AWS_BEARER_TOKEN_BEDROCK` is merely
  *present* — it checks that **before** it ever consults `_real_key()` (`:80-83`). So leaving both
  credentials in `.env` keeps you on the denied Bedrock path and the new key does nothing.
  `sh setkey.sh --prefer anthropic` comments the Bedrock token out (value preserved, so it's
  reversible) and writes the explicit `ROUTER_USE_BEDROCK=0`, which `_use_bedrock()` consults
  first of all. Then `sh start-all.sh stop && sh start-all.sh` and confirm `"backend": "anthropic"`.
- `sh setkey.sh --status` prints that precedence verdict without writing anything. It also flags a
  shell `export` of either variable, which **shadows `.env`** entirely — `_load_dotenv()`
  (`:55-69`) uses `os.environ.setdefault`, so it never overrides what the shell already set.
- **Or** get `bedrock:InvokeModel` allowed on that inference profile.

**What a 403 looks like from the outside, and why it is NOT an extraction bug.** With `/decompose`
down the browser sends a *classless* extraction request, so `logs/raft.log` reads:

```
[analyze] FLOW=raft_small … bbox=- crop=no mask=no … class=- frames=120 regions=1
```

`class=-` / `bbox=-` / `mask=no` is the signature. `raft_small` is the documented default for a
classless request — the extractor did the right thing with the nothing it was given. **Always
check `logs/router.log` before blaming the extractor.**

### The rest of Tier 2, once a credential works

| What | Command / action | Expect |
|---|---|---|
| Read the clip (Step 1) | `decompose_cli.py` on a flag+smoke+birds clip | 3 correctly-classed entries, no filename hints |
| Read the artwork (Step 10) | **Auto-label layers** button | one `/label` pass; each layer gets a depiction + class + confidence; capped at 12 layers **with a warning**, never a silent partial |
| Grade the render (Step 9) | **Judge & tune** button | a score, ≤3 iterations, bounded param deltas, reverts to the best pass |
| Everything at once | `node tests/step10-e2e.js` | **20 live checks** |

The live e2e is the only thing that can answer *"does the model recognise a river when the layer is
called `path2854`?"* — it labels the Scenery scene twice, once with real names and once with every
layer renamed `Layer N` / `pathNNNN`, then runs `assets/videos/Autumn.mp4` through the real file
picker onto the artwork it labelled blind.

```bash
sh start-all.sh
npm install --registry=https://registry.npmjs.org/ puppeteer-core   # once, into tests/
node tests/step10-e2e.js
```

**Measured previously: 11/12 objects still identified from pixels alone with all names destroyed**
(12/12 with names); motion classes acceptable 12/12 in every run — including the flag, which came
back `"flag"` / `cloth` / 85% while its layer was called `Layer 24`. A thin cloud is genuinely
unstable run-to-run (`cloth` 55% one run, `fluid` 55% the next) — that is a known limitation, not
a flake to chase.

---

## Tier 3 — the browser

Open `http://localhost:8000`. **If a change doesn't appear, it's the cache**: 19 hand-maintained
`?v=NN` refs in `index.html` (currently all `?v=54`). Hard-refresh or bump them all together:

```bash
grep -o '?v=[0-9]*' index.html | sort | uniq -c     # expect a single line: 19 ?v=NN
```

### The manual path

1. Pick **Poster** or **Scenery**. Expect *"Poster loaded — click the flag or the title, then pick
   a motion."*
2. Click an object → it highlights, a chip appears at the bottom, the Region Inspector opens.
3. Click a preset in the Motion Library → *"Applied "Flag Flutter" to "flag"."* — the second name is
   the layer's `data-name`, which is **lowercase** in the generated scenes (`js/poster.js`,
   `js/scenery.js`). With no object selected you get *"Motion "…" selected — now click an object to
   apply it."* instead (`js/main.js:82`).
4. **▶ Play** → every object with a motion animates at once, each with its own.
5. **+ Upload Video** with `assets/videos/flag.mp4` → an extraction overlay (flow vectors +
   distilled dials), then *"Motion "…" captured from video — click it, then apply to an object."*
6. Toggle **Cloth mode** on a flag → the geometry ripples *in place* instead of sliding.
7. **Export SVG** / **Export video**.

### How to tell, in the UI, which engine actually produced a swatch

**Hover the swatch chip in the Motion Library.** The tooltip (`tile.title`, `js/main.js:112` →
`swatchSummary`, `js/main.js:92-100`) is the only place in the app that states a swatch's
provenance, one line per swatch:

```
<kind> · <class> · <engine> · <frames> frames @ <fps>fps · confidence NN% (<confidence_of>)
    ⚠ <any warning>
```

So a dance clip that came out as `texture · unclassified · raft_small@mps` instead of
`skeleton · articulated · mediapipe_blazepose` tells you the class never arrived — check
`logs/router.log` for a 403 before blaming the extractor.

A **preset** has no `swatches` array at all, so the tooltip falls back to `m.desc` — and because a
preset carries no class, `_applicatorFor()` returns `''` and only then do the name-keyed curated
behaviours in `js/animate.js:100-115` get a turn. That is the gate that keeps real extracted motion
winning.

### The hands-off path (needs a working `:8771`)

Load a scene → **Auto-label layers** → **+ Upload Video** with a multi-motion clip. Expect:

> *One clip → 3 motions applied by what each layer IS, not what it is called*

and, critically, **the refusals named too**:

> *not placed: "seagull flight" (no free layer labelled flock)*

A refusal is a **pass**, not a failure — it means placement required class equality and didn't find
a match, instead of dumping the swatch on the nearest object.

### What to expect *not* to work

These are known and documented (`PROJECT_STATUS.md` §5), so don't file them as bugs:

- **Bird wing-flap and the `autumn-fall` look are synthetic**, not extracted. They only run for
  presets and unclassified motions. Never demo them as "extracted from the clip."
- **Light rain and smooth slow-mo water extract amplitude ≈ 0** and are dropped rather than
  invented. Optical flow genuinely cannot see them at 480 px.
- **No text box for text → motion.** Step 6's done-when is not met.
- **`:8772` is never called from the browser** — the browser reaches masking via `?preprocess=1` on
  `:8765`. `:8772` is CLI-only.
- **The mesh warp tears above ~1.8× lattice spacing.** Real captured amplitudes are 0.02–0.1,
  which is the range it was measured in.
- **Judge sessions are in-memory** — restarting `:8771` forgets a tuning run in progress.
- **`train-window` backward-parallax is not built.**

---

## Triage order when something looks wrong

1. `sh start-all.sh status` — all five UP?
2. `sh start-all.sh status` says UP but behaviour is stale → **restart**. The services do not
   hot-reload; a service started before your last commit is running the old code. Prove it:
   `curl -s 127.0.0.1:8765/engines | grep -c momask` should be `2`.
3. Unexpected engine → `curl "127.0.0.1:8765/route?cls=<class>"` and compare to the §1b table
   before assuming a fallback. Remember `cloth`/`fluid` are RAFT by design.
4. `tail -20 logs/router.log` — a 403 here explains a `class=-` extraction downstream.
5. `tail -5 logs/raft.log` — the `[analyze]` line names the flow engine, tracker, mask, class and
   frame count for every request.
6. Browser shows old behaviour → the `?v=NN` cache-bust.
7. Run Tier 0. If those twelve are green, the contracts and every routing/placement decision are
   intact and the problem is environmental.
