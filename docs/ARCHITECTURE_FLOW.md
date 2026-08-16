# MotionLife 2.0 — Architecture Flow

> Visual diagram: **`architecture.html`** (open in a browser; dark-themed, exportable to
> PNG/PDF). This file explains the flow in words. Companion: `docs/TASK_BREAKDOWN.md`.

## One-sentence version

> **Upload a video → a VLM finds every motion in it → SAM 2 + CoTracker measure each
> one for real → each is distilled into a reusable motion swatch → the swatch is
> mesh-warped onto the artwork → a VLM judges the rendered result and auto-tunes it
> until it looks right.**

---

## Row ① — CAPTURE → EXTRACT (understand the video, measure real motion)

What happens when a user uploads a video.

1. **Upload Video** *(browser)* — a short clip (flag, river, birds, smoke…).
2. **VLM Router** — `/decompose` *(Claude, server)* — the VLM watches the clip and
   answers *"what is moving, how many things, and what type?"* It returns a **list** of
   motions (this is what makes multi-motion clips work). → travels as **`motions[]`
   (Contract A)**.
3. **SAM 2** *(ML, server)* — the VLM's bounding box is approximate, so SAM 2 turns it
   into a **precise per-object mask** (just the flag, not the sky behind it).
4. **SEA-RAFT + CoTracker3** *(ML, server)* — the real motion measurement. SEA-RAFT
   produces clean optical flow; CoTracker3 tracks points **across the whole clip**
   (long-range, occlusion-aware). This is what makes motion look real instead of a small
   jiggle.
5. **Distill** *(server)* — compresses the tracks into a swatch and **keeps both bulk
   motion (travel) and residual (texture)** — the fix for "motion only moves a tiny
   distance."
6. **Motion Swatch** *(data)* — the reusable, editable result. → travels as **`swatch`
   (Contract B)**.

## The hop down
The finished **swatch** drops from Row ① into Row ② — into the Application Engine.

## Row ② — APPLY → JUDGE → AUTO-TUNE (paint it on, evaluate, self-correct)

7. **Application Engine** *(browser, `animate.js`)* — applies the swatch. It routes by
   the **motion class** (not the layer name — this retires the current hardcoding) and
   uses a **mesh warp (MLS/ARAP)** so the deformation is large and smooth without tearing.
8. **Canvas / Preview** *(browser)* — renders the animation and captures **N frames** of
   the result.
9. **VLM Judge** — `/judge` *(Claude, server)* — looks at the rendered frames vs. the
   reference clip and scores it: *"does this look real? what's wrong?"* Returns issues +
   **parameter deltas**. → travels as **report (Contract C)**.
10. **Auto-tune loop** *(orange dashed)* — the deltas flow back into the Application
    Engine; it re-applies → re-renders → re-judges, **2–3 times until the score passes.**
    This is the self-correcting loop.

## Side branch — artwork labeling
Independently: **Upload Artwork → VLM Layer Auto-label** — the VLM names the objects in
the SVG (flag, smoke, river…), and those **named objects** feed the Application Engine so
it can **auto-match each swatch to the right layer.** Replaces hand-labeling.

---

## Who does what (color code in the diagram)

| Color | Role | Responsibility |
|---|---|---|
| **Rose** | VLM (Claude) — the *brain* | Decide what's moving, name objects, judge quality. **Never produces the motion itself.** |
| **Amber** | ML models — the *muscle* | Actually measure the motion: SAM 2, SEA-RAFT, CoTracker3. |
| **Cyan** | Browser / app | Upload, apply, render, preview, export. |
| **Violet** | Motion data | The swatch flowing between stages. |
| **Orange dashed** | Feedback loop | Carries the judge's deltas back for auto-tune. |

## The three contracts (why parallel work is possible)

Fixed hand-off formats between stages — freeze these first and the three workstreams
build independently.

- **Contract A — decomposition list** (VLM Router → everything downstream)
  ```json
  { "motions": [
    { "id":"m1","object":"flag","medium":"cloth","class":"cloth_wave",
      "bbox":[x,y,w,h],"direction":"horizontal","speed_hint":0.6,"confidence":0.9 }
  ]}
  ```
- **Contract B — motion swatch** (extraction → application)
  ```json
  { "id","name","class","params":{...8...},"tracks":[[...]],"trackGrid":32,
    "bulk":{ "dx":0,"dy":0 },"residualScale":1.0,"fps":30,"confidence":0.9 }
  ```
- **Contract C — judge report** (judge → auto-tune)
  ```json
  { "score":6,"looks_real":false,"issues":["flag barely moves"],
    "suggest":{ "intensity":"+40%","amplitude":"+0.15" } }
  ```

## Why this design matters for Round 2
- **Multi-motion works with no filename hints** — retires the `DEMO_PROFILES` hardcoding.
- **Real long-range motion** (CoTracker + keeping bulk) — fixes "moves only a small distance."
- **Class-keyed application** — retires the name-keyed behavior hardcoding.
- **VLM judge + auto-tune** — reframes "we hand-tuned good defaults" as "the system
  evaluates its own output and self-corrects."
