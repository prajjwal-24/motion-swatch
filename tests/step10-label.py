"""Step 10 done-when, minus the model:

  "named swatches auto-applied to the right OBJECTS — no filename hints"

Being applied to the right object rests entirely on the router knowing which crop it
showed the model, and on refusing to act on anything the model said about a layer it was
never shown. contracts_selftest.py proves normalize_layer_labels in isolation; this proves
the WIRING in service/vlm_router.py around it — layer validation, the crop pipeline, the
LABEL_MAX_LAYERS cap, and exactly which layer ids the answer is normalized against — by
driving the real label_layers() with a scripted fake VLM in place of _client().

Why a fake and not the live model: what the model calls a flag is not under test. What is
under test is that a model answering wrongly, greedily, or about layers that were never
sent CANNOT get a swatch applied to the wrong object. That has to hold with expired
credentials, on a bad day for the model, and offline.

  routervenv/bin/python tests/step10-label.py

(routervenv, because vlm_router imports cv2 and anthropic. No credentials needed — the
 VLM is never called.)
"""
import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "service"))

import cv2                                     # noqa: E402
import numpy as np                             # noqa: E402
import contracts as C                          # noqa: E402
import vlm_router as R                         # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)


# ── a scripted stand-in for Claude ───────────────────────────────────────────
class _Block:
    type = "tool_use"

    def __init__(self, payload):
        self.input = payload


class _Msg:
    def __init__(self, payload):
        self.content = [_Block(payload)]


class FakeVLM:
    """Answers with the scripted labelling, and records the request it was sent."""

    def __init__(self, payload=None, blocks=None):
        self.payload = payload
        self.blocks = blocks
        self.requests = []

    messages = property(lambda self: self)

    def create(self, **kw):
        self.requests.append(kw)
        if self.blocks is not None:
            m = _Msg({})
            m.content = self.blocks
            return m
        return _Msg(self.payload if self.payload is not None else {"labels": []})


# ── an artwork with three findable objects ───────────────────────────────────
# Drawn rather than loaded so the test needs no assets: a flag block on the left, a
# scatter of birds top-right, a ground band along the bottom. Their boxes are the
# normalized bboxes a browser's collectLayers() would compute.
W, H = 480, 320
ART = np.full((H, W, 3), 235, np.uint8)
cv2.rectangle(ART, (40, 60), (150, 170), (60, 70, 200), -1)          # flag
for i in range(6):
    cv2.circle(ART, (300 + i * 22, 50 + (i % 3) * 14), 4, (30, 30, 30), -1)   # birds
cv2.rectangle(ART, (0, 270), (W, H), (90, 140, 90), -1)              # ground

BOX = {"flag": [40 / W, 60 / H, 110 / W, 110 / H],
       "birds": [295 / W, 40 / H, 130 / W, 50 / H],
       "ground": [0.0, 270 / H, 1.0, 50 / H]}


def art_b64(img=None):
    return base64.b64encode(cv2.imencode(".jpg", ART if img is None else img)[1].tobytes()).decode()


LAYERS = [{"id": "L1", "name": "Layer 3", "bbox": BOX["flag"]},
          {"id": "L2", "name": "path2847", "bbox": BOX["birds"]},
          {"id": "L3", "name": "Flag_x20_bottom", "bbox": BOX["ground"]}]

GOOD = {"labels": [
    {"id": "L1", "label": "a flag on a pole", "motion_class": "cloth", "confidence": 0.9,
     "notes": "rectangular cloth"},
    {"id": "L2", "label": "birds", "motion_class": "flock", "confidence": 0.72},
    {"id": "L3", "label": "ground", "motion_class": "", "confidence": 0.95},
]}


def run(payload=None, layers=None, image=None, blocks=None, **extra):
    """One label_layers() pass against a fake VLM. Returns (contract, warnings, fake)."""
    fake = FakeVLM(payload, blocks)
    R._client = lambda: fake
    body = {"image": art_b64() if image is None else image,
            "layers": LAYERS if layers is None else layers}
    body.update(extra)
    contract, warnings = R.label_layers(body)
    return contract, warnings, fake


def main():
    print("THE HAPPY PATH — one crop per layer, and the answer is a contract:")
    d, w, fake = run(GOOD)
    by = {l["id"]: l for l in d["labels"]}
    check("a labelling comes back as Contract D",
          d["kind"] == "layer_labels" and d["version"] == C.SCHEMA_VERSION)
    check("the whole illustration is sent ONCE, plus one crop per layer",
          sum(1 for c in fake.requests[0]["messages"][0]["content"] if c["type"] == "image") == 4,
          str([c["type"] for c in fake.requests[0]["messages"][0]["content"]]))
    check("...and each crop is announced by its layer id, so it cannot be misattributed",
          all(f"Layer {i}:" in [c.get("text") for c in fake.requests[0]["messages"][0]["content"]]
              for i in ("L1", "L2", "L3")))
    check("the tool call is FORCED, so there is no prose to parse",
          fake.requests[0]["tool_choice"] == {"type": "tool", "name": "report_layers"})
    check("the crops are JPEG, whatever the artwork arrived as",
          all(c["source"]["media_type"] == "image/jpeg"
              for c in fake.requests[0]["messages"][0]["content"][4:] if c["type"] == "image"))
    check("all three layers come back labelled", set(by) == {"L1", "L2", "L3"}, str(list(by)))
    check("cloth is mesh, flock is rigid, both derived not asked",
          by["L1"]["deforms"] == "mesh" and by["L2"]["deforms"] == "rigid")
    check("a layer the model says does not move keeps an empty class",
          by["L3"]["motion_class"] == "" and by["L3"]["applicator"] == "")
    check("the artwork's real pixel size travels with the labels",
          d["art"]["width"] == W and d["art"]["height"] == H, str(d["art"]))
    check("...and so does how many layers were actually sent",
          d["art"]["layers_sent"] == 3, str(d["art"]))
    check("a clean pass warns about nothing", w == [], str(w))

    print("\nTHE PROMPT tells the model to distrust the very thing we used to trust:")
    prompt = next(c["text"] for c in fake.requests[0]["messages"][0]["content"]
                  if c["type"] == "text")
    check("the file's layer name is given as a HINT ONLY", "HINT ONLY" in prompt)
    check("...and the model is told to trust the picture when they disagree",
          "trust the picture" in prompt)
    check("...and that most layers should NOT move", "Most layers should NOT move" in prompt)
    check("the taxonomy comes from MOTION_CLASSES, not from a second hand-written list",
          all(c in prompt for c in C.MOTION_CLASSES), str(list(C.MOTION_CLASSES)))
    check("the layer names are quoted verbatim, escapes and all",
          "'Flag_x20_bottom'" in prompt or '"Flag_x20_bottom"' in prompt)
    check("the class enum the tool offers is exactly MOTION_CLASSES plus ''",
          set(R.LABEL_TOOL["input_schema"]["properties"]["labels"]["items"]
              ["properties"]["motion_class"]["enum"]) == set(C.MOTION_CLASSES) | {""})

    print("\nA MODEL TALKING ABOUT LAYERS IT WAS NEVER SHOWN cannot move anything:")
    d, w, _ = run({"labels": [{"id": "L1", "label": "flag", "motion_class": "cloth",
                               "confidence": 0.9},
                              {"id": "L7", "label": "a dragon", "motion_class": "flock",
                               "confidence": 0.99}]})
    check("a label for an id that was never sent is dropped",
          [l["id"] for l in d["labels"]] == ["L1"] and any("unknown layer" in x for x in w),
          str(w))
    check("...and the two layers it stayed silent about are reported unlabelled",
          any("came back unlabelled" in x for x in w), str(w))

    # A layer the model never SAW is worth exactly as much as a layer that does not exist,
    # whichever way it dropped out: no usable box, or a box too thin to crop.
    for why, box, note in [
        ("no bbox at all", None, "no usable bbox"),
        ("a zero-area bbox", [0.5, 0.5, 0.0, 0.0], "no usable bbox"),
        ("a NaN bbox", [float("nan"), 0.1, 0.2, 0.2], "no usable bbox"),
        ("a sub-pixel bbox", [0.5, 0.5, 0.001, 0.001], "too small to crop"),
    ]:
        tiny = LAYERS[:2] + [{"id": "L4", "name": "hairline", "bbox": box}]
        d, w, fake = run({"labels": [{"id": "L4", "label": "a flag", "motion_class": "cloth",
                                      "confidence": 0.95}]}, layers=tiny)
        check(f"a layer with {why} is not shown to the model, and the skip is named",
              any(note in x for x in w), str(w))
        check(f"...and a confident label for that unseen layer is refused ({why})",
              d["labels"] == [] and any("unknown layer 'L4'" in x for x in w), str(w))
        check(f"...and its two visible siblings are still sent ({why})",
              len(fake.requests) == 1
              and sum(1 for c in fake.requests[0]["messages"][0]["content"]
                      if c["type"] == "image") == 3)

    print("\n_layer_bbox — a layer box is never WIDENED to cover something else:")
    check("a zero-area box is refused, where a seed_bbox would have grown to fill the frame",
          R._layer_bbox([0.5, 0.5, 0, 0]) is None
          and C._clamp_bbox([0.5, 0.5, 0, 0]) == [0.5, 0.5, 0.5, 0.5])
    check("a box overflowing the right edge is shrunk to fit, not moved",
          R._layer_bbox([0.8, 0.1, 0.5, 0.2]) == [0.8, 0.1, 0.2, 0.2])
    check("a NaN box is refused rather than becoming the whole artwork",
          R._layer_bbox([float("nan"), 0, 0.5, 0.5]) is None
          and C._clamp_bbox([float("nan"), 0, 0.5, 0.5]) == [0.0, 0.0, 1.0, 1.0])
    for bad in (None, [], [0.2, 0.2], "0,0,1,1", [0, 0, float("inf"), 0.5],
                [1.0, 0.1, 0.2, 0.2], [0.1, 0.1, -0.3, 0.2]):
        check(f"{bad!r} is not a layer box", R._layer_bbox(bad) is None)
    check("a legitimate full-artwork box is still accepted (it is a real box)",
          R._layer_bbox([0, 0, 1, 1]) == [0.0, 0.0, 1.0, 1.0])

    print("\nTHE CAP is reported, never silent:")
    many = [{"id": f"M{i}", "name": f"o{i}",
             "bbox": [0.02 + 0.03 * i, 0.1, 0.02, 0.2]} for i in range(R.LABEL_MAX_LAYERS + 4)]
    d, w, fake = run({"labels": []}, layers=many)
    sent = sum(1 for c in fake.requests[0]["messages"][0]["content"] if c["type"] == "image") - 1
    check(f"only the first {R.LABEL_MAX_LAYERS} layers reach the model",
          sent == R.LABEL_MAX_LAYERS, f"{sent} crops")
    check("...and the ones that did not are named as unlabelled, with the reason",
          any(f"first {R.LABEL_MAX_LAYERS}" in x for x in w)
          and any("came back unlabelled" in x for x in w), str(w))
    check("the cap the browser uses is the number the router publishes",
          R.LABEL_MAX_LAYERS == 12)

    print("\nBAD INPUT — label_layers never raises, it degrades and explains:")
    d, w, fake = run({"labels": []}, layers=[])
    check("no layers: an empty contract, one clear reason, and no VLM call",
          d["labels"] == [] and any("no labellable layers" in x for x in w)
          and not fake.requests, str(w))
    d, w, fake = run({"labels": []}, layers="all of them")
    check("layers that are not a list is the same as none, not a crash",
          d["labels"] == [] and not fake.requests)
    d, w, fake = run({"labels": []}, layers=[{"name": "no id here"}, LAYERS[0]])
    check("a layer with no id is skipped and named; its siblings still go",
          any("has no id" in x for x in w) and len(fake.requests) == 1, str(w))
    d, w, fake = run({"labels": []}, image="")
    check("no artwork image: an empty contract and no VLM call",
          d["labels"] == [] and any("could not be read" in x for x in w)
          and not fake.requests, str(w))
    d, w, fake = run({"labels": []}, image=base64.b64encode(b"II*\x00nope").decode())
    check("an unreadable artwork image is refused before a request is spent",
          d["labels"] == [] and not fake.requests, str(w))
    d, w, fake = run(blocks=[])
    check("a model that returns no tool_use gets an empty contract, not a guess",
          d["labels"] == [] and any("did not return structured" in x for x in w), str(w))
    check("...and the artwork dimensions survive even that",
          d["art"].get("width") == W, str(d["art"]))
    d, w, _ = run({"labels": [{"id": "L1", "motion_class": "cloth", "confidence": 0.9}]})
    check("a label with no text still yields a usable entry (the class is what matters)",
          d["labels"][0]["label"] == "" and d["labels"][0]["motion_class"] == "cloth")

    print("\n_crop — the geometry that decides WHICH object the model is asked about:")
    for name, box in BOX.items():
        crop = R._crop(ART, box)
        check(f"the {name} bbox crops to something the model can see", crop is not None)
    # WHICH object a crop landed on, measured rather than assumed: the flag is the only
    # red object and the ground the only green one, so a crop that landed on a neighbour
    # would not lead on its own colour. Self-calibrating — no hand-picked tolerance.
    def cast(box):
        c = cv2.imdecode(np.frombuffer(base64.b64decode(R._crop(ART, box)), np.uint8),
                         cv2.IMREAD_COLOR).astype(float)
        return {"red": c[:, :, 2].mean() - c[:, :, 0].mean(),
                "green": c[:, :, 1].mean() - c[:, :, 2].mean()}
    cast_of = {k: cast(b) for k, b in BOX.items()}
    check("...and the flag crop really landed on the flag: it is the reddest of the three",
          cast_of["flag"]["red"] == max(c["red"] for c in cast_of.values())
          and cast_of["flag"]["red"] > 20,
          " ".join(f"{k} r{c['red']:+.0f}" for k, c in cast_of.items()))
    check("...and the ground crop landed on the ground: it is the greenest",
          cast_of["ground"]["green"] == max(c["green"] for c in cast_of.values())
          and cast_of["ground"]["green"] > 20,
          " ".join(f"{k} g{c['green']:+.0f}" for k, c in cast_of.items()))
    check("a crop is downscaled to LABEL_CROP_W at most (cost is bounded per layer)",
          cv2.imdecode(np.frombuffer(base64.b64decode(R._crop(ART, [0, 0, 1, 1])), np.uint8),
                       cv2.IMREAD_COLOR).shape[1] == R.LABEL_CROP_W)
    check("a zero-size bbox yields no crop rather than a 0x0 encode error",
          R._crop(ART, [0.5, 0.5, 0.0, 0.0]) is None)
    check("a sub-pixel bbox yields no crop", R._crop(ART, [0.5, 0.5, 0.001, 0.001]) is None)
    check("a bbox running off the frame is clamped, and still crops",
          R._crop(ART, [0.9, 0.9, 0.5, 0.5]) is not None)
    small = np.full((6, 6, 3), 200, np.uint8)
    check("a 6px artwork still crops (no assumption about a minimum size)",
          R._crop(small, [0, 0, 1, 1]) is not None)
    check("...but a 2px one does not pretend to", R._crop(np.full((2, 2, 3), 9, np.uint8),
                                                          [0, 0, 1, 1]) is None)

    print("\nTHE ROUTE is wired and advertised:")
    ROUTER_SRC = open(os.path.join(HERE, "..", "service", "vlm_router.py")).read()
    check("/label is in the POST allowlist, next to /decompose and /judge",
          'self.path not in ("/decompose", "/label", "/judge", "/judge/reset")' in ROUTER_SRC)
    check("...and a POST to anything else names the four routes it does serve",
          'POST /decompose | /label | /judge | /judge/reset' in ROUTER_SRC)
    check("/health advertises /label, so the browser can find out it exists",
          '"/label"' in ROUTER_SRC.split("def do_GET")[-1] if "def do_GET" in ROUTER_SRC
          else False)
    check("...and publishes the cap and the deforms vocabulary the browser mirrors",
          "LABEL_MAX_LAYERS" in ROUTER_SRC.split("def do_GET")[-1]
          and "mesh_classes" in ROUTER_SRC)

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
