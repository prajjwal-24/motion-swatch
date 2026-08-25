"""Step 9 done-when, minus the model:

  "the tune loop is bounded, the deltas are clamped, and the SERVER decides when to stop"

contracts_selftest.py already proves the policy functions in isolation. This proves the
WIRING in service/vlm_router.py around them — session bookkeeping, the iteration cap,
apply_deltas plumbing, and what actually lands in the request body — by driving the real
judge() with a scripted fake VLM in place of _client().

Why a fake and not the live model: the model's taste is not what's under test here. Its
job is to return a verdict; this file's job is to prove that no verdict, however wrong or
greedy, can push a param past its cap or the loop past its budget. That must hold when
the credentials are expired, when the model is having an off day, and offline in CI.

  routervenv/bin/python tests/step9-judge-loop.py

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
    """Returns the scripted verdicts in order, and records every request it was sent."""

    def __init__(self, *verdicts):
        self.script = list(verdicts)
        self.requests = []

    messages = property(lambda self: self)

    def create(self, **kw):
        self.requests.append(kw)
        return _Msg(self.script.pop(0) if self.script else
                    {"verdict": "tune", "score": 0.5, "deltas": {"amplitude": 0.05},
                     "critique": "ran out of script"})


def verdict(score, deltas=None, v="tune", **extra):
    d = {"verdict": v, "score": score, "critique": f"scripted {score}",
         "deltas": deltas if deltas is not None else {"amplitude": 0.1}}
    d.update(extra)
    return d


def frame(w=200, h=120, shift=0):
    img = np.full((h, w, 3), 240, np.uint8)
    cv2.rectangle(img, (20 + shift, 30), (60 + shift, 90), (40, 80, 200), -1)
    return base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


FRAMES = [frame(shift=i * 6) for i in range(4)]
PARAMS = {"frequency": 1.0, "amplitude": 0.3, "direction": 90.0, "turbulence": 0.1,
          "damping": 0.2, "phaseSpread": 0.6, "driftX": 0.0, "driftY": 0.0,
          "leafFall": True}


def body(session, **kw):
    b = {"session": session, "class": "cloth", "applicator": "wave",
         "label": "waving flag", "params": dict(PARAMS), "frames": FRAMES}
    b.update(kw)
    return b


def run(session, *verdicts, **kw):
    """Judge repeatedly on one session until the server refuses. Returns (results, fake)."""
    fake = FakeVLM(*verdicts)
    R._client = lambda: fake
    R.JUDGE_SESSIONS.pop(session, None)
    out = []
    for _ in range(len(verdicts) + 2):          # deliberately over-run the script
        res, _w = R.judge(body(session, **kw))
        out.append(res)
        if res.get("stopped") or res.get("error"):
            break
    return out, fake


def main():
    print("THE LOOP — bounded by the server, not by the caller:")

    # improving run that reaches the bar
    res, fake = run("s-good", verdict(0.30), verdict(0.55), verdict(0.82))
    check("an improving run keeps going and stops at the quality bar",
          len(fake.requests) == 3 and res[2]["continue"] is False
          and "bar" in res[2]["reason"], f"{len(fake.requests)} calls, {res[-1].get('reason')}")

    # the cap: three calls even while still improving, and the 4th never reaches the VLM
    res, fake = run("s-cap", verdict(0.10), verdict(0.30), verdict(0.50), verdict(0.70))
    check(f"the cap holds at {C.JUDGE_MAX_ITERS} while the score is still climbing",
          len(fake.requests) == C.JUDGE_MAX_ITERS, f"{len(fake.requests)} calls")
    check("...and the over-budget call is refused without spending a VLM request",
          res[-1].get("stopped") is True and "cap" in res[-1]["reason"], str(res[-1])[:120])
    check("...and the refusal still reports the best iteration so far",
          res[-1]["best"] == 2, str(res[-1].get("best")))

    # stalling
    res, fake = run("s-stall", verdict(0.40), verdict(0.41), verdict(0.90))
    check("a stalled score stops the loop before the cap",
          len(fake.requests) == 2, f"{len(fake.requests)} calls")
    check("...and the run keeps the better of the two, not the last",
          res[-1]["best"] == 1, str(res[-1]["best"]))

    # regression: the last state is the worse one, so 'best' must not be last
    res, fake = run("s-regress", verdict(0.60), verdict(0.20))
    check("a regression stops the loop and points back at the good iteration",
          len(fake.requests) == 2 and res[-1]["best"] == 0, str(res[-1]["best"]))

    # wrong class
    res, fake = run("s-wrong", verdict(0.15, {}, v="wrong_class"), verdict(0.9))
    check("a wrong_class verdict spends exactly one request and stops",
          len(fake.requests) == 1 and res[-1].get("stopped") is True, f"{len(fake.requests)} calls")
    check("...and says the applicator is the problem, not the dials",
          "applicator is wrong" in res[-1]["reason"], res[-1]["reason"])

    # sessions are independent budgets
    R.JUDGE_SESSIONS.clear()
    run("s-a", verdict(0.1), verdict(0.2), verdict(0.3))
    _, fake_b = run("s-b", verdict(0.9))
    check("one exhausted session does not spend another's budget",
          len(fake_b.requests) == 1 and len(R.JUDGE_SESSIONS["s-a"]) == 3)
    check("...and reset clears a history so a fresh run is possible",
          (R.JUDGE_SESSIONS.pop("s-a", None) is not None) and "s-a" not in R.JUDGE_SESSIONS)

    print("\nCLAMPING — end to end, through the real judge():")
    fake = FakeVLM(verdict(0.2, {"amplitude": 9.9, "frequency": -7.0,
                                 "direction": 400.0, "nonsense": 3.0}))
    R._client = lambda: fake
    R.JUDGE_SESSIONS.pop("s-clamp", None)
    res, warns = R.judge(body("s-clamp"))
    d = res["verdict"]["deltas"]
    check("a greedy delta is capped at the per-iteration limit",
          d["amplitude"] == C.PARAM_DELTA_MAX["amplitude"]
          and d["frequency"] == -C.PARAM_DELTA_MAX["frequency"], str(d))
    check("a hallucinated param never reaches the renderer", "nonsense" not in d)
    check("the caps are reported, not applied silently",
          any("cap" in w for w in warns), str(warns))
    check("next_params is computed by the server, not asked of the caller",
          res["next_params"]["amplitude"] == round(PARAMS["amplitude"]
                                                  + C.PARAM_DELTA_MAX["amplitude"], 4),
          str(res["next_params"]["amplitude"]))
    check("...and it clamps to the param's absolute range too",
          0 <= res["next_params"]["frequency"] <= C.PARAM_RANGES["frequency"][1],
          str(res["next_params"]["frequency"]))
    check("the cap is applied BEFORE the addition, so a wild delta still lands nearby",
          res["next_params"]["direction"] == 90.0 + C.PARAM_DELTA_MAX["direction"],
          str(res["next_params"]["direction"]))
    # and separately: a heading near due-east must wrap rather than pin at 360
    fake = FakeVLM(verdict(0.2, {"direction": 40.0}))
    R._client = lambda: fake
    R.JUDGE_SESSIONS.pop("s-wrap", None)
    wrapped, _ = R.judge(body("s-wrap", params=dict(PARAMS, direction=340.0)))
    check("a heading past 360 wraps in next_params instead of pinning at due east",
          wrapped["next_params"]["direction"] == 20.0,
          str(wrapped["next_params"]["direction"]))
    check("renderer flags that are not dials survive the round trip",
          res["next_params"]["leafFall"] is True)
    check("the caller's own params come back unchanged for comparison",
          res["params"]["amplitude"] == PARAMS["amplitude"])

    print("\nWHAT ACTUALLY GETS SENT:")
    fake = FakeVLM(verdict(0.5))
    R._client = lambda: fake
    R.JUDGE_SESSIONS.pop("s-req", None)
    R.judge(body("s-req"))
    req = fake.requests[0]
    content = req["messages"][0]["content"]
    imgs = [c for c in content if c["type"] == "image"]
    text = " ".join(c["text"] for c in content if c["type"] == "text")
    check("a forced tool call, so the verdict cannot come back as prose",
          req["tool_choice"] == {"type": "tool", "name": "report_verdict"})
    check("every frame is sent — a judge shown one still cannot see motion",
          len(imgs) == len(FRAMES), f"{len(imgs)} images")
    check("frames are labelled in order so 'across the frames' means something",
          "RENDER frame 1/4" in text and "RENDER frame 4/4" in text)
    check("the prompt states the class and the applicator being judged",
          "cloth" in text and "wave" in text)
    check("the prompt lists the current param values",
          '"amplitude": 0.3' in text)
    check("the prompt states each dial's cap, generated from contracts",
          f"+/-{C.PARAM_DELTA_MAX['frequency']}" in text)
    check("the prompt says deltas are additive, not replacements",
          "ADDITIVE" in text)

    # The intent and the element it was applied to are different claims. Sending the layer
    # name as the intent had the model grading a cloth swatch as a "title reveal" — a fair
    # verdict on the wrong question, and the kind of failure that reads as model weakness.
    R.JUDGE_SESSIONS.pop("s-intent", None)
    fake = FakeVLM(verdict(0.5))
    R._client = lambda: fake
    R.judge(body("s-intent", label="waving flag", element="title"))
    t2 = " ".join(c["text"] for c in fake.requests[0]["messages"][0]["content"]
                  if c["type"] == "text")
    check("the motion's name is given as the intent",
          "Intended motion: waving flag" in t2)
    check("...and the layer it was applied to is named separately, not as the intent",
          "element named 'title'" in t2)

    # a hand-authored preset has no extracted class; printing class '' invites a guess
    R.JUDGE_SESSIONS.pop("s-noclass", None)
    fake = FakeVLM(verdict(0.5))
    R._client = lambda: fake
    R.judge(body("s-noclass", **{"class": "", "applicator": "wave", "label": "Waterfall Flow"}))
    t3 = " ".join(c["text"] for c in fake.requests[0]["messages"][0]["content"]
                  if c["type"] == "text")
    check("a motion with no class says so rather than printing an empty one",
          "class ''" not in t3 and "no motion class was recorded" in t3)
    check("...and its name still reaches the judge as the intent",
          "Intended motion: Waterfall Flow" in t3)
    check("with no reference, the score is labelled as plausibility only",
          "class_plausibility" in str(R.judge(body("s-req2"))[0]["score_of"]))

    fake = FakeVLM(verdict(0.5))
    R._client = lambda: fake
    R.JUDGE_SESSIONS.pop("s-ref", None)
    res, _ = R.judge(body("s-ref", reference=FRAMES[:3]))
    content = fake.requests[0]["messages"][0]["content"]
    text = " ".join(c["text"] for c in content if c["type"] == "text")
    check("with a reference clip, the score is labelled as a match to it",
          res["score_of"] == "match_to_reference", res["score_of"])
    check("...and both groups are sent, source first, clearly separated",
          text.index("SOURCE frame 1/3") < text.index("RENDER frame 1/4")
          and len([c for c in content if c["type"] == "image"]) == 7)

    print("\nFRAMES IN, WITHOUT A VLM CALL:")
    fake = FakeVLM(verdict(0.5))
    R._client = lambda: fake
    for name, kw in [("no frames at all", {"frames": []}),
                     ("a single still", {"frames": FRAMES[:1]}),
                     ("frames that are not base64", {"frames": ["!!!!", "????"]})]:
        R.JUDGE_SESSIONS.pop("s-bad", None)
        res, _ = R.judge(body("s-bad", **kw))
        check(f"{name} is refused, and no request is spent on it",
              bool(res.get("error")) and not fake.requests, res.get("error", "accepted it"))

    over = [frame(shift=i) for i in range(R.JUDGE_MAX_FRAMES + 5)]
    picked, notes = R._decode_frames(over)
    check(f"more frames than the {R.JUDGE_MAX_FRAMES}-frame budget are sampled, not truncated",
          len(picked) == R.JUDGE_MAX_FRAMES and any("sampled" in n for n in notes), str(notes))
    wide = base64.b64encode(
        cv2.imencode(".png", np.full((400, R.MAX_W * 2, 3), 200, np.uint8))[1].tobytes()).decode()
    got, _ = R._decode_frames([wide, wide])
    check(f"an oversized frame is downscaled to {R.MAX_W}px before it costs tokens",
          all(m == "image/jpeg" for m, _ in got)
          and cv2.imdecode(np.frombuffer(base64.b64decode(got[0][1]), np.uint8),
                           cv2.IMREAD_COLOR).shape[1] == R.MAX_W)
    got, _ = R._decode_frames(["data:image/png;base64," + wide, wide])
    check("a browser data: URL is accepted as readily as bare base64", len(got) == 2)

    # regression: the API rejects a declared media type that the bytes contradict with a
    # 400, so the media type is sniffed from the magic number and the header is ignored.
    small_jpeg = frame(w=R.MAX_W // 2)                 # small enough to skip re-encoding
    got, _ = R._decode_frames([small_jpeg, small_jpeg])
    check("a bare base64 JPEG is labelled image/jpeg, not assumed to be a PNG",
          [m for m, _ in got] == ["image/jpeg"] * 2, str([m for m, _ in got]))
    small_png = base64.b64encode(
        cv2.imencode(".png", np.full((80, 80, 3), 210, np.uint8))[1].tobytes()).decode()
    got, _ = R._decode_frames([small_png])
    check("...and a real PNG is still labelled image/png", got[0][0] == "image/png", got[0][0])
    got, _ = R._decode_frames(["data:image/png;base64," + small_jpeg])
    check("a data: URL that lies about its type is corrected, not forwarded to a 400",
          got[0][0] == "image/jpeg", got[0][0])
    check("a media type the API cannot read is never declared",
          R._sniff_media(b"II*\x00nonsense") is None)
    got, notes = R._decode_frames([base64.b64encode(b"II*\x00not an image").decode()])
    check("...and such a frame is dropped with a note rather than sent",
          not got and any("could not be read" in n for n in notes), str(notes))
    mixed, notes = R._decode_frames([FRAMES[0], "!!!not base64!!!", FRAMES[1]])
    check("one unreadable frame does not throw away the readable ones",
          len(mixed) == 2 and any("not valid base64" in n for n in notes), str(notes))

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
