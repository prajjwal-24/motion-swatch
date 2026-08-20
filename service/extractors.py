"""Pluggable motion-extractor registry (Step 4 of docs/BUILD_PLAN.md).

One dispatch table so each motion class can route to the best available model — and
adding a model = adding one Engine, without touching the server or the swatch schema.

Three capability slots (an Engine provides exactly one):
  FLOW        frames[T,H,W,3] in [0,1]  -> ndarray[T-1, 2, H, W]  px/frame   (RAFT-style dense flow)
  TRAJECTORY  frames[T,H,W,3] in [0,1]  -> list[144] of [x,y], ONE point per input frame,
              gy-major/gx-inner (matches grid_trajectories); coords are ÷W,H so they MAY
              fall outside 0..1 when a tracked point leaves the frame.
  PREPROC     (frames, fps)             -> frames  same shape/[0,1]          (e.g. motion magnify)

RULES that keep the server backward-compatible:
  * probe() is DEPENDENCY-FREE (importlib.find_spec + os.path only) — no heavy imports,
    so a missing/broken backend can never break server import or startup.
  * heavy imports + model loads happen ONLY inside factory(), lazily, cached on first use.
  * resolve(name, kind) falls back to the kind's `default` engine when a requested engine is
    unknown/unavailable. FLOW and PREPROC have defaults (raft_small, evm); TRAJECTORY has NO
    default, so resolve returns (None, reason) — the caller (server /analyze) must handle None
    (it does: it falls back to the RAFT-integrated grid).
"""
import os
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

FLOW, TRAJECTORY, PREPROC = "flow", "trajectory", "preproc"
GRID = 12


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class Engine:
    name: str
    kind: str                                  # FLOW | TRAJECTORY | PREPROC
    describe: str
    probe: Callable[[], tuple]                 # () -> (available: bool, reason: str)  — cheap
    factory: Callable[[], Any]                 # () -> impl callable                   — lazy, heavy
    default: bool = False
    _impl: Any = field(default=None, repr=False)

    def load(self):
        if self._impl is None:
            self._impl = self.factory()
        return self._impl


REGISTRY = {}


def register(e: Engine):
    REGISTRY[e.name] = e


def available_engines() -> list:
    """Cheap capability report (calls probe() only) — powers GET /engines."""
    out = []
    for e in REGISTRY.values():
        try:
            ok, why = e.probe()
        except Exception as ex:               # a broken probe must never crash the report
            ok, why = False, f"probe error: {ex}"
        out.append({"name": e.name, "kind": e.kind, "available": bool(ok),
                    "reason": why, "default": e.default, "describe": e.describe})
    return out


def resolve(name: Optional[str], kind: str):
    """(engine, fallback_reason). Unknown/unavailable -> the kind's default engine."""
    default = next((e for e in REGISTRY.values() if e.kind == kind and e.default), None)
    if not name:
        return default, None
    e = REGISTRY.get(name)
    if e is None or e.kind != kind:
        return default, f"{name!r} is not a registered {kind} engine"
    try:
        ok, why = e.probe()
    except Exception as ex:
        ok, why = False, str(ex)
    return (e, None) if ok else (default, f"{name} unavailable ({why})")


# ── FLOW: torchvision RAFT (core, cached, no downloads) ─────────────────────
def _torchvision_flow(which):
    def build():
        import numpy as np, torch
        from torchvision.models.optical_flow import (
            raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model = (raft_large(weights=Raft_Large_Weights.DEFAULT) if which == "raft_large"
                 else raft_small(weights=Raft_Small_Weights.DEFAULT)).eval().to(dev)

        def flow_fn(frames):
            t = torch.from_numpy(frames).permute(0, 3, 1, 2) * 2 - 1   # [-1,1] (torchvision)
            flows = []
            with torch.no_grad():
                for a, b in zip(t[:-1], t[1:]):
                    fl = model(a.unsqueeze(0).to(dev), b.unsqueeze(0).to(dev))[-1]
                    flows.append(fl[0].cpu().numpy())
            return np.stack(flows)
        return flow_fn
    return build


register(Engine("raft_small", FLOW, "torchvision RAFT-small (default, 12/12 on benchmark)",
                probe=lambda: (True, "torchvision cached") if _has("torchvision") else (False, "torchvision missing"),
                factory=_torchvision_flow("raft_small"), default=True))
register(Engine("raft_large", FLOW, "torchvision RAFT-large (sharper; scored 11/12 — not better end-to-end)",
                probe=lambda: (True, "torchvision cached") if _has("torchvision") else (False, "torchvision missing"),
                factory=_torchvision_flow("raft_large")))


# ── FLOW: SEA-RAFT (gated — vendored clone + weights) ───────────────────────
SEARAFT_REPO = os.environ.get("SEARAFT_REPO", os.path.join(os.path.dirname(__file__), "SEA-RAFT"))


def _searaft_probe():
    if not os.path.isdir(os.path.join(SEARAFT_REPO, "core")):
        return (False, "clone princeton-vl/SEA-RAFT into service/SEA-RAFT (or set SEARAFT_REPO)")
    for dep in ("einops", "scipy"):
        if not _has(dep):
            return (False, f"pip install {dep}")
    if not (_has("safetensors") or _has("huggingface_hub")):
        return (False, "pip install huggingface_hub safetensors (or provide a local .pth)")
    return (True, "vendored")


def _searaft_build():
    import sys, json, argparse, numpy as np, torch
    sys.path.append(os.path.join(SEARAFT_REPO, "core"))
    from raft import RAFT                                    # noqa: E402
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    # config MUST match the checkpoint architecture — README pairs spring-M.json with the
    # Tartan-C-T-TSKH-spring540x960-M weights (a mismatch silently loads wrong weights/errors).
    cfg_path = os.environ.get("SEARAFT_CFG", os.path.join(SEARAFT_REPO, "config/eval/spring-M.json"))
    args = argparse.Namespace(**json.load(open(cfg_path)))
    args.scale = 0                                           # disable test-time rescale (generic drop-in)
    hf = os.environ.get("SEARAFT_HF", "MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    model = RAFT.from_pretrained(hf, args=args).to(dev).eval()

    def flow_fn(frames):
        t = torch.from_numpy(frames).permute(0, 3, 1, 2) * 255.0   # SEA-RAFT wants 0-255
        flows = []
        with torch.no_grad():
            for a, b in zip(t[:-1], t[1:]):
                out = model(a.unsqueeze(0).to(dev), b.unsqueeze(0).to(dev),
                            iters=args.iters, test_mode=True)
                flows.append(out["flow"][-1][0].cpu().numpy())
        return np.stack(flows)
    return flow_fn


register(Engine("searaft", FLOW, "SEA-RAFT (sharper flow at motion boundaries; heavier)",
                probe=_searaft_probe, factory=_searaft_build))


# ── TRAJECTORY: CoTracker3 (gated on cached weights; verified on this machine) ──
def _cotracker_probe():
    ckpt = os.path.expanduser("~/.cache/torch/hub/checkpoints/scaled_offline.pth")
    hub = os.path.expanduser("~/.cache/torch/hub/facebookresearch_co-tracker_main")
    if os.path.exists(ckpt) and os.path.isdir(hub):
        return (True, "cotracker3 weights + hub repo cached")
    return (False, "cache miss — first run downloads ~97MB via torch.hub (needs network)")


def _cotracker_build():
    import numpy as np, torch
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev).eval()

    def traj_fn(frames):
        T, H, W, _ = frames.shape
        vid = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].to(dev) * 255.0  # [1,T,3,H,W] 0-255
        q = [[0.0, (gx + 0.5) / GRID * W, (gy + 0.5) / GRID * H]                  # (t,x,y), gy-major
             for gy in range(GRID) for gx in range(GRID)]
        queries = torch.tensor(q, dtype=torch.float32, device=dev)[None]          # [1,144,3]
        with torch.no_grad():
            tracks, _vis = model(vid, queries=queries)                            # tracks [1,T,144,2] px
        tr = tracks[0].cpu().numpy()
        return [[[round(float(tr[t, n, 0] / W), 4), round(float(tr[t, n, 1] / H), 4)]
                 for t in range(tr.shape[0])] for n in range(GRID * GRID)]
    return traj_fn


register(Engine("cotracker3", TRAJECTORY, "CoTracker3 — long-range point tracks (survives occlusion/large motion)",
                probe=_cotracker_probe, factory=_cotracker_build, default=False))


# ── PREPROC: Eulerian video magnification (pure numpy+cv2, no downloads) ─────
def _evm_build():
    import numpy as np, cv2

    def pre_fn(frames, fps, alpha=None, low=None, high=None):
        # Exceptions propagate to the caller (server /analyze), which surfaces a note
        # and uses raw frames — so a failed magnification is never a silent no-op.
        a = float(os.environ.get("MS_EVM_ALPHA", "12")) if alpha is None else alpha
        lo = float(os.environ.get("MS_EVM_LOW", "0.4")) if low is None else low
        hi = float(os.environ.get("MS_EVM_HIGH", "3.0")) if high is None else high
        T, H, W, _ = frames.shape
        if T < 6:
            return frames                                                          # too short to band-pass
        small = np.stack([cv2.pyrDown(cv2.pyrDown(f)) for f in frames])            # temporal @ 1/4 res
        fft = np.fft.rfft(small, axis=0)
        freqs = np.fft.rfftfreq(T, d=1.0 / max(1e-3, fps))
        fft[~((freqs >= lo) & (freqs <= hi))] = 0
        band = np.fft.irfft(fft, n=T, axis=0) * a                                  # amplified residual
        out = np.empty_like(frames)
        for i in range(T):
            up = band[i]
            for _ in range(2):
                up = cv2.pyrUp(up)
            up = cv2.resize(up, (W, H))
            out[i] = np.clip(frames[i] + up, 0, 1)
        return out.astype(np.float32)
    return pre_fn


register(Engine("evm", PREPROC, "Eulerian magnification — amplify tiny/subtle motion (rain, breathing, slow water)",
                probe=lambda: (True, "numpy+cv2") if _has("cv2") else (False, "opencv missing"),
                factory=_evm_build, default=True))
