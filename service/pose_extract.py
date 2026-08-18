"""Real human-pose extraction from a walking video — no hardcoding.
torchvision Keypoint R-CNN (17 COCO keypoints) on MPS. Outputs a per-frame
skeleton sequence (normalized) that becomes a 'character motion swatch'.
"""
import sys, json, cv2, numpy as np, torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

VID = sys.argv[1]; OUT = sys.argv[2]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
w = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
model = keypointrcnn_resnet50_fpn(weights=w).eval().to(dev)
prep = w.transforms()

COCO = ["nose","l_eye","r_eye","l_ear","r_ear","l_sho","r_sho","l_elb","r_elb",
        "l_wri","r_wri","l_hip","r_hip","l_knee","r_knee","l_ank","r_ank"]

cap = cv2.VideoCapture(VID)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frames = []
raw = []
i = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    # sample ~15 fps to keep it quick
    if i % max(1, round(fps/15)) == 0:
        raw.append(fr)
    i += 1
cap.release()
print(f"sampled {len(raw)} frames @~15fps from {i} total", file=sys.stderr)

seq = []
with torch.no_grad():
    for fr in raw:
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        t = prep(torch.from_numpy(rgb).permute(2,0,1)).to(dev)
        out = model([t])[0]
        if len(out["scores"]) == 0 or float(out["scores"][0]) < 0.9:
            seq.append(None); continue
        kp = out["keypoints"][0].cpu().numpy()  # (17,3) x,y,conf
        H, W = fr.shape[:2]
        # normalize to the person's bbox so the swatch is scale/position free
        box = out["boxes"][0].cpu().numpy()
        bw, bh = max(1, box[2]-box[0]), max(1, box[3]-box[1])
        norm = [[float((x-box[0])/bw), float((y-box[1])/bh), float(c)] for x,y,c in kp]
        seq.append(norm)

good = [s for s in seq if s]
print(f"detected pose in {len(good)}/{len(seq)} frames", file=sys.stderr)
json.dump({"joints": COCO, "fps": 15, "frames": seq}, open(OUT,"w"))
print("wrote", OUT, file=sys.stderr)
