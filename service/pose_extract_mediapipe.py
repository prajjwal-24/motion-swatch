"""Real human-pose extraction with MediaPipe Pose (BlazePose, 33 landmarks) —
the model recommended for character motion. No hardcoded motion.

Maps MediaPipe's landmarks to the joint names the duck rig expects, normalized
to the person's bounding box (scale/position free), and writes a character
motion swatch JSON.

Usage: python pose_extract_mediapipe.py <video.mp4> <out.json>
"""
import sys, json, cv2, numpy as np
import mediapipe as mp

VID, OUT = sys.argv[1], sys.argv[2]

# MediaPipe Pose landmark indices → our rig joint names
MP = mp.solutions.pose
IDX = {
    "nose": 0,
    "l_sho": 11, "r_sho": 12,
    "l_elb": 13, "r_elb": 14,
    "l_wri": 15, "r_wri": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ank": 27, "r_ank": 28,
}
NAMES = list(IDX.keys())

cap = cv2.VideoCapture(VID)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = max(1, round(fps / 15))               # sample ~15 fps
raw = []
i = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    if i % step == 0: raw.append(fr)
    i += 1
cap.release()
print(f"sampled {len(raw)} frames @~15fps from {i} total", file=sys.stderr)

seq = []
with MP.Pose(static_image_mode=False, model_complexity=2,
             min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for fr in raw:
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            seq.append(None); continue
        lms = res.pose_landmarks.landmark
        pts = {n: (lms[i].x, lms[i].y, lms[i].visibility) for n, i in IDX.items()}
        # normalize to the person's bbox from the joints we use
        xs = [p[0] for p in pts.values()]; ys = [p[1] for p in pts.values()]
        x0, y0 = min(xs), min(ys)
        bw, bh = max(1e-3, max(xs) - x0), max(1e-3, max(ys) - y0)
        norm = [[round((pts[n][0] - x0) / bw, 4), round((pts[n][1] - y0) / bh, 4),
                 round(float(pts[n][2]), 3)] for n in NAMES]
        seq.append(norm)

good = sum(1 for s in seq if s)
print(f"MediaPipe detected pose in {good}/{len(seq)} frames", file=sys.stderr)
json.dump({"joints": NAMES, "fps": 15, "engine": "mediapipe_blazepose", "frames": seq}, open(OUT, "w"))
print("wrote", OUT, file=sys.stderr)
