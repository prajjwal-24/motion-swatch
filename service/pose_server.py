"""Tiny pose-extraction HTTP endpoint (stdlib only, no framework).

POST /extract  with the raw video bytes in the request body
  -> runs MediaPipe Pose (BlazePose) and returns a character motion swatch JSON
     { joints:[...], fps:15, engine:"mediapipe_blazepose", frames:[[ [x,y,c]... ] ...] }

Run with the MediaPipe venv (Python <=3.12):
  /tmp/ms-test/mpvenv/bin/python service/pose_server.py    # serves on :8770
"""
import json, tempfile, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import mediapipe as mp

PORT = int(os.environ.get("POSE_PORT", "8770"))
MAX_FRAMES = 160          # cap so an uploaded clip stays snappy
MP = mp.solutions.pose
IDX = {"nose":0,"l_sho":11,"r_sho":12,"l_elb":13,"r_elb":14,"l_wri":15,"r_wri":16,
       "l_hip":23,"r_hip":24,"l_knee":25,"r_knee":26,"l_ank":27,"r_ank":28}
NAMES = list(IDX.keys())

def extract(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / 15))
    raw, i = [], 0
    while len(raw) < MAX_FRAMES:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0: raw.append(fr)
        i += 1
    cap.release()
    seq = []
    with MP.Pose(model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5) as pose:
        for fr in raw:
            res = pose.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                seq.append(None); continue
            lm = res.pose_landmarks.landmark
            pts = {n:(lm[i].x, lm[i].y, lm[i].visibility) for n,i in IDX.items()}
            xs=[p[0] for p in pts.values()]; ys=[p[1] for p in pts.values()]
            x0,y0=min(xs),min(ys); bw=max(1e-3,max(xs)-x0); bh=max(1e-3,max(ys)-y0)
            seq.append([[round((pts[n][0]-x0)/bw,4), round((pts[n][1]-y0)/bh,4),
                         round(float(pts[n][2]),3)] for n in NAMES])
    good = [s for s in seq if s]
    return {"joints":NAMES,"fps":15,"engine":"mediapipe_blazepose",
            "detected":len(good),"total":len(seq),"frames":seq}

class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def do_GET(self):
        self.send_response(200); self._cors()
        self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(b'{"ok":true,"engine":"mediapipe_blazepose"}')
    def do_POST(self):
        if self.path != "/extract":
            self.send_response(404); self._cors(); self.end_headers(); return
        n = int(self.headers.get("Content-Length","0"))
        data = self.rfile.read(n)
        tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); tf.write(data); tf.close()
        try:
            print(f"[pose] extracting from {len(data)//1024} KB clip…", file=sys.stderr)
            out = extract(tf.name)
            print(f"[pose] detected {out['detected']}/{out['total']} frames", file=sys.stderr)
            body = json.dumps(out).encode()
            self.send_response(200); self._cors()
            self.send_header("Content-Type","application/json"); self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(500); self._cors(); self.end_headers()
            self.wfile.write(json.dumps({"error":str(e)}).encode())
        finally:
            os.unlink(tf.name)
    def log_message(self, *a): pass

if __name__ == "__main__":
    print(f"pose_server on http://127.0.0.1:{PORT}  (POST /extract with video bytes)", file=sys.stderr)
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
