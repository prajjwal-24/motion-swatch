#!/bin/bash
# Start every MotionLife service + the web app, in the background, with logs.
#   sh start-all.sh          # start everything
#   sh start-all.sh stop     # stop everything
#   sh start-all.sh status   # what's up / down
#
# Credentials: put your key in .env first (see setkey.sh / .env.example).
# Logs land in ./logs/*.log  — tail them with:  tail -f logs/*.log
cd "$(dirname "$0")"
mkdir -p logs

PORTS="8000 8765 8770 8771 8772"

status() {
  for p in $PORTS; do
    r=$(curl -s -m 2 "http://127.0.0.1:$p/health" 2>/dev/null || curl -s -m 2 "http://127.0.0.1:$p/" 2>/dev/null)
    if [ -n "$r" ]; then printf "  :%-5s UP\n" "$p"; else printf "  :%-5s DOWN\n" "$p"; fi
  done
}

stop() {
  echo "Stopping…"
  pkill -f "server:app" 2>/dev/null
  pkill -f "pose_server.py" 2>/dev/null
  pkill -f "vlm_router.py" 2>/dev/null
  pkill -f "preprocess_server.py" 2>/dev/null
  pkill -f "http.server 8000" 2>/dev/null
  sleep 2; echo "Stopped."; status; exit 0
}

case "$1" in
  stop) stop ;;
  status) echo "MotionLife services:"; status; exit 0 ;;
esac

[ -f .env ] || echo "⚠️  no .env — the VLM router (:8771) will 500. Run: sh setkey.sh"

echo "Starting MotionLife…"

# 1. web app (static)
pgrep -f "http.server 8000" >/dev/null || \
  (python3 -m http.server 8000 >logs/web.log 2>&1 &) && echo "  :8000 web app"

# 2. RAFT flow + engine registry (torch, py3.13)
pgrep -f "server:app" >/dev/null || \
  (service/venv/bin/uvicorn --app-dir service server:app --host 127.0.0.1 --port 8765 \
     >logs/raft.log 2>&1 &) && echo "  :8765 RAFT + registry"

# 3. VLM router (needs .env key)
pgrep -f "vlm_router.py" >/dev/null || \
  (routervenv/bin/python service/vlm_router.py >logs/router.log 2>&1 &) && echo "  :8771 VLM router"

# 4. Preprocess (mask + camera) — reuses routervenv
pgrep -f "preprocess_server.py" >/dev/null || \
  (routervenv/bin/python service/preprocess_server.py >logs/prep.log 2>&1 &) && echo "  :8772 preprocess"

# 5. MediaPipe pose/hands/face — needs its own py<=3.12 venv (./mpvenv).
#    run-pose.sh creates it on first run (a few minutes), then it's instant.
if pgrep -f "pose_server.py" >/dev/null; then
  echo "  :8770 pose (already running)"
elif [ -d mpvenv ]; then
  (mpvenv/bin/python service/pose_server.py >logs/pose.log 2>&1 &) && echo "  :8770 MediaPipe pose"
else
  echo "  :8770 SKIPPED — no ./mpvenv yet. Build it once (takes a few min):"
  echo "        ./service/run-pose.sh      # creates ./mpvenv + starts the pose service"
fi

echo "Waiting for services to come up…"; sleep 10
echo "Status:"; status
echo
echo "App:  http://localhost:8000     Logs:  tail -f logs/*.log"
