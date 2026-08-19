#!/bin/bash
# Start the MotionLife VLM Router (Step 1). Samples frames from an uploaded clip and
# asks Claude vision to decompose it into distinct motions (Contract A) on :8771.
# First run creates ./routervenv and installs the Anthropic SDK + OpenCV.
set -e
cd "$(dirname "$0")/.."          # repo root
VENV="routervenv"

# ── Credentials (TODO — not wired yet) ───────────────────────────────────────
# The router talks to Claude via Amazon Bedrock by default (CLAUDE_CODE_USE_BEDROCK=1),
# reusing the standard AWS credential chain — no API key needed. It currently 500s
# because the default AWS profile's token is invalid/expired. To finish later, pick ONE:
#   (a) refresh your default AWS creds so botocore resolves a valid Bedrock token, or
#   (b) export AWS_PROFILE=<a profile with Bedrock access> before running this, or
#   (c) export ROUTER_USE_BEDROCK=0 ANTHROPIC_API_KEY=sk-ant-...  (direct Anthropic API)
if [ "$ROUTER_USE_BEDROCK" = "0" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "WARNING: ROUTER_USE_BEDROCK=0 but ANTHROPIC_API_KEY is not set — /decompose will 500." >&2
fi

if [ ! -d "$VENV" ]; then
  echo "Creating router venv …"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r service/requirements-router.txt
fi

echo "VLM Router starting on http://127.0.0.1:8771  (POST /decompose)"
exec "$VENV/bin/python" service/vlm_router.py
