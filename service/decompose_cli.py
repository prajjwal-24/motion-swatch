"""Decompose a video into motions from the command line (verifies Step 1).

  ANTHROPIC_API_KEY=sk-ant-... routervenv/bin/python service/decompose_cli.py assets/videos/flag.mp4

Prints the Contract-A JSON the VLM Router would return — handy for checking the
Step-1 done-when ("a flag+smoke+birds clip -> correctly-classed entries") without
running the HTTP server.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm_router as R

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: decompose_cli.py <video>", file=sys.stderr); sys.exit(2)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: export ANTHROPIC_API_KEY first", file=sys.stderr); sys.exit(1)
    contract, warnings = R.decompose(sys.argv[1])
    if warnings:
        contract["warnings"] = warnings
    print(json.dumps(contract, indent=2))
