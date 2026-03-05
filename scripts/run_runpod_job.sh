#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PAYLOAD="$ROOT_DIR/examples/runpod-job-motion-bundled-example.json"
DEFAULT_OUTPUT="$ROOT_DIR/final_output.mp4"
POLL_INTERVAL_SECONDS=30

usage() {
  cat <<'EOF'
Usage:
  RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... scripts/run_runpod_job.sh [payload.json] [output.mp4]

Arguments:
  payload.json   Optional. Defaults to examples/runpod-job-motion-bundled-example.json
  output.mp4     Optional. Defaults to final_output.mp4 in the repo root

Behavior:
  - Forces input.return_base64=true in the submitted payload
  - Submits the RunPod job
  - Polls job status every 30 seconds
  - Saves output.output_base64 to the requested mp4 path when complete
  - Raw driving-video payloads require CHAMP_POSE_EXTRACTOR to be configured on the worker
EOF
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_env "RUNPOD_API_KEY"
require_env "RUNPOD_ENDPOINT_ID"

PAYLOAD_PATH="${1:-$DEFAULT_PAYLOAD}"
OUTPUT_PATH="${2:-$DEFAULT_OUTPUT}"

if [[ ! -f "$PAYLOAD_PATH" ]]; then
  echo "Payload file not found: $PAYLOAD_PATH" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
REQUEST_JSON="$TMP_DIR/request.json"
SUBMIT_RESPONSE_JSON="$TMP_DIR/submit-response.json"
STATUS_RESPONSE_JSON="$TMP_DIR/status-response.json"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

python3 - "$PAYLOAD_PATH" "$REQUEST_JSON" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
payload = json.loads(source.read_text(encoding="utf-8"))
payload.setdefault("input", {})
payload["input"]["return_base64"] = True
destination.write_text(json.dumps(payload), encoding="utf-8")
PY

echo "Submitting RunPod job to endpoint: $RUNPOD_ENDPOINT_ID"
curl -sS -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @"$REQUEST_JSON" \
  > "$SUBMIT_RESPONSE_JSON"

JOB_ID="$(python3 - "$SUBMIT_RESPONSE_JSON" <<'PY'
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
job_id = response.get("id")
if not job_id:
    print(json.dumps(response, indent=2), file=sys.stderr)
    raise SystemExit("RunPod submission response did not include job id.")
print(job_id)
PY
)"

echo "Job submitted: $JOB_ID"

while true; do
  curl -sS -X GET "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$JOB_ID" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    > "$STATUS_RESPONSE_JSON"

  STATUS="$(python3 - "$STATUS_RESPONSE_JSON" <<'PY'
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(response.get("status", "UNKNOWN"))
PY
)"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Status: $STATUS"

  case "$STATUS" in
    COMPLETED)
      break
      ;;
    FAILED|CANCELLED|TIMED_OUT)
      python3 - "$STATUS_RESPONSE_JSON" <<'PY'
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(response, indent=2))
PY
      exit 1
      ;;
  esac

  sleep "$POLL_INTERVAL_SECONDS"
done

python3 - "$STATUS_RESPONSE_JSON" "$OUTPUT_PATH" <<'PY'
import base64
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output = response.get("output") or {}
encoded = output.get("output_base64")

if encoded:
    target = Path(sys.argv[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(encoded))
    print(f"Saved mp4 to: {target}")
    raise SystemExit(0)

upload = output.get("upload") or {}
if upload:
    print("Job completed, but no base64 payload was returned.")
    print(json.dumps(upload, indent=2))
    raise SystemExit(0)

print(json.dumps(response, indent=2))
raise SystemExit("Job completed, but no output_base64 field was present.")
PY
