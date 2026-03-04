#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PAYLOAD="$ROOT_DIR/examples/runpod-job-preprocess-video.json"
DEFAULT_MOTION_OUTPUT="$ROOT_DIR/motion_sequences.zip"
DEFAULT_AUDIO_OUTPUT="$ROOT_DIR/driving_audio.wav"
POLL_INTERVAL_SECONDS=30

usage() {
  cat <<'EOF'
Usage:
  RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... scripts/run_runpod_preprocess_job.sh [payload.json] [motion.zip] [audio.wav]

Arguments:
  payload.json   Optional. Defaults to examples/runpod-job-preprocess-video.json
  motion.zip     Optional. Defaults to motion_sequences.zip in the repo root
  audio.wav      Optional. Defaults to driving_audio.wav in the repo root

Behavior:
  - Forces input.return_base64=true in the submitted payload
  - Submits the RunPod preprocess job
  - Polls job status every 30 seconds
  - Saves motion_sequences_zip.base64 to the requested zip path when complete
  - Saves audio.base64 when present
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
MOTION_OUTPUT_PATH="${2:-$DEFAULT_MOTION_OUTPUT}"
AUDIO_OUTPUT_PATH="${3:-$DEFAULT_AUDIO_OUTPUT}"

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

echo "Submitting RunPod preprocess job to endpoint: $RUNPOD_ENDPOINT_ID"
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

python3 - "$STATUS_RESPONSE_JSON" "$MOTION_OUTPUT_PATH" "$AUDIO_OUTPUT_PATH" <<'PY'
import base64
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output = response.get("output") or {}
motion = output.get("motion_sequences_zip") or {}
audio = output.get("audio") or {}

motion_b64 = motion.get("base64")
if motion_b64:
    target = Path(sys.argv[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(motion_b64))
    print(f"Saved motion zip to: {target}")
else:
    upload = motion.get("upload") or {}
    if upload:
        print("Motion zip uploaded:")
        print(json.dumps(upload, indent=2))

audio_b64 = audio.get("base64")
if audio_b64:
    target = Path(sys.argv[3])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(audio_b64))
    print(f"Saved audio to: {target}")
elif audio:
    upload = audio.get("upload") or {}
    if upload:
        print("Audio uploaded:")
        print(json.dumps(upload, indent=2))

if not motion_b64 and not motion.get("upload"):
    print(json.dumps(response, indent=2))
    raise SystemExit("Job completed, but no motion zip payload was returned.")
PY
