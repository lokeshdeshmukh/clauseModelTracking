# RunPod deployment

This repository builds a RunPod Serverless worker for a Champ + VideoRetalking pipeline.

## What to deploy

Use a `Serverless` endpoint backed by a custom container image.

Create a new endpoint if:
- you do not already have a RunPod `Serverless` endpoint for this worker
- your current RunPod setup is a regular pod, not a serverless endpoint
- you want a clean endpoint for this new request schema

Reuse an existing endpoint if:
- it is already a `Serverless` endpoint
- it already runs this container contract
- you only need to roll the worker image forward

## Build the image

Build without model preloading:

```bash
docker build \
  --platform linux/amd64 \
  --build-arg PRELOAD_MODELS=0 \
  -t your-docker-user/clause-model-tracking:runpod-v1 .
```

Build with model preloading:

```bash
docker build \
  --platform linux/amd64 \
  --build-arg PRELOAD_MODELS=1 \
  -t your-docker-user/clause-model-tracking:runpod-v1 .
```

`PRELOAD_MODELS=1` creates a much larger image. SMPL is still a manual drop-in under `/workspace/champ/pretrained_models/smpl_models/`.

## Push the image

```bash
docker push your-docker-user/clause-model-tracking:runpod-v1
```

## Configure the RunPod endpoint

Recommended worker settings:

- Worker type: `Serverless`
- Container image: `your-docker-user/clause-model-tracking:runpod-v1`
- GPU: start with `A40`, `RTX 4090`, or better
- Volume/network storage: recommended if you do not preload models
- Idle timeout: long enough to avoid repeated cold-start downloads if models are not baked in

Optional environment variables:

- `CHAMP_POSE_EXTRACTOR`: absolute path to a working Champ video-to-motion extractor if your Champ fork differs from the assumed path
- `DOWNLOAD_MODELS_ON_START=1`: download missing weights when the worker starts
- `PIPELINE_KEEP_TEMP=1`: keep temp artifacts for debugging
- `PIPELINE_BASE64_OUTPUT_MAX_BYTES`: cap for inline base64 responses

If you deploy from GitHub, prefer `PRELOAD_MODELS=0` and let the worker fetch weights on startup or via attached storage. RunPod’s current GitHub integration documentation notes a 160-minute build limit and an 80 GB image limit, so baking all model assets into the image is the riskier path for this project.

## Deploy from GitHub

Your repo is already on GitHub, so the simplest path is usually:

1. Go to RunPod `Serverless`.
2. Click `New Endpoint`.
3. Choose `Import Git Repository`.
4. Select `lokeshdeshmukh/clauseModelTracking`.
5. Use branch `main`.
6. Keep `Dockerfile Path` as the repo root `Dockerfile`.
7. Choose endpoint type `Queue`.
8. Select the GPU types you want and deploy.

RunPod’s GitHub integration docs also note that updates are not pushed automatically after a normal commit. To update a GitHub-backed endpoint, create a new GitHub release or redeploy from the console.

## Request formats

Two input modes are supported:

1. `photo + video`
2. `photo + motion zip + audio`

Sample payloads are in [`examples/runpod-job-video.json`](/Volumes/Lokesh_1T_E/AI%20Projects/cloude_pipeline/examples/runpod-job-video.json) and [`examples/runpod-job-motion-audio.json`](/Volumes/Lokesh_1T_E/AI%20Projects/cloude_pipeline/examples/runpod-job-motion-audio.json).

The motion archive must unpack to a directory containing:

- `dwpose/`
- `smpl/`
- `depth/`
- `normal/`
- `semantic_map/`

## Invoke the endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @examples/runpod-job-video.json
```

## Poll result

```bash
curl -X GET "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

## Notes

- If the Champ repo you clone does not contain a compatible extractor, send precomputed motion sequences instead of raw driving video.
- If output videos are large, pass `output_upload_url` in the job input instead of `return_base64=true`.
