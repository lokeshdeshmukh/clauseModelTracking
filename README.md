# RunPod deployment

This repository builds a RunPod Serverless worker for a two-endpoint Champ + VideoRetalking pipeline.

Recommended architecture:

1. Endpoint A `preprocess`: accepts a driving video and produces `motion_sequences.zip` plus extracted audio
2. Endpoint B `inference`: accepts a reference photo plus `motion_sequences.zip` and audio/video, then renders the final video

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
- Volume/network storage: attach a RunPod network volume if you do not preload models
- Idle timeout: long enough to avoid repeated cold-start downloads if models are not baked in

Optional environment variables:

- `STORAGE_BACKEND=s3`: upload every finished video to S3 by default
- `AWS_PROFILE=schoollm`: use the named AWS profile if AWS config files are available inside the worker
- `S3_REGION=us-east-1`
- `S3_BUCKET=truefaceswapvideo-schoollm-738149121200`
- `S3_PREFIX=truefaceswapvideo`
- `MODEL_STORAGE_ROOT=/runpod-volume/models`: store Hugging Face and model assets on the RunPod network volume
- `RUNPOD_HANDLER=inference`: run the final photo+motion inference endpoint
- `RUNPOD_HANDLER=preprocess`: run the video preprocessing endpoint
- `CHAMP_POSE_EXTRACTOR`: absolute path to a working Champ video-to-motion extractor if your Champ fork differs from the assumed path
- `DOWNLOAD_MODELS_ON_START=1`: download missing weights when the worker starts
- `PIPELINE_KEEP_TEMP=1`: keep temp artifacts for debugging
- `PIPELINE_BASE64_OUTPUT_MAX_BYTES`: cap for inline base64 responses

If you deploy from GitHub, prefer `PRELOAD_MODELS=0` and let the worker fetch weights on startup onto the attached network volume. RunPod’s current GitHub integration documentation notes a 160-minute build limit and an 80 GB image limit, so baking all model assets into the image is the riskier path for this project.

With the S3 settings above, you do not need to send `output_upload_url` per request. The worker will upload outputs to:

```text
s3://truefaceswapvideo-schoollm-738149121200/truefaceswapvideo/<job_id>/final_output.mp4
```

You can override the final object key on a job with `output_s3_key`.

## Deploy from GitHub

Your repo is already on GitHub, so the simplest path is usually:

1. Go to RunPod `Serverless`.
2. Click `New Endpoint`.
3. Choose `Import Git Repository`.
4. Select `lokeshdeshmukh/clauseModelTracking`.
5. Use branch `main`.
6. Keep `Dockerfile Path` as the repo root `Dockerfile`.
7. Choose endpoint type `Queue`.
8. Attach a network volume so the worker can store multi-GB model files at `/runpod-volume`.
9. Select the GPU types you want and deploy.

RunPod’s GitHub integration docs also note that updates are not pushed automatically after a normal commit. To update a GitHub-backed endpoint, create a new GitHub release or redeploy from the console.

## Endpoint A: Preprocess video

Set:

- `RUNPOD_HANDLER=preprocess`

Purpose:

- input: `driving_video_url` or `driving_video_base64`
- output: `motion_sequences.zip`
- optional output: extracted `driving_audio.wav`

Sample payload:

- [`examples/runpod-job-preprocess-video.json`](/Volumes/Lokesh_1T_E/AI%20Projects/cloude_pipeline/examples/runpod-job-preprocess-video.json)

Important:

- Endpoint A requires a real Champ-compatible extractor.
- Set `CHAMP_POSE_EXTRACTOR` to the extractor script path in the container if the default upstream path does not exist.
- If no extractor is configured, Endpoint A will fail fast instead of pretending it can generate motion data.

The preprocess output zip must contain:

- `dwpose/`
- `depth/`
- `mask/`
- `normal/`
- `semantic_map/`

## Endpoint B: Inference

Set:

- `RUNPOD_HANDLER=inference`

Default supported request format:

1. `photo + motion zip + audio`

Optional request format:

2. `photo + motion zip + driving video`
   In this mode, the worker extracts audio from the supplied video and uses the provided motion zip for body motion.

Sample payloads are in [`examples/runpod-job-motion-audio.json`](/Volumes/Lokesh_1T_E/AI%20Projects/cloude_pipeline/examples/runpod-job-motion-audio.json) and [`examples/runpod-job-video.json`](/Volumes/Lokesh_1T_E/AI%20Projects/cloude_pipeline/examples/runpod-job-video.json).

The motion archive must unpack to a directory containing:

- `dwpose/`
- `depth/`
- `mask/`
- `normal/`
- `semantic_map/`

## Invoke the endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @examples/runpod-job-motion-audio.json
```

## Poll result

```bash
curl -X GET "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

## Notes

- If the Champ repo you clone does not contain a compatible extractor, use Endpoint A only after configuring `CHAMP_POSE_EXTRACTOR`.
- Endpoint B does not create motion sequences from raw video by default. It consumes the output of Endpoint A.
- If output videos are large, keep `return_base64=false` and rely on the default S3 upload.
- `AWS_PROFILE=schoollm` only works if the worker also has the matching AWS config and credentials available. On RunPod, standard AWS environment credentials are usually more reliable than profile-only configuration.
- If you do not attach a network volume and you rely on runtime model download, the worker can fail with `OSError: [Errno 28] No space left on device`.
