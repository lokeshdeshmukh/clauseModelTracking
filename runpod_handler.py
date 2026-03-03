"""
RunPod serverless entrypoint for the Champ + VideoRetalking pipeline.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

import boto3
import runpod
from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound

from download_models import missing_champ_artifacts, missing_retalking_artifacts, smpl_model_present
from pipeline import OUTPUTS_DIR, WORKSPACE, ensure_dirs, run_pipeline, validate_motion_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("runpod-handler")

INPUTS_DIR = Path(os.getenv("PIPELINE_INPUT_DIR", str(WORKSPACE / "inputs")))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("PIPELINE_DOWNLOAD_TIMEOUT_SECONDS", "600"))
BASE64_OUTPUT_MAX_BYTES = int(os.getenv("PIPELINE_BASE64_OUTPUT_MAX_BYTES", "16000000"))
DOWNLOAD_MODELS_ON_START = os.getenv("DOWNLOAD_MODELS_ON_START", "1")
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "").strip().lower()
AWS_PROFILE = os.getenv("AWS_PROFILE")
S3_REGION = os.getenv("S3_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "").strip("/")
S3_PRESIGN_TTL_SECONDS = int(os.getenv("S3_PRESIGN_TTL_SECONDS", "0"))
_MODELS_READY = False


def _coalesce(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _sanitize_filename(name: Optional[str], default_name: str) -> str:
    candidate = (name or default_name).strip()
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in candidate)
    return safe or default_name


def _sanitize_key_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-/" else "_" for ch in value.strip())
    return safe.strip("/") or "job"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _strip_data_uri(value: str) -> str:
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _download_to_file(url: str, destination: Path):
    log.info("Downloading %s -> %s", url, destination)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "runpod-champ-worker/1.0"},
    )
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        destination.write_bytes(response.read())


def _build_default_s3_key(job_id: str, output_path: Path) -> str:
    key_parts = []
    if S3_PREFIX:
        key_parts.append(S3_PREFIX)
    key_parts.append(_sanitize_key_part(job_id))
    key_parts.append(output_path.name)
    return "/".join(key_parts)


def _aws_session():
    session_kwargs: dict[str, Any] = {}
    if AWS_PROFILE:
        session_kwargs["profile_name"] = AWS_PROFILE
    if S3_REGION:
        session_kwargs["region_name"] = S3_REGION
    try:
        return boto3.Session(**session_kwargs)
    except ProfileNotFound as exc:
        raise RuntimeError(
            f"AWS profile '{AWS_PROFILE}' was requested but is not available in the worker. "
            "Provide mounted AWS config files or standard AWS credentials env vars."
        ) from exc


def _upload_output_to_s3(path: Path, bucket: str, key: str) -> dict[str, Any]:
    s3_client = _aws_session().client("s3")
    extra_args = {
        "ContentType": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }

    try:
        s3_client.upload_file(str(path), bucket, key, ExtraArgs=extra_args)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"S3 upload failed for s3://{bucket}/{key}: {exc}") from exc

    result = {
        "uploaded": True,
        "backend": "s3",
        "bucket": bucket,
        "key": key,
        "region": S3_REGION,
        "s3_uri": f"s3://{bucket}/{key}",
    }

    if S3_PRESIGN_TTL_SECONDS > 0:
        try:
            result["presigned_url"] = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=S3_PRESIGN_TTL_SECONDS,
            )
        except (BotoCoreError, ClientError) as exc:
            log.warning("Failed to generate presigned URL for s3://%s/%s: %s", bucket, key, exc)

    return result


def _upload_output_via_http(path: Path, upload_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    method = str(_coalesce(payload, "output_upload_method") or "PUT").upper()
    extra_headers = payload.get("output_upload_headers") or {}
    headers = {
        "Content-Type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        **extra_headers,
    }

    data = path.read_bytes()
    request = urllib.request.Request(upload_url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            status_code = getattr(response, "status", response.getcode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Output upload failed with HTTP {exc.code}: {body}") from exc

    return {
        "uploaded": True,
        "backend": "http",
        "method": method,
        "status_code": status_code,
        "destination": upload_url,
    }


def ensure_model_assets():
    global _MODELS_READY
    if _MODELS_READY:
        return

    missing_champ = missing_champ_artifacts()
    missing_retalking = missing_retalking_artifacts()
    if missing_champ or missing_retalking:
        if not _as_bool(DOWNLOAD_MODELS_ON_START):
            raise RuntimeError(
                "Model assets are missing and DOWNLOAD_MODELS_ON_START is disabled. "
                f"Missing Champ assets: {missing_champ or 'none'}. "
                f"Missing VideoRetalking assets: {missing_retalking or 'none'}."
            )

        log.info("Missing model assets detected. Downloading required weights now.")
        subprocess.run(
            ["python", str(WORKSPACE / "download_models.py"), "--champ", "--retalking"],
            check=True,
        )
        missing_champ = missing_champ_artifacts()
        missing_retalking = missing_retalking_artifacts()
        if missing_champ or missing_retalking:
            raise RuntimeError(
                "Model bootstrap completed but required assets are still missing. "
                f"Missing Champ assets: {missing_champ or 'none'}. "
                f"Missing VideoRetalking assets: {missing_retalking or 'none'}."
            )

    if not smpl_model_present():
        log.warning(
            "SMPL_NEUTRAL.pkl is not present under /workspace/champ/pretrained_models/smpl_models/. "
            "Champ preprocessing/inference may fail until it is added."
        )

    _MODELS_READY = True


def _decode_base64_to_file(encoded_value: str, destination: Path):
    try:
        decoded = base64.b64decode(_strip_data_uri(encoded_value), validate=True)
    except binascii.Error as exc:
        raise ValueError(f"Invalid base64 payload for {destination.name}") from exc
    destination.write_bytes(decoded)


def _materialize_input_file(
    payload: dict[str, Any],
    destination_dir: Path,
    *,
    url_keys: tuple[str, ...],
    base64_keys: tuple[str, ...],
    filename_keys: tuple[str, ...],
    default_name: str,
    required: bool,
) -> Optional[Path]:
    url = _coalesce(payload, *url_keys)
    encoded_value = _coalesce(payload, *base64_keys)

    if url is None and encoded_value is None:
        if required:
            raise ValueError(
                f"Missing required input. Provide one of: {', '.join(url_keys + base64_keys)}"
            )
        return None

    filename = _sanitize_filename(_coalesce(payload, *filename_keys), default_name)
    destination = destination_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    if url is not None:
        _download_to_file(url, destination)
    else:
        _decode_base64_to_file(encoded_value, destination)

    return destination


def _looks_like_motion_dir(path: Path) -> bool:
    required = ("dwpose", "smpl", "depth", "normal", "semantic_map")
    return all((path / name).exists() for name in required)


def _extract_motion_sequences(archive_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(destination_dir)

    if _looks_like_motion_dir(destination_dir):
        return destination_dir

    children = [path for path in destination_dir.iterdir() if path.is_dir()]
    if len(children) == 1 and _looks_like_motion_dir(children[0]):
        return children[0]

    raise ValueError(
        "Motion sequences archive is missing required directories: "
        "dwpose, smpl, depth, normal, semantic_map"
    )


def _encode_output_base64(path: Path) -> str:
    size = path.stat().st_size
    if size > BASE64_OUTPUT_MAX_BYTES:
        raise ValueError(
            f"Output file is {size} bytes which exceeds PIPELINE_BASE64_OUTPUT_MAX_BYTES="
            f"{BASE64_OUTPUT_MAX_BYTES}. Provide output_upload_url instead."
        )
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _upload_output(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    upload_url = _coalesce(payload, "output_upload_url", "upload_url")
    if upload_url:
        if upload_url.startswith("s3://"):
            bucket_and_key = upload_url[5:]
            if "/" not in bucket_and_key:
                raise ValueError("S3 upload URL must be in the form s3://bucket/key")
            bucket, key = bucket_and_key.split("/", 1)
            return _upload_output_to_s3(path, bucket, key)
        return _upload_output_via_http(path, upload_url, payload)

    if STORAGE_BACKEND == "s3":
        if not S3_BUCKET:
            raise RuntimeError("STORAGE_BACKEND=s3 requires S3_BUCKET to be set.")
        job_id = str(payload.get("job_id") or "runpod-job")
        key = str(payload.get("output_s3_key") or _build_default_s3_key(job_id, path))
        return _upload_output_to_s3(path, S3_BUCKET, key)

    return {}


def _prepare_inputs(job_id: str, payload: dict[str, Any]) -> dict[str, Optional[Path]]:
    job_input_dir = INPUTS_DIR / job_id
    shutil.rmtree(job_input_dir, ignore_errors=True)
    ensure_dirs(job_input_dir)

    photo_path = _materialize_input_file(
        payload,
        job_input_dir,
        url_keys=("reference_photo_url", "photo_url"),
        base64_keys=("reference_photo_base64", "photo_base64"),
        filename_keys=("reference_photo_filename", "photo_filename"),
        default_name="reference_photo.png",
        required=True,
    )

    video_path = _materialize_input_file(
        payload,
        job_input_dir,
        url_keys=("driving_video_url", "video_url"),
        base64_keys=("driving_video_base64", "video_base64"),
        filename_keys=("driving_video_filename", "video_filename"),
        default_name="driving_video.mp4",
        required=False,
    )

    audio_path = _materialize_input_file(
        payload,
        job_input_dir,
        url_keys=("audio_url",),
        base64_keys=("audio_base64",),
        filename_keys=("audio_filename",),
        default_name="driving_audio.wav",
        required=False,
    )

    motion_archive = _materialize_input_file(
        payload,
        job_input_dir,
        url_keys=("motion_sequences_url",),
        base64_keys=("motion_sequences_base64",),
        filename_keys=("motion_sequences_filename",),
        default_name="motion_sequences.zip",
        required=False,
    )

    motion_dir = None
    if motion_archive is not None:
        motion_dir = _extract_motion_sequences(motion_archive, job_input_dir / "motion_sequences")
        validate_motion_sequences(motion_dir)

    return {
        "job_input_dir": job_input_dir,
        "photo_path": photo_path,
        "video_path": video_path,
        "audio_path": audio_path,
        "motion_dir": motion_dir,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input") or {}
    job_id = str(job.get("id") or payload.get("job_id") or "runpod-job")
    log.info("Received job %s", job_id)
    ensure_model_assets()
    prepared = _prepare_inputs(job_id, payload)

    width = int(payload.get("width", 512))
    height = int(payload.get("height", 768))
    steps = int(payload.get("steps", 20))
    guidance_scale = float(payload.get("guidance_scale", 3.5))
    seed = int(payload.get("seed", 42))
    keep_temp = _as_bool(payload.get("keep_temp", False))

    output_path = run_pipeline(
        reference_photo=prepared["photo_path"],
        driving_video=prepared["video_path"],
        audio_path=prepared["audio_path"],
        motion_sequences_dir=prepared["motion_dir"],
        output_dir=OUTPUTS_DIR,
        job_id=job_id,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        keep_temp=keep_temp,
    )

    result = {
        "job_id": job_id,
        "output_path": str(output_path),
        "output_file_name": output_path.name,
        "output_size_bytes": output_path.stat().st_size,
    }

    upload_result = _upload_output(output_path, payload)
    if upload_result:
        result["upload"] = upload_result

    if _as_bool(payload.get("return_base64", False)):
        result["output_base64"] = _encode_output_base64(output_path)

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    ensure_dirs(INPUTS_DIR, OUTPUTS_DIR)
    ensure_model_assets()
    runpod.serverless.start({"handler": handler})
