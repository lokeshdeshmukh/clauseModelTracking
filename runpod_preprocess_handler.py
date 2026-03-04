"""
RunPod serverless entrypoint for Champ motion preprocessing.

Endpoint A:
  - input: driving video, optional reference photo
  - output: motion_sequences.zip and optional extracted audio
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
from pathlib import Path
from typing import Any, Optional

import boto3
import runpod
from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound

from download_models import (
    MODEL_STORAGE_ROOT,
    missing_preprocess_artifacts,
    prepare_storage_layout,
    smpl_model_present,
)
from pipeline import (
    OUTPUTS_DIR,
    WORKSPACE,
    ensure_dirs,
    extract_audio,
    extract_pose_sequences,
    has_native_pose_extractor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("runpod-preprocess-handler")

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
        headers={"User-Agent": "runpod-champ-preprocess-worker/1.0"},
    )
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        destination.write_bytes(response.read())


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


def _upload_to_s3(path: Path, bucket: str, key: str) -> dict[str, Any]:
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


def _upload_via_http(path: Path, upload_url: str, payload: dict[str, Any], method_key: str) -> dict[str, Any]:
    method = str(payload.get(method_key) or "PUT").upper()
    headers = {
        "Content-Type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
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


def _encode_output_base64(path: Path) -> str:
    size = path.stat().st_size
    if size > BASE64_OUTPUT_MAX_BYTES:
        raise ValueError(
            f"Output file is {size} bytes which exceeds PIPELINE_BASE64_OUTPUT_MAX_BYTES="
            f"{BASE64_OUTPUT_MAX_BYTES}. Upload the file instead of returning base64."
        )
    return base64.b64encode(path.read_bytes()).decode("ascii")


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


def _default_output_key(job_id: str, filename: str) -> str:
    key_parts = []
    if S3_PREFIX:
        key_parts.append(S3_PREFIX)
    key_parts.append(_sanitize_key_part(job_id))
    key_parts.append(filename)
    return "/".join(key_parts)


def _upload_artifact(
    path: Path,
    payload: dict[str, Any],
    *,
    job_id: str,
    upload_url_keys: tuple[str, ...],
    method_key: str,
    default_s3_key_key: str,
) -> dict[str, Any]:
    upload_url = _coalesce(payload, *upload_url_keys)
    if upload_url:
        if upload_url.startswith("s3://"):
            bucket_and_key = upload_url[5:]
            if "/" not in bucket_and_key:
                raise ValueError("S3 upload URL must be in the form s3://bucket/key")
            bucket, key = bucket_and_key.split("/", 1)
            return _upload_to_s3(path, bucket, key)
        return _upload_via_http(path, upload_url, payload, method_key)

    if STORAGE_BACKEND == "s3":
        if not S3_BUCKET:
            raise RuntimeError("STORAGE_BACKEND=s3 requires S3_BUCKET to be set.")
        key = str(payload.get(default_s3_key_key) or _default_output_key(job_id, path.name))
        return _upload_to_s3(path, S3_BUCKET, key)

    return {}


def ensure_preprocess_assets():
    global _MODELS_READY
    if _MODELS_READY:
        return

    prepare_storage_layout()
    log.info("Model storage root: %s", MODEL_STORAGE_ROOT)

    if not has_native_pose_extractor():
        raise RuntimeError(
            "Endpoint A requires a real Champ video-to-motion extractor. "
            "Set CHAMP_POSE_EXTRACTOR to a valid script in the container or add the "
            "upstream extractor path to the image."
        )

    missing_preprocess = missing_preprocess_artifacts()
    if missing_preprocess:
        if not _as_bool(DOWNLOAD_MODELS_ON_START):
            raise RuntimeError(
                "Preprocess assets are missing and DOWNLOAD_MODELS_ON_START is disabled. "
                f"Missing preprocess assets: {missing_preprocess}."
            )
        log.info("Ensuring Champ preprocessing assets are available.")
        subprocess.run(
            ["python", str(WORKSPACE / "download_models.py"), "--champ", "--preprocess"],
            check=True,
        )
        missing_preprocess = missing_preprocess_artifacts()
        if missing_preprocess:
            raise RuntimeError(
                "Preprocess asset bootstrap completed but required assets are still missing. "
                f"Missing preprocess assets: {missing_preprocess}."
            )

    if not smpl_model_present():
        log.warning(
            "SMPL_NEUTRAL.pkl is not present under /workspace/champ/pretrained_models/smpl_models/. "
            "Champ preprocessing may fail until it is seeded from SMPL_MODEL_URL or added manually."
        )

    _MODELS_READY = True


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
        required=False,
    )

    video_path = _materialize_input_file(
        payload,
        job_input_dir,
        url_keys=("driving_video_url", "video_url"),
        base64_keys=("driving_video_base64", "video_base64"),
        filename_keys=("driving_video_filename", "video_filename"),
        default_name="driving_video.mp4",
        required=True,
    )

    return {
        "job_input_dir": job_input_dir,
        "photo_path": photo_path,
        "video_path": video_path,
    }


def _zip_motion_sequences(source_dir: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_base = destination.with_suffix("")
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=source_dir))
    if archive_path != destination:
        shutil.move(str(archive_path), str(destination))
    return destination


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input") or {}
    job_id = str(job.get("id") or payload.get("job_id") or "runpod-preprocess-job")
    log.info("Received preprocess job %s", job_id)
    ensure_preprocess_assets()
    prepared = _prepare_inputs(job_id, payload)

    keep_temp = _as_bool(payload.get("keep_temp", False))
    extract_audio_enabled = _as_bool(payload.get("extract_audio", True))

    output_dir = OUTPUTS_DIR / job_id
    temp_dir = OUTPUTS_DIR / f"{job_id}_temp"
    ensure_dirs(output_dir, temp_dir)

    try:
        motion_dir = extract_pose_sequences(
            prepared["video_path"],
            temp_dir,
            reference_image=prepared["photo_path"],
        )
        motion_zip = _zip_motion_sequences(motion_dir, output_dir / "motion_sequences.zip")

        audio_path = None
        if extract_audio_enabled:
            extracted_audio = extract_audio(prepared["video_path"], temp_dir)
            audio_path = output_dir / "driving_audio.wav"
            shutil.copy2(extracted_audio, audio_path)

        result: dict[str, Any] = {
            "job_id": job_id,
            "motion_sequences_zip": {
                "path": str(motion_zip),
                "file_name": motion_zip.name,
                "size_bytes": motion_zip.stat().st_size,
            },
        }

        motion_upload = _upload_artifact(
            motion_zip,
            payload,
            job_id=job_id,
            upload_url_keys=("motion_upload_url", "output_upload_url"),
            method_key="motion_upload_method",
            default_s3_key_key="motion_output_s3_key",
        )
        if motion_upload:
            result["motion_sequences_zip"]["upload"] = motion_upload

        if _as_bool(payload.get("return_base64", False)) or _as_bool(payload.get("return_motion_base64", False)):
            result["motion_sequences_zip"]["base64"] = _encode_output_base64(motion_zip)

        if audio_path is not None:
            result["audio"] = {
                "path": str(audio_path),
                "file_name": audio_path.name,
                "size_bytes": audio_path.stat().st_size,
            }
            audio_upload = _upload_artifact(
                audio_path,
                payload,
                job_id=job_id,
                upload_url_keys=("audio_upload_url",),
                method_key="audio_upload_method",
                default_s3_key_key="audio_output_s3_key",
            )
            if audio_upload:
                result["audio"]["upload"] = audio_upload

            if _as_bool(payload.get("return_base64", False)) or _as_bool(payload.get("return_audio_base64", False)):
                result["audio"]["base64"] = _encode_output_base64(audio_path)

        metadata_path = output_dir / "preprocess_result.json"
        metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        if keep_temp or os.getenv("PIPELINE_KEEP_TEMP") == "1":
            log.info("Keeping preprocess temp directory for inspection: %s", temp_dir)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    ensure_dirs(INPUTS_DIR, OUTPUTS_DIR)
    ensure_preprocess_assets()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
