"""
pipeline.py - Champ + VideoRetalking end-to-end pipeline.

Inputs:
  - reference photo (required)
  - driving video and/or precomputed motion sequences
  - optional separate audio track

Output:
  - final animated video with body motion and lip sync
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")

WORKSPACE = Path(os.getenv("PIPELINE_WORKSPACE", "/workspace"))
CHAMP_DIR = Path(os.getenv("CHAMP_DIR", str(WORKSPACE / "champ")))
RETALKING_DIR = Path(os.getenv("RETALKING_DIR", str(WORKSPACE / "video-retalking")))
SCRIPTS_DIR = WORKSPACE / "scripts"
OUTPUTS_DIR = Path(os.getenv("PIPELINE_OUTPUT_DIR", str(WORKSPACE / "outputs")))
TEMP_DIR = Path(os.getenv("PIPELINE_TEMP_DIR", str(WORKSPACE / "temp")))

CHAMP_INFERENCE = CHAMP_DIR / "inference.py"
RETALKING_INFERENCE = RETALKING_DIR / "inference.py"
DEFAULT_POSE_EXTRACTOR = CHAMP_DIR / "scripts" / "data_processors" / "extract_data_from_video.py"
FALLBACK_POSE_EXTRACTOR = SCRIPTS_DIR / "extract_pose_fallback.py"

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg"}
MOTION_SEQUENCE_SUBDIRS = ("dwpose", "smpl", "depth", "normal", "semantic_map")


def run(cmd: list[str], cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None):
    """Run a subprocess command and raise on failure."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    log.info("CMD: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}"
        )


def ensure_dirs(*dirs: Path):
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_runtime_layout():
    ensure_dirs(OUTPUTS_DIR, TEMP_DIR, SCRIPTS_DIR)

    missing = [
        path
        for path in (CHAMP_INFERENCE, RETALKING_INFERENCE)
        if not path.exists()
    ]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing required runtime files: {formatted}. "
            "Build the RunPod image with the upstream repositories cloned into /workspace."
        )


def _sanitize_job_id(job_id: Optional[str]) -> str:
    raw = job_id or f"job_{int(time.time())}"
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return clean or f"job_{int(time.time())}"


def _validate_photo(photo: Path):
    if not photo.exists():
        raise FileNotFoundError(f"Reference photo not found: {photo}")
    if photo.suffix.lower() not in PHOTO_SUFFIXES:
        raise ValueError(f"Unsupported photo format: {photo.suffix}")


def _validate_video(video: Path):
    if not video.exists():
        raise FileNotFoundError(f"Driving video not found: {video}")
    if video.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video format: {video.suffix}")


def _validate_audio(audio_path: Path):
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported audio format: {audio_path.suffix}")


def validate_motion_sequences(motion_sequences_dir: Path):
    if not motion_sequences_dir.exists():
        raise FileNotFoundError(f"Motion sequences directory not found: {motion_sequences_dir}")
    if not motion_sequences_dir.is_dir():
        raise ValueError(f"Motion sequences path is not a directory: {motion_sequences_dir}")

    missing = [
        subdir for subdir in MOTION_SEQUENCE_SUBDIRS
        if not (motion_sequences_dir / subdir).exists()
    ]
    if missing:
        raise ValueError(
            "Motion sequences directory is incomplete. Missing: "
            + ", ".join(missing)
        )


def validate_inputs(
    photo: Path,
    driving_video: Optional[Path] = None,
    motion_sequences_dir: Optional[Path] = None,
    audio_path: Optional[Path] = None,
):
    _validate_photo(photo)

    if driving_video is None and motion_sequences_dir is None:
        raise ValueError("Provide either a driving video or a motion sequences directory.")
    if driving_video is None and audio_path is None:
        raise ValueError("Provide either a driving video or a separate audio file.")

    if driving_video is not None:
        _validate_video(driving_video)
    if motion_sequences_dir is not None:
        validate_motion_sequences(motion_sequences_dir)
    if audio_path is not None:
        _validate_audio(audio_path)

    log.info("Inputs validated.")


def extract_audio(driving_video: Path, output_dir: Path) -> Path:
    """Extract WAV audio from the driving video using FFmpeg."""
    log.info("Stage 1: extracting audio")
    audio_path = output_dir / "driving_audio.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(driving_video),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
    )
    if not audio_path.exists():
        raise RuntimeError("Audio extraction failed: output file not created.")
    log.info("Audio extracted -> %s", audio_path)
    return audio_path


def _resolve_pose_extractor() -> Path:
    override = os.getenv("CHAMP_POSE_EXTRACTOR")
    candidates = [Path(override)] if override else []
    candidates.extend([DEFAULT_POSE_EXTRACTOR, FALLBACK_POSE_EXTRACTOR])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No compatible pose extractor found. Set CHAMP_POSE_EXTRACTOR or "
        "provide precomputed motion sequences."
    )


def extract_pose_sequences(driving_video: Path, output_dir: Path) -> Path:
    """
    Use a compatible pose extraction script to derive the motion guidance
    directories expected by Champ.
    """
    log.info("Stage 2: extracting motion sequences")
    pose_dir = output_dir / "pose_sequences"
    pose_dir.mkdir(parents=True, exist_ok=True)

    pose_extractor = _resolve_pose_extractor()
    extractor_cwd = CHAMP_DIR if pose_extractor.is_relative_to(CHAMP_DIR) else WORKSPACE
    run(
        [
            "python",
            str(pose_extractor),
            "--video_path",
            str(driving_video),
            "--output_dir",
            str(pose_dir),
            "--pretrained_model_path",
            str(CHAMP_DIR / "pretrained_models" / "dwpose"),
        ],
        cwd=extractor_cwd,
    )

    validate_motion_sequences(pose_dir)
    log.info("Motion sequences extracted -> %s", pose_dir)
    return pose_dir


def _render_champ_config(
    reference_photo: Path,
    pose_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> str:
    pretrained_dir = CHAMP_DIR / "pretrained_models"
    return "\n".join(
        [
            "weight_dtype: 'fp16'",
            f"exp_name: '{output_dir.name}'",
            "",
            f"pretrained_vae_path: '{pretrained_dir / 'sd-vae-ft-mse'}'",
            f"pretrained_base_model_path: '{pretrained_dir / 'stable-diffusion-v1-5'}'",
            f"image_encoder_path: '{pretrained_dir / 'image_encoder'}'",
            f"denoising_unet_path: '{pretrained_dir / 'champ' / 'unet' / 'diffusion_pytorch_model.bin'}'",
            f"reference_unet_path: '{pretrained_dir / 'champ' / 'reference_unet' / 'diffusion_pytorch_model.bin'}'",
            f"pose_guider_path: '{pretrained_dir / 'champ' / 'pose_guider' / 'diffusion_pytorch_model.bin'}'",
            f"motion_module_path: '{pretrained_dir / 'champ' / 'motion_module' / 'pytorch_model.bin'}'",
            "",
            f"source_image: '{reference_photo}'",
            f"driving_smpl_path: '{pose_dir / 'smpl'}'",
            f"driving_pose_dir: '{pose_dir / 'dwpose'}'",
            f"driving_depth_dir: '{pose_dir / 'depth'}'",
            f"driving_normal_dir: '{pose_dir / 'normal'}'",
            f"driving_mask_dir: '{pose_dir / 'semantic_map'}'",
            "",
            f"output_dir: '{output_dir}'",
            f"seed: {seed}",
            "guidance_types: ['pose', 'depth', 'normal', 'seg', 'smpl']",
            "cond_type: 'smpl'",
            "guidance_encoder_kwargs:",
            "  guidance_embedding_channels: 320",
            "  guidance_input_channels: 3",
            "  block_out_channels: [16, 32, 96, 256]",
            "",
            f"W: {width}",
            f"H: {height}",
            f"S: {steps}",
            "ddim_eta: 1.0",
            "fps: 6",
            f"guidance_scale: {guidance_scale}",
            f"steps: {steps}",
            "num_frames: 16",
            "n_motion_frames: 16",
            "fusion_blocks: 'full'",
            "skip: 0",
            "use_text: false",
            "text: ''",
            "zero_snr: false",
            "",
            "do_audio: false",
            "",
            "weight_type:",
            "  pose: 1.0",
            "  smpl: 1.0",
            "  depth: 1.0",
            "  normal: 1.0",
            "  seg: 1.0",
        ]
    )


def _write_champ_config(
    reference_photo: Path,
    pose_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> Path:
    config_path = output_dir / "champ_inference.yaml"
    config_path.write_text(
        _render_champ_config(
            reference_photo=reference_photo,
            pose_dir=pose_dir,
            output_dir=output_dir,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        ),
        encoding="utf-8",
    )
    return config_path


def run_champ(
    reference_photo: Path,
    pose_dir: Path,
    output_dir: Path,
    width: int = 512,
    height: int = 768,
    steps: int = 20,
    guidance_scale: float = 3.5,
    seed: int = 42,
) -> Path:
    """Run Champ inference to animate the reference photo using motion sequences."""
    log.info("Stage 3: Champ body animation")
    animated_dir = output_dir / "champ_output"
    animated_dir.mkdir(parents=True, exist_ok=True)

    config_path = _write_champ_config(
        reference_photo=reference_photo,
        pose_dir=pose_dir,
        output_dir=animated_dir,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    run(["python", str(CHAMP_INFERENCE), "--config", str(config_path)], cwd=CHAMP_DIR)

    animated_video = _find_output_video(animated_dir)
    if animated_video is None:
        frame_dir = _find_frame_dir(animated_dir)
        if frame_dir is None:
            raise RuntimeError(f"Champ did not produce a video or frames under {animated_dir}")
        animated_video = _stitch_frames_to_video(frame_dir, output_dir)

    log.info("Champ animation complete -> %s", animated_video)
    return animated_video


def _find_output_video(directory: Path) -> Optional[Path]:
    for ext in ("*.mp4", "*.avi", "*.mov"):
        matches = sorted(directory.rglob(ext))
        if matches:
            return matches[0]
    return None


def _find_frame_dir(directory: Path) -> Optional[Path]:
    for candidate in [directory, *sorted(path for path in directory.rglob("*") if path.is_dir())]:
        has_png = any(candidate.glob("*.png"))
        has_jpg = any(candidate.glob("*.jpg")) or any(candidate.glob("*.jpeg"))
        if has_png or has_jpg:
            return candidate
    return None


def _stitch_frames_to_video(frames_dir: Path, output_dir: Path, fps: int = 25) -> Path:
    """Stitch PNG or JPG frames into an MP4 video using FFmpeg."""
    log.info("Stitching frames into video from %s", frames_dir)
    frames = sorted(frames_dir.glob("*.png"))
    pattern = "*.png"
    if not frames:
        frames = sorted(frames_dir.glob("*.jpg")) or sorted(frames_dir.glob("*.jpeg"))
        pattern = "*.jpg" if any(frame.suffix.lower() == ".jpg" for frame in frames) else "*.jpeg"
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")

    out_video = output_dir / "animated_body.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            str(frames_dir / pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(out_video),
        ]
    )
    return out_video


def run_retalking(animated_video: Path, audio_path: Path, output_dir: Path) -> Path:
    """Run VideoRetalking to add speech-synced lips to the animated video."""
    log.info("Stage 4: VideoRetalking lip sync")
    final_video = output_dir / "final_output.mp4"

    run(
        [
            "python",
            str(RETALKING_INFERENCE),
            "--face",
            str(animated_video),
            "--audio",
            str(audio_path),
            "--outfile",
            str(final_video),
            "--LNet_batch_size",
            "16",
        ],
        cwd=RETALKING_DIR,
    )

    if not final_video.exists():
        raise RuntimeError("VideoRetalking failed: output file not created.")

    log.info("Lip sync complete -> %s", final_video)
    return final_video


def run_pipeline(
    reference_photo: str | Path,
    driving_video: Optional[str | Path] = None,
    output_dir: str | Path = OUTPUTS_DIR,
    width: int = 512,
    height: int = 768,
    steps: int = 20,
    guidance_scale: float = 3.5,
    seed: int = 42,
    motion_sequences_dir: Optional[str | Path] = None,
    audio_path: Optional[str | Path] = None,
    job_id: Optional[str] = None,
    keep_temp: bool = False,
) -> Path:
    """
    Run the full photo-to-video pipeline.

    You can either provide:
      - a driving video and let the worker extract audio + motion sequences, or
      - precomputed motion sequences plus a separate audio file.
    """
    ensure_runtime_layout()
    t_start = time.time()

    reference_photo = Path(reference_photo)
    driving_video_path = Path(driving_video) if driving_video is not None else None
    motion_sequences_path = Path(motion_sequences_dir) if motion_sequences_dir is not None else None
    audio_input_path = Path(audio_path) if audio_path is not None else None
    output_dir = Path(output_dir)

    validate_inputs(
        photo=reference_photo,
        driving_video=driving_video_path,
        motion_sequences_dir=motion_sequences_path,
        audio_path=audio_input_path,
    )

    safe_job_id = _sanitize_job_id(job_id)
    job_dir = output_dir / safe_job_id
    temp_dir = TEMP_DIR / safe_job_id
    ensure_dirs(output_dir, job_dir, temp_dir)

    log.info("=" * 60)
    log.info("Pipeline start [job=%s]", safe_job_id)
    log.info("Reference photo: %s", reference_photo)
    if driving_video_path is not None:
        log.info("Driving video: %s", driving_video_path)
    if motion_sequences_path is not None:
        log.info("Motion sequences: %s", motion_sequences_path)
    if audio_input_path is not None:
        log.info("Separate audio: %s", audio_input_path)
    log.info("=" * 60)

    try:
        resolved_audio_path = audio_input_path
        if resolved_audio_path is None:
            if driving_video_path is None:
                raise ValueError("A driving video is required when no separate audio file is provided.")
            resolved_audio_path = extract_audio(driving_video_path, temp_dir)

        pose_dir = motion_sequences_path
        if pose_dir is None:
            if driving_video_path is None:
                raise ValueError("A driving video is required when no motion sequences directory is provided.")
            pose_dir = extract_pose_sequences(driving_video_path, temp_dir)

        animated_video = run_champ(
            reference_photo=reference_photo,
            pose_dir=pose_dir,
            output_dir=temp_dir,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        final_video = run_retalking(animated_video, resolved_audio_path, job_dir)
    finally:
        if keep_temp or os.getenv("PIPELINE_KEEP_TEMP") == "1":
            log.info("Keeping temp directory for inspection: %s", temp_dir)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("Pipeline done (%.1fs)", elapsed)
    log.info("Output -> %s", final_video)
    log.info("=" * 60)
    return final_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Champ + VideoRetalking pipeline for local and RunPod jobs."
    )
    parser.add_argument("--photo", required=True, help="Path to the reference photo")
    parser.add_argument("--video", help="Path to the driving video")
    parser.add_argument("--audio", help="Optional separate audio file")
    parser.add_argument(
        "--motion_dir",
        help="Optional precomputed motion sequences directory containing dwpose/smpl/depth/normal/semantic_map",
    )
    parser.add_argument("--output_dir", default=str(OUTPUTS_DIR))
    parser.add_argument("--job_id", help="Optional stable job identifier")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_temp", action="store_true", help="Keep temp artifacts for debugging")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    result = run_pipeline(
        reference_photo=args.photo,
        driving_video=args.video,
        audio_path=args.audio,
        motion_sequences_dir=args.motion_dir,
        output_dir=args.output_dir,
        job_id=args.job_id,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        keep_temp=args.keep_temp,
    )
    print(f"\nDone. Output saved to: {result}")
