"""
pipeline.py  —  Champ + VideoRetalking End-to-End Pipeline
===========================================================
Takes:  reference_photo (jpg/png)  +  driving_video (mp4/mov)
Gives:  final animated video with body motion AND lip sync

Stages:
  1. Extract audio from driving video          (FFmpeg)
  2. Extract pose sequences from driving video (DWPose + SMPL via Champ)
  3. Animate reference photo with poses        (Champ)
  4. Sync lips to audio on animated video      (VideoRetalking)
"""

import os
import sys
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE       = Path("/workspace")
CHAMP_DIR       = WORKSPACE / "champ"
RETALKING_DIR   = WORKSPACE / "video-retalking"
SCRIPTS_DIR     = WORKSPACE / "scripts"
OUTPUTS_DIR     = WORKSPACE / "outputs"
TEMP_DIR        = WORKSPACE / "temp"

CHAMP_INFERENCE     = CHAMP_DIR / "inference.py"
RETALKING_INFERENCE = RETALKING_DIR / "inference.py"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def run(cmd: list[str], cwd: Optional[Path] = None, env: Optional[dict] = None):
    """Run a subprocess command and raise on failure."""
    _env = os.environ.copy()
    if env:
        _env.update(env)
    log.info("CMD: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                            env=_env, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")


def ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def validate_inputs(photo: Path, video: Path):
    if not photo.exists():
        raise FileNotFoundError(f"Reference photo not found: {photo}")
    if photo.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError(f"Unsupported photo format: {photo.suffix}")
    if not video.exists():
        raise FileNotFoundError(f"Driving video not found: {video}")
    if video.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise ValueError(f"Unsupported video format: {video.suffix}")
    log.info("✅ Inputs validated.")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Extract Audio
# ══════════════════════════════════════════════════════════════════════════════

def extract_audio(driving_video: Path, output_dir: Path) -> Path:
    """Extract WAV audio from the driving video using FFmpeg."""
    log.info("━━ Stage 1: Extracting audio ━━")
    audio_path = output_dir / "driving_audio.wav"
    run([
        "ffmpeg", "-y",
        "-i", str(driving_video),
        "-vn",                      # no video
        "-acodec", "pcm_s16le",     # PCM WAV
        "-ar", "16000",             # 16 kHz (required by retalking)
        "-ac", "1",                 # mono
        str(audio_path),
    ])
    if not audio_path.exists():
        raise RuntimeError("Audio extraction failed — output file not created.")
    log.info("✅ Audio extracted → %s", audio_path)
    return audio_path


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Extract Pose Sequences (DWPose + SMPL via Champ utils)
# ══════════════════════════════════════════════════════════════════════════════

def extract_pose_sequences(driving_video: Path, output_dir: Path) -> Path:
    """
    Use Champ's built-in pose extraction script to derive:
      - skeleton (DWPose)
      - depth maps
      - normal maps
      - semantic segmentation
    from the driving video.
    """
    log.info("━━ Stage 2: Extracting pose sequences ━━")
    pose_dir = output_dir / "pose_sequences"
    pose_dir.mkdir(parents=True, exist_ok=True)

    # Champ provides scripts/data_processors/extract_data_from_smpl.py
    # and scripts/data_processors/dwpose_detector.py
    # We call the unified extraction helper:
    pose_extractor = CHAMP_DIR / "scripts" / "data_processors" / "extract_data_from_video.py"
    
    if not pose_extractor.exists():
        # Fallback to dwpose-only extraction
        pose_extractor = SCRIPTS_DIR / "extract_pose_fallback.py"

    run([
        "python", str(pose_extractor),
        "--video_path",  str(driving_video),
        "--output_dir",  str(pose_dir),
        "--pretrained_model_path",
            str(CHAMP_DIR / "pretrained_models" / "dwpose"),
    ], cwd=CHAMP_DIR)

    log.info("✅ Pose sequences extracted → %s", pose_dir)
    return pose_dir


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Champ Body Animation
# ══════════════════════════════════════════════════════════════════════════════

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
    """
    Run Champ inference to animate the reference photo using
    extracted pose sequences.
    """
    log.info("━━ Stage 3: Champ body animation ━━")
    animated_dir = output_dir / "champ_output"
    animated_dir.mkdir(parents=True, exist_ok=True)

    config_path = WORKSPACE / "configs" / "champ_inference.yaml"

    run([
        "python", str(CHAMP_INFERENCE),
        "--config",        str(config_path),
        "--reference_image_path", str(reference_photo),
        "--motion_seqs_dir",      str(pose_dir),
        "--output_dir",           str(animated_dir),
        "--width",         str(width),
        "--height",        str(height),
        "--num_inference_steps", str(steps),
        "--guidance_scale",      str(guidance_scale),
        "--seed",          str(seed),
    ], cwd=CHAMP_DIR)

    # Champ outputs frames or a video — find it
    animated_video = _find_output_video(animated_dir)
    if animated_video is None:
        # Champ may output frames — stitch them
        animated_video = _stitch_frames_to_video(animated_dir, output_dir)

    log.info("✅ Champ animation complete → %s", animated_video)
    return animated_video


def _find_output_video(directory: Path) -> Optional[Path]:
    for ext in ["*.mp4", "*.avi", "*.mov"]:
        matches = list(directory.glob(ext))
        if matches:
            return matches[0]
    return None


def _stitch_frames_to_video(frames_dir: Path, output_dir: Path,
                             fps: int = 25) -> Path:
    """Stitch PNG/JPG frames into an MP4 video using FFmpeg."""
    log.info("Stitching frames into video...")
    # Detect frame naming pattern
    frames = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")

    out_video = output_dir / "animated_body.mp4"
    run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(frames_dir / "*.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_video),
    ])
    return out_video


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — VideoRetalking Lip Sync
# ══════════════════════════════════════════════════════════════════════════════

def run_retalking(
    animated_video: Path,
    audio_path: Path,
    output_dir: Path,
) -> Path:
    """
    Run VideoRetalking to replace the mouth region in the animated
    video with speech-synced lips from the audio track.
    """
    log.info("━━ Stage 4: VideoRetalking lip sync ━━")
    final_video = output_dir / "final_output.mp4"

    run([
        "python", str(RETALKING_INFERENCE),
        "--face",        str(animated_video),
        "--audio",       str(audio_path),
        "--outfile",     str(final_video),
        "--LNet_batch_size", "16",    # adjust per GPU VRAM
    ], cwd=RETALKING_DIR)

    if not final_video.exists():
        raise RuntimeError("VideoRetalking failed — output file not created.")

    log.info("✅ Lip sync complete → %s", final_video)
    return final_video


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    reference_photo: str | Path,
    driving_video:   str | Path,
    output_dir:      str | Path = OUTPUTS_DIR,
    width:  int   = 512,
    height: int   = 768,
    steps:  int   = 20,
    guidance_scale: float = 3.5,
    seed:   int   = 42,
) -> Path:
    """
    Full end-to-end pipeline:
      photo + driving_video → final animated + lip-synced video
    
    Returns path to the final output video.
    """
    t_start = time.time()

    reference_photo = Path(reference_photo)
    driving_video   = Path(driving_video)
    output_dir      = Path(output_dir)

    # ── Setup ──────────────────────────────────────────────
    job_id  = f"job_{int(time.time())}"
    job_dir = output_dir / job_id
    temp    = TEMP_DIR / job_id
    ensure_dirs(job_dir, temp)

    log.info("="*60)
    log.info("🚀 Pipeline START  [job: %s]", job_id)
    log.info("   Photo : %s", reference_photo)
    log.info("   Video : %s", driving_video)
    log.info("="*60)

    # ── Validate inputs ────────────────────────────────────
    validate_inputs(reference_photo, driving_video)

    # ── Stage 1: Audio extraction ──────────────────────────
    audio_path = extract_audio(driving_video, temp)

    # ── Stage 2: Pose extraction ───────────────────────────
    pose_dir = extract_pose_sequences(driving_video, temp)

    # ── Stage 3: Champ animation ───────────────────────────
    animated_video = run_champ(
        reference_photo, pose_dir, temp,
        width=width, height=height,
        steps=steps, guidance_scale=guidance_scale,
        seed=seed,
    )

    # ── Stage 4: Lip sync ──────────────────────────────────
    final_video = run_retalking(animated_video, audio_path, job_dir)

    # ── Cleanup temp ───────────────────────────────────────
    shutil.rmtree(temp, ignore_errors=True)

    elapsed = time.time() - t_start
    log.info("="*60)
    log.info("✅ Pipeline DONE  (%.1fs)", elapsed)
    log.info("   Output → %s", final_video)
    log.info("="*60)

    return final_video


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Champ + VideoRetalking: animate a photo with a driving video"
    )
    parser.add_argument("--photo",          required=True,  help="Path to reference photo")
    parser.add_argument("--video",          required=True,  help="Path to driving video")
    parser.add_argument("--output_dir",     default=str(OUTPUTS_DIR))
    parser.add_argument("--width",          type=int,   default=512)
    parser.add_argument("--height",         type=int,   default=768)
    parser.add_argument("--steps",          type=int,   default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    result = run_pipeline(
        reference_photo = args.photo,
        driving_video   = args.video,
        output_dir      = args.output_dir,
        width           = args.width,
        height          = args.height,
        steps           = args.steps,
        guidance_scale  = args.guidance_scale,
        seed            = args.seed,
    )
    print(f"\n🎬 Done! Output saved to: {result}")
