"""
Wrapper around Champ's released preprocessing scripts.

This script converts a driving video into the motion directory layout expected by
Champ inference:
  - dwpose/
  - depth/
  - mask/
  - normal/
  - semantic_map/

It optionally takes a reference image. If omitted, the first frame of the driving
video is used as the reference for SMPL transfer and rendering.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


WORKSPACE = Path(os.getenv("PIPELINE_WORKSPACE", "/workspace"))
CHAMP_DIR = Path(os.getenv("CHAMP_DIR", str(WORKSPACE / "champ")))
FOURD_HUMANS_DIR = Path(os.getenv("FOURD_HUMANS_DIR", str(WORKSPACE / "4D-Humans")))
DEFAULT_MODEL_STORAGE_ROOT = Path(os.getenv("MODEL_STORAGE_ROOT", str(WORKSPACE / "model-storage")))
PREPROCESS_HOME = Path(
    os.getenv("CHAMP_PREPROCESS_HOME", str(DEFAULT_MODEL_STORAGE_ROOT / "preprocess-home"))
)
DWPose_DIR = CHAMP_DIR / "DWPose"
BLENDER_BIN = Path(os.getenv("BLENDER_BIN", shutil.which("blender") or "/usr/bin/blender"))
PRETRAINED_DIR = CHAMP_DIR / "pretrained_models"
SMPL_MODEL_SOURCE = PRETRAINED_DIR / "smpl_models" / "SMPL_NEUTRAL.pkl"
FOURD_HUMANS_SMPL_PATH = PREPROCESS_HOME / ".cache" / "4DHumans" / "data" / "smpl" / "SMPL_NEUTRAL.pkl"
REQUIRED_OUTPUTS = ("dwpose", "depth", "mask", "normal", "semantic_map")


def log(message: str):
    print(f"[extract_champ_motion] {message}", flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    log("CMD: " + " ".join(str(part) for part in cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(str(part) for part in cmd)}"
        )


def ensure_exists(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def ensure_dwp_pose_layout(pretrained_model_path: Path):
    ckpt_dir = DWPose_DIR / "ControlNet-v1-1-nightly" / "annotator" / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        source = pretrained_model_path / filename
        destination = ckpt_dir / filename
        ensure_exists(source, f"DWPose weight {filename}")
        if destination.is_symlink() or destination.exists():
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            destination.unlink()
        destination.symlink_to(source)


def ensure_smpl_model():
    if not SMPL_MODEL_SOURCE.exists():
        raise FileNotFoundError(
            "SMPL_NEUTRAL.pkl is required for Champ preprocessing. "
            f"Expected it at {SMPL_MODEL_SOURCE}."
        )

    FOURD_HUMANS_SMPL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FOURD_HUMANS_SMPL_PATH.is_symlink() or FOURD_HUMANS_SMPL_PATH.exists():
        if FOURD_HUMANS_SMPL_PATH.is_symlink() and FOURD_HUMANS_SMPL_PATH.resolve() == SMPL_MODEL_SOURCE.resolve():
            return
        FOURD_HUMANS_SMPL_PATH.unlink()
    FOURD_HUMANS_SMPL_PATH.symlink_to(SMPL_MODEL_SOURCE)


def build_preprocess_env() -> dict[str, str]:
    env = {
        "HOME": str(PREPROCESS_HOME),
        "PYTHONPATH": os.pathsep.join(
            [
                str(CHAMP_DIR),
                str(FOURD_HUMANS_DIR),
                os.environ.get("PYTHONPATH", ""),
            ]
        ).strip(os.pathsep),
    }
    return env


def prepare_reference_image(reference_image_path: Path | None, video_path: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if reference_image_path is not None:
        shutil.copy2(reference_image_path, destination)
        return

    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "select=eq(n\\,0)",
            "-vframes",
            "1",
            str(destination),
        ]
    )


def split_video_frames(video_path: Path, destination_dir: Path):
    destination_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            str(destination_dir / "%04d.png"),
        ]
    )


def validate_motion_outputs(path: Path):
    missing = [name for name in REQUIRED_OUTPUTS if not (path / name).exists()]
    if missing:
        raise RuntimeError(
            "Champ preprocessing did not produce the expected outputs. Missing: "
            + ", ".join(missing)
        )


def normalize_smpl_group_archive(smpl_results_dir: Path):
    group_path = smpl_results_dir / "smpls_group.npz"
    if not group_path.exists():
        raise FileNotFoundError(f"Missing grouped SMPL results archive: {group_path}")

    grouped = np.load(group_path, allow_pickle=True)
    if "scaled_focal_length" in grouped.files:
        return

    frame_paths = sorted(
        path
        for path in smpl_results_dir.iterdir()
        if path.suffix == ".npy"
    )
    if not frame_paths:
        raise RuntimeError(f"No per-frame SMPL results found in {smpl_results_dir}")

    focal_lengths = []
    for frame_path in frame_paths:
        frame_result = np.load(frame_path, allow_pickle=True).item()
        if "scaled_focal_length" not in frame_result:
            raise KeyError(
                f"scaled_focal_length missing from per-frame SMPL result: {frame_path}"
            )
        focal_lengths.append(frame_result["scaled_focal_length"])

    np.savez(
        str(group_path),
        smpl=grouped["smpl"],
        camera=grouped["camera"],
        scaled_focal_length=np.asarray(focal_lengths),
    )
    log(f"Normalized {group_path.name} with scaled_focal_length for smpl_transfer")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--reference_image_path")
    parser.add_argument("--device", default=os.getenv("CHAMP_PREPROCESS_DEVICE", "0"))
    parser.add_argument("--keep_workdir", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    pretrained_model_path = Path(args.pretrained_model_path)
    reference_image_path = Path(args.reference_image_path) if args.reference_image_path else None

    ensure_exists(video_path, "driving video")
    ensure_exists(CHAMP_DIR / "scripts" / "data_processors" / "smpl" / "generate_smpls.py", "Champ SMPL preprocessing script")
    ensure_exists(FOURD_HUMANS_DIR / "setup.py", "4D-Humans checkout")
    ensure_exists(DWPose_DIR / "ControlNet-v1-1-nightly" / "annotator" / "dwpose" / "__init__.py", "DWPose checkout")
    ensure_exists(BLENDER_BIN, "Blender executable")

    ensure_dwp_pose_layout(pretrained_model_path)
    ensure_smpl_model()

    work_root = output_dir.parent / "_champ_preprocess_work"
    if work_root.exists():
        shutil.rmtree(work_root)
    reference_root = work_root / "reference_imgs"
    reference_images_dir = reference_root / "images"
    driving_root = work_root / "driving_video"
    driving_images_dir = driving_root / "images"
    transferred_root = work_root / "transferred_result"

    reference_name = "reference.png"
    reference_image_output = reference_images_dir / reference_name

    env = build_preprocess_env()
    figure_transfer_enabled = os.getenv("CHAMP_SMPL_FIGURE_TRANSFER", "1").strip().lower() in {"1", "true", "yes", "on"}
    view_transfer_enabled = os.getenv("CHAMP_SMPL_VIEW_TRANSFER", "1").strip().lower() in {"1", "true", "yes", "on"}
    smooth_smpl_enabled = os.getenv("CHAMP_SMOOTH_SMPL", "0").strip().lower() in {"1", "true", "yes", "on"}

    try:
        prepare_reference_image(reference_image_path, video_path, reference_image_output)
        split_video_frames(video_path, driving_images_dir)

        run(
            [
                sys.executable,
                "-m",
                "scripts.data_processors.smpl.generate_smpls",
                "--reference_imgs_folder",
                str(reference_root),
                "--driving_video_path",
                str(driving_root),
                "--device",
                str(args.device),
            ],
            cwd=CHAMP_DIR,
            env=env,
        )
        normalize_smpl_group_archive(driving_root / "smpl_results")

        if smooth_smpl_enabled:
            run(
                [
                    str(BLENDER_BIN),
                    "--background",
                    "--python",
                    str(CHAMP_DIR / "scripts" / "data_processors" / "smpl" / "smooth_smpls.py"),
                    "--smpls_group_path",
                    str(driving_root / "smpl_results" / "smpls_group.npz"),
                    "--smoothed_result_path",
                    str(driving_root / "smpl_results" / "smpls_group.npz"),
                ],
                cwd=CHAMP_DIR,
                env=env,
            )

        smpl_transfer_cmd = [
            sys.executable,
            "-m",
            "scripts.data_processors.smpl.smpl_transfer",
            "--reference_path",
            str(reference_root / "smpl_results" / Path(reference_name).with_suffix(".npy")),
            "--driving_path",
            str(driving_root),
            "--output_folder",
            str(transferred_root),
            "--device",
            str(args.device),
        ]
        if figure_transfer_enabled:
            smpl_transfer_cmd.append("--figure_transfer")
        if view_transfer_enabled:
            smpl_transfer_cmd.append("--view_transfer")
        run(smpl_transfer_cmd, cwd=CHAMP_DIR, env=env)

        run(
            [
                str(BLENDER_BIN),
                str(CHAMP_DIR / "scripts" / "data_processors" / "smpl" / "blend" / "smpl_rendering.blend"),
                "--background",
                "--python",
                str(CHAMP_DIR / "scripts" / "data_processors" / "smpl" / "render_condition_maps.py"),
                "--driving_path",
                str(transferred_root / "smpl_results"),
                "--reference_path",
                str(reference_image_output),
                "--device",
                str(args.device),
            ],
            cwd=CHAMP_DIR,
            env=env,
        )

        run(
            [
                sys.executable,
                "-m",
                "scripts.data_processors.dwpose.generate_dwpose",
                "--input",
                str(transferred_root / "normal"),
                "--output",
                str(transferred_root / "dwpose"),
            ],
            cwd=CHAMP_DIR,
            env=env,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_OUTPUTS:
            source = transferred_root / name
            destination = output_dir / name
            ensure_exists(source, f"generated {name} directory")
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)

        validate_motion_outputs(output_dir)
        log(f"Motion sequences saved to {output_dir}")
        return 0
    finally:
        if args.keep_workdir or os.getenv("CHAMP_KEEP_PREPROCESS_WORKDIR") == "1":
            log(f"Keeping preprocess work directory for inspection: {work_root}")
        else:
            shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
