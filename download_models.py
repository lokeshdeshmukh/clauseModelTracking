"""
download_models.py

Download the model weights required by the Champ + VideoRetalking worker.
Use this during image build time or on first container boot.
"""

from __future__ import annotations

import os
import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

WORKSPACE = Path("/workspace")
CHAMP_DIR = Path("/workspace/champ")
RETALKING_DIR = Path("/workspace/video-retalking")
RUNPOD_VOLUME_DIR = Path("/runpod-volume")
DEFAULT_MODEL_STORAGE_ROOT = (
    RUNPOD_VOLUME_DIR / "models" if RUNPOD_VOLUME_DIR.exists() else WORKSPACE / "model-storage"
)
MODEL_STORAGE_ROOT = Path(os.getenv("MODEL_STORAGE_ROOT", str(DEFAULT_MODEL_STORAGE_ROOT)))
PRETRAINED_DIR = MODEL_STORAGE_ROOT / "champ" / "pretrained_models"
WEIGHTS_DIR = MODEL_STORAGE_ROOT / "video-retalking" / "checkpoints"
PREPROCESS_HOME = MODEL_STORAGE_ROOT / "preprocess-home"
FOURD_HUMANS_CACHE = PREPROCESS_HOME / ".cache" / "4DHumans"
FOURD_HUMANS_HMR2_CKPT = (
    FOURD_HUMANS_CACHE
    / "logs"
    / "train"
    / "multiruns"
    / "hmr2"
    / "0"
    / "checkpoints"
    / "epoch=35-step=1000000.ckpt"
)
DETECTRON2_MODEL_PATH = PRETRAINED_DIR / "detectron2" / "model_final_f05665.pkl"
DWPose_CKPT_DIR = CHAMP_DIR / "DWPose" / "ControlNet-v1-1-nightly" / "annotator" / "ckpts"
MODEL_STORAGE_MIN_FREE_GB = float(os.getenv("MODEL_STORAGE_MIN_FREE_GB", "60"))
RETALKING_WEIGHTS_REPO = os.getenv("RETALKING_WEIGHTS_REPO", "camenduru/video-retalking")


def log(message: str):
    print(f"\n[download_models] {message}", flush=True)


def prepare_storage_layout():
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESS_HOME.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(MODEL_STORAGE_ROOT / "hf-cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_STORAGE_ROOT / "hf-cache" / "hub"))

    champ_target = CHAMP_DIR / "pretrained_models"
    retalking_target = RETALKING_DIR / "checkpoints"
    _ensure_symlink(champ_target, PRETRAINED_DIR)
    _ensure_symlink(retalking_target, WEIGHTS_DIR)
    if DWPose_CKPT_DIR.parent.exists():
        DWPose_CKPT_DIR.mkdir(parents=True, exist_ok=True)
        for filename in ("dw-ll_ucoco_384.onnx", "yolox_l.onnx"):
            source = PRETRAINED_DIR / "dwpose" / filename
            destination = DWPose_CKPT_DIR / filename
            if not source.exists():
                continue
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(source)
    ensure_storage_capacity()


def ensure_storage_capacity():
    usage = shutil.disk_usage(MODEL_STORAGE_ROOT)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < MODEL_STORAGE_MIN_FREE_GB:
        raise RuntimeError(
            f"Insufficient free space at {MODEL_STORAGE_ROOT}. "
            f"Free space: {free_gb:.2f} GB, required minimum: {MODEL_STORAGE_MIN_FREE_GB:.2f} GB. "
            "Attach or resize a RunPod network volume and point MODEL_STORAGE_ROOT to it "
            "(recommended: /runpod-volume/models), or preload models outside the worker."
        )


def _ensure_symlink(target_path: Path, source_path: Path):
    if target_path.is_symlink():
        if target_path.resolve() == source_path.resolve():
            return
        target_path.unlink()
    elif target_path.exists():
        if target_path.is_dir():
            source_path.mkdir(parents=True, exist_ok=True)
            for child in target_path.iterdir():
                destination = source_path / child.name
                if destination.exists():
                    continue
                shutil.move(str(child), str(destination))
            target_path.rmdir()
        else:
            raise RuntimeError(
                f"Cannot replace non-directory path with symlink: {target_path}"
            )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.symlink_to(source_path, target_is_directory=True)


def snapshot(repo_id: str, destination: Path, **kwargs):
    snapshot_download(repo_id=repo_id, local_dir=str(destination), **kwargs)


def missing_champ_artifacts() -> list[str]:
    required = {
        "stable-diffusion-v1-5": PRETRAINED_DIR / "stable-diffusion-v1-5" / "model_index.json",
        "sd-vae-ft-mse": PRETRAINED_DIR / "sd-vae-ft-mse" / "config.json",
        "image_encoder": PRETRAINED_DIR / "image_encoder" / "model_index.json",
        "champ": PRETRAINED_DIR / "champ",
        "dwpose detector": PRETRAINED_DIR / "dwpose" / "yolox_l.onnx",
        "dwpose pose": PRETRAINED_DIR / "dwpose" / "dw-ll_ucoco_384.onnx",
    }
    return [name for name, path in required.items() if not path.exists()]


def missing_retalking_artifacts() -> list[str]:
    required = {
        "DNet.pt": WEIGHTS_DIR / "DNet.pt",
        "LNet.pth": WEIGHTS_DIR / "LNet.pth",
        "GFPGANv1.3.pth": WEIGHTS_DIR / "GFPGANv1.3.pth",
        "RetinaFace-R50.pth": WEIGHTS_DIR / "RetinaFace-R50.pth",
        "shape_predictor_68_face_landmarks.dat": WEIGHTS_DIR / "shape_predictor_68_face_landmarks.dat",
    }
    return [name for name, path in required.items() if not path.exists()]


def missing_preprocess_artifacts() -> list[str]:
    required = {
        "detectron2 model": DETECTRON2_MODEL_PATH,
        "HMR2 checkpoint": FOURD_HUMANS_HMR2_CKPT,
    }
    return [name for name, path in required.items() if not path.exists()]


def smpl_model_present() -> bool:
    return (PRETRAINED_DIR / "smpl_models" / "SMPL_NEUTRAL.pkl").exists()


def _download_file(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def _preprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(PREPROCESS_HOME)
    return env


def download_champ_models():
    log("Downloading Champ model weights")
    prepare_storage_layout()
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    log("  -> Stable Diffusion 1.5 base model")
    snapshot(
        "runwayml/stable-diffusion-v1-5",
        PRETRAINED_DIR / "stable-diffusion-v1-5",
        ignore_patterns=[
            "*.safetensors",
            "*.ckpt",
            "*.msgpack",
            "onnx/*",
            "flax/*",
        ],
    )

    log("  -> SD VAE")
    snapshot("stabilityai/sd-vae-ft-mse", PRETRAINED_DIR / "sd-vae-ft-mse")

    log("  -> CLIP image encoder")
    snapshot(
        "lambdalabs/sd-image-variations-diffusers",
        PRETRAINED_DIR / "image_encoder",
        ignore_patterns=["*.msgpack", "rust_model.gguf"],
    )

    log("  -> Champ motion guidance weights")
    snapshot("fudan-generative-ai/champ", PRETRAINED_DIR / "champ")

    log("  -> DWPose weights")
    dwpose_dir = PRETRAINED_DIR / "dwpose"
    dwpose_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("dw-ll_ucoco_384.onnx", "yolox_l.onnx"):
        hf_hub_download(repo_id="yzd-v/DWPose", filename=filename, local_dir=str(dwpose_dir))

    log("  -> SMPL placeholder note")
    smpl_dir = PRETRAINED_DIR / "smpl_models"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    note_path = smpl_dir / "README.txt"
    note_path.write_text(
        "SMPL model requires manual download.\n"
        "Register at https://smpl.is.tue.mpg.de/ and place SMPL_NEUTRAL.pkl here.\n",
        encoding="utf-8",
    )
    log(f"  -> Manual step required: {note_path}")

    log("Champ model download complete")


def download_preprocess_models():
    log("Downloading Champ preprocessing dependencies")
    prepare_storage_layout()

    detectron2_dir = PRETRAINED_DIR / "detectron2"
    detectron2_dir.mkdir(parents=True, exist_ok=True)
    if not DETECTRON2_MODEL_PATH.exists():
        log("  -> Detectron2 model")
        _download_file(
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl",
            DETECTRON2_MODEL_PATH,
        )

    if not FOURD_HUMANS_HMR2_CKPT.exists():
        log("  -> 4D-Humans checkpoint bundle")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from hmr2.models import download_models; download_models()",
            ],
            check=True,
            env=_preprocess_env(),
        )

    smpl_dir = PRETRAINED_DIR / "smpl_models"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    fourd_humans_smpl_dir = FOURD_HUMANS_CACHE / "data" / "smpl"
    fourd_humans_smpl_dir.mkdir(parents=True, exist_ok=True)
    note_text = (
        "SMPL model requires manual download.\n"
        "Register at https://smpl.is.tue.mpg.de/ and place SMPL_NEUTRAL.pkl here.\n"
    )
    if not smpl_model_present():
        (smpl_dir / "README.txt").write_text(note_text, encoding="utf-8")
        (fourd_humans_smpl_dir / "README.txt").write_text(note_text, encoding="utf-8")
        log(f"  -> Manual step required: {smpl_dir / 'README.txt'}")
    else:
        source = PRETRAINED_DIR / "smpl_models" / "SMPL_NEUTRAL.pkl"
        destination = fourd_humans_smpl_dir / "SMPL_NEUTRAL.pkl"
        if destination.is_symlink() and destination.resolve() == source.resolve():
            pass
        else:
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(source)

    prepare_storage_layout()
    log("Champ preprocessing dependency download complete")


def download_retalking_models():
    log("Downloading VideoRetalking checkpoints")
    prepare_storage_layout()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        "BFM.zip",
        "DNet.pt",
        "ENet.pth",
        "expression.mat",
        "face3d_pretrain_epoch_20.pth",
        "GFPGANv1.3.pth",
        "GPEN-BFR-512.pth",
        "LNet.pth",
        "ParseNet-latest.pth",
        "RetinaFace-R50.pth",
        "shape_predictor_68_face_landmarks.dat",
    ]

    for filename in checkpoints:
        log(f"  -> {filename}")
        source_filename = (
            filename
            if RETALKING_WEIGHTS_REPO == "camenduru/video-retalking"
            else f"checkpoints/{filename}"
        )
        destination_dir = WEIGHTS_DIR if RETALKING_WEIGHTS_REPO == "camenduru/video-retalking" else RETALKING_DIR
        hf_hub_download(
            repo_id=RETALKING_WEIGHTS_REPO,
            filename=source_filename,
            local_dir=str(destination_dir),
        )

    bfm_zip = WEIGHTS_DIR / "BFM.zip"
    if bfm_zip.exists():
        log("  -> Extracting BFM.zip")
        subprocess.run(["unzip", "-o", str(bfm_zip), "-d", str(WEIGHTS_DIR)], check=True)

    log("VideoRetalking checkpoint download complete")


if __name__ == "__main__":
    prepare_storage_layout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--champ", action="store_true", help="Download Champ weights")
    parser.add_argument("--preprocess", action="store_true", help="Download preprocessing assets")
    parser.add_argument("--retalking", action="store_true", help="Download VideoRetalking weights")
    parser.add_argument("--all", action="store_true", help="Download both sets of weights")
    args = parser.parse_args()

    if args.all or args.champ:
        download_champ_models()
    if args.all or args.preprocess:
        download_preprocess_models()
    if args.all or args.retalking:
        download_retalking_models()
    if not any((args.all, args.champ, args.preprocess, args.retalking)):
        print("Usage: python download_models.py [--champ] [--preprocess] [--retalking] [--all]")
