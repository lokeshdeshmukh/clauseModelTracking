"""
download_models.py

Download the model weights required by the Champ + VideoRetalking worker.
Use this during image build time or on first container boot.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

WORKSPACE = Path("/workspace")
CHAMP_DIR = Path("/workspace/champ")
RETALKING_DIR = Path("/workspace/video-retalking")
PRETRAINED_DIR = CHAMP_DIR / "pretrained_models"
WEIGHTS_DIR = RETALKING_DIR / "checkpoints"


def log(message: str):
    print(f"\n[download_models] {message}", flush=True)


def snapshot(repo_id: str, destination: Path, **kwargs):
    snapshot_download(repo_id=repo_id, local_dir=str(destination), **kwargs)


def download_champ_models():
    log("Downloading Champ model weights")
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    log("  -> Stable Diffusion 1.5 base model")
    snapshot("runwayml/stable-diffusion-v1-5", PRETRAINED_DIR / "stable-diffusion-v1-5")

    log("  -> SD VAE")
    snapshot("stabilityai/sd-vae-ft-mse", PRETRAINED_DIR / "sd-vae-ft-mse")

    log("  -> CLIP image encoder")
    snapshot(
        "lambdalabs/sd-image-variations-diffusers",
        PRETRAINED_DIR / "image_encoder",
        ignore_patterns=["*.msgpack", "rust_model.gguf"],
    )

    log("  -> Champ motion guidance weights")
    snapshot("fudan-generative-vision/champ", PRETRAINED_DIR / "champ")

    log("  -> DWPose weights")
    dwpose_dir = PRETRAINED_DIR / "dwpose"
    dwpose_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("dw-ll_ucoco_384.onnx", "det_nano_192x192.onnx"):
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


def download_retalking_models():
    log("Downloading VideoRetalking checkpoints")
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
        hf_hub_download(
            repo_id="OpenTalker/video-retalking",
            filename=f"checkpoints/{filename}",
            local_dir=str(RETALKING_DIR),
        )

    bfm_zip = WEIGHTS_DIR / "BFM.zip"
    if bfm_zip.exists():
        log("  -> Extracting BFM.zip")
        subprocess.run(["unzip", "-o", str(bfm_zip), "-d", str(WEIGHTS_DIR)], check=True)

    log("VideoRetalking checkpoint download complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--champ", action="store_true", help="Download Champ weights")
    parser.add_argument("--retalking", action="store_true", help="Download VideoRetalking weights")
    parser.add_argument("--all", action="store_true", help="Download both sets of weights")
    args = parser.parse_args()

    if args.all or args.champ:
        download_champ_models()
    if args.all or args.retalking:
        download_retalking_models()
    if not any((args.all, args.champ, args.retalking)):
        print("Usage: python download_models.py [--champ] [--retalking] [--all]")
