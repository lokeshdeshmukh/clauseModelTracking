"""
download_models.py
Downloads all required model weights for Champ + VideoRetalking pipeline.
Run once at Docker build time or first launch.
"""

import os
import argparse
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# ── Paths ──────────────────────────────────────────────────────────────────────
CHAMP_DIR       = Path("/workspace/champ")
RETALKING_DIR   = Path("/workspace/video-retalking")
PRETRAINED_DIR  = CHAMP_DIR / "pretrained_models"
WEIGHTS_DIR     = RETALKING_DIR / "checkpoints"


def log(msg: str):
    print(f"\n[download_models] {msg}", flush=True)


# ── Champ weights ──────────────────────────────────────────────────────────────
def download_champ_models():
    log("Downloading Champ model weights from HuggingFace...")

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    # Stable Diffusion 1.5 (base)
    log("  → Stable Diffusion 1.5 (base UNet)...")
    snapshot_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir=str(PRETRAINED_DIR / "stable-diffusion-v1-5"),
        ignore_patterns=["*.bin", "*.safetensors"],   # config only first pass
    )
    # Full SD weights needed by Champ
    snapshot_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir=str(PRETRAINED_DIR / "stable-diffusion-v1-5"),
    )

    # image_encoder (CLIP)
    log("  → CLIP image encoder...")
    snapshot_download(
        repo_id="lambdalabs/sd-image-variations-diffusers",
        local_dir=str(PRETRAINED_DIR / "image_encoder"),
        ignore_patterns=["*.msgpack", "rust_model.gguf"],
    )

    # Champ-specific weights
    log("  → Champ motion guidance weights...")
    snapshot_download(
        repo_id="fudan-generative-vision/champ",
        local_dir=str(PRETRAINED_DIR / "champ"),
    )

    # DWPose (pose detector)
    log("  → DWPose weights...")
    dwpose_dir = PRETRAINED_DIR / "dwpose"
    dwpose_dir.mkdir(parents=True, exist_ok=True)
    for fname, repo in [
        ("dw-ll_ucoco_384.onnx",   "yzd-v/DWPose"),
        ("det_nano_192x192.onnx",  "yzd-v/DWPose"),
    ]:
        hf_hub_download(repo_id=repo, filename=fname,
                        local_dir=str(dwpose_dir))

    # SMPL-related (shape guidance)
    log("  → SMPL model files...")
    smpl_dir = PRETRAINED_DIR / "smpl_models"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    # Note: SMPL requires manual registration at https://smpl.is.tue.mpg.de/
    # Place SMPL_NEUTRAL.pkl in /workspace/champ/pretrained_models/smpl_models/
    smpl_note = smpl_dir / "README.txt"
    smpl_note.write_text(
        "⚠️  SMPL model requires manual download.\n"
        "Register at: https://smpl.is.tue.mpg.de/\n"
        "Download SMPL_NEUTRAL.pkl and place it here.\n"
    )
    log("  ⚠️  SMPL requires manual download — see pretrained_models/smpl_models/README.txt")

    log("✅ Champ models downloaded.")


# ── VideoRetalking weights ─────────────────────────────────────────────────────
def download_retalking_models():
    log("Downloading VideoRetalking checkpoints...")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        # (filename, repo_id, subfolder or None)
        ("BFM.zip",                     "OpenTalker/video-retalking", "checkpoints"),
        ("DNet.pt",                     "OpenTalker/video-retalking", "checkpoints"),
        ("ENet.pth",                    "OpenTalker/video-retalking", "checkpoints"),
        ("expression.mat",              "OpenTalker/video-retalking", "checkpoints"),
        ("face3d_pretrain_epoch_20.pth","OpenTalker/video-retalking", "checkpoints"),
        ("GFPGANv1.3.pth",             "OpenTalker/video-retalking", "checkpoints"),
        ("GPEN-BFR-512.pth",           "OpenTalker/video-retalking", "checkpoints"),
        ("LNet.pth",                    "OpenTalker/video-retalking", "checkpoints"),
        ("ParseNet-latest.pth",         "OpenTalker/video-retalking", "checkpoints"),
        ("RetinaFace-R50.pth",         "OpenTalker/video-retalking", "checkpoints"),
        ("shape_predictor_68_face_landmarks.dat",
                                        "OpenTalker/video-retalking", "checkpoints"),
    ]

    for fname, repo, subfolder in checkpoints:
        log(f"  → {fname}")
        try:
            hf_hub_download(
                repo_id=repo,
                filename=f"{subfolder}/{fname}" if subfolder else fname,
                local_dir=str(RETALKING_DIR),
            )
        except Exception as e:
            log(f"  ⚠️  Failed to download {fname}: {e}")

    # Unzip BFM if present
    bfm_zip = WEIGHTS_DIR / "BFM.zip"
    if bfm_zip.exists():
        log("  → Extracting BFM.zip...")
        subprocess.run(["unzip", "-o", str(bfm_zip), "-d", str(WEIGHTS_DIR)],
                       check=True)

    log("✅ VideoRetalking checkpoints downloaded.")


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--champ",     action="store_true", help="Download Champ weights")
    parser.add_argument("--retalking", action="store_true", help="Download VideoRetalking weights")
    parser.add_argument("--all",       action="store_true", help="Download everything")
    args = parser.parse_args()

    if args.all or args.champ:
        download_champ_models()
    if args.all or args.retalking:
        download_retalking_models()

    if not any([args.champ, args.retalking, args.all]):
        print("Usage: python download_models.py [--champ] [--retalking] [--all]")
