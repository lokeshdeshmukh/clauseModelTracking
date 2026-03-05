#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] Running inference smoke checks"

python /workspace/champ/inference.py --help >/dev/null
echo "[smoke] Champ inference imports are healthy"

python - <<'PY'
import tempfile
import zipfile
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

import pipeline
import runpod_handler


required_motion_dirs = ("dwpose", "depth", "mask", "normal", "semantic_map")
def make_tiny_png_bytes() -> bytes:
    with tempfile.TemporaryDirectory(prefix="champ-smoke-png-") as td:
        path = Path(td) / "tiny.png"
        Image.new("RGB", (2, 2), color=(0, 0, 0)).save(path, format="PNG")
        return path.read_bytes()

tiny_png = make_tiny_png_bytes()
required_config_keys = (
    "data",
    "base_model_path",
    "vae_model_path",
    "image_encoder_path",
    "ckpt_dir",
    "motion_module_path",
    "num_inference_steps",
    "guidance_scale",
    "guidance_types",
    "noise_scheduler_kwargs",
    "unet_additional_kwargs",
    "guidance_encoder_kwargs",
)

tmp_root = Path(tempfile.mkdtemp(prefix="champ-smoke-"))
reference = tmp_root / "reference.png"
reference.touch()
motion_root = tmp_root / "motion"
for name in required_motion_dirs:
    folder = motion_root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "0001.png").write_bytes(tiny_png)

cfg_text = pipeline._render_champ_config(
    reference_photo=reference,
    pose_dir=motion_root,
    output_dir=tmp_root / "out",
    width=512,
    height=768,
    steps=20,
    guidance_scale=3.5,
    seed=42,
)
cfg = OmegaConf.create(cfg_text)

for key in required_config_keys:
    if key not in cfg:
        raise RuntimeError(f"Generated Champ config is missing key: {key}")

if cfg.data.ref_image_path != str(reference):
    raise RuntimeError("Generated Champ config has incorrect data.ref_image_path")
if cfg.data.guidance_data_folder != str(motion_root):
    raise RuntimeError("Generated Champ config has incorrect data.guidance_data_folder")
if str(cfg.image_encoder_path) != str(pipeline.CHAMP_DIR / "pretrained_models" / "image_encoder" / "image_encoder"):
    raise RuntimeError("Generated Champ config has incorrect image_encoder_path")

archive = tmp_root / "motion.zip"
with zipfile.ZipFile(archive, "w") as zf:
    zf.writestr("__MACOSX/._dummy", "")
    for motion_dir in required_motion_dirs:
        zf.writestr(f"wrapper/motion-01/{motion_dir}/0001.png", tiny_png)

extracted_root = tmp_root / "extract"
resolved_motion = runpod_handler._extract_motion_sequences(archive, extracted_root)
runpod_handler.validate_motion_sequences(resolved_motion)

print("[smoke] Champ config schema and motion archive extraction checks passed")
PY

echo "[smoke] All checks passed"
