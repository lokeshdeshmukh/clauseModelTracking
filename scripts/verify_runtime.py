#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import sys
from pathlib import Path


WORKSPACE = Path("/workspace")
CHAMP_DIR = WORKSPACE / "champ"
FOURD_HUMANS_DIR = WORKSPACE / "4D-Humans"
DWPOSE_DIR = CHAMP_DIR / "DWPose"


def ok(message: str):
    print(f"[verify] OK   {message}")


def fail(message: str):
    print(f"[verify] FAIL {message}")


def run_check(name: str, fn, failures: list[str]):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        fail(f"{name}: {exc}")
        failures.append(name)
    else:
        ok(name)


def check_path(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"{description} missing at {path}")


def check_binary(name: str):
    if shutil.which(name) is None:
        raise FileNotFoundError(f"{name} not found in PATH")


def import_module(name: str):
    importlib.import_module(name)


def import_hmr2():
    sys.path.insert(0, str(CHAMP_DIR))
    sys.path.insert(0, str(FOURD_HUMANS_DIR))
    from hmr2.models import download_models  # noqa: F401


def check_torch_family_versions():
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "torchaudio")
    }
    normalized = {name: version.split("+", 1)[0] for name, version in versions.items()}
    major_minor = {
        name: ".".join(version.split(".")[:2])
        for name, version in normalized.items()
    }
    if len(set(major_minor.values())) != 1:
        raise RuntimeError(
            "Torch package mismatch: "
            + ", ".join(f"{name}={versions[name]}" for name in ("torch", "torchvision", "torchaudio"))
        )


def main() -> int:
    failures: list[str] = []

    checks = [
        ("python binary", lambda: check_binary("python")),
        ("ffmpeg binary", lambda: check_binary("ffmpeg")),
        ("blender binary", lambda: check_binary("blender")),
        ("champ checkout", lambda: check_path(CHAMP_DIR / "scripts" / "data_processors" / "smpl" / "generate_smpls.py", "Champ SMPL generator")),
        ("4D-Humans checkout", lambda: check_path(FOURD_HUMANS_DIR / "setup.py", "4D-Humans setup.py")),
        ("DWPose checkout", lambda: check_path(DWPOSE_DIR / "ControlNet-v1-1-nightly" / "annotator" / "dwpose" / "__init__.py", "DWPose annotator")),
        ("torch import", lambda: import_module("torch")),
        ("torchvision import", lambda: import_module("torchvision")),
        ("torchaudio import", lambda: import_module("torchaudio")),
        ("torch family version match", check_torch_family_versions),
        ("detectron2 import", lambda: import_module("detectron2")),
        ("pytorch_lightning import", lambda: import_module("pytorch_lightning")),
        ("dlib import", lambda: import_module("dlib")),
        ("hmr2 import chain", import_hmr2),
    ]

    for name, fn in checks:
        run_check(name, fn, failures)

    if failures:
        print(f"[verify] {len(failures)} checks failed: {', '.join(failures)}")
        return 1

    print("[verify] Runtime verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
