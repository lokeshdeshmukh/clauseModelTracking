"""
Select the RunPod worker mode from environment.

Set RUNPOD_HANDLER to:
  - inference  -> photo + motion/audio endpoint
  - preprocess -> video -> motion zip/audio endpoint
"""

from __future__ import annotations

import os


def main():
    mode = os.getenv("RUNPOD_HANDLER", "inference").strip().lower()

    if mode == "inference":
        from runpod_handler import main as handler_main
    elif mode == "preprocess":
        from runpod_preprocess_handler import main as handler_main
    else:
        raise SystemExit(
            f"Unsupported RUNPOD_HANDLER={mode!r}. Expected 'inference' or 'preprocess'."
        )

    handler_main()


if __name__ == "__main__":
    main()
