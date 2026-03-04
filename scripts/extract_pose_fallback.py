"""
Fallback pose extraction entrypoint.

This does not attempt to recreate Champ preprocessing internally. It exists so the
worker fails with a clear action instead of a missing-file error when the expected
extractor is not present in the Champ checkout.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained_model_path", required=True)
    parser.parse_args()

    print(
        "No compatible Champ pose extractor was found in the cloned repository.\n"
        "This worker supports precomputed Champ motion directories with\n"
        "dwpose/, depth/, mask/, normal/, and semantic_map/.\n"
        "Either set CHAMP_POSE_EXTRACTOR to a working extractor script or provide\n"
        "those precomputed motion sequences to the RunPod job input.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
