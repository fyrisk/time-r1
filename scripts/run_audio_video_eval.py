#!/usr/bin/env python3
"""Command-line entry for evaluating audio-video temporal localization.

Example usage::

    python scripts/run_audio_video_eval.py \
        --qa-root /data5/fy/omni-reason-ground/qa \
        --video-root /data5/fy/data/mybenchvideo \
        --output outputs/omni_eval.jsonl \
        --model-module /path/to/my_model_impl.py

``my_model_impl.py`` must expose a ``build_model()`` function that returns
an instance of :class:`omni_eval.model_interface.TemporalLocalizationModel`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from omni_eval.evaluator import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal localization QA tasks")
    parser.add_argument(
        "--qa-root",
        type=str,
        required=True,
        help="Root directory containing QA JSON files",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        required=True,
        help="Root directory containing MP4 files aligned with the QA set",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/omni_eval.jsonl",
        help="Where to store per-sample predictions in JSONL format",
    )
    parser.add_argument(
        "--model-module",
        type=str,
        default=None,
        help="Optional Python file that defines build_model() -> TemporalLocalizationModel",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="IoU thresholds for reporting (default: 0.3 0.5 0.7)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_evaluation(
        qa_root=args.qa_root,
        video_root=args.video_root,
        output_path=args.output,
        model_module=args.model_module,
        thresholds=args.thresholds,
    )

    print("\n==== Evaluation Summary ====")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
