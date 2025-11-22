"""Evaluation loop for audio-video temporal localization."""

from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from .data_loader import QASample, load_qa_samples
from .metrics import compute_iou, summarize_iou
from .model_interface import (
    EchoGroundTruthModel,
    RawPrediction,
    TemporalLocalizationModel,
    normalize_prediction,
    seconds_to_mmss,
)


@dataclass
class EvaluationResult:
    qid: str
    pred: tuple[float, float]
    target: tuple[float, float]
    iou: float
    latency: float
    qa_path: Path
    video_path: Path
    raw_output: RawPrediction


class TemporalGroundingEvaluator:
    def __init__(self, thresholds: Sequence[float] = (0.3, 0.5, 0.7)):
        self.thresholds = thresholds

    def evaluate(
        self,
        model: TemporalLocalizationModel,
        samples: Iterable[QASample],
    ) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        for sample in samples:
            start_time = time.perf_counter()
            raw_pred = model.predict_timestamps(
                str(sample.video_path), sample.question, qa_sample=sample.metadata
            )
            latency = time.perf_counter() - start_time

            pred = normalize_prediction(raw_pred)
            iou = compute_iou(pred, sample.target)
            results.append(
                EvaluationResult(
                    qid=sample.qid,
                    pred=pred,
                    target=sample.target,
                    iou=iou,
                    latency=latency,
                    qa_path=sample.qa_path,
                    video_path=sample.video_path,
                    raw_output=raw_pred,
                )
            )
        return results

    def save_jsonl(self, results: Iterable[EvaluationResult], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for res in results:
                f.write(
                    json.dumps(
                        {
                            "qid": res.qid,
                            "pred": [seconds_to_mmss(res.pred[0]), seconds_to_mmss(res.pred[1])],
                            "pred_seconds": res.pred,
                            "target": [seconds_to_mmss(res.target[0]), seconds_to_mmss(res.target[1])],
                            "target_seconds": res.target,
                            "iou": res.iou,
                            "latency": res.latency,
                            "qa_path": str(res.qa_path),
                            "video_path": str(res.video_path),
                            "raw_output": res.raw_output,
                        }
                    )
                    + "\n"
                )

    def summarize(self, results: Iterable[EvaluationResult]) -> Mapping[str, float]:
        ious = [res.iou for res in results]
        return summarize_iou(ious, thresholds=self.thresholds)


def _load_user_model(module_path: Path) -> TemporalLocalizationModel:
    spec = importlib.util.spec_from_file_location("user_model", module_path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "build_model"):
        raise AttributeError(
            "The provided module must define a build_model() function returning TemporalLocalizationModel"
        )
    model = module.build_model()
    if not isinstance(model, TemporalLocalizationModel):
        raise TypeError("build_model() must return a TemporalLocalizationModel instance")
    return model


def build_model(model_module: str | None) -> TemporalLocalizationModel:
    if model_module:
        return _load_user_model(Path(model_module))
    return EchoGroundTruthModel()


def run_evaluation(
    qa_root: str,
    video_root: str,
    output_path: str,
    model_module: str | None = None,
    thresholds: Sequence[float] = (0.3, 0.5, 0.7),
) -> Mapping[str, float]:
    model = build_model(model_module)
    samples = list(load_qa_samples(qa_root, video_root))

    evaluator = TemporalGroundingEvaluator(thresholds=thresholds)
    results = evaluator.evaluate(model, samples)
    evaluator.save_jsonl(results, Path(output_path))
    return evaluator.summarize(results)
