"""Evaluation loop for audio-video temporal localization."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from .data_loader import QASample, load_qa_samples, sample_dedup_key
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
    sample_key: str
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
        output_path: Path,
        processed_keys: set[str] | None = None,
        concurrent: bool = False,
        max_workers: int | None = None,
    ) -> List[EvaluationResult]:
        """Run evaluation and stream results to a JSONL file.

        Args:
            model: Temporal localization backend.
            samples: Iterable QA samples to process.
            output_path: JSONL file for incremental writes (append mode).
            processed_keys: Optional set of deduplication keys to skip.
            concurrent: Whether to run inference concurrently (default False).
            max_workers: Optional worker cap when ``concurrent`` is True.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_keys = processed_keys or set()
        writer_lock = threading.Lock()
        results: List[EvaluationResult] = []

        def _serialize(res: EvaluationResult) -> str:
            return json.dumps(
                {
                    "qid": res.qid,
                    "sample_key": res.sample_key,
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

        def _process(sample: QASample) -> EvaluationResult | None:
            key = sample_dedup_key(sample)
            if key in processed_keys or sample.qid in processed_keys:
                return None

            start_time = time.perf_counter()
            raw_pred = model.predict_timestamps(
                str(sample.video_path), sample.question, qa_sample=sample.metadata
            )
            latency = time.perf_counter() - start_time

            pred = normalize_prediction(raw_pred)
            iou = compute_iou(pred, sample.target)
            res = EvaluationResult(
                qid=sample.qid,
                sample_key=key,
                pred=pred,
                target=sample.target,
                iou=iou,
                latency=latency,
                qa_path=sample.qa_path,
                video_path=sample.video_path,
                raw_output=raw_pred,
            )

            line = _serialize(res)
            with writer_lock:
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            return res

        if concurrent:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for res in executor.map(_process, samples):
                    if res is not None:
                        results.append(res)
        else:
            for sample in samples:
                res = _process(sample)
                if res is not None:
                    results.append(res)
        return results

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
    concurrent: bool = False,
    max_workers: int | None = None,
) -> Mapping[str, float]:
    model = build_model(model_module)
    samples = list(load_qa_samples(qa_root, video_root))

    output_path_obj = Path(output_path)
    existing_results, processed_keys = load_existing_results(output_path_obj)

    evaluator = TemporalGroundingEvaluator(thresholds=thresholds)
    new_results = evaluator.evaluate(
        model,
        samples,
        output_path_obj,
        processed_keys=processed_keys,
        concurrent=concurrent,
        max_workers=max_workers,
    )
    all_results = existing_results + new_results
    return evaluator.summarize(all_results)


def load_existing_results(output_path: Path) -> tuple[List[EvaluationResult], set[str]]:
    """Load previously saved results to support resume logic."""

    if not output_path.exists():
        return [], set()

    results: List[EvaluationResult] = []
    processed_keys: set[str] = set()

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("sample_key") or record.get("qid")
            if key:
                processed_keys.add(str(key))
            if "qid" in record:
                processed_keys.add(str(record["qid"]))
            try:
                pred = tuple(record.get("pred_seconds", record.get("pred", (0, 0))))
                target = tuple(record.get("target_seconds", record.get("target", (0, 0))))
            except Exception:
                continue

            results.append(
                EvaluationResult(
                    qid=str(record.get("qid", "")),
                    sample_key=str(key) if key else "",
                    pred=(float(pred[0]), float(pred[1])),
                    target=(float(target[0]), float(target[1])),
                    iou=float(record.get("iou", 0.0)),
                    latency=float(record.get("latency", 0.0)),
                    qa_path=Path(record.get("qa_path", "")),
                    video_path=Path(record.get("video_path", "")),
                    raw_output=record.get("raw_output", {}),
                )
            )

    return results, processed_keys
