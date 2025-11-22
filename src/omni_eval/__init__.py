"""Utilities for evaluating audio-video temporal localization.

This package provides dataset loaders, model interfaces, metrics, and
an evaluator loop for running multimodal LLMs that return timestamp
predictions.
"""

from .data_loader import QASample, load_qa_samples
from .evaluator import EvaluationResult, TemporalGroundingEvaluator
from .metrics import compute_iou, summarize_iou
from .model_interface import TemporalLocalizationModel, normalize_prediction, seconds_to_mmss

__all__ = [
    "QASample",
    "load_qa_samples",
    "EvaluationResult",
    "TemporalGroundingEvaluator",
    "compute_iou",
    "summarize_iou",
    "TemporalLocalizationModel",
    "normalize_prediction",
    "seconds_to_mmss",
]
