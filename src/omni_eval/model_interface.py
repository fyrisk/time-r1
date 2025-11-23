"""Model interface abstractions for temporal grounding.

The project can use remote APIs or locally hosted MLLMs; both should
implement :class:`TemporalLocalizationModel` so that evaluation logic
can remain agnostic to how predictions are generated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Sequence, Tuple


TimestampPrediction = Tuple[float, float]
RawTimestamp = float | str
RawPrediction = Sequence[RawTimestamp] | TimestampPrediction


class TemporalLocalizationModel(ABC):
    """Abstract interface for models that output temporal spans.

    Subclasses should return a tuple of timestamps (start, end). Each
    timestamp can be either a float representing seconds or a string in
    ``MM:SS`` format.
    """

    @abstractmethod
    def predict_timestamps(
        self,
        video_path: str,
        question: str,
        qa_sample: Mapping[str, Any] | None = None,
    ) -> RawPrediction:
        """Run inference on a video and question.

        Args:
            video_path: Absolute path to the video file that includes
                audio (``.mp4`` as provided by the dataset).
            question: The natural language query to send to the model.
            qa_sample: Full QA payload from the dataset for additional
                context (e.g., category or reference answer). The field is
                optional so that API and local backends can ignore it.

        Returns:
            A start/end pair. The method may return a tuple/list of two
            strings (``MM:SS``), two floats (seconds), or a mix of both.
        """


def _mmss_to_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected MM:SS format but got: {value}")
    minutes = int(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds


def normalize_prediction(raw: RawPrediction) -> TimestampPrediction:
    """Convert user/model timestamps into a consistent (start, end) pair.

    The function accepts either ``(start, end)`` or ``[start, end]`` where
    each element is a ``float`` (seconds) or ``MM:SS`` string. It ensures
    ``start <= end``; if the inputs are reversed they will be swapped.
    """

    if not isinstance(raw, Iterable):
        raise TypeError("Prediction must be an iterable with two timestamps")

    raw_list = list(raw)
    if len(raw_list) != 2:
        raise ValueError(f"Prediction must contain two timestamps, got {raw_list}")

    start_raw, end_raw = raw_list

    def _to_seconds(value: RawTimestamp) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return _mmss_to_seconds(value.strip())
        raise TypeError(f"Unsupported timestamp type: {type(value)}")

    start = _to_seconds(start_raw)
    end = _to_seconds(end_raw)

    if start > end:
        start, end = end, start
    return start, end


def seconds_to_mmss(seconds: float) -> str:
    """Format seconds into ``MM:SS`` with zero padding."""

    whole_seconds = int(seconds)
    minutes = whole_seconds // 60
    remaining = whole_seconds % 60
    return f"{minutes:02d}:{remaining:02d}"


class EchoGroundTruthModel(TemporalLocalizationModel):
    """A tiny helper model for debugging the evaluation pipeline.

    The model echoes the ground-truth timestamps passed in ``qa_sample``
    so you can verify that file loading and metric computation behave as
    expected before wiring in a real MLLM backend.
    """

    def predict_timestamps(
        self,
        video_path: str,
        question: str,
        qa_sample: Mapping[str, Any] | None = None,
    ) -> RawPrediction:
        if not qa_sample or "time" not in qa_sample:
            raise ValueError("EchoGroundTruthModel requires 'time' in qa_sample")
        return qa_sample["time"]
