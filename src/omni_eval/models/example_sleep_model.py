"""Example model that waits and returns random timestamps within 30 seconds.

This module implements the required ``build_model`` interface expected by
``scripts/run_audio_video_eval.py``. The model sleeps for three seconds to
simulate latency (e.g., from a remote API) and then returns a random
``MM:SS`` start/end pair between ``00:00`` and ``00:30``.
"""

from __future__ import annotations

import random
import time

from omni_eval.model_interface import TemporalLocalizationModel, seconds_to_mmss


class SleepyRandomTemporalModel(TemporalLocalizationModel):
    """Return random timestamps after a fixed delay."""

    def __init__(self, delay_seconds: float = 3.0, max_seconds: float = 30.0):
        self.delay_seconds = delay_seconds
        self.max_seconds = max_seconds

    def predict_timestamps(self, video_path: str, question: str, qa_sample=None):
        time.sleep(self.delay_seconds)
        start = random.uniform(0, self.max_seconds)
        end = random.uniform(start, self.max_seconds)
        return seconds_to_mmss(start), seconds_to_mmss(end)


def build_model() -> TemporalLocalizationModel:
    """Factory for the sleepy random model."""

    return SleepyRandomTemporalModel()
