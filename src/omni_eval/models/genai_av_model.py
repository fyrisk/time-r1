"""GenAI (Gemini) SDK-backed temporal localization model for MP4 inputs.

This module adapts the reference multimodal snippet to the
``TemporalLocalizationModel`` interface. It feeds the raw MP4 bytes to a
Gemini endpoint and parses ``MM:SS`` spans from the text response.
"""

from __future__ import annotations

import os
import re
from typing import Tuple

from dotenv import load_dotenv
import genai
from genai import types

from omni_eval.model_interface import TemporalLocalizationModel


def normalize_mmss(ts: str) -> str:
    """Zero-pad timestamps expressed as ``M:SS`` or ``MM:SS``."""

    minutes, seconds = ts.split(":")
    return f"{int(minutes):02d}:{int(seconds):02d}"


def parse_mmss_pair(text: str) -> Tuple[str, str]:
    """Parse the first two ``MM:SS`` timestamps from a string."""

    matches = re.findall(r"\b(\d{1,2}:\d{2})\b", text)
    start = normalize_mmss(matches[0]) if len(matches) > 0 else "00:00"
    end = normalize_mmss(matches[1]) if len(matches) > 1 else start
    return start, end


class GenAIAudioVideoTemporalModel(TemporalLocalizationModel):
    """Temporal localization model powered by the GenAI (Gemini) SDK."""

    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model = model

    def predict_timestamps(self, video_path: str, question: str, qa_sample=None):
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        prompt = (
            "You are an audio-video temporal localization assistant. "
            "Given the video and the question, reply only with a start and end "
            "timestamp in MM:SS format as '<start> to <end>'."
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")
                    ),
                    types.Part.from_text(f"{prompt}\nQuestion: {question}"),
                ],
            )
        ]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
        )
        text = getattr(response, "text", "") or ""
        start, end = parse_mmss_pair(text)
        return start, end


def build_model() -> TemporalLocalizationModel:
    """Instantiate a GenAI-backed model using environment variables.

    Expected environment variables:
    - ``GEMINI_API_KEY``: API key for the GenAI client.
    - ``GENAI_AV_MODEL`` (optional): model name, defaults to ``gemini-2.5-pro``.
    """

    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model_name = os.environ.get("GENAI_AV_MODEL", "gemini-2.5-pro")
    return GenAIAudioVideoTemporalModel(client=client, model=model_name)
