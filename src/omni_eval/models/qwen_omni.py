"""Local Qwen Omni audio-video temporal localization example.

This module wraps a locally hosted ``Qwen2.5-Omni`` model so it can be used
with the evaluation CLI. It follows the same ``build_model`` contract as the
other examples while reusing the official processor utilities to prepare
multimodal inputs that include both audio and video.
"""

from __future__ import annotations

import os
import re
from typing import Tuple

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from omni_eval.model_interface import TemporalLocalizationModel
from qwen_omni_utils import process_mm_info


def _parse_mmss_pair(text: str) -> Tuple[str, str]:
    """Parse two ``MM:SS`` timestamps from model output, defaulting to 00:00."""

    matches = re.findall(r"\b(\d{1,2}:\d{2})\b", text)
    start = matches[0] if len(matches) > 0 else "00:00"
    end = matches[1] if len(matches) > 1 else start

    def _pad(ts: str) -> str:
        minutes, seconds = ts.split(":")
        return f"{int(minutes):02d}:{int(seconds):02d}"

    return _pad(start), _pad(end)


class QwenOmniTemporalModel(TemporalLocalizationModel):
    """Temporal localization powered by a locally hosted Qwen Omni model."""

    def __init__(self, model_path: str | None = None, system_prompt: str | None = None):
        self.model_path = model_path or os.environ.get("QWEN_OMNI_PATH", "/data5/fy/models/cache/Qwen/Qwen2.5-Omni-7B")
        self.system_prompt = system_prompt or "You are an audio-video temporal localization assistant."

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path, local_files_only=True)

    def predict_timestamps(self, video_path: str, question: str, qa_sample=None):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        output = self.model.generate(**inputs, use_audio_in_video=True, return_audio=False)
        decoded = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        message = decoded[0] if decoded else ""
        start, end = _parse_mmss_pair(message)
        return start, end


def build_model() -> TemporalLocalizationModel:
    """Factory for the Qwen Omni temporal localization backend."""

    return QwenOmniTemporalModel()
