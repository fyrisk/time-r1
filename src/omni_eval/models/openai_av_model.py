"""OpenAI API-backed temporal localization model for audio-video QA.

This module shows how to wire a multimodal OpenAI client into the
``TemporalLocalizationModel`` interface. It extracts video frames and the
full audio track, sends them as mixed content, and parses ``MM:SS``
timestamps from the response.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
from typing import List, Tuple

import cv2
from dotenv import load_dotenv
from openai import OpenAI

from omni_eval.model_interface import TemporalLocalizationModel


def extract_video_frames(
    path: str, seconds_per_frame: float = 1.0, max_frames: int = 20
) -> List[str]:
    """Extract JPEG frames every ``seconds_per_frame`` seconds and return base64 strings."""

    frames: List[str] = []
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        return frames

    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    while True:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        frames.append(base64.b64encode(buffer).decode("utf-8"))

        curr_frame += frames_to_skip
        if len(frames) >= max_frames:
            break

    video.release()
    return frames


def extract_full_audio_base64(path: str) -> Tuple[str, float]:
    """Extract the full audio track as 16kHz mono WAV and return base64 + duration."""

    workdir = tempfile.mkdtemp(prefix="av_full_")
    full_wav = os.path.join(workdir, "full.wav")

    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", full_wav],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            full_wav,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(probe.stdout.strip() or 0.0)

    with open(full_wav, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return b64, duration


def normalize_mmss(ts: str) -> str:
    """Ensure timestamps are zero-padded ``MM:SS``."""

    minutes, seconds = ts.split(":")
    return f"{int(minutes):02d}:{int(seconds):02d}"


def parse_mmss_pair(text: str) -> Tuple[str, str]:
    """Parse the first two MM:SS timestamps from a string, defaulting to 00:00."""

    matches = re.findall(r"\b(\d{1,2}:\d{2})\b", text)
    start = normalize_mmss(matches[0]) if len(matches) > 0 else "00:00"
    end = normalize_mmss(matches[1]) if len(matches) > 1 else start
    return start, end


class OpenAIAudioVideoTemporalModel(TemporalLocalizationModel):
    """Temporal localization model powered by an OpenAI-compatible API."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        seconds_per_frame: float = 1.0,
        max_frames: int = 20,
    ):
        self.client = client
        self.model = model
        self.seconds_per_frame = seconds_per_frame
        self.max_frames = max_frames

    def predict_timestamps(self, video_path: str, question: str, qa_sample=None):
        frames = extract_video_frames(
            video_path, seconds_per_frame=self.seconds_per_frame, max_frames=self.max_frames
        )
        audio_b64, audio_duration = extract_full_audio_base64(video_path)

        content = []
        content.append(
            {
                "type": "text",
                "text": (
                    "You are an audio-video temporal localization assistant. "
                    "Answer only with a start and end timestamp in MM:SS format."
                ),
            }
        )

        for frame_b64 in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})

        content.append({"type": "text", "text": f"[Full audio ~{audio_duration:.1f}s]"})
        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"},
            }
        )

        content.append({"type": "text", "text": f"Question: {question}\nReturn: <start_mm:ss> to <end_mm:ss>."})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        message = completion.choices[0].message.content or ""
        start, end = parse_mmss_pair(message)
        return start, end


def build_model() -> TemporalLocalizationModel:
    """Create a default OpenAIAudioVideoTemporalModel using environment variables."""

    load_dotenv()
    client = OpenAI(base_url=os.environ.get("BASE_URL"), api_key=os.environ.get("API_KEY"))
    model_name = os.environ.get("OPENAI_AV_MODEL", "gpt-4o-mini")
    return OpenAIAudioVideoTemporalModel(client=client, model=model_name)
