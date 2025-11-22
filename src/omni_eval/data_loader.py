"""Dataset loader for the omni-reason-ground QA format."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Sequence

from .model_interface import TimestampPrediction, normalize_prediction

LOGGER = logging.getLogger(__name__)


@dataclass
class QASample:
    qid: str
    question: str
    answer: str
    target: TimestampPrediction
    qa_path: Path
    video_path: Path
    metadata: Mapping[str, object]


def _generate_qid(relative_path: Path, idx: int) -> str:
    return f"{relative_path.as_posix()}#{idx}"


def _load_json(path: Path) -> Sequence[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_qa_samples(qa_root: str, video_root: str) -> Iterator[QASample]:
    """Yield QA samples with resolved video paths.

    Args:
        qa_root: Root directory of QA JSON files
            (e.g., ``/data5/fy/omni-reason-ground/qa``).
        video_root: Root directory of corresponding MP4 files
            (e.g., ``/data5/fy/data/mybenchvideo``).

    Each JSON file is expected to contain a list of QA dicts with a
    ``time`` field of ``["MM:SS", "MM:SS"]``. The function pairs each QA
    entry with the MP4 file that shares the same relative prefix.
    """

    qa_root_path = Path(qa_root).expanduser().resolve()
    video_root_path = Path(video_root).expanduser().resolve()

    json_files: List[Path] = sorted(qa_root_path.rglob("*.json"))
    for json_path in json_files:
        relative_dir = json_path.parent.relative_to(qa_root_path)
        video_stem = json_path.stem
        video_path = video_root_path / relative_dir / f"{video_stem}.mp4"

        if not video_path.exists():
            LOGGER.warning("Missing video for %s", json_path)
            continue

        try:
            qa_entries = _load_json(json_path)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {json_path}") from exc

        for idx, qa in enumerate(qa_entries):
            if "question" not in qa or "time" not in qa:
                # Skip malformed QA pairs but keep going for the rest.
                continue

            try:
                target = normalize_prediction(qa["time"])
            except Exception:
                # Ignore samples with invalid time formats.
                continue

            qid = _generate_qid(relative_dir / json_path.name, idx)
            yield QASample(
                qid=qid,
                question=str(qa["question"]),
                answer=str(qa.get("answer", "")),
                target=target,
                qa_path=json_path,
                video_path=video_path,
                metadata=qa,
            )
