#!/usr/bin/env python3
"""Create a filtered QA subset by category prefix and sample count.

Usage example::

    python scripts/sample_qa_subset.py \
        --qa-root /data5/fy/omni-reason-ground/qa \
        --output-root /data5/fy/omni-reason-ground/small_qa \
        --target-codes 1.1 1.2 2.1 \
        --k 3 \
        --seed 123

The script keeps the original relative directory structure and JSON
filenames when writing the sampled subset under ``--output-root``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a QA subset by category code")
    parser.add_argument(
        "--qa-root",
        type=str,
        required=True,
        help="Root directory containing full QA JSON files",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Where to write the sampled QA files while preserving subdirectories",
    )
    parser.add_argument(
        "--target-codes",
        type=str,
        nargs="+",
        required=True,
        help="Category code prefixes to filter on (e.g., 1.1 1.2 2.1)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of QA entries to sample per category code",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling determinism",
    )
    return parser.parse_args()


def extract_category_prefix(chosen_category: str) -> str:
    """Return the numeric prefix from a category string.

    Examples:
        "1.1. Causal Reasoning" -> "1.1"
        "2.1 Something" -> "2.1"
    """

    if not chosen_category:
        return ""
    first_token = chosen_category.split()[0]
    return first_token.rstrip(".")


def load_candidates(
    qa_root: Path, target_codes: Sequence[str]
) -> Dict[str, List[tuple[Path, int, dict]]]:
    """Collect QA entries grouped by target code prefix."""

    candidates: Dict[str, List[tuple[Path, int, dict]]] = {
        code: [] for code in target_codes
    }
    for json_path in qa_root.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to read {json_path}: {exc}") from exc

        for idx, item in enumerate(data):
            cat_prefix = extract_category_prefix(str(item.get("chosen_category", "")))
            for code in target_codes:
                if cat_prefix.startswith(code):
                    candidates[code].append((json_path, idx, item))
    return candidates


def sample_indices(
    rng: random.Random, items: Sequence[tuple[Path, int, dict]], k: int
) -> List[tuple[Path, int, dict]]:
    if not items:
        return []
    if k >= len(items):
        return list(items)
    return rng.sample(items, k)


def write_subset(
    qa_root: Path,
    output_root: Path,
    selected: Iterable[tuple[Path, int, dict]],
) -> Dict[Path, int]:
    """Write selected QA entries preserving structure.

    Returns a mapping of source JSON path to number of entries written.
    """

    per_file_indices: Dict[Path, set[int]] = defaultdict(set)

    for src_path, idx, item in selected:
        per_file_indices[src_path].add(idx)

    counts: Dict[Path, int] = {}
    for src_path, indices in per_file_indices.items():
        data = json.loads(src_path.read_text(encoding="utf-8"))
        keep_set = set(indices)
        trimmed = [entry for i, entry in enumerate(data) if i in keep_set]
        if not trimmed:
            continue

        rel_path = src_path.relative_to(qa_root)
        dest_path = output_root / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(
            json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        counts[rel_path] = len(trimmed)
    return counts


def main() -> None:
    args = parse_args()
    qa_root = Path(args.qa_root)
    output_root = Path(args.output_root)
    target_codes = [code.rstrip(".") for code in args.target_codes]

    rng = random.Random(args.seed)
    candidates = load_candidates(qa_root, target_codes)

    selected: list[tuple[Path, int, dict]] = []
    for code in target_codes:
        items = candidates.get(code, [])
        sampled = sample_indices(rng, items, args.k)
        selected.extend(sampled)
        print(f"Code {code}: selected {len(sampled)} / {len(items)} available")

    counts = write_subset(qa_root, output_root, selected)

    print("\nWrote subset files:")
    for rel_path, num in sorted(counts.items()):
        print(f"  {rel_path} -> {num} qa entries")

    print("\nDone.")


if __name__ == "__main__":
    main()
