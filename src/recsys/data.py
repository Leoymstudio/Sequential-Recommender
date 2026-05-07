"""Data loading and prediction writing helpers."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class InteractionRow:
    user_id: str
    parent_asin: str
    rating: float
    timestamp: int
    history: str


def category_csv_path(data_dir: str | Path, category: str, split: str) -> Path:
    return Path(data_dir) / f"{category}.{split}.csv.gz"


def category_meta_path(data_dir: str | Path, category: str) -> Path:
    return Path(data_dir) / f"meta_{category}.jsonl.gz"


def iter_interactions(path: str | Path) -> Iterator[InteractionRow]:
    """Yield interaction rows from one gzipped CSV split."""

    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield InteractionRow(
                user_id=row["user_id"],
                parent_asin=row["parent_asin"],
                rating=float(row.get("rating") or 0.0),
                timestamp=int(float(row.get("timestamp") or 0)),
                history=row.get("history") or "",
            )


def iter_meta(path: str | Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def split_history(history: str) -> list[str]:
    return history.split() if history else []


def tail_history(history: str, max_items: int) -> list[str]:
    """Return at most max_items right-most history items without full allocation."""

    if not history or max_items <= 0:
        return []
    parts = history.rsplit(" ", max_items)
    if len(parts) > max_items:
        return parts[-max_items:]
    return parts


def write_prediction_row(f, user_id: str, predictions: list[str], ground_truth: str) -> None:
    payload = {
        "user_id": user_id,
        "predictions": predictions,
        "ground_truth": ground_truth,
    }
    f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
