"""Evaluation metrics for single-ground-truth top-k recommendation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence


def ndcg_at_k(predictions: Sequence[str], ground_truth: str, k: int = 10) -> float:
    """Return NDCG@k for one relevant item.

    PRD rule: IDCG@10 is 1.0. If the ground-truth item appears at one-based
    rank i in the top-k list, score is 1 / log2(i + 1), otherwise 0.
    """

    if not ground_truth:
        return 0.0
    for rank, item in enumerate(predictions[:k], start=1):
        if item == ground_truth:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def mean_ndcg_at_k(rows: Iterable[tuple[Sequence[str], str]], k: int = 10) -> float:
    total = 0.0
    count = 0
    for predictions, ground_truth in rows:
        total += ndcg_at_k(predictions, ground_truth, k=k)
        count += 1
    return total / count if count else 0.0


def evaluate_prediction_file(path: str | Path, k: int = 10) -> dict[str, float | int]:
    path = Path(path)
    total = 0.0
    count = 0
    hits = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            score = ndcg_at_k(row.get("predictions", []), row.get("ground_truth", ""), k=k)
            total += score
            count += 1
            if score > 0.0:
                hits += 1
    return {
        "rows": count,
        f"hit@{k}": hits / count if count else 0.0,
        f"ndcg@{k}": total / count if count else 0.0,
    }
