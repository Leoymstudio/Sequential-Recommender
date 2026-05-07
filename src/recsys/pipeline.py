"""Training, evaluation, and prediction pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .data import category_csv_path, category_meta_path, iter_interactions, write_prediction_row
from .metrics import ndcg_at_k
from .recommenders import HybridParams, HybridSequentialRecommender, PopularityRecommender


def build_model(
    category: str,
    data_dir: str | Path,
    model_name: str = "hybrid",
    use_meta: bool = False,
    params: HybridParams | None = None,
) -> HybridSequentialRecommender | PopularityRecommender:
    train_path = category_csv_path(data_dir, category, "train")
    if model_name == "popularity":
        model = PopularityRecommender(top_k=(params.top_k if params else 10))
        model.fit(iter_interactions(train_path))
        return model
    if model_name != "hybrid":
        raise ValueError(f"Unknown model: {model_name}")

    model = HybridSequentialRecommender(params=params)
    model.fit(iter_interactions(train_path))
    if use_meta:
        meta_path = category_meta_path(data_dir, category)
        if meta_path.exists():
            model.fit_metadata(meta_path)
    return model


def predict_split(
    model,
    category: str,
    split: str,
    data_dir: str | Path,
    output_dir: str | Path,
    k: int = 10,
) -> dict[str, float | int | str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = category_csv_path(data_dir, category, split)
    pred_path = output_dir / f"{category}_{split}_pred.jsonl"

    total = 0.0
    hits = 0
    rows = 0
    started = time.perf_counter()
    with pred_path.open("w", encoding="utf-8") as f:
        for row in iter_interactions(split_path):
            predictions = model.recommend(row.history, k=k)
            score = ndcg_at_k(predictions, row.parent_asin, k=k)
            total += score
            hits += int(score > 0.0)
            rows += 1
            write_prediction_row(f, row.user_id, predictions, row.parent_asin)

    elapsed = time.perf_counter() - started
    return {
        "category": category,
        "split": split,
        "rows": rows,
        f"hit@{k}": hits / rows if rows else 0.0,
        f"ndcg@{k}": total / rows if rows else 0.0,
        "prediction_file": str(pred_path),
        "seconds": elapsed,
    }


def run_category(
    category: str,
    data_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "hybrid",
    use_meta: bool = False,
    k: int = 10,
    seed: int = 2026,
    splits: tuple[str, ...] = ("valid", "test"),
) -> dict:
    started = time.perf_counter()
    params = HybridParams(top_k=k, seed=seed)
    model = build_model(category, data_dir, model_name=model_name, use_meta=use_meta, params=params)

    split_metrics = [
        predict_split(model, category, split, data_dir, output_dir, k=k)
        for split in splits
    ]
    result = {
        "category": category,
        "model": model_name,
        "use_meta": use_meta,
        "seed": seed,
        "top_k": k,
        "splits": split_metrics,
        "seconds_total": time.perf_counter() - started,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{category}_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metrics_file"] = str(metrics_path)
    return result
