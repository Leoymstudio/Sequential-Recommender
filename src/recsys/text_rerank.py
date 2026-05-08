"""Text embedding rerankers for metadata-enhanced recommendation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DEFAULT_CATEGORIES
from .data import (
    InteractionRow,
    category_csv_path,
    category_meta_path,
    iter_interactions,
    iter_meta,
    tail_history,
    write_prediction_row,
)
from .metrics import ndcg_at_k
from .pipeline import build_model
from .recommenders import HybridParams


TEXT_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._-]{1,}", re.IGNORECASE)
TEXT_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "the",
    "this",
    "that",
    "with",
    "your",
    "you",
    "our",
    "new",
    "pack",
    "set",
    "inch",
    "black",
    "white",
    "amazon",
}


@dataclass(frozen=True)
class TextRerankConfig:
    backend: str = "hashing"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 256
    batch_size: int = 128
    candidate_k: int = 50
    max_history_items: int = 3
    base_rank_weight: float = 1.0
    text_score_weight: float = 0.05
    use_meta: bool = True
    device: str = "auto"
    cache_dir: str = "text_cache"
    local_files_only: bool = False
    seed: int = 2026


class SentenceTransformerUnavailableError(RuntimeError):
    pass


class TextEmbeddingIndex:
    def __init__(self, item_ids: list[str], embeddings: np.ndarray):
        self.item_ids = item_ids
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.item_to_row = {item: idx for idx, item in enumerate(item_ids)}

    def vector_for_item(self, item: str) -> np.ndarray | None:
        idx = self.item_to_row.get(item)
        if idx is None:
            return None
        return self.embeddings[idx]

    def query_from_history(self, history: str, max_items: int) -> np.ndarray | None:
        vectors = []
        for item in tail_history(history, max_items):
            vec = self.vector_for_item(item)
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return None
        query = np.mean(np.stack(vectors, axis=0), axis=0)
        norm = float(np.linalg.norm(query))
        if norm <= 1e-12:
            return None
        return (query / norm).astype(np.float32, copy=False)


def run_text_rerank_category(
    category: str,
    data_dir: str | Path,
    output_dir: str | Path,
    splits: tuple[str, ...] = ("valid",),
    config: TextRerankConfig | None = None,
) -> dict:
    cfg = config or TextRerankConfig()
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = build_model(
        category=category,
        data_dir=data_dir,
        model_name="hybrid",
        use_meta=cfg.use_meta,
        params=HybridParams(top_k=max(cfg.candidate_k, 10), seed=cfg.seed),
    )
    item_universe = getattr(base, "item_universe")
    texts = load_item_texts(category, data_dir, item_universe)
    index, backend_info = build_text_index(category, texts, output_dir, cfg)

    split_metrics = [
        _predict_text_split(
            category=category,
            split=split,
            data_dir=data_dir,
            output_dir=output_dir,
            base=base,
            index=index,
            cfg=cfg,
        )
        for split in splits
    ]

    result = {
        "category": category,
        "model": "text_rerank",
        "config": cfg.__dict__,
        "backend": backend_info,
        "splits": split_metrics,
        "seconds_total": time.perf_counter() - started,
    }
    metrics_path = output_dir / f"{category}_text_rerank_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metrics_file"] = str(metrics_path)
    return result


def run_text_rerank_grid(
    data_dir: str | Path,
    output_dir: str | Path,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    splits: tuple[str, ...] = ("valid",),
    config: TextRerankConfig | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or TextRerankConfig()
    results = []
    for category in categories:
        result = run_text_rerank_category(
            category=category,
            data_dir=data_dir,
            output_dir=output_dir / category,
            splits=splits,
            config=cfg,
        )
        results.append(result)

    summary = _flatten_text_results(results)
    _write_text_summary(output_dir, summary)
    payload = {
        "output_dir": str(output_dir),
        "model": "text_rerank",
        "config": cfg.__dict__,
        "results": results,
        "summary": summary,
    }
    summary_path = output_dir / "text_rerank_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["summary_file"] = str(summary_path)
    return payload


def load_item_texts(category: str, data_dir: str | Path, item_universe: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    meta_path = category_meta_path(data_dir, category)
    for obj in iter_meta(meta_path):
        item = obj.get("parent_asin")
        if item not in item_universe:
            continue
        text = format_item_text(obj)
        if text:
            texts[item] = text
    for item in item_universe:
        texts.setdefault(item, item)
    return texts


def format_item_text(obj: dict) -> str:
    chunks: list[str] = []
    for key in ("title", "store", "main_category"):
        value = obj.get(key)
        if isinstance(value, str):
            chunks.append(value)
    for key in ("categories", "features", "description"):
        value = obj.get(key)
        if isinstance(value, list):
            chunks.extend(str(part) for part in value[:8])
        elif isinstance(value, str):
            chunks.append(value)
    return " ".join(chunks)[:2048]


def build_text_index(
    category: str,
    texts: dict[str, str],
    output_dir: Path,
    cfg: TextRerankConfig,
) -> tuple[TextEmbeddingIndex, dict]:
    cache_dir = output_dir / cfg.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_name(category, cfg, len(texts))
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        item_ids = [str(item) for item in data["item_ids"].tolist()]
        embeddings = data["embeddings"]
        return TextEmbeddingIndex(item_ids, embeddings), {
            "backend": cfg.backend,
            "cache": str(cache_path),
            "loaded_from_cache": True,
            "items": len(item_ids),
            "dim": int(embeddings.shape[1]),
        }

    item_ids = sorted(texts)
    corpus = [texts[item] for item in item_ids]
    if cfg.backend == "hashing":
        embeddings = encode_hashing(corpus, dim=cfg.embedding_dim)
        backend_info = {"backend": "hashing", "dim": cfg.embedding_dim}
    elif cfg.backend == "sentence-transformer":
        embeddings, backend_info = encode_sentence_transformer(corpus, cfg)
    else:
        raise ValueError(f"Unknown text backend: {cfg.backend}")

    np.savez_compressed(cache_path, item_ids=np.array(item_ids), embeddings=embeddings.astype(np.float32))
    backend_info.update(
        {
            "cache": str(cache_path),
            "loaded_from_cache": False,
            "items": len(item_ids),
            "dim": int(embeddings.shape[1]),
        }
    )
    return TextEmbeddingIndex(item_ids, embeddings), backend_info


def encode_hashing(corpus: list[str], dim: int = 256) -> np.ndarray:
    embeddings = np.zeros((len(corpus), dim), dtype=np.float32)
    for row, text in enumerate(corpus):
        counts = {}
        for token in tokenize_text(text):
            counts[token] = counts.get(token, 0) + 1.0
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            col = value % dim
            sign = 1.0 if ((value >> 8) & 1) else -1.0
            embeddings[row, col] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(embeddings[row]))
        if norm > 1e-12:
            embeddings[row] /= norm
    return embeddings


def encode_sentence_transformer(corpus: list[str], cfg: TextRerankConfig) -> tuple[np.ndarray, dict]:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise SentenceTransformerUnavailableError(
            "sentence-transformers is not installed. Install it before using --text-backend sentence-transformer."
        ) from exc

    device = None
    if cfg.device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ModuleNotFoundError:
            device = "cpu"
    else:
        device = cfg.device
    model = SentenceTransformer(
        cfg.model_name,
        device=device,
        local_files_only=cfg.local_files_only,
    )
    embeddings = model.encode(
        corpus,
        batch_size=cfg.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32, copy=False), {
        "backend": "sentence-transformer",
        "model_name": cfg.model_name,
        "device": device,
    }


def tokenize_text(text: str) -> list[str]:
    tokens = []
    for raw in TEXT_TOKEN_RE.findall(text.lower()):
        token = raw.strip("._-+")
        if len(token) < 2 or token in TEXT_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _predict_text_split(
    category: str,
    split: str,
    data_dir: str | Path,
    output_dir: Path,
    base,
    index: TextEmbeddingIndex,
    cfg: TextRerankConfig,
) -> dict:
    split_path = category_csv_path(data_dir, category, split)
    pred_path = output_dir / f"{category}_{split}_text_rerank_pred.jsonl"
    rows = 0
    hits = 0
    total = 0.0
    started = time.perf_counter()
    with pred_path.open("w", encoding="utf-8") as f:
        for row in iter_interactions(split_path):
            candidates = base.recommend(row.history, k=cfg.candidate_k)
            predictions = rerank_text_candidates(row, candidates, index, cfg)
            score = ndcg_at_k(predictions, row.parent_asin, k=10)
            total += score
            hits += int(score > 0.0)
            rows += 1
            write_prediction_row(f, row.user_id, predictions, row.parent_asin)

    return {
        "category": category,
        "split": split,
        "rows": rows,
        "hit@10": hits / rows if rows else 0.0,
        "ndcg@10": total / rows if rows else 0.0,
        "prediction_file": str(pred_path),
        "seconds": time.perf_counter() - started,
    }


def rerank_text_candidates(
    row: InteractionRow,
    candidates: list[str],
    index: TextEmbeddingIndex,
    cfg: TextRerankConfig,
) -> list[str]:
    query = index.query_from_history(row.history, cfg.max_history_items)
    if query is None:
        return candidates[:10]

    scored = []
    for rank, item in enumerate(candidates, start=1):
        vec = index.vector_for_item(item)
        text_score = float(np.dot(query, vec)) if vec is not None else 0.0
        base_score = 1.0 / math.log2(rank + 1)
        final_score = cfg.base_rank_weight * base_score + cfg.text_score_weight * text_score
        scored.append((item, final_score))
    ranked = sorted(scored, key=lambda kv: (-kv[1], kv[0]))
    return [item for item, _ in ranked[:10]]


def _flatten_text_results(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        backend = result.get("backend", {})
        for split in result.get("splits", []):
            rows.append(
                {
                    "category": result["category"],
                    "split": split["split"],
                    "backend": backend.get("backend", ""),
                    "model_name": backend.get("model_name", ""),
                    "device": backend.get("device", ""),
                    "rows": split["rows"],
                    "hit@10": split["hit@10"],
                    "ndcg@10": split["ndcg@10"],
                    "seconds_total": result.get("seconds_total", 0.0),
                    "prediction_file": split["prediction_file"],
                }
            )
    return rows


def _write_text_summary(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path = output_dir / "text_rerank_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "text_rerank_summary.md"
    headers = ["category", "split", "backend", "device", "hit@10", "ndcg@10", "seconds_total"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {category} | {split} | {backend} | {device} | {hit:.6f} | {ndcg:.6f} | {seconds:.2f} |".format(
                category=row["category"],
                split=row["split"],
                backend=row["backend"],
                device=row["device"],
                hit=row["hit@10"],
                ndcg=row["ndcg@10"],
                seconds=row["seconds_total"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cache_name(category: str, cfg: TextRerankConfig, item_count: int) -> str:
    model = cfg.model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{category}_{cfg.backend}_{model}_{cfg.embedding_dim}_{item_count}.npz"
