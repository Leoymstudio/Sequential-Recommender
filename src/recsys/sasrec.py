"""Optional PyTorch SASRec reranker.

This module is intentionally import-safe when torch is not installed. The
standard-library baseline remains the default path; SASRec is enabled by the
`sasrec-rerank` CLI command after installing PyTorch in the conda environment.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .data import InteractionRow, category_csv_path, iter_interactions, tail_history, write_prediction_row
from .metrics import ndcg_at_k
from .pipeline import build_model
from .recommenders import HybridParams

try:  # pragma: no cover - exercised only in advanced environments.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class SasRecConfig:
    max_len: int = 50
    hidden_size: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 1
    batch_size: int = 256
    negatives: int = 64
    lr: float = 0.001
    candidate_k: int = 100
    max_train_rows: int = 0
    seed: int = 2026
    device: str = "auto"


class TorchUnavailableError(RuntimeError):
    pass


def require_torch():
    if torch is None or nn is None:
        raise TorchUnavailableError(
            "PyTorch is not installed. Install it in the recom environment before running sasrec-rerank."
        )
    return torch, nn


class ItemMapper:
    def __init__(self, items: Iterable[str]):
        ordered = sorted(set(items))
        self.item_to_id = {item: idx for idx, item in enumerate(ordered, start=1)}
        self.id_to_item = {idx: item for item, idx in self.item_to_id.items()}

    @property
    def size(self) -> int:
        return len(self.item_to_id) + 1

    def encode_history(self, history: str, max_len: int) -> list[int]:
        ids = [self.item_to_id[item] for item in tail_history(history, max_len) if item in self.item_to_id]
        return ids[-max_len:]

    def encode_item(self, item: str) -> int:
        return self.item_to_id.get(item, 0)

    def encode_candidates(self, items: list[str]) -> list[int]:
        return [self.item_to_id.get(item, 0) for item in items]


if nn is not None:

    class SASRecModel(nn.Module):
        def __init__(self, num_items: int, config: SasRecConfig):
            super().__init__()
            self.config = config
            self.item_embedding = nn.Embedding(num_items, config.hidden_size, padding_idx=0)
            self.position_embedding = nn.Embedding(config.max_len, config.hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
            self.dropout = nn.Dropout(config.dropout)
            self.layer_norm = nn.LayerNorm(config.hidden_size)

        def encode(self, seq: "torch.Tensor") -> "torch.Tensor":
            batch_size, seq_len = seq.shape
            positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, seq_len)
            x = self.item_embedding(seq) + self.position_embedding(positions)
            x = self.layer_norm(self.dropout(x))
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool),
                diagonal=1,
            )
            padding_mask = seq.eq(0)
            encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
            lengths = seq.ne(0).sum(dim=1).clamp(min=1) - 1
            return encoded[torch.arange(batch_size, device=seq.device), lengths]

        def score(self, seq: "torch.Tensor", candidates: "torch.Tensor") -> "torch.Tensor":
            hidden = self.encode(seq)
            cand_emb = self.item_embedding(candidates)
            return torch.einsum("bh,bch->bc", hidden, cand_emb)

else:

    class SASRecModel:  # pragma: no cover
        pass


def run_sasrec_rerank_category(
    category: str,
    data_dir: str | Path,
    output_dir: str | Path,
    use_meta: bool = True,
    splits: tuple[str, ...] = ("valid",),
    config: SasRecConfig | None = None,
) -> dict:
    torch_mod, _ = require_torch()
    cfg = config or SasRecConfig()
    random.seed(cfg.seed)
    torch_mod.manual_seed(cfg.seed)

    device = _resolve_device(cfg.device)
    started = time.perf_counter()
    base = build_model(
        category=category,
        data_dir=data_dir,
        model_name="hybrid",
        use_meta=use_meta,
        params=HybridParams(top_k=max(cfg.candidate_k, 10), seed=cfg.seed),
    )

    mapper = ItemMapper(getattr(base, "item_universe"))
    model = SASRecModel(mapper.size, cfg).to(device)
    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = torch_mod.nn.BCEWithLogitsLoss()

    train_path = category_csv_path(data_dir, category, "train")
    train_metrics = _train_sasrec(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_path=train_path,
        mapper=mapper,
        cfg=cfg,
        device=device,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metrics = [
        _predict_sasrec_split(
            model=model,
            base=base,
            mapper=mapper,
            category=category,
            split=split,
            data_dir=data_dir,
            output_dir=output_dir,
            cfg=cfg,
            device=device,
        )
        for split in splits
    ]

    result = {
        "category": category,
        "model": "sasrec_rerank",
        "use_meta": use_meta,
        "seed": cfg.seed,
        "resolved_device": str(device),
        "config": cfg.__dict__,
        "train": train_metrics,
        "splits": split_metrics,
        "seconds_total": time.perf_counter() - started,
    }
    metrics_path = output_dir / f"{category}_sasrec_rerank_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metrics_file"] = str(metrics_path)
    return result


def _train_sasrec(model, optimizer, loss_fn, train_path: Path, mapper: ItemMapper, cfg: SasRecConfig, device) -> dict:
    torch_mod, _ = require_torch()
    losses: list[float] = []
    rows_seen = 0
    model.train()

    for epoch in range(cfg.epochs):
        batch_seq: list[list[int]] = []
        batch_pos: list[int] = []
        for row in iter_interactions(train_path):
            pos = mapper.encode_item(row.parent_asin)
            seq = mapper.encode_history(row.history, cfg.max_len)
            if not pos or not seq:
                continue
            batch_seq.append(seq)
            batch_pos.append(pos)
            rows_seen += 1
            if len(batch_seq) >= cfg.batch_size:
                losses.append(_train_batch(model, optimizer, loss_fn, batch_seq, batch_pos, mapper, cfg, device))
                batch_seq, batch_pos = [], []
            if cfg.max_train_rows and rows_seen >= cfg.max_train_rows:
                break
        if batch_seq:
            losses.append(_train_batch(model, optimizer, loss_fn, batch_seq, batch_pos, mapper, cfg, device))

    return {
        "rows_seen": rows_seen,
        "epochs": cfg.epochs,
        "loss_last": losses[-1] if losses else 0.0,
        "loss_mean": sum(losses) / len(losses) if losses else 0.0,
    }


def _train_batch(model, optimizer, loss_fn, seqs, positives, mapper: ItemMapper, cfg: SasRecConfig, device) -> float:
    torch_mod, _ = require_torch()
    seq_tensor = _pad_sequences(seqs, cfg.max_len, device)
    candidates = []
    labels = []
    for pos in positives:
        negs = _sample_negatives(mapper.size, pos, cfg.negatives)
        candidates.append([pos] + negs)
        labels.append([1.0] + [0.0] * len(negs))
    cand_tensor = torch_mod.tensor(candidates, dtype=torch_mod.long, device=device)
    label_tensor = torch_mod.tensor(labels, dtype=torch_mod.float32, device=device)

    optimizer.zero_grad(set_to_none=True)
    logits = model.score(seq_tensor, cand_tensor)
    loss = loss_fn(logits, label_tensor)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def _predict_sasrec_split(model, base, mapper: ItemMapper, category: str, split: str, data_dir, output_dir, cfg, device) -> dict:
    split_path = category_csv_path(data_dir, category, split)
    pred_path = Path(output_dir) / f"{category}_{split}_sasrec_rerank_pred.jsonl"
    rows = 0
    hits = 0
    total = 0.0
    started = time.perf_counter()
    model.eval()

    with pred_path.open("w", encoding="utf-8") as f:
        pending: list[tuple[InteractionRow, list[str]]] = []
        for row in iter_interactions(split_path):
            base_candidates = base.recommend(row.history, k=cfg.candidate_k)
            pending.append((row, base_candidates))
            if len(pending) >= cfg.batch_size:
                batch_total, batch_hits, batch_rows = _write_prediction_batch(
                    f, model, mapper, pending, cfg, device
                )
                total += batch_total
                hits += batch_hits
                rows += batch_rows
                pending = []
        if pending:
            batch_total, batch_hits, batch_rows = _write_prediction_batch(
                f, model, mapper, pending, cfg, device
            )
            total += batch_total
            hits += batch_hits
            rows += batch_rows

    return {
        "category": category,
        "split": split,
        "rows": rows,
        "hit@10": hits / rows if rows else 0.0,
        "ndcg@10": total / rows if rows else 0.0,
        "prediction_file": str(pred_path),
        "seconds": time.perf_counter() - started,
    }


def _write_prediction_batch(f, model, mapper: ItemMapper, batch, cfg: SasRecConfig, device) -> tuple[float, int, int]:
    predictions_batch = _rerank_candidate_batch(model, mapper, batch, cfg, device)
    total = 0.0
    hits = 0
    rows = 0
    for (row, _), predictions in zip(batch, predictions_batch):
        score = ndcg_at_k(predictions, row.parent_asin, k=10)
        total += score
        hits += int(score > 0.0)
        rows += 1
        write_prediction_row(f, row.user_id, predictions, row.parent_asin)
    return total, hits, rows


def _rerank_candidate_batch(model, mapper: ItemMapper, batch, cfg: SasRecConfig, device) -> list[list[str]]:
    torch_mod, _ = require_torch()
    seqs: list[list[int]] = []
    candidate_texts: list[list[str]] = []
    candidate_ids: list[list[int]] = []
    fallback_indexes: list[int] = []

    for idx, (row, candidates) in enumerate(batch):
        seq_ids = mapper.encode_history(row.history, cfg.max_len)
        encoded = [(item, mapper.encode_item(item)) for item in candidates]
        encoded = [(item, item_id) for item, item_id in encoded if item_id]
        if not seq_ids or not encoded:
            fallback_indexes.append(idx)
            seqs.append([0])
            candidate_texts.append(candidates[:10])
            candidate_ids.append([0])
            continue
        seqs.append(seq_ids)
        candidate_texts.append([item for item, _ in encoded])
        candidate_ids.append([item_id for _, item_id in encoded])

    max_candidates = max(len(ids) for ids in candidate_ids)
    padded_candidates = [ids + [0] * (max_candidates - len(ids)) for ids in candidate_ids]
    seq_tensor = _pad_sequences(seqs, cfg.max_len, device)
    cand_tensor = torch_mod.tensor(padded_candidates, dtype=torch_mod.long, device=device)
    with torch_mod.no_grad():
        scores_batch = model.score(seq_tensor, cand_tensor).detach().cpu().tolist()

    fallback_set = set(fallback_indexes)
    results: list[list[str]] = []
    for idx, ((_, original_candidates), texts, ids, scores) in enumerate(
        zip(batch, candidate_texts, candidate_ids, scores_batch)
    ):
        if idx in fallback_set:
            results.append(original_candidates[:10])
            continue
        scored = [
            (item, score)
            for item, item_id, score in zip(texts, ids, scores)
            if item_id
        ]
        ranked = sorted(scored, key=lambda kv: (-kv[1], kv[0]))
        final = [item for item, _ in ranked[:10]]
        if len(final) < 10:
            seen = set(final)
            for item in original_candidates:
                if item in seen:
                    continue
                final.append(item)
                seen.add(item)
                if len(final) >= 10:
                    break
        results.append(final[:10])
    return results


def _rerank_candidates(model, mapper: ItemMapper, row: InteractionRow, candidates: list[str], cfg: SasRecConfig, device) -> list[str]:
    torch_mod, _ = require_torch()
    seq_ids = mapper.encode_history(row.history, cfg.max_len)
    encoded_candidates = [(item, mapper.encode_item(item)) for item in candidates]
    encoded_candidates = [(item, idx) for item, idx in encoded_candidates if idx]
    if not seq_ids or not encoded_candidates:
        return candidates[:10]

    seq_tensor = _pad_sequences([seq_ids], cfg.max_len, device)
    cand_ids = torch_mod.tensor([[idx for _, idx in encoded_candidates]], dtype=torch_mod.long, device=device)
    with torch_mod.no_grad():
        scores = model.score(seq_tensor, cand_ids).squeeze(0).detach().cpu().tolist()

    ranked = sorted(
        zip([item for item, _ in encoded_candidates], scores),
        key=lambda kv: (-kv[1], kv[0]),
    )
    final = [item for item, _ in ranked[:10]]
    if len(final) < 10:
        seen = set(final)
        for item in candidates:
            if item in seen:
                continue
            final.append(item)
            seen.add(item)
            if len(final) >= 10:
                break
    return final[:10]


def _pad_sequences(seqs: list[list[int]], max_len: int, device):
    torch_mod, _ = require_torch()
    padded = []
    for seq in seqs:
        trimmed = seq[-max_len:]
        padded.append([0] * (max_len - len(trimmed)) + trimmed)
    return torch_mod.tensor(padded, dtype=torch_mod.long, device=device)


def _sample_negatives(num_items: int, positive: int, n: int) -> list[int]:
    negs: list[int] = []
    while len(negs) < n:
        item = random.randint(1, num_items - 1)
        if item != positive:
            negs.append(item)
    return negs


def _resolve_device(requested: str):
    torch_mod, _ = require_torch()
    if requested == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(requested)
