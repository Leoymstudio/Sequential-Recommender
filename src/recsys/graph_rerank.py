"""LightGCN-style graph reranking over hybrid candidates."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CATEGORIES
from .data import category_csv_path, iter_interactions, write_prediction_row
from .metrics import ndcg_at_k
from .pipeline import build_model
from .recommenders import HybridParams

try:  # pragma: no cover - optional advanced dependency.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class GraphRerankConfig:
    embedding_dim: int = 64
    layers: int = 1
    epochs: int = 1
    batch_size: int = 4096
    lr: float = 0.003
    reg: float = 1e-6
    candidate_k: int = 50
    base_rank_weight: float = 1.0
    graph_score_weight: float = 0.03
    use_meta: bool = True
    max_train_rows: int = 0
    seed: int = 2026
    device: str = "auto"


class TorchUnavailableError(RuntimeError):
    pass


def require_torch():
    if torch is None or nn is None:
        raise TorchUnavailableError("PyTorch is not installed. Install it before running graph-rerank.")
    return torch, nn


class GraphData:
    def __init__(self):
        self.user_to_id: dict[str, int] = {}
        self.item_to_id: dict[str, int] = {}
        self.id_to_item: dict[int, str] = {}
        self.edge_users: list[int] = []
        self.edge_items: list[int] = []
        self.user_positive_items: dict[int, set[int]] = {}

    @property
    def num_users(self) -> int:
        return len(self.user_to_id)

    @property
    def num_items(self) -> int:
        return len(self.item_to_id)

    def user_id(self, raw: str) -> int:
        idx = self.user_to_id.get(raw)
        if idx is None:
            idx = len(self.user_to_id)
            self.user_to_id[raw] = idx
        return idx

    def item_id(self, raw: str) -> int:
        idx = self.item_to_id.get(raw)
        if idx is None:
            idx = len(self.item_to_id)
            self.item_to_id[raw] = idx
            self.id_to_item[idx] = raw
        return idx


if nn is not None:

    class GraphMFModel(nn.Module):
        def __init__(self, num_users: int, num_items: int, dim: int):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, dim)
            self.item_embedding = nn.Embedding(num_items, dim)
            nn.init.normal_(self.user_embedding.weight, std=0.02)
            nn.init.normal_(self.item_embedding.weight, std=0.02)

        def bpr_loss(self, users, positives, negatives, reg: float):
            user_vec = self.user_embedding(users)
            pos_vec = self.item_embedding(positives)
            neg_vec = self.item_embedding(negatives)
            pos_scores = (user_vec * pos_vec).sum(dim=1)
            neg_scores = (user_vec * neg_vec).sum(dim=1)
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            if reg > 0:
                loss = loss + reg * (
                    user_vec.pow(2).sum(dim=1)
                    + pos_vec.pow(2).sum(dim=1)
                    + neg_vec.pow(2).sum(dim=1)
                ).mean()
            return loss

else:

    class GraphMFModel:  # pragma: no cover
        pass


def run_graph_rerank_category(
    category: str,
    data_dir: str | Path,
    output_dir: str | Path,
    splits: tuple[str, ...] = ("valid",),
    config: GraphRerankConfig | None = None,
) -> dict:
    torch_mod, _ = require_torch()
    cfg = config or GraphRerankConfig()
    random.seed(cfg.seed)
    torch_mod.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
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
    graph = load_graph_data(category, data_dir, getattr(base, "item_universe"), cfg)
    model = GraphMFModel(graph.num_users, graph.num_items, cfg.embedding_dim).to(device)
    train_metrics = train_graph_model(model, graph, cfg, device)
    user_emb, item_emb = lightgcn_propagate(model, graph, cfg, device)

    split_metrics = [
        predict_graph_split(
            category=category,
            split=split,
            data_dir=data_dir,
            output_dir=output_dir,
            base=base,
            graph=graph,
            user_emb=user_emb,
            item_emb=item_emb,
            cfg=cfg,
        )
        for split in splits
    ]
    result = {
        "category": category,
        "model": "graph_rerank",
        "resolved_device": str(device),
        "config": cfg.__dict__,
        "graph": {
            "users": graph.num_users,
            "items": graph.num_items,
            "edges": len(graph.edge_users),
        },
        "train": train_metrics,
        "splits": split_metrics,
        "seconds_total": time.perf_counter() - started,
    }
    metrics_path = output_dir / f"{category}_graph_rerank_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metrics_file"] = str(metrics_path)
    return result


def run_graph_rerank_grid(
    data_dir: str | Path,
    output_dir: str | Path,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    splits: tuple[str, ...] = ("valid",),
    config: GraphRerankConfig | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or GraphRerankConfig()
    results = []
    for category in categories:
        result = run_graph_rerank_category(
            category=category,
            data_dir=data_dir,
            output_dir=output_dir / category,
            splits=splits,
            config=cfg,
        )
        results.append(result)
    summary = flatten_graph_results(results)
    write_graph_summary(output_dir, summary)
    payload = {
        "output_dir": str(output_dir),
        "model": "graph_rerank",
        "config": cfg.__dict__,
        "results": results,
        "summary": summary,
    }
    summary_path = output_dir / "graph_rerank_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["summary_file"] = str(summary_path)
    return payload


def load_graph_data(category: str, data_dir: str | Path, item_universe: set[str], cfg: GraphRerankConfig) -> GraphData:
    graph = GraphData()
    for item in sorted(item_universe):
        graph.item_id(item)
    rows = 0
    for row in iter_interactions(category_csv_path(data_dir, category, "train")):
        user_idx = graph.user_id(row.user_id)
        item_idx = graph.item_id(row.parent_asin)
        graph.edge_users.append(user_idx)
        graph.edge_items.append(item_idx)
        graph.user_positive_items.setdefault(user_idx, set()).add(item_idx)
        rows += 1
        if cfg.max_train_rows and rows >= cfg.max_train_rows:
            break
    return graph


def train_graph_model(model, graph: GraphData, cfg: GraphRerankConfig, device) -> dict:
    torch_mod, _ = require_torch()
    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=cfg.lr)
    edge_count = len(graph.edge_users)
    losses: list[float] = []
    edge_indexes = list(range(edge_count))
    model.train()
    for _ in range(cfg.epochs):
        random.shuffle(edge_indexes)
        for start in range(0, edge_count, cfg.batch_size):
            indexes = edge_indexes[start : start + cfg.batch_size]
            users = [graph.edge_users[idx] for idx in indexes]
            positives = [graph.edge_items[idx] for idx in indexes]
            negatives = [sample_negative(graph, user, positive) for user, positive in zip(users, positives)]
            user_t = torch_mod.tensor(users, dtype=torch_mod.long, device=device)
            pos_t = torch_mod.tensor(positives, dtype=torch_mod.long, device=device)
            neg_t = torch_mod.tensor(negatives, dtype=torch_mod.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.bpr_loss(user_t, pos_t, neg_t, cfg.reg)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    return {
        "edges": edge_count,
        "epochs": cfg.epochs,
        "loss_last": losses[-1] if losses else 0.0,
        "loss_mean": sum(losses) / len(losses) if losses else 0.0,
    }


def lightgcn_propagate(model, graph: GraphData, cfg: GraphRerankConfig, device):
    torch_mod, _ = require_torch()
    model.eval()
    with torch_mod.no_grad():
        user_emb = model.user_embedding.weight
        item_emb = model.item_embedding.weight
        all_emb = torch_mod.cat([user_emb, item_emb], dim=0)
        embeddings = [all_emb]
        adjacency = build_normalized_adjacency(graph, device)
        current = all_emb
        for _ in range(cfg.layers):
            current = torch_mod.sparse.mm(adjacency, current)
            embeddings.append(current)
        final = torch_mod.stack(embeddings, dim=0).mean(dim=0)
        user_final = final[: graph.num_users].detach().cpu()
        item_final = final[graph.num_users :].detach().cpu()
    return user_final, item_final


def build_normalized_adjacency(graph: GraphData, device):
    torch_mod, _ = require_torch()
    num_nodes = graph.num_users + graph.num_items
    user_degrees = [0] * graph.num_users
    item_degrees = [0] * graph.num_items
    for user, item in zip(graph.edge_users, graph.edge_items):
        user_degrees[user] += 1
        item_degrees[item] += 1

    rows = []
    cols = []
    values = []
    for user, item in zip(graph.edge_users, graph.edge_items):
        item_node = graph.num_users + item
        weight = 1.0 / math.sqrt(max(user_degrees[user], 1) * max(item_degrees[item], 1))
        rows.extend([user, item_node])
        cols.extend([item_node, user])
        values.extend([weight, weight])
    indices = torch_mod.tensor([rows, cols], dtype=torch_mod.long, device=device)
    vals = torch_mod.tensor(values, dtype=torch_mod.float32, device=device)
    with torch_mod.sparse.check_sparse_tensor_invariants(False):
        return torch_mod.sparse_coo_tensor(
            indices,
            vals,
            (num_nodes, num_nodes),
            device=device,
        ).coalesce()


def predict_graph_split(category, split, data_dir, output_dir, base, graph, user_emb, item_emb, cfg):
    split_path = category_csv_path(data_dir, category, split)
    pred_path = Path(output_dir) / f"{category}_{split}_graph_rerank_pred.jsonl"
    rows = 0
    hits = 0
    total = 0.0
    started = time.perf_counter()
    with pred_path.open("w", encoding="utf-8") as f:
        for row in iter_interactions(split_path):
            candidates = base.recommend(row.history, k=cfg.candidate_k)
            predictions = rerank_graph_candidates(row.user_id, candidates, graph, user_emb, item_emb, cfg)
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


def rerank_graph_candidates(user_id: str, candidates: list[str], graph: GraphData, user_emb, item_emb, cfg: GraphRerankConfig) -> list[str]:
    user_idx = graph.user_to_id.get(user_id)
    if user_idx is None:
        return candidates[:10]
    scores = []
    raw_scores = []
    user_vec = user_emb[user_idx]
    for item in candidates:
        item_idx = graph.item_to_id.get(item)
        if item_idx is None:
            raw_scores.append(0.0)
        else:
            raw_scores.append(float((user_vec * item_emb[item_idx]).sum().item()))
    mean_score = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
    variance = sum((score - mean_score) ** 2 for score in raw_scores) / len(raw_scores) if raw_scores else 0.0
    std_score = math.sqrt(variance) if variance > 1e-12 else 1.0
    for rank, (item, raw_score) in enumerate(zip(candidates, raw_scores), start=1):
        base_score = 1.0 / math.log2(rank + 1)
        graph_score = (raw_score - mean_score) / std_score
        final_score = cfg.base_rank_weight * base_score + cfg.graph_score_weight * graph_score
        scores.append((item, final_score))
    ranked = sorted(scores, key=lambda kv: (-kv[1], kv[0]))
    return [item for item, _ in ranked[:10]]


def sample_negative(graph: GraphData, user: int, positive: int) -> int:
    positives = graph.user_positive_items.get(user, set())
    for _ in range(64):
        item = random.randint(0, graph.num_items - 1)
        if item not in positives:
            return item
    item = random.randint(0, graph.num_items - 1)
    return item if item != positive else (item + 1) % graph.num_items


def flatten_graph_results(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        for split in result.get("splits", []):
            rows.append(
                {
                    "category": result["category"],
                    "split": split["split"],
                    "device": result.get("resolved_device", ""),
                    "users": result.get("graph", {}).get("users", 0),
                    "items": result.get("graph", {}).get("items", 0),
                    "edges": result.get("graph", {}).get("edges", 0),
                    "hit@10": split["hit@10"],
                    "ndcg@10": split["ndcg@10"],
                    "seconds_total": result.get("seconds_total", 0.0),
                    "prediction_file": split["prediction_file"],
                }
            )
    return rows


def write_graph_summary(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path = output_dir / "graph_rerank_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = output_dir / "graph_rerank_summary.md"
    headers = ["category", "split", "device", "edges", "hit@10", "ndcg@10", "seconds_total"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {category} | {split} | {device} | {edges} | {hit:.6f} | {ndcg:.6f} | {seconds:.2f} |".format(
                category=row["category"],
                split=row["split"],
                device=row["device"],
                edges=row["edges"],
                hit=row["hit@10"],
                ndcg=row["ndcg@10"],
                seconds=row["seconds_total"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_device(requested: str):
    torch_mod, _ = require_torch()
    if requested == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(requested)
