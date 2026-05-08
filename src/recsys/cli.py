"""Command-line interface for the recommendation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CATEGORIES, DEFAULT_SEED, DEFAULT_TOP_K
from .experiments import run_experiment_grid
from .metrics import evaluate_prediction_file
from .pipeline import run_category
from .sasrec import SasRecConfig, TorchUnavailableError, run_sasrec_rerank_category, run_sasrec_rerank_grid
from .text_rerank import (
    SentenceTransformerUnavailableError,
    TextRerankConfig,
    run_text_rerank_category,
    run_text_rerank_grid,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Amazon Reviews 2023 sequential recommender")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", default="data", help="Directory containing *.csv.gz and meta_*.jsonl.gz")
        p.add_argument("--output-dir", default="outputs", help="Directory for predictions and metrics")
        p.add_argument("--model", default="hybrid", choices=("hybrid", "popularity"))
        p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--use-meta", action="store_true", help="Use meta_*.jsonl.gz lexical metadata index")
        p.add_argument(
            "--splits",
            nargs="+",
            default=["valid", "test"],
            choices=("valid", "test"),
            help="Splits to predict/evaluate",
        )

    p_cat = sub.add_parser("run-category", help="Train and evaluate one category")
    add_common(p_cat)
    p_cat.add_argument("--category", required=True, choices=DEFAULT_CATEGORIES)

    p_all = sub.add_parser("run-all", help="Train and evaluate all PRD categories")
    add_common(p_all)
    p_all.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), choices=DEFAULT_CATEGORIES)

    p_eval = sub.add_parser("evaluate-file", help="Evaluate an existing *_pred.jsonl file")
    p_eval.add_argument("--predictions", required=True)
    p_eval.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    p_exp = sub.add_parser("run-experiments", help="Run PRD baseline and ablation experiments")
    p_exp.add_argument("--data-dir", default="data")
    p_exp.add_argument("--output-dir", default="experiments")
    p_exp.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), choices=DEFAULT_CATEGORIES)
    p_exp.add_argument("--splits", nargs="+", default=["valid"], choices=("valid", "test"))
    p_exp.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_exp.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p_sas = sub.add_parser("sasrec-rerank", help="Train optional PyTorch SASRec reranker over hybrid candidates")
    p_sas.add_argument("--category", required=True, choices=DEFAULT_CATEGORIES)
    p_sas.add_argument("--data-dir", default="data")
    p_sas.add_argument("--output-dir", default="outputs_sasrec")
    p_sas.add_argument("--splits", nargs="+", default=["valid"], choices=("valid", "test"))
    p_sas.add_argument("--use-meta", action="store_true")
    add_sasrec_args(p_sas)

    p_sas_grid = sub.add_parser("sasrec-grid", help="Run SASRec reranker across multiple categories")
    p_sas_grid.add_argument("--data-dir", default="data")
    p_sas_grid.add_argument("--output-dir", default="experiments_sasrec")
    p_sas_grid.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), choices=DEFAULT_CATEGORIES)
    p_sas_grid.add_argument("--splits", nargs="+", default=["valid"], choices=("valid", "test"))
    p_sas_grid.add_argument("--use-meta", action="store_true")
    add_sasrec_args(p_sas_grid)

    p_text = sub.add_parser("text-rerank", help="Rerank hybrid candidates with metadata text embeddings")
    p_text.add_argument("--category", required=True, choices=DEFAULT_CATEGORIES)
    p_text.add_argument("--data-dir", default="data")
    p_text.add_argument("--output-dir", default="outputs_text")
    p_text.add_argument("--splits", nargs="+", default=["valid"], choices=("valid", "test"))
    add_text_rerank_args(p_text)

    p_text_grid = sub.add_parser("text-grid", help="Run text reranker across multiple categories")
    p_text_grid.add_argument("--data-dir", default="data")
    p_text_grid.add_argument("--output-dir", default="experiments_text")
    p_text_grid.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), choices=DEFAULT_CATEGORIES)
    p_text_grid.add_argument("--splits", nargs="+", default=["valid"], choices=("valid", "test"))
    add_text_rerank_args(p_text_grid)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "evaluate-file":
        metrics = evaluate_prediction_file(args.predictions, k=args.top_k)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-category":
        result = run_category(
            category=args.category,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            use_meta=args.use_meta,
            k=args.top_k,
            seed=args.seed,
            splits=tuple(args.splits),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-all":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for category in args.categories:
            result = run_category(
                category=category,
                data_dir=args.data_dir,
                output_dir=output_dir,
                model_name=args.model,
                use_meta=args.use_meta,
                k=args.top_k,
                seed=args.seed,
                splits=tuple(args.splits),
            )
            results.append(result)
        summary_path = output_dir / "summary_metrics.json"
        summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"summary_file": str(summary_path), "results": results}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-experiments":
        result = run_experiment_grid(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            categories=tuple(args.categories),
            splits=tuple(args.splits),
            top_k=args.top_k,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "sasrec-rerank":
        config = sasrec_config_from_args(args)
        try:
            result = run_sasrec_rerank_category(
                category=args.category,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                use_meta=args.use_meta,
                splits=tuple(args.splits),
                config=config,
            )
        except TorchUnavailableError as exc:
            parser.exit(1, f"{exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "sasrec-grid":
        config = sasrec_config_from_args(args)
        try:
            result = run_sasrec_rerank_grid(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                categories=tuple(args.categories),
                use_meta=args.use_meta,
                splits=tuple(args.splits),
                config=config,
            )
        except TorchUnavailableError as exc:
            parser.exit(1, f"{exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-rerank":
        config = text_rerank_config_from_args(args)
        try:
            result = run_text_rerank_category(
                category=args.category,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                splits=tuple(args.splits),
                config=config,
            )
        except SentenceTransformerUnavailableError as exc:
            parser.exit(1, f"{exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "text-grid":
        config = text_rerank_config_from_args(args)
        try:
            result = run_text_rerank_grid(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                categories=tuple(args.categories),
                splits=tuple(args.splits),
                config=config,
            )
        except SentenceTransformerUnavailableError as exc:
            parser.exit(1, f"{exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2

def add_sasrec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--base-rank-weight", type=float, default=1.0)
    parser.add_argument("--sasrec-score-weight", type=float, default=0.03)
    parser.add_argument("--loss", choices=("ce", "bce"), default="ce")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="0 means use all train rows; set a small value for quick smoke tests",
    )
    parser.add_argument("--device", default="auto")


def sasrec_config_from_args(args: argparse.Namespace) -> SasRecConfig:
    return SasRecConfig(
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        negatives=args.negatives,
        lr=args.lr,
        candidate_k=args.candidate_k,
        base_rank_weight=args.base_rank_weight,
        sasrec_score_weight=args.sasrec_score_weight,
        loss=args.loss,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
        device=args.device,
    )


def add_text_rerank_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text-backend", choices=("hashing", "sentence-transformer"), default="hashing")
    parser.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-history-items", type=int, default=3)
    parser.add_argument("--base-rank-weight", type=float, default=1.0)
    parser.add_argument("--text-score-weight", type=float, default=0.05)
    parser.add_argument("--no-meta", action="store_true", help="Disable lexical metadata in first-stage hybrid model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default="text_cache")
    parser.add_argument("--local-files-only", action="store_true", help="Load SentenceTransformer from local cache only")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def text_rerank_config_from_args(args: argparse.Namespace) -> TextRerankConfig:
    return TextRerankConfig(
        backend=args.text_backend,
        model_name=args.text_model,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        candidate_k=args.candidate_k,
        max_history_items=args.max_history_items,
        base_rank_weight=args.base_rank_weight,
        text_score_weight=args.text_score_weight,
        use_meta=not args.no_meta,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
