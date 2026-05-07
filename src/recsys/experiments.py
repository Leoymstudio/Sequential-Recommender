"""Experiment runners and report helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CATEGORIES
from .pipeline import run_category


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model: str
    use_meta: bool


DEFAULT_EXPERIMENTS = (
    ExperimentSpec("popularity", "popularity", False),
    ExperimentSpec("hybrid_no_meta", "hybrid", False),
    ExperimentSpec("hybrid_meta", "hybrid", True),
)


def run_experiment_grid(
    data_dir: str | Path,
    output_dir: str | Path,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    splits: tuple[str, ...] = ("valid",),
    top_k: int = 10,
    seed: int = 2026,
    experiments: tuple[ExperimentSpec, ...] = DEFAULT_EXPERIMENTS,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in experiments:
        exp_dir = output_dir / spec.name
        for category in categories:
            result = run_category(
                category=category,
                data_dir=data_dir,
                output_dir=exp_dir,
                model_name=spec.model,
                use_meta=spec.use_meta,
                k=top_k,
                seed=seed,
                splits=splits,
            )
            result["experiment"] = spec.name
            results.append(result)

    summary = flatten_results(results, top_k=top_k)
    write_summary_files(output_dir, summary, top_k=top_k)
    payload = {
        "output_dir": str(output_dir),
        "splits": list(splits),
        "top_k": top_k,
        "seed": seed,
        "results": results,
        "summary": summary,
    }
    (output_dir / "experiments.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def flatten_results(results: list[dict], top_k: int = 10) -> list[dict[str, str | int | float | bool]]:
    rows: list[dict[str, str | int | float | bool]] = []
    for result in results:
        for split_result in result.get("splits", []):
            rows.append(
                {
                    "experiment": result.get("experiment", result.get("model", "")),
                    "category": result["category"],
                    "split": split_result["split"],
                    "model": result["model"],
                    "use_meta": bool(result["use_meta"]),
                    "rows": int(split_result["rows"]),
                    f"hit@{top_k}": float(split_result[f"hit@{top_k}"]),
                    f"ndcg@{top_k}": float(split_result[f"ndcg@{top_k}"]),
                    "seconds_total": float(result.get("seconds_total", 0.0)),
                    "prediction_file": split_result["prediction_file"],
                }
            )
    return rows


def write_summary_files(output_dir: Path, rows: list[dict], top_k: int = 10) -> None:
    if not rows:
        return

    csv_path = output_dir / "experiments_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "experiments_summary.md"
    headers = ["experiment", "category", "split", f"hit@{top_k}", f"ndcg@{top_k}", "seconds_total"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | :--- | :--- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {category} | {split} | {hit:.6f} | {ndcg:.6f} | {seconds:.2f} |".format(
                experiment=row["experiment"],
                category=row["category"],
                split=row["split"],
                hit=row[f"hit@{top_k}"],
                ndcg=row[f"ndcg@{top_k}"],
                seconds=row["seconds_total"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
