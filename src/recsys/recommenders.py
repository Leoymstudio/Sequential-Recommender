"""Lightweight recommenders for Amazon Reviews sequential splits."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .data import InteractionRow, iter_meta, tail_history


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._-]{2,}", re.IGNORECASE)
STOPWORDS = {
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
    "set",
    "pack",
    "new",
    "inch",
    "black",
    "white",
    "amazon",
}


@dataclass(frozen=True)
class HybridParams:
    top_k: int = 10
    seed: int = 2026
    train_history_tail: int = 24
    predict_history_tail: int = 16
    last_neighbors: int = 160
    pair_neighbors: int = 80
    cooc_neighbors: int = 60
    popularity_fallback: int = 5000
    meta_history_items: int = 2
    meta_candidates_per_token: int = 10
    pair_weight: float = 5.0
    last_weight: float = 3.0
    cooc_weight: float = 1.4
    popularity_weight: float = 0.18
    meta_weight: float = 0.08


class PopularityRecommender:
    """Simple non-personalized recommender used as a baseline and fallback."""

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.item_counts: Counter[str] = Counter()
        self.popular_items: list[tuple[str, float]] = []

    def fit(self, rows: Iterable[InteractionRow]) -> "PopularityRecommender":
        for row in rows:
            self.item_counts[row.parent_asin] += 1.0
        self.popular_items = [(item, float(count)) for item, count in self.item_counts.most_common()]
        return self

    def recommend(self, history: str | Iterable[str], k: int | None = None) -> list[str]:
        limit = k or self.top_k
        seen = set(history.split() if isinstance(history, str) else history)
        recs: list[str] = []
        for item, _ in self.popular_items:
            if item in seen:
                continue
            recs.append(item)
            if len(recs) >= limit:
                break
        return recs


class MetadataIndex:
    """Tiny lexical metadata index for candidate expansion.

    It uses only standard-library tokenization. Metadata is not the main signal,
    but it gives the model a PRD-aligned way to use titles/categories when
    candidate transitions are sparse.
    """

    def __init__(self, max_tokens_per_item: int = 32):
        self.max_tokens_per_item = max_tokens_per_item
        self.item_tokens: dict[str, tuple[str, ...]] = {}
        self.token_items: dict[str, list[tuple[str, float]]] = {}

    def fit(
        self,
        meta_path: str | Path,
        item_universe: set[str],
        item_popularity: Counter[str],
        token_item_limit: int = 120,
    ) -> "MetadataIndex":
        token_scores: dict[str, Counter[str]] = defaultdict(Counter)
        for obj in iter_meta(meta_path):
            item = obj.get("parent_asin")
            if item not in item_universe:
                continue
            tokens = self._extract_tokens(obj)
            if not tokens:
                continue
            self.item_tokens[item] = tuple(tokens)
            quality = self._quality_score(obj, item_popularity.get(item, 0.0))
            for token in tokens:
                token_scores[token][item] += quality

        self.token_items = {
            token: [(item, float(score)) for item, score in counter.most_common(token_item_limit)]
            for token, counter in token_scores.items()
        }
        return self

    def candidates_for_history(
        self,
        history_tail: list[str],
        max_history_items: int,
        per_token: int,
    ) -> Counter[str]:
        scores: Counter[str] = Counter()
        if not self.item_tokens or max_history_items <= 0 or per_token <= 0:
            return scores

        for recency, item in enumerate(reversed(history_tail[-max_history_items:]), start=1):
            weight = 1.0 / recency
            for token in self.item_tokens.get(item, ())[: self.max_tokens_per_item]:
                for candidate, score in self.token_items.get(token, ())[:per_token]:
                    scores[candidate] += weight * score
        return scores

    def _extract_tokens(self, obj: dict) -> list[str]:
        chunks: list[str] = []
        title = obj.get("title")
        if isinstance(title, str):
            chunks.append(title)
        categories = obj.get("categories")
        if isinstance(categories, list):
            chunks.extend(str(part) for part in categories[-4:])
        store = obj.get("store")
        if isinstance(store, str):
            chunks.append(store)

        seen: set[str] = set()
        tokens: list[str] = []
        for raw in TOKEN_RE.findall(" ".join(chunks).lower()):
            token = raw.strip("._-+")
            if len(token) < 3 or token in STOPWORDS or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= self.max_tokens_per_item:
                break
        return tokens

    @staticmethod
    def _quality_score(obj: dict, popularity: float) -> float:
        avg = obj.get("average_rating")
        num = obj.get("rating_number")
        try:
            avg_rating = float(avg) if avg is not None else 3.5
        except (TypeError, ValueError):
            avg_rating = 3.5
        try:
            rating_number = float(num) if num is not None else 0.0
        except (TypeError, ValueError):
            rating_number = 0.0
        return 0.25 + math.log1p(popularity) + (avg_rating / 5.0) * math.log1p(rating_number)


class HybridSequentialRecommender:
    """Sequential transition, recency co-occurrence, popularity, and metadata."""

    def __init__(self, params: HybridParams | None = None):
        self.params = params or HybridParams()
        self.item_counts: Counter[str] = Counter()
        self.transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.pair_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.cooc_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.item_universe: set[str] = set()
        self.popular_items: list[tuple[str, float]] = []
        self.popularity_scores: dict[str, float] = {}
        self.transitions: dict[str, list[tuple[str, float]]] = {}
        self.pair_transitions: dict[tuple[str, str], list[tuple[str, float]]] = {}
        self.cooc: dict[str, list[tuple[str, float]]] = {}
        self.meta_index: MetadataIndex | None = None
        random.seed(self.params.seed)

    def fit(self, rows: Iterable[InteractionRow]) -> "HybridSequentialRecommender":
        for row in rows:
            target = row.parent_asin
            if not target:
                continue
            self.item_universe.add(target)
            rating_weight = 1.0 + max(row.rating - 3.0, -2.0) * 0.05
            self.item_counts[target] += rating_weight

            history_tail = tail_history(row.history, self.params.train_history_tail)
            if not history_tail:
                continue
            self.item_universe.update(history_tail)

            last_item = history_tail[-1]
            self.transition_counts[last_item][target] += rating_weight

            if len(history_tail) >= 2:
                self.pair_counts[(history_tail[-2], last_item)][target] += rating_weight

            for distance, item in enumerate(reversed(history_tail), start=1):
                if item == target:
                    continue
                self.cooc_counts[item][target] += rating_weight / math.sqrt(distance)

        self._finalize()
        return self

    def fit_metadata(self, meta_path: str | Path) -> "HybridSequentialRecommender":
        self.meta_index = MetadataIndex()
        self.meta_index.fit(meta_path, self.item_universe, self.item_counts)
        return self

    def recommend(self, history: str | Iterable[str], k: int | None = None) -> list[str]:
        limit = k or self.params.top_k
        if isinstance(history, str):
            seen = set(history.split() if history else [])
            history_tail = tail_history(history, self.params.predict_history_tail)
        else:
            items = list(history)
            seen = set(items)
            history_tail = items[-self.params.predict_history_tail :]

        scores: Counter[str] = Counter()

        if history_tail:
            last = history_tail[-1]
            self._add_candidates(scores, self.transitions.get(last, ()), self.params.last_weight)

            if len(history_tail) >= 2:
                pair = (history_tail[-2], last)
                self._add_candidates(scores, self.pair_transitions.get(pair, ()), self.params.pair_weight)

            for distance, item in enumerate(reversed(history_tail), start=1):
                weight = self.params.cooc_weight / math.sqrt(distance)
                self._add_candidates(scores, self.cooc.get(item, ()), weight)

            if self.meta_index:
                meta_scores = self.meta_index.candidates_for_history(
                    history_tail=history_tail,
                    max_history_items=self.params.meta_history_items,
                    per_token=self.params.meta_candidates_per_token,
                )
                for item, score in meta_scores.items():
                    scores[item] += self.params.meta_weight * math.log1p(score)

        for item in list(scores):
            scores[item] += self.params.popularity_weight * self.popularity_scores.get(item, 0.0)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        recommendations: list[str] = []
        for item, _ in ranked:
            if item in seen:
                continue
            recommendations.append(item)
            if len(recommendations) >= limit:
                break

        if len(recommendations) < limit:
            already = seen | set(recommendations)
            for item, _ in self.popular_items[: self.params.popularity_fallback]:
                if item in already:
                    continue
                recommendations.append(item)
                already.add(item)
                if len(recommendations) >= limit:
                    break
            if len(recommendations) < limit:
                for item, _ in self.popular_items[self.params.popularity_fallback :]:
                    if item in already:
                        continue
                    recommendations.append(item)
                    already.add(item)
                    if len(recommendations) >= limit:
                        break
        return recommendations[:limit]

    def _finalize(self) -> None:
        self.popular_items = [(item, float(count)) for item, count in self.item_counts.most_common()]
        self.popularity_scores = {item: math.log1p(count) for item, count in self.popular_items}
        self.transitions = self._prune(self.transition_counts, self.params.last_neighbors)
        self.pair_transitions = self._prune(self.pair_counts, self.params.pair_neighbors)
        self.cooc = self._prune(self.cooc_counts, self.params.cooc_neighbors)

        self.transition_counts = {}
        self.pair_counts = {}
        self.cooc_counts = {}

    @staticmethod
    def _prune(counts: dict, limit: int) -> dict:
        return {
            key: [(item, float(score)) for item, score in counter.most_common(limit)]
            for key, counter in counts.items()
        }

    @staticmethod
    def _add_candidates(scores: Counter[str], candidates: Iterable[tuple[str, float]], weight: float) -> None:
        for item, raw_score in candidates:
            scores[item] += weight * math.log1p(raw_score)
