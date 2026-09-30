"""Permission-aware answer caching, model routing, and usage measurement.

The cache runs *after* access-filtered retrieval. It saves generation work, while
still paying for retrieval and authorization on every request.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, sqrt
import re
from time import monotonic
from typing import Any, Callable, Sequence

from .access import ACLStore, can_read
from .domain import Answer, Chunk, Principal


def percentile(values: Sequence[float], q: float) -> float:
    """Linearly interpolate between adjacent ordered values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low, high = floor(rank), ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


AccessCheck = Callable[[Principal, Chunk], bool]


def default_access_check(principal: Principal, chunk: Chunk) -> bool:
    """Use the project's tenant, role/user, and classification ACL semantics."""
    return can_read(principal, chunk.acl)


def _normalized(question: str) -> str:
    return " ".join(question.casefold().split())


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return -1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


@dataclass(frozen=True)
class CacheScope:
    tenant_id: str
    user_id: str
    roles: tuple[str, ...]
    clearance: int
    corpus_revision: str
    index_revision: str
    pipeline_revision: str
    evidence: tuple[tuple[str, str, int, int], ...]


@dataclass(frozen=True)
class CacheHit:
    answer: Answer
    match_type: str
    similarity: float


@dataclass(frozen=True)
class _Entry:
    scope: CacheScope
    question: str
    embedding: tuple[float, ...] | None
    answer: Answer
    expires_at: float


class AnswerCache:
    """In-memory answer cache with security-scoped exact and semantic matching.

    Set ``semantic_threshold`` only after calibrating it with held-out questions;
    the default disables semantic reuse. Matching requires the identical current
    evidence set, including document, chunk, content, and ACL versions.
    """

    def __init__(
        self,
        semantic_threshold: float | None = None,
        access_check: AccessCheck = default_access_check,
        acl_store: ACLStore | None = None,
        max_entries: int = 1000,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if semantic_threshold is not None and not 0.0 < semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be within (0, 1]")
        if max_entries < 1 or ttl_seconds <= 0:
            raise ValueError("cache size and TTL must be positive")
        self.semantic_threshold = semantic_threshold
        self.access_check = access_check
        self.acl_store = acl_store
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries: list[_Entry] = []

    def scope(
        self,
        principal: Principal,
        chunks: Sequence[Chunk],
        *,
        corpus_revision: str,
        index_revision: str,
        pipeline_revision: str,
    ) -> CacheScope:
        evidence = tuple(
            sorted(
                (
                    chunk.document_id,
                    chunk.chunk_id,
                    chunk.version,
                    self._acl_revision(chunk),
                )
                for chunk in chunks
            )
        )
        return CacheScope(
            principal.tenant_id,
            principal.user_id,
            tuple(sorted(principal.roles)),
            principal.clearance,
            corpus_revision,
            index_revision,
            pipeline_revision,
            evidence,
        )

    def _acl_revision(self, chunk: Chunk) -> int:
        if self.acl_store is None:
            return chunk.acl.revision
        current = self.acl_store.get(chunk.document_id)
        return current.revision if current is not None else -1

    def _authorized(self, principal: Principal, chunk: Chunk) -> bool:
        if not self.access_check(principal, chunk):
            return False
        if self.acl_store is None:
            return True
        current = self.acl_store.get(chunk.document_id)
        return current is not None and can_read(principal, current)

    def _safe(self, principal: Principal, chunks: Sequence[Chunk], answer: Answer) -> bool:
        current = {(chunk.document_id, chunk.chunk_id, chunk.version): chunk for chunk in chunks}
        for citation in answer.citations:
            chunk = current.get((citation.document_id, citation.chunk_id, citation.version))
            if chunk is None or not self._authorized(principal, chunk):
                return False
        return all(self._authorized(principal, chunk) for chunk in chunks)

    def put(
        self,
        principal: Principal,
        chunks: Sequence[Chunk],
        question: str,
        answer: Answer,
        *,
        corpus_revision: str,
        index_revision: str,
        pipeline_revision: str,
        embedding: tuple[float, ...] | None = None,
    ) -> None:
        if not answer.text.strip() or not answer.citations:
            raise ValueError("empty or uncited answers must not be cached")
        if not self._safe(principal, chunks, answer):
            raise ValueError("Answer or evidence is not authorized for this principal")
        scope = self.scope(
            principal, chunks, corpus_revision=corpus_revision,
            index_revision=index_revision, pipeline_revision=pipeline_revision,
        )
        question_key = _normalized(question)
        self._evict_expired()
        self._entries = [entry for entry in self._entries if not (entry.scope == scope and entry.question == question_key)]
        self._entries.append(_Entry(scope, question_key, embedding, answer, self.clock() + self.ttl_seconds))
        self._entries = self._entries[-self.max_entries:]

    def get(
        self,
        principal: Principal,
        chunks: Sequence[Chunk],
        question: str,
        *,
        corpus_revision: str,
        index_revision: str,
        pipeline_revision: str,
        embedding: tuple[float, ...] | None = None,
    ) -> CacheHit | None:
        self._evict_expired()
        if not all(self._authorized(principal, chunk) for chunk in chunks):
            return None
        scope = self.scope(
            principal, chunks, corpus_revision=corpus_revision,
            index_revision=index_revision, pipeline_revision=pipeline_revision,
        )
        candidates = [entry for entry in self._entries if entry.scope == scope and self._safe(principal, chunks, entry.answer)]
        question_key = _normalized(question)
        for entry in reversed(candidates):
            if entry.question == question_key:
                return CacheHit(entry.answer, "exact", 1.0)
        if self.semantic_threshold is None or embedding is None:
            return None
        ranked = sorted(
            ((_cosine(embedding, entry.embedding), entry) for entry in candidates if entry.embedding is not None),
            key=lambda pair: pair[0], reverse=True,
        )
        if ranked and ranked[0][0] >= self.semantic_threshold:
            return CacheHit(ranked[0][1].answer, "semantic", ranked[0][0])
        return None

    def _evict_expired(self) -> None:
        now = self.clock()
        self._entries = [entry for entry in self._entries if entry.expires_at > now]


THRESHOLD_CANDIDATES = tuple(round(0.85 + i * 0.01, 2) for i in range(13))


def calibrate_semantic_threshold(
    paraphrase_similarities: Sequence[float],
    near_miss_similarities: Sequence[float],
    candidates: Sequence[float] = THRESHOLD_CANDIDATES,
) -> tuple[float, float]:
    """Choose the lowest candidate with zero false hits; report paraphrase recall."""
    if not paraphrase_similarities or not near_miss_similarities:
        raise ValueError("both calibration sets must be nonempty")
    for threshold in sorted(candidates):
        if not 0 < threshold <= 1:
            raise ValueError("thresholds must be within (0, 1]")
        if all(score < threshold for score in near_miss_similarities):
            recall = sum(score >= threshold for score in paraphrase_similarities) / len(paraphrase_similarities)
            return threshold, recall
    raise ValueError("no candidate threshold avoids false hits")


@dataclass(frozen=True)
class RouteDecision:
    model_id: str
    reason: str


class ModelRouter:
    """Simple, auditable policy; evaluate quality before claiming cost savings."""

    def __init__(self, cheap_model: str, strong_model: str) -> None:
        if not cheap_model or not strong_model or cheap_model == strong_model:
            raise ValueError("Choose two distinct nonempty model IDs")
        self.cheap_model = cheap_model
        self.strong_model = strong_model

    def route(self, question: str, chunks: Sequence[Chunk]) -> RouteDecision:
        if not chunks:
            return RouteDecision(self.strong_model, "no_evidence")
        if len({chunk.document_id for chunk in chunks}) > 1:
            return RouteDecision(self.strong_model, "multiple_documents")
        if re.search(r"\b(?:compare|across|why|synthesis|synthesize|synthesise|relationship|trade[- ]off)\b", question.casefold()):
            return RouteDecision(self.strong_model, "complex_question")
        return RouteDecision(self.cheap_model, "single_document_fact")


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        if self.input_per_million < 0 or self.output_per_million < 0:
            raise ValueError("Token prices must be nonnegative")


class OptimizationMetrics:
    """Per-request accounting. Price configuration must identify its own date."""

    def __init__(self, prices: dict[str, ModelPrice], price_date: str) -> None:
        if not price_date:
            raise ValueError("Provide a pricing snapshot date")
        self.prices = dict(prices)
        self.price_date = price_date
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        *,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cache_status: str,
        route_reason: str,
        extra_usage: Sequence[tuple[str, int, int]] = (),
    ) -> None:
        if model_id not in self.prices:
            raise ValueError(f"No price configured for {model_id}")
        if min(input_tokens, output_tokens, latency_ms) < 0:
            raise ValueError("Usage and latency must be nonnegative")
        if cache_status not in {"miss", "exact", "semantic"}:
            raise ValueError("Unknown cache status")
        price = self.prices[model_id]
        cost = (input_tokens * price.input_per_million + output_tokens * price.output_per_million) / 1_000_000
        for extra_model, extra_input, extra_output in extra_usage:
            if extra_model not in self.prices or min(extra_input, extra_output) < 0:
                raise ValueError("invalid extra model usage")
            extra_price = self.prices[extra_model]
            cost += (extra_input * extra_price.input_per_million + extra_output * extra_price.output_per_million) / 1_000_000
        self.events.append({
            "model_id": model_id,
            "input_tokens": input_tokens + sum(item[1] for item in extra_usage),
            "output_tokens": output_tokens + sum(item[2] for item in extra_usage),
            "latency_ms": latency_ms,
            "cache_status": cache_status,
            "route_reason": route_reason,
            "extra_usage": tuple(extra_usage),
            "cost": cost,
        })

    def summary(self) -> dict[str, object]:
        count = len(self.events)
        if not count:
            return {"requests": 0, "cache_hit_rate": 0.0, "total_cost": 0.0, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "price_date": self.price_date}
        latencies = sorted(float(event["latency_ms"]) for event in self.events)
        total_cost = sum(float(event["cost"]) for event in self.events)
        hits = sum(event["cache_status"] != "miss" for event in self.events)
        return {
            "requests": count,
            "cache_hit_rate": hits / count,
            "total_cost": total_cost,
            "p50_latency_ms": percentile(latencies, 0.5),
            "p95_latency_ms": percentile(latencies, 0.95),
            "price_date": self.price_date,
        }
