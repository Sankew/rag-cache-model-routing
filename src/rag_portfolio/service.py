"""Permission-aware retrieval with optional answer caching and model routing."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .access import ACLStore, can_read, payload_prefilter, secure_retrieve
from .baseline import Embedder, Generator, MemoryIndex
from .domain import Answer, Chunk, Principal, ScoredChunk
from .optimization import AnswerCache, ModelRouter, OptimizationMetrics


@dataclass(frozen=True)
class ServiceResult:
    answer: Answer
    cache_status: str
    route_reason: str
    latency_ms: float


class SecureRAGService:
    def __init__(
        self,
        *,
        index: MemoryIndex,
        acl_store: ACLStore,
        embedder: Embedder,
        generators: dict[str, Generator],
        default_model: str,
        top_k: int = 5,
        cache: AnswerCache | None = None,
        router: ModelRouter | None = None,
        metrics: OptimizationMetrics | None = None,
    ) -> None:
        if default_model not in generators or top_k < 1:
            raise ValueError("configure a default generator and positive top_k")
        if cache is not None and cache.acl_store is not acl_store:
            raise ValueError("cache must use the authoritative ACL store")
        self.index, self.acl_store, self.embedder = index, acl_store, embedder
        self.generators, self.default_model, self.top_k = generators, default_model, top_k
        self.cache, self.router, self.metrics = cache, router, metrics

    def answer(self, principal: Principal, question: str) -> ServiceResult:
        if not question.strip():
            raise ValueError("question must not be empty")
        started = perf_counter()
        vector = self.embedder.embed(question)
        ranked = self.index.query(
            vector, limit=max(1, len(self.index.all_chunks())),
            predicate=payload_prefilter(principal),
        )
        chunks = secure_retrieve(principal, (row.chunk for row in ranked), self.acl_store, limit=self.top_k)
        score_by_id = {row.chunk.chunk_id: row.score for row in ranked}
        evidence = [ScoredChunk(chunk, score_by_id[chunk.chunk_id]) for chunk in chunks]
        cache_args = dict(
            corpus_revision="demo-v1", index_revision=str(self.index.revision),
            pipeline_revision="cache-router-v1", embedding=vector,
        )
        if self.cache is not None:
            hit = self.cache.get(principal, chunks, question, **cache_args)
            if hit is not None:
                elapsed = (perf_counter() - started) * 1000
                self._record(hit.answer, elapsed, hit.match_type, "cache_hit", ())
                return ServiceResult(hit.answer, hit.match_type, "cache_hit", elapsed)

        decision = self.router.route(question, chunks) if self.router else None
        model_id = decision.model_id if decision else self.default_model
        if model_id not in self.generators:
            raise ValueError(f"router selected unconfigured model {model_id}")
        answer = self.generators[model_id].generate(question, evidence)
        extra_usage: tuple[tuple[str, int, int], ...] = ()
        reason = decision.reason if decision else "baseline"
        if self.router and model_id == self.router.cheap_model and not answer.citations:
            extra_usage = ((model_id, answer.input_tokens, answer.output_tokens),)
            answer = self.generators[self.router.strong_model].generate(question, evidence)
            reason += "+escalated"
        self._validate(principal, answer, chunks)
        if not answer.citations:
            answer = Answer(
                "I don't know based on the available documents.", (), answer.model_id,
                answer.input_tokens, answer.output_tokens,
            )
        if self.cache is not None and answer.text.strip() and answer.citations:
            self.cache.put(principal, chunks, question, answer, **cache_args)
        elapsed = (perf_counter() - started) * 1000
        self._record(answer, elapsed, "miss", reason, extra_usage)
        return ServiceResult(answer, "miss", reason, elapsed)

    def _validate(self, principal: Principal, answer: Answer, chunks: tuple[Chunk, ...]) -> None:
        available = {(chunk.document_id, chunk.chunk_id, chunk.version) for chunk in chunks}
        for citation in answer.citations:
            if (citation.document_id, citation.chunk_id, citation.version) not in available:
                raise ValueError("generator cited evidence outside authorized retrieval")
            current = self.acl_store.get(citation.document_id)
            if current is None or not can_read(principal, current):
                raise ValueError("cited document access was revoked")

    def _record(
        self, answer: Answer, elapsed: float, status: str, reason: str,
        extra_usage: tuple[tuple[str, int, int], ...],
    ) -> None:
        if self.metrics is not None:
            cached = status != "miss"
            self.metrics.record(
                model_id=answer.model_id,
                input_tokens=0 if cached else answer.input_tokens,
                output_tokens=0 if cached else answer.output_tokens,
                latency_ms=elapsed, cache_status=status, route_reason=reason,
                extra_usage=extra_usage,
            )
