from dataclasses import replace

from rag_portfolio.access import ACLStore
from rag_portfolio.baseline import HashingEmbedder, MemoryIndex, ingest_document
from rag_portfolio.domain import ACL, Answer, Citation, Principal
from rag_portfolio.optimization import AnswerCache, ModelPrice, ModelRouter, OptimizationMetrics
from rag_portfolio.service import SecureRAGService


class FixedGenerator:
    def __init__(self, model_id):
        self.model_id = model_id

    def generate(self, question, evidence):
        if not evidence:
            return Answer("No evidence", (), self.model_id, 5, 2)
        chunk = evidence[0].chunk
        return Answer("20 days", (Citation(chunk.document_id, chunk.chunk_id, chunk.version),), self.model_id, 10, 2)


class UncitedGenerator:
    def __init__(self, model_id):
        self.model_id = model_id

    def generate(self, question, evidence):
        return Answer("20 days", (), self.model_id, 10, 2)


def test_service_cache_and_revocation():
    embedder = HashingEmbedder()
    index = MemoryIndex()
    store = ACLStore()
    acl = ACL("tenant", frozenset({"staff"}), frozenset())
    chunks = ingest_document("leave", "Annual leave is 20 days per year.", acl, embedder)
    index.upsert(chunks)
    store.update(chunks[0].document_id, acl)
    user = Principal("alice", "tenant", frozenset({"staff"}))
    metrics = OptimizationMetrics({"small": ModelPrice(1, 2), "strong": ModelPrice(5, 10)}, "test")
    service = SecureRAGService(
        index=index, acl_store=store, embedder=embedder,
        generators={name: FixedGenerator(name) for name in ("small", "strong")},
        default_model="strong", top_k=1, cache=AnswerCache(acl_store=store),
        router=ModelRouter("small", "strong"), metrics=metrics,
    )
    first = service.answer(user, "How many annual leave days?")
    second = service.answer(user, "How many annual leave days?")
    assert first.cache_status == "miss" and first.answer.model_id == "small"
    assert second.cache_status == "exact" and second.answer == first.answer
    assert metrics.summary()["cache_hit_rate"] == 0.5

    store.update(chunks[0].document_id, replace(acl, allowed_roles=frozenset(), revision=2))
    revoked = service.answer(user, "How many annual leave days?")
    assert revoked.cache_status == "miss"
    assert revoked.answer.citations == ()


def test_uncited_escalation_abstains_and_counts_both_calls():
    embedder = HashingEmbedder()
    index = MemoryIndex()
    store = ACLStore()
    acl = ACL("tenant", frozenset({"staff"}), frozenset())
    chunks = ingest_document("leave", "Annual leave is 20 days per year.", acl, embedder)
    index.upsert(chunks)
    store.update(chunks[0].document_id, acl)
    metrics = OptimizationMetrics({"small": ModelPrice(1, 2), "strong": ModelPrice(5, 10)}, "test")
    service = SecureRAGService(
        index=index, acl_store=store, embedder=embedder,
        generators={name: UncitedGenerator(name) for name in ("small", "strong")},
        default_model="strong", top_k=1, cache=AnswerCache(acl_store=store),
        router=ModelRouter("small", "strong"), metrics=metrics,
    )
    user = Principal("alice", "tenant", frozenset({"staff"}))

    result = service.answer(user, "How many annual leave days?")
    assert result.answer.text == "I don't know based on the available documents."
    assert result.answer.citations == ()
    assert result.route_reason == "single_document_fact+escalated"
    assert metrics.summary()["total_cost"] == 0.000084

    repeated = service.answer(user, "How many annual leave days?")
    assert repeated.cache_status == "miss"
