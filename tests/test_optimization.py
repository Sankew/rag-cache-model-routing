"""Behavioral tests for cache isolation, invalidation, routing, and accounting."""

from dataclasses import replace

import pytest

from rag_portfolio.access import ACLStore
from rag_portfolio.domain import ACL, Answer, Chunk, Citation, Principal
from rag_portfolio.optimization import AnswerCache, ModelPrice, ModelRouter, OptimizationMetrics, calibrate_semantic_threshold


def fixture_data():
    principal = Principal("alice", "tenant-a", frozenset({"engineer"}), clearance=2)
    acl = ACL("tenant-a", frozenset({"engineer"}), frozenset(), classification=1, revision=1)
    chunk = Chunk("chunk-1", "doc-1", "The deadline is Friday.", acl, 1, (1.0, 0.0))
    answer = Answer("Friday.", (Citation("doc-1", "chunk-1", 1),), "small", 20, 3)
    return principal, chunk, answer


REVISIONS = {"corpus_revision": "corpus-1", "index_revision": "index-1", "pipeline_revision": "pipeline-1"}


def test_cache_is_scoped_to_principal_tenant_and_revisions():
    principal, chunk, answer = fixture_data()
    cache = AnswerCache()
    cache.put(principal, [chunk], "  When is the DEADLINE? ", answer, **REVISIONS)
    hit = cache.get(principal, [chunk], "when is the deadline?", **REVISIONS)
    assert hit is not None and hit.match_type == "exact" and hit.answer == answer

    another_user = replace(principal, user_id="bob")
    assert cache.get(another_user, [chunk], "when is the deadline?", **REVISIONS) is None
    another_tenant = replace(principal, tenant_id="tenant-b")
    assert cache.get(another_tenant, [chunk], "when is the deadline?", **REVISIONS) is None
    assert cache.get(principal, [chunk], "when is the deadline?", **(REVISIONS | {"index_revision": "index-2"})) is None
    assert cache.get(principal, [chunk], "when is the deadline?", **(REVISIONS | {"pipeline_revision": "pipeline-2"})) is None


def test_acl_revocation_and_source_changes_invalidate_cache():
    principal, chunk, answer = fixture_data()
    store = ACLStore()
    store.update(chunk.document_id, chunk.acl)

    cache = AnswerCache(acl_store=store)
    cache.put(principal, [chunk], "deadline?", answer, **REVISIONS)
    assert cache.get(principal, [chunk], "deadline?", **REVISIONS) is not None

    revoked = replace(chunk.acl, allowed_roles=frozenset(), revision=2)
    store.update(chunk.document_id, revoked)
    assert cache.get(principal, [chunk], "deadline?", **REVISIONS) is None

    restored = replace(chunk.acl, revision=3)
    store.update(chunk.document_id, restored)
    assert cache.get(principal, [chunk], "deadline?", **REVISIONS) is None
    changed_acl = replace(chunk, acl=restored)
    assert cache.get(principal, [changed_acl], "deadline?", **REVISIONS) is None
    changed_source = replace(chunk, version=2)
    assert cache.get(principal, [changed_source], "deadline?", **REVISIONS) is None


def test_cache_rejects_unauthorized_or_uncited_evidence():
    principal, chunk, answer = fixture_data()
    cache = AnswerCache()
    denied = replace(chunk, acl=replace(chunk.acl, classification=3))
    with pytest.raises(ValueError):
        cache.put(principal, [denied], "deadline?", answer, **REVISIONS)
    with pytest.raises(ValueError):
        cache.put(principal, [chunk], "deadline?", replace(answer, citations=(Citation("other", "missing", 1),)), **REVISIONS)


def test_semantic_cache_hits_only_within_calibrated_threshold_and_scope():
    principal, chunk, answer = fixture_data()
    cache = AnswerCache(semantic_threshold=0.98)
    cache.put(principal, [chunk], "When is the deadline?", answer, embedding=(1.0, 0.0), **REVISIONS)
    hit = cache.get(principal, [chunk], "What date is it due?", embedding=(0.999, 0.02), **REVISIONS)
    assert hit is not None and hit.match_type == "semantic"
    assert cache.get(principal, [chunk], "Was it cancelled?", embedding=(0.7, 0.7), **REVISIONS) is None
    assert cache.get(replace(principal, user_id="bob"), [chunk], "What date is it due?", embedding=(1.0, 0.0), **REVISIONS) is None
    with pytest.raises(ValueError):
        AnswerCache(semantic_threshold=1.2)


def test_router_uses_evidence_and_records_reason():
    _, chunk, _ = fixture_data()
    router = ModelRouter("small", "large")
    assert router.route("When is the deadline?", [chunk]).model_id == "small"
    assert router.route("Why was the deadline changed?", [chunk]).reason == "complex_question"
    assert router.route("When is the deadline?", []).reason == "no_evidence"
    other = replace(chunk, document_id="doc-2", chunk_id="chunk-2")
    assert router.route("When is the deadline?", [chunk, other]).reason == "multiple_documents"


def test_router_complex_terms_match_whole_words_or_phrases():
    _, chunk, _ = fixture_data()
    router = ModelRouter("small", "large")
    assert router.route("Can we compare the policy values?", [chunk]).reason == "complex_question"
    assert router.route("What is the trade-off?", [chunk]).reason == "complex_question"
    assert router.route("What is the acrossing status?", [chunk]).reason == "single_document_fact"
    assert router.route("Whysoever is the deadline?", [chunk]).reason == "single_document_fact"


def test_metrics_use_recorded_tokens_and_pricing_snapshot():
    metrics = OptimizationMetrics({"small": ModelPrice(1.0, 2.0)}, "2026-09-27")
    metrics.record(model_id="small", input_tokens=1000, output_tokens=500, latency_ms=100, cache_status="miss", route_reason="single_document_fact")
    metrics.record(model_id="small", input_tokens=0, output_tokens=0, latency_ms=10, cache_status="exact", route_reason="cache_hit")
    summary = metrics.summary()
    assert summary["requests"] == 2
    assert summary["cache_hit_rate"] == 0.5
    assert summary["total_cost"] == pytest.approx(0.002)
    assert summary["p50_latency_ms"] == 55
    assert summary["p95_latency_ms"] == pytest.approx(95.5)


def test_cache_ttl_and_capacity():
    principal, chunk, answer = fixture_data()
    now = [0.0]
    cache = AnswerCache(max_entries=1, ttl_seconds=10, clock=lambda: now[0])
    cache.put(principal, [chunk], "first", answer, **REVISIONS)
    cache.put(principal, [chunk], "second", answer, **REVISIONS)
    assert cache.get(principal, [chunk], "first", **REVISIONS) is None
    assert cache.get(principal, [chunk], "second", **REVISIONS) is not None
    now[0] = 10.0
    assert cache.get(principal, [chunk], "second", **REVISIONS) is None


def test_semantic_threshold_calibration_rejects_near_misses():
    threshold, recall = calibrate_semantic_threshold([0.96, 0.98], [0.92, 0.94], [0.93, 0.95, 0.97])
    assert threshold == 0.95
    assert recall == 1.0


def test_escalation_cost_includes_both_models():
    metrics = OptimizationMetrics({"small": ModelPrice(1, 2), "large": ModelPrice(3, 4)}, "2026-09-27")
    metrics.record(model_id="large", input_tokens=1000, output_tokens=500, latency_ms=100,
                   cache_status="miss", route_reason="escalated", extra_usage=(("small", 1000, 500),))
    assert metrics.summary()["total_cost"] == pytest.approx(0.007)
