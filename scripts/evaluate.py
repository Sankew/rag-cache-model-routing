"""Deterministic 100-request synthetic cache and router evaluation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from rag_portfolio.access import ACLStore
from rag_portfolio.baseline import ExtractiveDemoGenerator, HashingEmbedder, MemoryIndex, ingest_document
from rag_portfolio.domain import ACL, Answer, Principal
from rag_portfolio.optimization import AnswerCache, ModelPrice, ModelRouter, OptimizationMetrics
from rag_portfolio.service import SecureRAGService


class MeteredDemoGenerator:
    """Same extractive logic for both routes; counts tokens for accounting only."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.extractive = ExtractiveDemoGenerator()

    def generate(self, question, evidence) -> Answer:
        raw = self.extractive.generate(question, evidence)
        input_tokens = len(question.split()) + sum(len(hit.chunk.text.split()) for hit in evidence)
        return Answer(raw.text, raw.citations, self.model_id, input_tokens, len(raw.text.split()))


@dataclass(frozen=True)
class WorkloadCase:
    question: str
    kind: str
    policy_id: int


def build_workload() -> tuple[WorkloadCase, ...]:
    """25 first asks, 50 repeats, 24 near misses, and one multi-doc probe."""
    cases: list[WorkloadCase] = []
    for policy_id in range(1, 26):
        base = f"What is policy {policy_id:02d} standard value?"
        if policy_id == 1:
            probe = WorkloadCase(
                "Compare policy 01 standard value with policy 02 standard value?",
                "multi_document", policy_id,
            )
        elif policy_id <= 5:
            probe = WorkloadCase(
                f"Why is policy {policy_id:02d} exception value {200 + policy_id} units?",
                "near_miss", policy_id,
            )
        else:
            probe = WorkloadCase(
                f"What is policy {policy_id:02d} exception value?",
                "near_miss", policy_id,
            )
        cases.extend((
            WorkloadCase(base, "first", policy_id),
            WorkloadCase(base, "repeat", policy_id),
            probe,
            WorkloadCase(base, "repeat", policy_id),
        ))
    return tuple(cases)


def run() -> dict[str, object]:
    embedder = HashingEmbedder(dimensions=1024)
    index = MemoryIndex()
    acl_store = ACLStore()
    acl = ACL("sample", frozenset({"reader"}), frozenset(), classification=0)
    for policy_id in range(1, 26):
        text = (
            f"Policy {policy_id:02d} standard value is {100 + policy_id} units. "
            f"Policy {policy_id:02d} exception value is {200 + policy_id} units."
        )
        chunks = ingest_document(f"policy-{policy_id:02d}", text, acl, embedder)
        index.upsert(chunks)
        acl_store.update(chunks[0].document_id, acl)
    principal = Principal("reader-1", "sample", frozenset({"reader"}))
    workload = build_workload()
    counts = Counter(case.kind for case in workload)
    prices = {"small-demo": ModelPrice(1.0, 2.0), "strong-demo": ModelPrice(5.0, 10.0)}

    def measure(optimized: bool) -> dict[str, object]:
        metrics = OptimizationMetrics(prices, "illustrative-input")
        generators = {model: MeteredDemoGenerator(model) for model in prices}
        cache = AnswerCache(acl_store=acl_store) if optimized else None
        router = ModelRouter("small-demo", "strong-demo") if optimized else None
        services = {
            limit: SecureRAGService(
                index=index, acl_store=acl_store, embedder=embedder,
                generators=generators, default_model="strong-demo", top_k=limit,
                cache=cache, router=router, metrics=metrics,
            )
            for limit in (1, 2)
        }
        answers = [
            services[2 if case.kind == "multi_document" else 1].answer(principal, case.question)
            for case in workload
        ]
        summary = metrics.summary()
        return {
            "requests": summary["requests"],
            "cache_hits": sum(result.cache_status != "miss" for result in answers),
            "cache_hit_rate": summary["cache_hit_rate"],
            "near_miss_cache_hits": sum(
                result.cache_status != "miss"
                for case, result in zip(workload, answers) if case.kind == "near_miss"
            ),
            "routing_decisions": dict(sorted(Counter(result.route_reason for result in answers).items())),
            "estimated_generation_cost_usd": round(summary["total_cost"], 8),
            "latency_p50_ms": round(summary["p50_latency_ms"], 3),
            "latency_p95_ms": round(summary["p95_latency_ms"], 3),
            "answers": [result.answer.text for result in answers],
        }

    baseline = measure(False)
    optimized = measure(True)
    agreement = sum(a == b for a, b in zip(baseline["answers"], optimized["answers"]))
    near_miss_answer_differences = sum(
        baseline["answers"][offset] != baseline["answers"][offset + 2]
        for offset in range(0, len(workload), 4)
        if workload[offset + 2].kind == "near_miss"
    )
    baseline.pop("answers")
    optimized.pop("answers")
    return {
        "dataset": f"{len({case.policy_id for case in workload})} synthetic policy documents; {len(workload)} ordered requests",
        "workload": {
            "first_asks": counts["first"],
            "exact_repeats": counts["repeat"],
            "exact_repeat_rate": counts["repeat"] / len(workload),
            "unique_near_misses": len({case.question for case in workload if case.kind == "near_miss"}),
            "complex_routing_probes": sum(case.kind == "near_miss" and case.question.startswith("Why ") for case in workload),
            "multi_document_probes": counts["multi_document"],
        },
        "models": "both routes use the same deterministic extractive generator",
        "prices": "illustrative USD per million tokens; not provider prices",
        "answer_agreement": {"count": agreement, "total": len(workload)},
        "near_miss_answer_differences": near_miss_answer_differences,
        "baseline": baseline,
        "optimized": optimized,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
