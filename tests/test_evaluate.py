from collections import Counter

from scripts.evaluate import build_workload, run


def test_workload_has_documented_repeats_and_distinct_near_misses():
    cases = build_workload()
    assert len(cases) == 100
    assert Counter(case.kind for case in cases) == {
        "first": 25, "repeat": 50, "near_miss": 24, "multi_document": 1,
    }
    for offset in range(0, 100, 4):
        first, repeat_one, probe, repeat_two = cases[offset : offset + 4]
        assert first.question == repeat_one.question == repeat_two.question
        assert probe.question != first.question
        assert first.policy_id == probe.policy_id
    assert len({case.question for case in cases if case.kind == "near_miss"}) == 24


def test_report_checks_cache_isolation_and_answer_agreement():
    report = run()
    assert report["workload"]["exact_repeat_rate"] == 0.5
    assert report["workload"]["unique_near_misses"] == 24
    assert report["workload"]["multi_document_probes"] == 1
    assert report["answer_agreement"] == {"count": 100, "total": 100}
    assert report["near_miss_answer_differences"] == 24
    baseline, optimized = report["baseline"], report["optimized"]
    assert baseline["cache_hits"] == 0
    assert optimized["cache_hits"] == 50
    assert optimized["near_miss_cache_hits"] == 0
    assert optimized["routing_decisions"] == {
        "cache_hit": 50,
        "complex_question": 4,
        "multiple_documents": 1,
        "single_document_fact": 45,
    }
    assert optimized["estimated_generation_cost_usd"] < baseline["estimated_generation_cost_usd"]
    for group in (baseline, optimized):
        assert group["latency_p50_ms"] <= group["latency_p95_ms"]
