#!/usr/bin/env python3
"""Validate and summarize the frozen CUAD full-document vs dense-RAG run."""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).parent
MANIFEST = ROOT / "test_manifest.json"
RESULTS = ROOT / "results" / "test_metrics.jsonl"
SUMMARY = ROOT / "results" / "run_summary.json"
OUTPUT = ROOT / "results" / "analysis.json"
PRICE_PER_MILLION = 0.20
BOOTSTRAPS = 20_000
ORDERS_PER_CONTRACT = 20
SEED = 261126


def percentile_interval(values: list[float], alpha: float = 0.05) -> list[float]:
    return [quantile(values, alpha / 2), quantile(values, 1 - alpha / 2)]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def row_price(row: dict) -> float:
    return (
        row["input_tokens"] + row["output_tokens"]
    ) * PRICE_PER_MILLION / 1_000_000


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    summary = json.loads(SUMMARY.read_text())
    manifest_contracts = manifest["contracts"]
    expected = {
        question["id"]: (contract["title"], bool(question["answerable"]))
        for contract in manifest_contracts
        for question in contract["questions"]
    }
    records = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
    by_qid: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in records:
        qid, method = row["qid"], row["method"]
        if qid not in expected:
            raise ValueError(f"Unexpected question ID: {qid}")
        if method in by_qid[qid]:
            raise ValueError(f"Duplicate method row: {qid}, {method}")
        if method not in {"full_document", "dense_rag"}:
            raise ValueError(f"Unexpected method: {method}")
        title, answerable = expected[qid]
        if row["contract_title"] != title or row["answerable"] != answerable:
            raise ValueError(f"Manifest mismatch: {qid}")
        if row["input_tokens"] <= 0 or row["output_tokens"] < 0:
            raise ValueError(f"Invalid token counts: {qid}, {method}")
        if not 0 <= row["exact_match"] <= 1 or not 0 <= row["token_f1"] <= 1:
            raise ValueError(f"Invalid score: {qid}, {method}")
        by_qid[qid][method] = row
    if len(by_qid) != len(expected):
        raise ValueError(f"Expected {len(expected)} questions, found {len(by_qid)}")
    for qid, pair in by_qid.items():
        if set(pair) != {"full_document", "dense_rag"}:
            raise ValueError(f"Incomplete paired result: {qid}")

    by_contract: dict[str, list[str]] = defaultdict(list)
    for qid, (title, _answerable) in expected.items():
        by_contract[title].append(qid)

    metric_summary = {}
    diffs_by_contract: dict[str, list[float]] = defaultdict(list)
    for method in ("full_document", "dense_rag"):
        method_rows = [pair[method] for pair in by_qid.values()]
        metric_summary[method] = {}
        for label, selected in (
            ("all", method_rows),
            ("answerable", [row for row in method_rows if row["answerable"]]),
            ("no_answer", [row for row in method_rows if not row["answerable"]]),
        ):
            metric_summary[method][label] = {
                "n": len(selected),
                "exact_match": statistics.mean(r["exact_match"] for r in selected),
                "token_f1": statistics.mean(r["token_f1"] for r in selected),
            }
        metric_summary[method]["input_tokens"] = sum(r["input_tokens"] for r in method_rows)
        metric_summary[method]["output_tokens"] = sum(r["output_tokens"] for r in method_rows)
        metric_summary[method]["price_equivalent_usd"] = sum(row_price(r) for r in method_rows)
        metric_summary[method]["mean_end_to_end_seconds"] = statistics.mean(
            r["end_to_end_seconds"] for r in method_rows
        )
        metric_summary[method]["median_end_to_end_seconds"] = statistics.median(
            r["end_to_end_seconds"] for r in method_rows
        )

    f1_by_contract = {
        method: defaultdict(list) for method in ("full_document", "dense_rag")
    }
    unclipped_f1_by_contract = {
        method: defaultdict(list) for method in ("full_document", "dense_rag")
    }
    output_cap_hits = {
        method: {"all": 0, "answerable": 0, "no_answer": 0}
        for method in ("full_document", "dense_rag")
    }
    any_method_cap_hits = 0
    for qid, pair in by_qid.items():
        if any(pair[method]["output_tokens"] >= summary["max_new_tokens"] for method in pair):
            any_method_cap_hits += 1
        for method in ("full_document", "dense_rag"):
            row = pair[method]
            if row["output_tokens"] >= summary["max_new_tokens"]:
                output_cap_hits[method]["all"] += 1
                output_cap_hits[method]["answerable" if row["answerable"] else "no_answer"] += 1
            if row["answerable"]:
                title = row["contract_title"]
                f1_by_contract[method][title].append(row["token_f1"])
                if all(pair[m]["output_tokens"] < summary["max_new_tokens"] for m in pair):
                    unclipped_f1_by_contract[method][title].append(row["token_f1"])
        if pair["full_document"]["answerable"]:
            title = pair["full_document"]["contract_title"]
            diffs_by_contract[title].append(
                pair["dense_rag"]["token_f1"] - pair["full_document"]["token_f1"]
            )
    titles = sorted(by_contract)
    rng = random.Random(SEED)
    bootstrap_diffs = []
    bootstrap_method_f1 = {method: [] for method in ("full_document", "dense_rag")}
    bootstrap_unclipped_diffs = []
    for _ in range(BOOTSTRAPS):
        sampled_titles = rng.choices(titles, k=len(titles))
        sampled = [d for title in sampled_titles for d in diffs_by_contract[title]]
        bootstrap_diffs.append(statistics.mean(sampled))
        for method in ("full_document", "dense_rag"):
            method_values = [
                value for title in sampled_titles
                for value in f1_by_contract[method][title]
            ]
            bootstrap_method_f1[method].append(statistics.mean(method_values))
        unclipped_full = [
            value for title in sampled_titles
            for value in unclipped_f1_by_contract["full_document"][title]
        ]
        unclipped_rag = [
            value for title in sampled_titles
            for value in unclipped_f1_by_contract["dense_rag"][title]
        ]
        bootstrap_unclipped_diffs.append(
            statistics.mean(unclipped_rag) - statistics.mean(unclipped_full)
        )
    observed_f1_diff = (
        metric_summary["dense_rag"]["answerable"]["token_f1"]
        - metric_summary["full_document"]["answerable"]["token_f1"]
    )
    unclipped_answerable_n = sum(
        len(values) for values in unclipped_f1_by_contract["full_document"].values()
    )
    unclipped_f1_diff = (
        sum(v for vals in unclipped_f1_by_contract["dense_rag"].values() for v in vals)
        / unclipped_answerable_n
        - sum(v for vals in unclipped_f1_by_contract["full_document"].values() for v in vals)
        / unclipped_answerable_n
    )

    # Random question orders expose how the relative per-question costs vary.
    # Cost is a model-token price equivalent; local embedding/index work is separate.
    order_rng = random.Random(SEED)
    curves = []
    first_crossings = []
    sustained_crossings = []
    contract_time_diffs = []
    q_count = len(next(iter(by_contract.values())))
    if any(len(ids) != q_count for ids in by_contract.values()):
        raise ValueError("Selected contracts do not have a common question count")
    for title in titles:
        ids = sorted(by_contract[title])
        pair_costs = {
            qid: {method: row_price(by_qid[qid][method]) for method in ("full_document", "dense_rag")}
            for qid in ids
        }
        full_times = [by_qid[qid]["full_document"]["end_to_end_seconds"] for qid in ids]
        rag_times = [by_qid[qid]["dense_rag"]["end_to_end_seconds"] for qid in ids]
        contract_time_diffs.append(statistics.mean(f - r for f, r in zip(full_times, rag_times)))
        for _ in range(ORDERS_PER_CONTRACT):
            shuffled = list(ids)
            order_rng.shuffle(shuffled)
            cumulative = {"full_document": 0.0, "dense_rag": 0.0}
            curve = {0: {"full_document": 0.0, "dense_rag": 0.0}}
            crossing = None
            costs_by_k = [curve[0]]
            for k, qid in enumerate(shuffled, start=1):
                for method in cumulative:
                    cumulative[method] += pair_costs[qid][method]
                curve[k] = cumulative.copy()
                costs_by_k.append(cumulative.copy())
                if crossing is None and cumulative["dense_rag"] < cumulative["full_document"]:
                    crossing = k
            first_crossings.append(crossing)
            sustained = next(
                (
                    k for k in range(1, q_count + 1)
                    if all(costs_by_k[j]["dense_rag"] < costs_by_k[j]["full_document"]
                           for j in range(k, q_count + 1))
                ),
                None,
            )
            sustained_crossings.append(sustained)
            curves.append(curve)

    curve_rows = []
    for k in range(q_count + 1):
        full = [curve[k]["full_document"] for curve in curves]
        rag = [curve[k]["dense_rag"] for curve in curves]
        curve_rows.append({
            "questions_per_contract": k,
            "full_document_median_usd": statistics.median(full),
            "dense_rag_median_usd": statistics.median(rag),
            "full_document_p05_usd": quantile(full, 0.05),
            "full_document_p95_usd": quantile(full, 0.95),
            "dense_rag_p05_usd": quantile(rag, 0.05),
            "dense_rag_p95_usd": quantile(rag, 0.95),
            "share_contract_orders_rag_cheaper": sum(r < f for r, f in zip(rag, full)) / len(full) if k else None,
        })
    observed_crossings = [x for x in first_crossings if x is not None]
    observed_sustained = [x for x in sustained_crossings if x is not None]

    total_full = metric_summary["full_document"]["price_equivalent_usd"]
    total_rag = metric_summary["dense_rag"]["price_equivalent_usd"]
    mean_contract_latency_saving = statistics.mean(contract_time_diffs)
    answerable_coverage = [
        pair["dense_rag"]["gold_span_coverage"]
        for pair in by_qid.values()
        if pair["dense_rag"]["gold_span_coverage"] is not None
    ]
    out = {
        "validation": {
            "manifest_question_count": len(expected),
            "result_question_count": len(by_qid),
            "result_row_count": len(records),
            "complete_pairs": sum(len(pair) == 2 for pair in by_qid.values()),
            "contract_count": len(by_contract),
            "rows_per_contract": sorted(set(map(len, by_contract.values()))),
            "answerability_mismatches": 0,
            "duplicate_rows": 0,
            "all_scores_and_tokens_in_range": True,
            "chunk_audit_max_tokens_including_specials": 483,
            "embedding_limit_tokens": 512,
        },
        "results_recomputed": metric_summary,
        "derived": {
            "dense_rag_input_token_reduction_fraction": 1 - metric_summary["dense_rag"]["input_tokens"] / metric_summary["full_document"]["input_tokens"],
            "dense_rag_output_token_reduction_fraction": 1 - metric_summary["dense_rag"]["output_tokens"] / metric_summary["full_document"]["output_tokens"],
            "dense_rag_price_equivalent_reduction_fraction": 1 - total_rag / total_full,
            "answerable_token_f1_difference_dense_minus_full": observed_f1_diff,
            "answerable_token_f1_difference_contract_bootstrap_95_ci": percentile_interval(bootstrap_diffs),
            "answerable_token_f1_by_method_contract_bootstrap_95_ci": {
                method: percentile_interval(values)
                for method, values in bootstrap_method_f1.items()
            },
            "answerable_token_f1_noninferiority_margin": -0.05,
            "passes_prespecified_quality_gate": percentile_interval(bootstrap_diffs)[0] > -0.05,
            "output_cap_hit_counts": {
                "max_new_tokens": summary["max_new_tokens"],
                "by_method_and_answerability": output_cap_hits,
                "question_pairs_with_either_method_at_cap": any_method_cap_hits,
            },
            "unclipped_answerable_sensitivity": {
                "question_count": unclipped_answerable_n,
                "dense_minus_full_token_f1_difference": unclipped_f1_diff,
                "contract_bootstrap_95_ci": percentile_interval(bootstrap_unclipped_diffs),
                "interpretation": "Diagnostic subset excluding an answerable question if either method reached the shared output limit; not a replacement for the primary analysis.",
            },
            "dense_rag_answerable_max_gold_span_character_coverage_mean": statistics.mean(answerable_coverage),
            "price_equivalent_delta_usd": total_full - total_rag,
            "price_equivalent_is_not_an_api_invoice": True,
        },
        "question_order_cost_curve": {
            "random_orders_per_contract": ORDERS_PER_CONTRACT,
            "seed": SEED,
            "curve_rows": curve_rows,
            "contract_order_first_crossing_question_count_median_when_crossed": statistics.median(observed_crossings) if observed_crossings else None,
            "contract_order_first_crossing_question_count_p05_p95_when_crossed": [quantile(observed_crossings, 0.05), quantile(observed_crossings, 0.95)] if observed_crossings else None,
            "contract_orders_with_any_crossing": len(observed_crossings),
            "contract_orders_with_any_crossing_fraction": len(observed_crossings) / len(first_crossings),
            "contract_order_sustained_crossing_question_count_median": statistics.median(observed_sustained) if observed_sustained else None,
            "contract_order_sustained_crossing_p05_p95": [quantile(observed_sustained, 0.05), quantile(observed_sustained, 0.95)] if observed_sustained else None,
            "contract_orders_with_sustained_crossing": len(observed_sustained),
            "contract_orders_with_sustained_crossing_fraction": len(observed_sustained) / len(sustained_crossings),
            "contract_orders_with_no_crossing_by_final_question": sum(x is None for x in first_crossings),
            "contract_order_paths_rag_cheaper_at_question_1_fraction": curve_rows[1]["share_contract_orders_rag_cheaper"],
            "contract_order_paths_rag_cheaper_at_question_41_fraction": curve_rows[-1]["share_contract_orders_rag_cheaper"],
            "interpretation": "Order variability across sampled questions, not a confidence interval. Local index build time and storage are excluded from these token-price curves.",
        },
        "reported_run_summary_matches_recomputed_totals": {
            "dense_rag_price_equivalent": abs(summary["results"]["dense_rag"]["price_equivalent_usd"] - total_rag) < 1e-8,
            "full_document_price_equivalent": abs(summary["results"]["full_document"]["price_equivalent_usd"] - total_full) < 1e-8,
        },
        "local_index_measurements": {
            "index_build_seconds_across_30_contracts": summary["index_build_seconds"],
            "index_vector_bytes": summary["index_vector_bytes"],
            "index_text_bytes": summary["index_text_bytes"],
            "combined_vector_plus_text_payload_bytes": summary["index_payload_bytes"],
            "average_index_build_seconds_per_contract": summary["index_build_seconds"] / len(by_contract),
            "mean_per_question_end_to_end_seconds_saved_by_rag": metric_summary["full_document"]["mean_end_to_end_seconds"] - metric_summary["dense_rag"]["mean_end_to_end_seconds"],
            "mean_latency_break_even_questions_using_average_build_and_saved_time": (
                (summary["index_build_seconds"] / len(by_contract)) / mean_contract_latency_saving
                if mean_contract_latency_saving > 0 else None
            ),
            "per_contract_mean_latency_savings_seconds_median": statistics.median(contract_time_diffs),
            "note": "Payload is raw vectors plus chunk text; it excludes Python/object and vector-index metadata. Retrieval is exact NumPy cosine scanning, not an ANN index.",
        },
    }
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "validation": out["validation"],
        "results_recomputed": out["results_recomputed"],
        "derived": out["derived"],
        "first_crossing_median_when_crossed": out["question_order_cost_curve"]["contract_order_first_crossing_question_count_median_when_crossed"],
        "first_crossing_p05_p95_when_crossed": out["question_order_cost_curve"]["contract_order_first_crossing_question_count_p05_p95_when_crossed"],
        "no_crossing_order_count": out["question_order_cost_curve"]["contract_orders_with_no_crossing_by_final_question"],
        "sustained_crossing_fraction": out["question_order_cost_curve"]["contract_orders_with_sustained_crossing_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
