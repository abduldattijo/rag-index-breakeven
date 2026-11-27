#!/usr/bin/env python3
"""Create the article figure from the validated local analysis output."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
ANALYSIS = ROOT / "results" / "analysis.json"
OUT = ROOT / "figures" / "rag_index_cost_quality.png"


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text())
    rows = analysis["question_order_cost_curve"]["curve_rows"][1:]
    x = np.array([row["questions_per_contract"] for row in rows])

    fig, (ax_cost, ax_quality) = plt.subplots(
        1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    colors = {"full": "#245a81", "rag": "#df7b28"}

    for label, key, low_key, high_key, color in (
        (
            "Full contract prompt",
            "full_document_median_usd",
            "full_document_p05_usd",
            "full_document_p95_usd",
            colors["full"],
        ),
        (
            "Dense retrieval",
            "dense_rag_median_usd",
            "dense_rag_p05_usd",
            "dense_rag_p95_usd",
            colors["rag"],
        ),
    ):
        median = np.array([row[key] for row in rows]) * 100
        low = np.array([row[low_key] for row in rows]) * 100
        high = np.array([row[high_key] for row in rows]) * 100
        ax_cost.plot(x, median, label=label, color=color, linewidth=2.4)
        ax_cost.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)

    ax_cost.set_title("Median token price per contract")
    ax_cost.set_xlabel("Questions asked about the same contract")
    ax_cost.set_ylabel("Total token price, US cents")
    ax_cost.set_xticks([1, 5, 10, 20, 30, 41])
    ax_cost.grid(axis="y", alpha=0.22)
    ax_cost.legend(frameon=False, loc="upper left")

    derived = analysis["derived"]
    diff = derived["answerable_token_f1_difference_dense_minus_full"] * 100
    lo, hi = [value * 100 for value in derived["answerable_token_f1_difference_contract_bootstrap_95_ci"]]
    gate = derived["answerable_token_f1_noninferiority_margin"] * 100
    ax_quality.axvline(0, color="#777777", linewidth=1)
    ax_quality.axvline(gate, color="#a13d36", linestyle="--", linewidth=1.8)
    ax_quality.errorbar(
        diff,
        0,
        xerr=[[diff - lo], [hi - diff]],
        fmt="o",
        color=colors["rag"],
        markersize=8,
        capsize=6,
        linewidth=2,
    )
    ax_quality.set_xlim(-14, 1)
    ax_quality.set_ylim(-0.65, 0.65)
    ax_quality.set_yticks([])
    ax_quality.set_xlabel("Dense RAG minus full prompt, F1 points")
    ax_quality.set_title("Answerable question quality")
    ax_quality.grid(axis="x", alpha=0.22)
    ax_quality.text(
        0.03,
        0.92,
        f"Difference {diff:.1f} points\n95 percent interval\n{lo:.1f} to {hi:.1f}",
        transform=ax_quality.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )
    ax_quality.text(
        gate - 0.2,
        -0.43,
        f"Quality limit\n{gate:.0f} points",
        ha="right",
        va="center",
        fontsize=8.5,
        color="#8d302b",
    )

    fig.tight_layout(w_pad=2.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
