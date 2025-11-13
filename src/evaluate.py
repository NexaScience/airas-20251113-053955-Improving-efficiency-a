"""Evaluation & visualisation across all experiment variations.

This script MUST be called **after** all individual runs have finished.
It aggregates the per-run `results.json` files, computes comparison metrics,
and produces publication-ready figures under `<results_dir>/images/`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=str, help="Root directory containing all run sub-directories")
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------

def _load_results(results_dir: Path) -> List[Dict]:
    runs: List[Dict] = []
    for sub in results_dir.iterdir():
        if not sub.is_dir():
            continue
        res_file = sub / "results.json"
        if res_file.exists():
            with open(res_file, "r") as f:
                runs.append(json.load(f))
    if not runs:
        raise RuntimeError(f"No results.json found under {results_dir}")
    return runs


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def _plot_final_score_bar(runs: List[Dict], images_dir: Path):
    plt.figure(figsize=(6, 4))
    names = [r["run_id"] for r in runs]
    scores = [r["best_score"] for r in runs]

    sns.barplot(x=names, y=scores, palette="Set2")
    plt.ylabel("Best Score")
    plt.xlabel("Run ID")
    plt.title("Final Best Score Across Variations")
    # annotate each bar
    for idx, score in enumerate(scores):
        plt.text(idx, score + 0.01 * max(scores), f"{score:.2f}", ha="center")
    plt.tight_layout()
    fname = images_dir / "best_score_comparison.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def _plot_optimisation_curves(runs: List[Dict], images_dir: Path):
    plt.figure(figsize=(6, 4))
    for r in runs:
        hist = r["trial_history"]
        y = [d["best_so_far"] for d in hist]
        x = list(range(1, len(y) + 1))
        sns.lineplot(x=x, y=y, marker="o", label=r["run_id"])
        # annotate last point
        plt.annotate(f"{y[-1]:.2f}", xy=(x[-1], y[-1]), xytext=(5, 5), textcoords="offset points")

    plt.xlabel("Trial")
    plt.ylabel("Best-so-far Score")
    plt.title("Optimisation Progress Comparison")
    plt.legend()
    plt.tight_layout()
    fname = images_dir / "optimisation_curve_comparison.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Statistics helper
# -----------------------------------------------------------------------------

def _compute_summary(runs: List[Dict]):
    best_scores = {r["run_id"]: r["best_score"] for r in runs}
    fastest = min(runs, key=lambda d: d["wall_clock_seconds"])
    summary = {
        "best_scores": best_scores,
        "fastest_run": fastest["run_id"],
        "fastest_wall_clock_seconds": fastest["wall_clock_seconds"],
    }
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = _parse_args()
    results_dir = Path(args.results_dir).expanduser()
    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True, parents=True)

    runs = _load_results(results_dir)

    # Figures
    _plot_final_score_bar(runs, images_dir)
    _plot_optimisation_curves(runs, images_dir)

    # Summary JSON
    summary = _compute_summary(runs)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
