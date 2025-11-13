import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# Absolute imports inside namespace package.
from src.preprocess import get_task  # type: ignore
from src.model import (
    TailWeightedLogisticCompressor,
    BOILOptimiser,
    ATWBOILOptimiser,
    ATWBOILNoLogisticOptimiser,
    ATWBOILSparseGPOptimiser,
    FreezeThawBOOptimiser,
)
from src.utils import set_global_seed, tee_stdout_stderr


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _parse_arguments():
    parser = argparse.ArgumentParser(description="Run one experiment variation (single HPO run)")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the JSON/YAML file with run config")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to save all artefacts for this run-id")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


_OPTIMISER_MAP = {
    "boil": BOILOptimiser,
    "boil-original": BOILOptimiser,
    "atw-boil": ATWBOILOptimiser,
    "atw-boil-nologistic": ATWBOILNoLogisticOptimiser,
    "atw-boil-sparsegp": ATWBOILSparseGPOptimiser,
    "freezethaw-bo": FreezeThawBOOptimiser,
}


def _instantiate_optimiser(config: Dict[str, Any], compressor: TailWeightedLogisticCompressor):
    algo_key = config["algorithm"].lower()
    if algo_key not in _OPTIMISER_MAP:
        raise ValueError(f"Unsupported algorithm {config['algorithm']}")
    cls = _OPTIMISER_MAP[algo_key]
    return cls(config, compressor)


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main():
    args = _parse_arguments()
    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load run-specific config
    # ---------------------------------------------------------------------
    if args.config_path.endswith(".json"):
        with open(args.config_path, "r") as f:
            run_cfg = json.load(f)
    else:
        import yaml  # local import to keep global dependencies clear

        with open(args.config_path, "r") as f:
            run_cfg = yaml.safe_load(f)

    run_id: str = run_cfg.get("run_id", str(uuid.uuid4()))
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout / stderr *within this process* as well so train.py keeps
    # its own tee (main.py already tees outer level). This gives us clean logs
    # even if users run train.py directly.
    tee_stdout_stderr(run_dir / "stdout.log", run_dir / "stderr.log")

    set_global_seed(run_cfg.get("seed", 17))

    # ---------------------------------------------------------------------
    # Prepare Task + Objective
    # ---------------------------------------------------------------------
    task = get_task(run_cfg)

    # ---------------------------------------------------------------------
    # Compressor shared by BOIL and ATW-BOIL
    # ---------------------------------------------------------------------
    compressor = TailWeightedLogisticCompressor(
        curve_length=task.curve_length,
        midpoint=run_cfg.get("compressor", {}).get("midpoint", 0.0),
        growth=run_cfg.get("compressor", {}).get("growth", 1.0),
        # alpha is ignored by BOILOptimiser internally
        alpha_initial=run_cfg.get("compressor", {}).get("alpha", 0.0),
    )

    optimiser = _instantiate_optimiser(run_cfg, compressor)

    # ---------------------------------------------------------------------
    # Run Hyper-Parameter Optimisation
    # ---------------------------------------------------------------------
    t0 = time.time()
    best_score, best_hparams, trial_history = optimiser.optimise(task)
    wall_clock = time.time() - t0

    # ---------------------------------------------------------------------
    # Save artefacts
    # ---------------------------------------------------------------------
    metrics = {
        "run_id": run_id,
        "algorithm": run_cfg["algorithm"],
        "best_score": float(best_score),
        "best_hyperparams": best_hparams,
        "trial_history": trial_history,  # list[dict]
        "wall_clock_seconds": wall_clock,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Produce simple optimisation-curve figure for this individual run
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import seaborn as sns

    scores = [d["best_so_far"] for d in trial_history]
    trials = list(range(1, len(scores) + 1))

    plt.figure(figsize=(6, 4))
    sns.lineplot(x=trials, y=scores, marker="o", label=run_id)
    plt.xlabel("Trial")
    plt.ylabel("Best-so-far Score")
    plt.title(f"Optimisation Progress – {run_id}")
    # Annotate final value
    plt.annotate(f"{scores[-1]:.2f}", xy=(trials[-1], scores[-1]), xytext=(5, 5), textcoords="offset points")
    plt.legend()
    plt.tight_layout()

    img_dir = run_dir / "images"
    img_dir.mkdir(exist_ok=True, parents=True)
    fig_name = "best_score.pdf"
    plt.savefig(img_dir / fig_name, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Final stdout JSON (must be single line)
    # ------------------------------------------------------------------
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()