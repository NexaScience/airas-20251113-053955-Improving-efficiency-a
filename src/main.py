"""Top-level orchestration script.

Usage:
  # Smoke test
  uv run python -m src.main --smoke-test --results-dir <path>

  # Full experiment
  uv run python -m src.main --full-experiment --results-dir <path>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from src.utils import tee_subprocess


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true")
    group.add_argument("--full-experiment", action="store_true")
    p.add_argument("--results-dir", required=True, type=str)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
TRAIN_MODULE = "src.train"
EVALUATE_MODULE = "src.evaluate"


def _load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_tmp_run_config(run_cfg: Dict, scratch_dir: Path) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_dir / f"{run_cfg['run_id']}.yaml"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(run_cfg, f)
    return tmp_path


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------

def main():
    args = _parse_args()
    results_root = Path(args.results_dir).expanduser()
    results_root.mkdir(parents=True, exist_ok=True)

    # Pick config file
    cfg_file = CONFIG_DIR / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    master_cfg = _load_yaml(cfg_file)
    experiments: List[Dict] = master_cfg.get("experiments", [])
    if not experiments:
        print("No experiments found in config", file=sys.stderr)
        sys.exit(1)

    scratch_dir = results_root / "tmp_cfg"
    scratch_dir.mkdir(exist_ok=True)

    # Sequential execution
    for run_cfg in experiments:
        run_id = run_cfg["run_id"]
        run_dir = results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        run_config_path = _write_tmp_run_config(run_cfg, scratch_dir)
        cmd = [
            sys.executable,
            "-m",
            TRAIN_MODULE,
            "--config-path",
            str(run_config_path),
            "--results-dir",
            str(results_root),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        tee_subprocess(proc, run_dir / "stdout.log", run_dir / "stderr.log")
        if proc.returncode != 0:
            print(f"Run {run_id} failed with code {proc.returncode}", file=sys.stderr)
            sys.exit(proc.returncode)

    # After all runs, aggregate & evaluate
    cmd_eval = [sys.executable, "-m", EVALUATE_MODULE, "--results-dir", str(results_root)]
    subprocess.run(cmd_eval, check=True)


if __name__ == "__main__":
    main()
