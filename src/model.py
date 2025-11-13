"""Core algorithmic components including the learning-curve compressor and
BOIL / ATW-BOIL optimisers.

All heavy-lifting lives here and is shared by every experimental variation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch
from torch import Tensor
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# -----------------------------------------------------------------------------
# Compressor
# -----------------------------------------------------------------------------


class TailWeightedLogisticCompressor:
    """Implements s_α curve compression (ATW-BOIL)."""

    def __init__(self, curve_length: int, midpoint: float = 0.0, growth: float = 1.0, alpha_initial: float = 0.0):
        self.curve_length = int(curve_length)
        self.midpoint = midpoint
        self.growth = growth
        self.alpha = alpha_initial  # mutable only for ATW-BOIL optimiser

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def compress(self, curve: List[float], alpha_override: float | None = None) -> float:
        alpha = self.alpha if alpha_override is None else alpha_override
        y = np.asarray(curve, dtype=float)
        T = len(y)
        t = np.arange(1, T + 1, dtype=float)
        logistic = 1.0 / (1.0 + np.exp(-self.growth * ((t / T) - self.midpoint)))
        weights = ((t / T) + 1e-9) ** alpha * logistic
        weights /= weights.sum()
        return float((weights * y).sum())


# -----------------------------------------------------------------------------
# Helper utilities for hyperparameter sampling
# -----------------------------------------------------------------------------


def _generate_sobol_points(bounds: Tensor, n: int) -> Tensor:
    from botorch.sampling import SobolQMCNormalSampler, SobolQMCSequence

    dim = bounds.size(1)
    sobol = torch.quasirandom.SobolEngine(dim, scramble=True)
    samples = sobol.draw(n)
    return bounds[0] + (bounds[1] - bounds[0]) * samples


# -----------------------------------------------------------------------------
# Base optimiser shared logic
# -----------------------------------------------------------------------------


class BOILBaseOptimiser:
    """Shared logic between original BOIL and ATW-BOIL."""

    def __init__(self, config: Dict[str, Any], compressor: TailWeightedLogisticCompressor):
        self.config = config
        self.compressor = compressor
        self.max_trials = int(config.get("hpo", {}).get("max_trials", 20))
        self.initial_trials = max(3, int(0.2 * self.max_trials))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Search-space parsing
        self.param_names, self.bounds = self._parse_search_space(config["search_space"])
        self.dim = self.bounds.shape[1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimise(self, task):
        X, y = self._initial_design(task)
        best_so_far = y.min().item()  # lower is better if we cast as negative reward
        history = [
            {
                "trial": i + 1,
                "hyperparams": self._tensor_to_hparams(xi),
                "compressed_score": yi.item(),
                "best_so_far": best_so_far,
            }
            for i, (xi, yi) in enumerate(zip(X, y))
        ]

        for trial in range(self.initial_trials, self.max_trials):
            model = self._fit_gp(X, y)
            candidate = self._select_next(model)
            cand_np = candidate.detach().cpu()
            hparams = self._tensor_to_hparams(cand_np)
            curve = task.evaluate(hparams)
            score = self._curve_to_scalar(curve)  # lower is better by convention (negative)
            X = torch.cat([X, candidate], dim=0)
            y = torch.cat([y, torch.tensor([[score]], dtype=torch.double, device=self.device)])
            best_so_far = min(best_so_far, score)
            history.append(
                {
                    "trial": trial + 1,
                    "hyperparams": hparams,
                    "compressed_score": score,
                    "best_so_far": best_so_far,
                }
            )

        # Convert back to *maximise* convention for user-friendly metrics
        best_idx = int(torch.argmin(y))
        best_x = X[best_idx]
        best_hparams = self._tensor_to_hparams(best_x)
        best_score = -float(y[best_idx].item())
        # change sign of best_so_far inside history as well
        for h in history:
            h["best_so_far"] = -h["best_so_far"]
            h["compressed_score"] = -h["compressed_score"]
        return best_score, best_hparams, history

    # ------------------------------------------------------------------
    # Internals – can be overridden
    # ------------------------------------------------------------------
    def _curve_to_scalar(self, curve: List[float]) -> float:
        """Return NEGATIVE scalar because BO maximises by default; we minimise."""
        return -self.compressor.compress(curve)

    def _select_next(self, model):
        bounds = self.bounds.to(self.device)
        acq = qExpectedImprovement(model=model, best_f=torch.min(model.train_targets), maximize=False)
        candidate, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=5, raw_samples=64)
        return candidate.detach()

    # ------------------------------------------------------------------
    # GP utils
    # ------------------------------------------------------------------
    def _fit_gp(self, X: Tensor, y: Tensor):
        gp = SingleTaskGP(X, y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp

    # ------------------------------------------------------------------
    # Initial design utilities
    # ------------------------------------------------------------------
    def _initial_design(self, task):
        bounds = self.bounds
        X_list, y_list = [], []
        sobol_pts = _generate_sobol_points(bounds, self.initial_trials)
        for xi in sobol_pts:
            hparams = self._tensor_to_hparams(xi)
            curve = task.evaluate(hparams)
            score = self._curve_to_scalar(curve)
            X_list.append(xi)
            y_list.append(score)
        X = torch.stack(X_list).to(torch.double)
        y = torch.tensor(y_list, dtype=torch.double).view(-1, 1)
        return X, y

    # ------------------------------------------------------------------
    # Search-space conversion
    # ------------------------------------------------------------------
    def _parse_search_space(self, space_cfg: Dict[str, Dict]):
        names = []
        lows = []
        highs = []
        for name, spec in space_cfg.items():
            names.append(name)
            bound_low, bound_high = float(spec["bounds"][0]), float(spec["bounds"][1])
            lows.append(bound_low)
            highs.append(bound_high)
        bounds = torch.tensor([lows, highs], dtype=torch.double)
        return names, bounds

    def _tensor_to_hparams(self, t: Tensor):
        hp = {}
        for idx, name in enumerate(self.param_names):
            hp[name] = float(t[idx].item())
        return hp


# -----------------------------------------------------------------------------
# Original BOIL – alpha is fixed 0
# -----------------------------------------------------------------------------


class BOILOptimiser(BOILBaseOptimiser):
    pass  # no extra change needed. compressor.alpha stays at whatever (0)


# -----------------------------------------------------------------------------
# Adaptive Tail-Weighted BOIL implementation
# -----------------------------------------------------------------------------


class ATWBOILOptimiser(BOILBaseOptimiser):
    """Extends BOIL by jointly optimising α via GP marginal likelihood."""

    def __init__(self, config, compressor):
        super().__init__(config, compressor)
        # bounds for alpha hyper-parameter during compression
        self.alpha_bounds = tuple(config.get("compressor", {}).get("alpha_bounds", [0.0, 3.0]))
        # Use the compressor's alpha attribute as mutable hyper-param.

    # ------------------------------------------------------------------
    # Override to update alpha each GP refit via grid search (simple impl)
    # ------------------------------------------------------------------
    def _curve_to_scalar(self, curve):
        # grid-search 3 values around current alpha for simplicity
        alphas = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1], num=5)
        best_alpha = self.compressor.alpha
        best_ll = math.inf
        for a in alphas:
            self.compressor.alpha = float(a)
            score = super()._curve_to_scalar(curve)
            if score < best_ll:
                best_ll = score
                best_alpha = a
        self.compressor.alpha = best_alpha
        return best_ll
