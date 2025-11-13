"""Core algorithmic components including compressors and optimisers for every
experimental variation (baseline, proposed, ablations).
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
import gpytorch

# -----------------------------------------------------------------------------
# Compressor(s)
# -----------------------------------------------------------------------------


class TailWeightedLogisticCompressor:
    """Implements the general s_α learning-curve compression.

    Parameters
    ----------
    curve_length : int
        Expected length of incoming learning curves (used only for sanity).
    midpoint : float
        Mid-point of the logistic mask (original BOIL parameter).
    growth : float
        Growth factor of the logistic mask.
    alpha_initial : float
        Initial value for the tail-weight exponent α.
    """

    def __init__(self, curve_length: int, midpoint: float = 0.0, growth: float = 1.0, alpha_initial: float = 0.0):
        self.curve_length = int(curve_length)
        self.midpoint = midpoint
        self.growth = growth
        self.alpha = alpha_initial  # mutable – ATW variants update this

    # ------------------------------------------------------------------
    # Core method
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


class TailOnlyCompressor(TailWeightedLogisticCompressor):
    """Same as above but *without* the logistic component (ablation)."""

    def compress(self, curve: List[float], alpha_override: float | None = None) -> float:  # type: ignore[override]
        alpha = self.alpha if alpha_override is None else alpha_override
        y = np.asarray(curve, dtype=float)
        T = len(y)
        t = np.arange(1, T + 1, dtype=float)
        weights = ((t / T) + 1e-9) ** alpha
        weights /= weights.sum()
        return float((weights * y).sum())


# -----------------------------------------------------------------------------
# Helper utilities for hyperparameter sampling
# -----------------------------------------------------------------------------


def _generate_sobol_points(bounds: Tensor, n: int) -> Tensor:
    from botorch.sampling import SobolQMCSequence

    dim = bounds.size(1)
    sobol = torch.quasirandom.SobolEngine(dim, scramble=True)
    samples = sobol.draw(n)
    return bounds[0] + (bounds[1] - bounds[0]) * samples


# -----------------------------------------------------------------------------
# Base optimiser shared logic
# -----------------------------------------------------------------------------


class BOILBaseOptimiser:
    """Shared logic between BOIL variants and other baselines."""

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
        best_so_far = y.min().item()  # lower is better because we negate scores
        history = [
            {
                "trial": i + 1,
                "hyperparams": self._tensor_to_hparams(xi),
                "compressed_score": -yi.item(),  # user-facing positive metric
                "best_so_far": -best_so_far,
            }
            for i, (xi, yi) in enumerate(zip(X, y))
        ]

        for trial in range(self.initial_trials, self.max_trials):
            model = self._fit_gp(X, y)
            candidate = self._select_next(model)
            cand_np = candidate.detach().cpu()
            hparams = self._tensor_to_hparams(cand_np)
            curve = task.evaluate(hparams)
            score = self._curve_to_scalar(curve)  # NEGATIVE scalar
            X = torch.cat([X, candidate], dim=0)
            y = torch.cat([y, torch.tensor([[score]], dtype=torch.double, device=self.device)])
            best_so_far = min(best_so_far, score)
            history.append(
                {
                    "trial": trial + 1,
                    "hyperparams": hparams,
                    "compressed_score": -score,
                    "best_so_far": -best_so_far,
                }
            )

        # Identify best
        best_idx = int(torch.argmin(y))
        best_x = X[best_idx]
        best_hparams = self._tensor_to_hparams(best_x)
        best_score = -float(y[best_idx].item())
        return best_score, best_hparams, history

    # ------------------------------------------------------------------
    # Internals – can be overridden
    # ------------------------------------------------------------------
    def _curve_to_scalar(self, curve: List[float]) -> float:
        """Return NEGATIVE scalar (we minimise in internal GP)."""
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
        X_list, y_list = [], []
        sobol_pts = _generate_sobol_points(self.bounds, self.initial_trials)
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
# Original BOIL – α fixed to 0
# -----------------------------------------------------------------------------


class BOILOptimiser(BOILBaseOptimiser):
    pass  # behaviour already matches original BOIL (α = 0 in compressor)


# -----------------------------------------------------------------------------
# Adaptive Tail-Weighted BOIL implementation
# -----------------------------------------------------------------------------


class ATWBOILOptimiser(BOILBaseOptimiser):
    """Extends BOIL by updating α via a lightweight grid search around the current
    value after every curve evaluation. This keeps the implementation simple
    while still allowing α to adapt.
    """

    def __init__(self, config, compressor):
        super().__init__(config, compressor)
        self.alpha_bounds = tuple(config.get("compressor", {}).get("alpha_bounds", [0.0, 3.0]))

    # ------------------------------------------------------------------
    # Override: choose α that maximises the compressed score on the *latest*
    # curve (greedy). More sophisticated joint optimisation is possible but not
    # necessary for this implementation.
    # ------------------------------------------------------------------
    def _curve_to_scalar(self, curve):
        # Coarse grid search (5 points) within bounds
        grid = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1], num=5)
        best_alpha = grid[0]
        best_val = -math.inf
        for a in grid:
            val = self.compressor.compress(curve, alpha_override=a)
            if val > best_val:
                best_val = val
                best_alpha = a
        # Update compressor for *future* curves.
        self.compressor.alpha = best_alpha
        return -best_val  # negate for internal minimisation


# -----------------------------------------------------------------------------
# No-logistic ablation
# -----------------------------------------------------------------------------


class ATWBOILNoLogisticOptimiser(ATWBOILOptimiser):
    """Same as ATW-BOIL but without the logistic term (ablation study)."""

    def __init__(self, config, _):
        # Replace compressor with tail-only version
        compressor = TailOnlyCompressor(
            curve_length=config.get("task", {}).get("curve_length", 50),
            alpha_initial=config.get("compressor", {}).get("alpha", 0.0),
        )
        super().__init__(config, compressor)


# -----------------------------------------------------------------------------
# Sparse GP variant for scalability
# -----------------------------------------------------------------------------


class _SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ATWBOILSparseGPOptimiser(ATWBOILOptimiser):
    """Uses a variational sparse GP with a user-specified number of inducing points."""

    def __init__(self, config, compressor):
        super().__init__(config, compressor)
        self.n_inducing = int(config.get("gp", {}).get("inducing_points", 1024))

    # Override GP fitting
    def _fit_gp(self, X: Tensor, y: Tensor):
        # Choose inducing points as a subset of X (Sobol order guarantees spread)
        n_inducing = min(self.n_inducing, X.size(0))
        inducing = X[:n_inducing].contiguous()
        model = _SparseGPModel(inducing)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.train()
        likelihood.train()

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X.size(0))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        iters = 200 if X.size(0) > 100 else 100
        for i in range(iters):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, y.squeeze(-1))
            loss.backward()
            optimizer.step()
        model.eval()
        likelihood.eval()

        # Botorch expects a model object with .posterior; we wrap via ModelListGP-like dummy.
        class _Wrapper(torch.nn.Module):
            def __init__(self, model, likelihood):
                super().__init__()
                self.model = model
                self.likelihood = likelihood

            @property
            def train_targets(self):
                return y.squeeze(-1)

            def posterior(self, X_test):
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    preds = self.likelihood(self.model(X_test))
                return preds

        return _Wrapper(model, likelihood)


# -----------------------------------------------------------------------------
# Freeze-Thaw BO baseline (simplified)
# -----------------------------------------------------------------------------


class FreezeThawBOOptimiser(BOILBaseOptimiser):
    """Implements a *very* lightweight version of Freeze-Thaw BO: we approximate a
    partially-observed learning curve by its last value. In a real
    implementation the GP would be defined over both configuration and time,
    but for the purposes of this executable code we keep things simple and fast.
    """

    # We re-use everything from the base class except the curve-to-scalar mapping.
    def _curve_to_scalar(self, curve: List[float]) -> float:  # noqa: D401
        # Use the *last* observed performance as the scalar – mimics freeze-thaw.
        return -float(curve[-1])