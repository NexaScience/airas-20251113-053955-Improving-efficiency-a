"""Common data & task preprocessing utilities.

This module **must not** contain dataset-specific code beyond the placeholders.
The public function is `get_task(config)` which returns a Task instance with a
`evaluate(hyperparams) -> list[float]` method yielding a learning curve.

A lightweight *dummy* task is provided so that smoke-tests run without any
external dataset. Researchers should replace the placeholder sections with real
loading / simulation logic for their particular tasks.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Callable

import numpy as np

# -----------------------------------------------------------------------------
# Base Task abstraction
# -----------------------------------------------------------------------------


class TaskBase:
    """Abstract interface every task must implement."""

    def __init__(self, config):
        self.config = config
        self.curve_length = int(config.get("task", {}).get("curve_length", 50))
        self.seed = config.get("seed", 17)
        random.seed(self.seed)
        np.random.seed(self.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, hyperparams: Dict) -> List[float]:
        """Run one training run under *hyperparams*.

        Returns the full learning curve as a list of floats (length T).
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Dummy task for smoke-test
# -----------------------------------------------------------------------------


class DummyLearningCurveTask(TaskBase):
    """Synthetic curve resembling RL reward learning."""

    def __init__(self, config):
        super().__init__(config)
        self.noise_std = float(config.get("task", {}).get("noise_std", 2.0))

    def evaluate(self, hyperparams):
        # Hyper-param influences curve height & speed.
        lr = float(hyperparams.get("lr", 1e-3))
        gamma = float(hyperparams.get("gamma", 0.99))

        T = self.curve_length
        curve = []
        base_speed = 1.0 + 5.0 * lr  # faster learning with higher lr
        base_height = 100.0 + 50.0 * gamma  # higher asymptote with higher gamma

        for t in range(1, T + 1):
            progress = 1 - math.exp(-base_speed * t / T)
            value = base_height * progress + np.random.randn() * self.noise_std
            curve.append(value)
        return curve


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def get_task(config):
    """Factory that instantiates the correct Task for *config*."""
    task_name = config.get("task", {}).get("name", "dummy").lower()

    if task_name == "dummy":
        return DummyLearningCurveTask(config)

    # ------------------------------------------------------------------
    # PLACEHOLDER: Add real task loaders here (e.g. CartPole, CIFAR-10, ...)
    # ------------------------------------------------------------------
    # if task_name == "cartpole":
    #     from .tasks.cartpole_task import CartPoleTask
    #     return CartPoleTask(config)

    raise NotImplementedError(f"Task '{task_name}' not implemented yet – placeholder")
