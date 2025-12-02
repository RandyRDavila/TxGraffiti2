from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import shutil

# PuLP is used as a fallback backend if SciPy is not present.
import pulp  # type: ignore


@dataclass(frozen=True)
class LPSolution:
    """Solution container for an affine fit y ~ a·x + b."""
    a: np.ndarray
    b: float


class LPBackend:
    """Abstracts the 'min sum of slacks' LP: sense ∈ {'lower','upper'}."""
    def solve(self, X: np.ndarray, y: np.ndarray, *, sense: str) -> LPSolution:  # pragma: no cover - abstract
        raise NotImplementedError


class SciPyBackend(LPBackend):
    """
    In-process HiGHS via scipy.optimize.linprog.
    Minimize sum(s_i) subject to:
      lower:   y_i >= a·x_i + b  ⇒  -a·x_i - b - s_i = -y_i, s_i ≥ 0
      upper:   y_i <= a·x_i + b  ⇒   a·x_i + b - s_i =  y_i, s_i ≥ 0
    """
    def __init__(self) -> None:
        from scipy.optimize import linprog as _linprog  # lazy import
        self._linprog = _linprog

    def solve(self, X: np.ndarray, y: np.ndarray, *, sense: str) -> LPSolution:
        if sense not in {"lower", "upper"}:
            raise ValueError("sense must be 'lower' or 'upper'")

        n, k = X.shape
        ones = np.ones((n, 1))
        I = np.eye(n)

        # Variables: a (k), b (1), s (n >= 0)
        c = np.r_[np.zeros(k + 1), np.ones(n)]  # minimize sum s_i
        if sense == "upper":
            A_eq = np.hstack([X, ones, -I]); b_eq = y
        else:  # lower
            A_eq = np.hstack([-X, -ones, -I]); b_eq = -y

        bounds = [(None, None)] * (k + 1) + [(0, None)] * n

        res = self._linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")

        x = res.x
        a = x[:k].astype(float, copy=False)
        b = float(x[k])
        return LPSolution(a=a, b=b)


class PuLPBackend(LPBackend):
    """
    External solver fallback (CBC or GLPK via PuLP).
    Keeps the same formulation/optimality as SciPyBackend.
    """
    def solve(self, X: np.ndarray, y: np.ndarray, *, sense: str) -> LPSolution:
        if sense not in {"lower", "upper"}:
            raise ValueError("sense must be 'lower' or 'upper'")

        n, k = X.shape

        # Pick an available CLI solver
        solver = None
        cbc = shutil.which("cbc")
        if cbc:
            solver = pulp.COIN_CMD(path=cbc, msg=False)
        glp = shutil.which("glpsol")
        if solver is None and glp:
            solver = pulp.GLPK_CMD(path=glp, msg=False)
        if solver is None:
            raise RuntimeError("No LP solver found (need SciPy or CBC/GLPK).")

        prob = pulp.LpProblem("sum_slack", pulp.LpMinimize)

        a_vars = [pulp.LpVariable(f"a_{j}") for j in range(k)]
        b_var = pulp.LpVariable("b")
        s_vars = [pulp.LpVariable(f"s_{i}", lowBound=0) for i in range(n)]

        # Objective: minimize sum(s_i)
        prob += pulp.lpSum(s_vars)

        # Constraints to encode slacks
        for i in range(n):
            lhs = pulp.lpSum(a_vars[j] * float(X[i, j]) for j in range(k)) + b_var
            if sense == "upper":
                # a·x + b - s = y
                prob += lhs - float(y[i]) == s_vars[i]
            else:
                # y - (a·x + b) = s
                prob += float(y[i]) - lhs == s_vars[i]

        status = prob.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"LP not optimal: {pulp.LpStatus[status]}")

        a = np.array([float(v.value()) for v in a_vars], dtype=float)
        b = float(b_var.value())
        return LPSolution(a=a, b=b)


def best_available_backend() -> LPBackend:
    """Prefer SciPy (in-process) else fallback to PuLP (external solver)."""
    try:
        import scipy  # noqa: F401
        return SciPyBackend()
    except Exception:
        return PuLPBackend()
