# src/txgraffiti2025/generators/lp.py

"""
Linear programming-based conjecture generator (R2 family).

Fits linear LOWER/UPPER bounds on a target column using a sum-of-slacks LP:

- LOWER bound (≥): minimize ∑ s_i  s.t. y_i - (a·x_i + b) = s_i,  s_i ≥ 0
- UPPER bound (≤): minimize ∑ s_i  s.t. (a·x_i + b) - y_i = s_i,  s_i ≥ 0

Returns Conjecture objects of the form:
    H ⇒ target ≥ (Σ_j a_j·feat_j) + b
    H ⇒ target ≤ (Σ_j a_j·feat_j) + b

Features/target are DataFrame column names (strings).
Hypotheses are txgraffiti2025 Predicates.

Requires a MILP/LP solver on PATH (CBC or GLPK).
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pulp
import shutil

from txgraffiti2025.forms.utils import Expr, Const, to_expr  # Expr nodes & constants
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.utils.safe_generator import safe_generator


__all__ = ["lp_bounds", "LPConfig"]


# ---------------------------------------------------------------------
# Solver detection
# ---------------------------------------------------------------------

def _get_available_solver():
    cbc = shutil.which("cbc")
    if cbc:
        return pulp.COIN_CMD(path=cbc, msg=False)
    glpk = shutil.which("glpsol")
    if glpk:
        pulp.LpSolverDefault.msg = 0
        return pulp.GLPK_CMD(path=glpk, msg=False)
    raise RuntimeError("No LP solver found (install CBC or GLPK)")


# ---------------------------------------------------------------------
# Core LP: sum-of-slacks hyperplane fit
# ---------------------------------------------------------------------

def _solve_sum_slack_lp(X: np.ndarray, y: np.ndarray, *, sense: str) -> Tuple[np.ndarray, float]:
    """
    Solve the sum-of-slacks LP for an UPPER or LOWER bounding hyperplane.

    sense: "upper" -> (a·x + b) - y == s, s >= 0  (so y <= a·x + b)
           "lower" -> y - (a·x + b) == s, s >= 0  (so y >= a·x + b)
    """
    n, k = X.shape
    prob = pulp.LpProblem("sum_slack", pulp.LpMinimize)

    a = [pulp.LpVariable(f"a_{j}", lowBound=None) for j in range(k)]
    b = pulp.LpVariable("b", lowBound=None)
    s = [pulp.LpVariable(f"s_{i}", lowBound=0) for i in range(n)]

    # minimize total slack
    prob += pulp.lpSum(s)

    # constraints
    for i in range(n):
        lhs = pulp.lpSum(a[j] * float(X[i, j]) for j in range(k)) + b
        yi = float(y[i])
        if sense == "upper":
            prob += lhs - yi == s[i]
        elif sense == "lower":
            prob += yi - lhs == s[i]
        else:
            raise ValueError("sense must be 'upper' or 'lower'")

    solver = _get_available_solver()
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP did not solve optimally: {pulp.LpStatus[status]}")

    a_sol = np.array([v.value() for v in a], dtype=float)
    b_sol = float(b.value())
    return a_sol, b_sol


# ---------------------------------------------------------------------
# Config & utilities
# ---------------------------------------------------------------------

@dataclass
class LPConfig:
    features: Sequence[str]
    target: str
    direction: str = "both"        # "both" | "upper" | "lower"
    max_denominator: int = 50      # nice-looking rationals for printing
    tol: float = 1e-9              # drop |coef| < tol
    min_support: int = 3           # need this many valid rows under hypothesis

def _numify(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _prepare_arrays(
    df: pd.DataFrame, features: Sequence[str], target: str, mask: pd.Series
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    sub = df.loc[mask]
    cols = [c for c in features if c in sub.columns]
    X_df = pd.concat([_numify(sub[c]) for c in cols], axis=1)
    y_s = _numify(sub[target]) if target in sub.columns else pd.Series(np.nan, index=sub.index)
    valid = (~X_df.isna().any(axis=1)) & y_s.notna()
    X = X_df.loc[valid].to_numpy(dtype=float)
    y = y_s.loc[valid].to_numpy(dtype=float)
    return X, y, sub.index[valid]

def _rhs_expr_from_linear(
    coefs: np.ndarray,
    intercept: float,
    features: Sequence[str],
    *,
    max_denominator: int,
    tol: float,
) -> Expr:
    """
    Build an Expr for  (Σ_j a_j * feature_j) + b,
    with pretty Fractions for readability.
    """
    # start with intercept
    rhs: Expr = Const(Fraction(intercept).limit_denominator(max_denominator))
    for a, col in zip(coefs, features):
        if abs(a) < tol:
            continue
        rhs = rhs + (Const(Fraction(a).limit_denominator(max_denominator)) * to_expr(col))
    return rhs


# ---------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------

@safe_generator
def lp_bounds(
    df: pd.DataFrame,
    *,
    hypothesis: Optional[Predicate],
    config: LPConfig,
) -> Iterator[Conjecture]:
    """
    Generate linear programming bounds under a hypothesis.

    Parameters
    ----------
    df : DataFrame
        Data containing numeric columns.

    hypothesis : Predicate or None
        Class restriction H. If None, applies to all rows.

    config : LPConfig
        features, target, direction ('both' | 'upper' | 'lower'),
        max_denominator (for pretty Fractions), tol, min_support.

    Yields
    ------
    Conjecture
        One or two Conjectures of the form:
            H ⇒ target ≥ Σ a_j·feat_j + b      (lower)
            H ⇒ target ≤ Σ a_j·feat_j + b      (upper)

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti.example_data import graph_data as df
    >>> from txgraffiti2025.processing.pre.hypotheses import detect_base_hypothesis
    >>> base = detect_base_hypothesis(df)
    >>> from txgraffiti2025.generators.lp import lp_bounds, LPConfig
    >>> cfg = LPConfig(features=["order", "matching_number"], target="domination_number", direction="both")
    >>> list(lp_bounds(df, hypothesis=base, config=cfg))[:2]  # doctest: +ELLIPSIS
    [Conjecture(...), Conjecture(...)]
    """
    # build mask
    mask = (hypothesis.mask(df) if hypothesis is not None else pd.Series(True, index=df.index)).astype(bool)
    X, y, valid_idx = _prepare_arrays(df, config.features, config.target, mask)

    if len(valid_idx) < config.min_support:
        return

    # Which directions?
    directions = []
    if config.direction in ("both", "upper"):
        directions.append("upper")
    if config.direction in ("both", "lower"):
        directions.append("lower")
    if not directions:
        return

    for sense in directions:
        a_sol, b_sol = _solve_sum_slack_lp(X, y, sense=sense)
        rhs_expr = _rhs_expr_from_linear(
            a_sol, b_sol, config.features,
            max_denominator=config.max_denominator, tol=config.tol
        )
        lhs_expr = to_expr(config.target)

        if sense == "upper":
            rel = Le(lhs_expr, rhs_expr)
            name = f"lp_upper_{config.target}_vs_{'_'.join(config.features)}"
        else:
            rel = Ge(lhs_expr, rhs_expr)
            name = f"lp_lower_{config.target}_vs_{'_'.join(config.features)}"

        yield Conjecture(relation=rel, condition=hypothesis, name=name)
