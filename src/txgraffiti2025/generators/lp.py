# src/txgraffiti2025/generators/lp.py

"""
Linear programming-based conjecture generator (R2 family).

Fits linear LOWER/UPPER bounds on a target column using a sum-of-slacks LP:

- LOWER bound (≥): minimize ∑ s_i  s.t. y_i - (a·x_i + b) = s_i,  s_i ≥ 0
- UPPER bound (≤): minimize ∑ s_i  s.t. (a·x_i + b) - y_i = s_i,  s_i ≥ 0

Returns Conjecture objects of the form:
    H ⇒ target ≥ (Σ_j a_j·feat_j) + b
    H ⇒ target ≤ (Σ_j a_j·feat_j) + b

Features may be DataFrame column names (strings) **or** Expr objects.
Hypotheses are txgraffiti2025 Predicates.

Requires a MILP/LP solver on PATH (CBC or GLPK).
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Optional, Sequence, Tuple, Union

import math
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

    # Some solvers can return None for variables in degenerate cases.
    def _val(v):
        vv = v.value()
        return float(vv) if vv is not None else float("nan")

    a_sol = np.array([_val(v) for v in a], dtype=float)
    b_sol = _val(b)
    return a_sol, b_sol


# ---------------------------------------------------------------------
# Config & utilities
# ---------------------------------------------------------------------

@dataclass
class LPConfig:
    # Features can be raw column names or Exprs (e.g., sqrt(x), ln(x), x**2)
    features: Sequence[Union[str, Expr]]
    target: Union[str, Expr]
    direction: str = "both"        # "both" | "upper" | "lower"
    max_denominator: int = 50      # nice-looking rationals for printing
    tol: float = 1e-9              # drop |coef| < tol
    min_support: int = 3           # need this many valid rows under hypothesis


def _numify(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _as_expr(x: Union[str, Expr]) -> Expr:
    return to_expr(x)


def _prepare_arrays(
    df: pd.DataFrame,
    features: Sequence[Union[str, Expr]],
    target: Union[str, Expr],
    mask: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, pd.Index, Sequence[Expr]]:
    """
    Evaluate each feature (string->column, Expr->eval) under the mask to build X and y.
    Returns:
        X (n×k), y (n,), valid_idx, features_exprs (the evaluated Expr objects)
    """
    sub = df.loc[mask]
    if len(sub) == 0:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float), sub.index[:0], []

    feat_exprs: list[Expr] = []
    cols: list[pd.Series] = []

    for f in features:
        ex = _as_expr(f)
        try:
            s = _numify(ex.eval(sub))
        except KeyError:
            # Feature references missing columns: skip
            continue
        cols.append(s)
        feat_exprs.append(ex)

    if not cols:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float), sub.index[:0], []

    X_df = pd.concat(cols, axis=1)
    y_s = _numify(_as_expr(target).eval(sub))

    valid = (~X_df.isna().any(axis=1)) & y_s.notna()
    X = X_df.loc[valid].to_numpy(dtype=float)
    y = y_s.loc[valid].to_numpy(dtype=float)
    return X, y, sub.index[valid], feat_exprs


def _rhs_expr_from_linear(
    coefs: np.ndarray,
    intercept: float,
    features_exprs: Sequence[Expr],
    *,
    max_denominator: int,
    tol: float,
) -> Optional[Expr]:
    """
    Build an Expr for  (Σ_j a_j * feature_j) + b,
    with pretty Fractions for readability.

    Returns None if there is no usable term (all coefs non-finite/tiny and intercept ~ 0).
    """
    # sanitize intercept
    b = 0.0 if (intercept is None or not math.isfinite(intercept)) else float(intercept)
    rhs: Expr = Const(Fraction(b).limit_denominator(max_denominator))

    used_any = False
    for a, ex in zip(coefs, features_exprs):
        # drop non-finite or tiny coefficients
        if a is None or not math.isfinite(a) or abs(a) < tol:
            continue
        frac = Fraction(float(a)).limit_denominator(max_denominator)
        rhs = rhs + (Const(frac) * ex)
        used_any = True

    if not used_any and abs(b) < tol:
        return None
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
        features (str|Expr), target (str|Expr), direction, max_denominator, tol, min_support.

    Yields
    ------
    Conjecture
        One or two Conjectures of the form:
            H ⇒ target ≥ Σ a_j·feat_j + b      (lower)
            H ⇒ target ≤ Σ a_j·feat_j + b      (upper)
    """
    # build mask
    mask = (hypothesis.mask(df) if hypothesis is not None else pd.Series(True, index=df.index)).astype(bool)

    X, y, valid_idx, feat_exprs = _prepare_arrays(df, config.features, config.target, mask)
    if len(valid_idx) < config.min_support or X.size == 0:
        return

    # Which directions?
    directions = []
    if config.direction in ("both", "upper"):
        directions.append("upper")
    if config.direction in ("both", "lower"):
        directions.append("lower")
    if not directions:
        return

    lhs_expr = _as_expr(config.target)

    for sense in directions:
        a_sol, b_sol = _solve_sum_slack_lp(X, y, sense=sense)

        # sanitize solver outputs
        a_sol = np.array([
            0.0 if (aj is None or not math.isfinite(aj)) else float(aj)
            for aj in a_sol
        ], dtype=float)
        b_sol = 0.0 if (b_sol is None or not math.isfinite(b_sol)) else float(b_sol)

        rhs_expr = _rhs_expr_from_linear(
            a_sol, b_sol, feat_exprs,
            max_denominator=config.max_denominator, tol=config.tol
        )
        if rhs_expr is None:
            continue

        if sense == "upper":
            rel = Le(lhs_expr, rhs_expr)
            name = "lp_upper"
        else:
            rel = Ge(lhs_expr, rhs_expr)
            name = "lp_lower"

        # Slightly more descriptive name
        feat_names = "_".join(repr(e) for e in feat_exprs)
        yield Conjecture(relation=rel, condition=hypothesis, name=f"{name}_{repr(lhs_expr)}_vs_{feat_names}")
