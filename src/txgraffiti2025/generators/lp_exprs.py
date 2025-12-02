# src/txgraffiti2025/generators/lp_exprs.py
"""
LP-based conjecture generator (R2) with Expr features.

- Accepts features as column names or Expr trees (sqrt("x"), log("x"), ("x")**2, ...).
- Evaluates only on rows satisfying the hypothesis mask.
- Guards against domain losses (log/sqrt), NaNs/±inf, and low support.
- Emits RHS as a generic Expr (Const/ColumnTerm/BinOp/UnaryOp) for full downstream compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Optional, Sequence, Tuple, Union, List

import math
import numpy as np
import pandas as pd
import pulp
import shutil

from txgraffiti2025.forms.utils import Expr, Const, to_expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.utils.safe_generator import safe_generator

__all__ = ["lp_bounds", "LPConfig"]


# ───────────────────────── Solver detection ───────────────────────── #

def _get_available_solver():
    cbc = shutil.which("cbc")
    if cbc:
        return pulp.COIN_CMD(path=cbc, msg=False)
    glpk = shutil.which("glpsol")
    if glpk:
        pulp.LpSolverDefault.msg = 0
        return pulp.GLPK_CMD(path=glpk, msg=False)
    raise RuntimeError("No LP solver found (install CBC or GLPK)")


# ─────────────────────── Core LP: sum-of-slacks ────────────────────── #

def _solve_sum_slack_lp(X: np.ndarray, y: np.ndarray, *, sense: str) -> Tuple[np.ndarray, float]:
    """
    sense: "upper" -> (a·x + b) - y == s, s >= 0  (so y <= a·x + b)
           "lower" -> y - (a·x + b) == s, s >= 0  (so y >= a·x + b)
    """
    n, k = X.shape
    prob = pulp.LpProblem("sum_slack", pulp.LpMinimize)

    a = [pulp.LpVariable(f"a_{j}", lowBound=None) for j in range(k)]
    b = pulp.LpVariable("b", lowBound=None)
    s = [pulp.LpVariable(f"s_{i}", lowBound=0) for i in range(n)]

    prob += pulp.lpSum(s)

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

    def _val(v):
        vv = v.value()
        return float(vv) if vv is not None else float("nan")

    a_sol = np.array([_val(v) for v in a], dtype=float)
    b_sol = _val(b)
    return a_sol, b_sol


# ───────────────────────── Config & guards ────────────────────────── #

@dataclass
class LPConfig:
    """
    Parameters
    ----------
    features : Sequence[Union[str, Expr]]
        Column names or Expr trees (e.g., sqrt("x"), log("x"), ("x")**2).
    target : str
        Target column name.
    direction : {"both","upper","lower"}
        Which bounds to emit. Default "both".
    max_denominator : int
        Limit for Fraction pretty-printing of coefficients/intercepts.
    tol : float
        Drop |coef| < tol when synthesizing RHS.
    min_support : int
        Minimum valid rows under hypothesis mask after dropping NaNs/±inf.
    min_valid_frac : float
        Minimum fraction of valid rows among masked rows; guards domain wipeouts.
    warn : bool
        If True, prints concise guard messages.
    name_prefix : Optional[str]
        Prefix in conjecture names (default "lp").
    """
    features: Sequence[Union[str, Expr]]
    target: str
    direction: str = "both"
    max_denominator: int = 50
    tol: float = 1e-9
    min_support: int = 3
    min_valid_frac: float = 0.20
    warn: bool = False
    name_prefix: Optional[str] = "lp"


def _finite_series(s: pd.Series) -> pd.Series:
    return s.notna() & np.isfinite(s.to_numpy(dtype=float))


def _evaluate_features_under_mask(
    df: pd.DataFrame,
    features: Sequence[Union[str, Expr]],
    mask: pd.Series,
) -> Tuple[pd.DataFrame, List[Expr], List[str], int, List[float]]:
    """
    Returns:
      X_df                 : evaluated features on sub-df (masked)
      feats_expr           : Expr objects used
      feat_names           : repr(expr) for pretty names
      masked_count         : rows after mask (before NaN/±inf drop)
      per_feat_valid_frac  : fraction valid per single feature (finite only)
    """
    sub = df.loc[mask]
    feats_expr: List[Expr] = [to_expr(f) for f in features]
    feat_names: List[str] = [repr(f) for f in feats_expr]

    cols: List[pd.Series] = []
    per_feat_valid_frac: List[float] = []
    for e in feats_expr:
        # silence domain warnings; NaN/±inf handled by masks
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            s = pd.to_numeric(e.eval(sub), errors="coerce")
        cols.append(s)
        per_feat_valid_frac.append(float(_finite_series(s).mean()) if len(sub) else 0.0)

    if len(cols) == 0:
        return pd.DataFrame(index=sub.index), [], [], len(sub), per_feat_valid_frac

    X_df = pd.concat(cols, axis=1)
    X_df.columns = feat_names
    return X_df, feats_expr, feat_names, len(sub), per_feat_valid_frac


def _prepare_arrays_flex(
    df: pd.DataFrame,
    *,
    target: str,
    mask: pd.Series,
    features: Sequence[Union[str, Expr]],
    cfg: LPConfig,
) -> Tuple[np.ndarray, np.ndarray, pd.Index, List[Expr], List[str]]:
    sub = df.loc[mask]
    if target not in sub.columns:
        if cfg.warn:
            print(f"[LP] target '{target}' not in DataFrame; skip.")
        return np.empty((0, 0)), np.empty((0,)), sub.index[:0], [], []

    X_df, feats_expr, feat_names, masked_count, per_feat_frac = _evaluate_features_under_mask(
        df, features, mask
    )
    if X_df.shape[1] == 0:
        if cfg.warn:
            print("[LP] no usable features; skip.")
        return np.empty((0, 0)), np.empty((0,)), sub.index[:0], [], []

    # target (coerce; do not raise)
    y_s = pd.to_numeric(sub[target], errors="coerce")

    # require finite across ALL features and target
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        X_vals = X_df.to_numpy(dtype=float, copy=False)
        X_finite = np.isfinite(X_vals).all(axis=1)
        y_finite = _finite_series(y_s)

    valid = X_finite & y_finite
    valid_count = int(valid.sum())
    valid_frac = (valid_count / masked_count) if masked_count > 0 else 0.0

    if valid_count < cfg.min_support or valid_frac < cfg.min_valid_frac:
        if cfg.warn:
            pf = ", ".join(f"{n}:{f:.2f}" for n, f in zip(feat_names, per_feat_frac))
            print(
                f"[LP] insufficient support (valid={valid_count}, masked={masked_count}, "
                f"frac={valid_frac:.2f}; per-feature valid={{ {pf} }}); skip."
            )
        return np.empty((0, 0)), np.empty((0,)), sub.index[:0], [], []

    X = X_vals[valid, :]  # already float; finite-only
    y = y_s.loc[valid].to_numpy(dtype=float)

    # belt & suspenders
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        if cfg.warn:
            print("[LP] non-finite detected post-filter; skip.")
        return np.empty((0, 0)), np.empty((0,)), sub.index[:0], [], []

    return X, y, sub.index[valid], feats_expr, feat_names


# ─────────────── RHS construction (generic Expr) ─────────────── #

def _rhs_expr_from_linear_exprs(
    coefs: np.ndarray,
    intercept: float,
    feats_expr: Sequence[Expr],
    *,
    max_denominator: int,
    tol: float,
) -> Optional[Expr]:
    """
    Build RHS as a generic Expr: (Σ_j a_j * feat_expr_j) + b
    (Avoid LinearForm so downstream refiners that pattern-match Expr trees are happy.)
    """
    b = 0.0 if (intercept is None or not math.isfinite(intercept)) else float(intercept)
    rhs: Expr = Const(Fraction(b).limit_denominator(max_denominator))

    used_any = False
    for a, e in zip(coefs, feats_expr):
        if a is None or not math.isfinite(a) or abs(a) < tol:
            continue
        used_any = True
        frac = Fraction(float(a)).limit_denominator(max_denominator)
        rhs = rhs + (Const(frac) * e)

    if not used_any and abs(b) < tol:
        return None
    return rhs


# ───────────────────────── Public generator ────────────────────────── #

@safe_generator
def lp_bounds(
    df: pd.DataFrame,
    *,
    hypothesis: Optional[Predicate],
    config: LPConfig,
) -> Iterator[Conjecture]:
    """
    Generate LP-based bounds under a hypothesis.

    Yields Conjectures:
        H ⇒ target ≥ Σ a_j·feat_j + b      (lower)
        H ⇒ target ≤ Σ a_j·feat_j + b      (upper)
    """
    mask = (hypothesis.mask(df) if hypothesis is not None else pd.Series(True, index=df.index)).astype(bool)

    X, y, valid_idx, feats_expr, feat_names = _prepare_arrays_flex(
        df, target=config.target, mask=mask, features=config.features, cfg=config
    )
    if len(valid_idx) < config.min_support or X.size == 0:
        return

    # defensive: ensure X/y still finite
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        if config.warn:
            print("[LP] skipping due to non-finite X/y after preparation.")
        return

    directions: List[str] = []
    if config.direction in ("both", "upper"):
        directions.append("upper")
    if config.direction in ("both", "lower"):
        directions.append("lower")
    if not directions:
        return

    for sense in directions:
        try:
            a_sol, b_sol = _solve_sum_slack_lp(X, y, sense=sense)
        except Exception as e:
            if config.warn:
                print(f"[LP] solver error ({sense}): {e}")
            continue

        # sanitize solver outputs
        a_sol = np.array(
            [0.0 if (aj is None or not math.isfinite(aj)) else float(aj) for aj in a_sol],
            dtype=float
        )
        b_sol = 0.0 if (b_sol is None or not math.isfinite(b_sol)) else float(b_sol)

        rhs_expr = _rhs_expr_from_linear_exprs(
            a_sol, b_sol, feats_expr,
            max_denominator=config.max_denominator, tol=config.tol
        )
        if rhs_expr is None:
            continue

        lhs_expr = to_expr(config.target)
        rel = Le(lhs_expr, rhs_expr) if sense == "upper" else Ge(lhs_expr, rhs_expr)

        prefix = (config.name_prefix or "lp").strip()
        name = f"{prefix}_{'upper' if sense=='upper' else 'lower'}_{config.target}_vs_{'_'.join(feat_names)}"

        yield Conjecture(relation=rel, condition=hypothesis, name=name)
