# src/txgraffiti2025/workbench/conj_lp_two_feature.py

from __future__ import annotations
from typing import Iterable, Tuple, List, Sequence, Callable, Optional, Dict
import numpy as np
import pandas as pd
import shutil, pulp

from txgraffiti2025.forms.utils import to_expr, Const, floor, ceil, sqrt, log
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate

from .caches import _EvalCache
from .conj_single_feature import to_frac_const
from .config import GenerationConfig


# ───────────────────────── solver helpers ───────────────────────── #

def _get_available_solver():
    cbc = shutil.which("cbc")
    if cbc:
        return pulp.COIN_CMD(path=cbc, msg=False)
    glpk = shutil.which("glpsol")
    if glpk:
        pulp.LpSolverDefault.msg = 0
        return pulp.GLPK_CMD(path=glpk, msg=False)
    raise RuntimeError("No LP solver found (install CBC or GLPK)")


def _solve_sum_slack_lp(X: np.ndarray, y: np.ndarray, *, sense: str) -> Tuple[np.ndarray, float]:
    """
    Sum-of-slacks LP fitting a linear inequality with intercept.
    sense: "upper" -> (a·x + b) - y == s_i >= 0  (so y <= a·x + b)
           "lower" -> y - (a·x + b) == s_i >= 0  (so y >= a·x + b)
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


# ─────────────────────── two-feature LP generator ─────────────────────── #

TransformName = str

def generate_lp_two_feature_bounds(
    df: pd.DataFrame,
    target_col: str,
    *,
    hyps: Iterable[Predicate],
    features: Sequence[str],
    config: GenerationConfig,
    require_finite: bool = True,
    require_min_rows: int = 3,
    nonneg_coeffs: bool = False,           # soft filter; add LP bounds if you want hard constraints
    rationalize: bool = True,
    # Transformed secondary features to try
    secondary_transforms: Sequence[TransformName] = ("identity",),
    allow_same_feature_when_transformed: bool = True,  # allow (x, sqrt(x)), (x, x**2), (x, log(x))
    # log variants
    log_base: Optional[float] = None,      # None -> natural log, 10 -> log10, 2 -> log2, or any float base
    log_epsilon: float = 0.0,              # clamp for numeric eval via forms.utils.LogOp semantics
) -> Tuple[List[Conjecture], List[Conjecture]]:
    """
    LP-based two-feature bounds with whole-expression ceil/floor and transformed pairs.

    For each slice H and each unordered pair (x, y) from `features`, try (x, T(y)) for
    T in `secondary_transforms` with domain checks (sqrt/log require y>0 unless you
    set `log_epsilon>0`, which clamps inside the symbolic op). For every valid triple,
    solve *two* LPs to get tight feasible upper/lower bounds; emit base conjectures,
    and when `config.use_floor_ceil_if_true` is True, also emit floor/ceil-wrapped
    variants iff they are valid for **all** rows in the slice.
    """
    # numeric transforms for the *secondary* feature
    def _id_num(a: np.ndarray) -> np.ndarray: return a
    def _sq_num(a: np.ndarray) -> np.ndarray: return np.square(a, dtype=float)
    def _log_num(a: np.ndarray) -> np.ndarray: return np.log(a)

    NUM_XFORM: Dict[TransformName, Callable[[np.ndarray], np.ndarray]] = {
        "identity": _id_num,
        "sqrt": None,   # use cache.sqrt_col for numeric; domain handled below
        "square": _sq_num,
        "log": _log_num,
    }

    # symbolic transforms for the *secondary* feature
    def _id_sym(e): return e
    def _sqrt_sym(e): return sqrt(e)
    def _square_sym(e): return e ** to_frac_const(2)   # matches your symbolic power style
    def _log_sym(e): return log(e, base=log_base, epsilon=log_epsilon)

    SYM_XFORM: Dict[TransformName, Callable] = {
        "identity": _id_sym,
        "sqrt": _sqrt_sym,
        "square": _square_sym,
        "log": _log_sym,
    }

    # sanitize transform list
    allowed = []
    for t in secondary_transforms:
        if t not in SYM_XFORM:
            continue
        allowed.append(t)
    if not allowed:
        allowed = ["identity"]

    target_expr = to_expr(target_col)
    lowers: List[Conjecture] = []
    uppers: List[Conjecture] = []

    # pre-evaluate target once; slice by mask for speed
    t_full = target_expr.eval(df).values.astype(float, copy=False)

    # Unordered base pairs (x, y), exclude the target itself
    base_feats = [f for f in dict.fromkeys(features) if f != target_col]
    base_pairs = [(base_feats[i], base_feats[j]) for i in range(len(base_feats)) for j in range(i, len(base_feats))]

    for H in hyps:
        Hmask = H.mask(df).reindex(df.index, fill_value=False).astype(bool, copy=False).to_numpy()
        if not np.any(Hmask):
            continue
        dfH = df.loc[Hmask]
        cache = _EvalCache(dfH)
        t_arr = t_full[Hmask]

        for x, y in base_pairs:
            # identity transform on secondary is only considered when x != y
            candidate_specs: List[Tuple[str, str, TransformName]] = []
            if x != y:
                candidate_specs.append((x, y, "identity"))
            elif not allow_same_feature_when_transformed:
                continue

            # add transformed secondaries even when x == y (if allowed)
            for T in allowed:
                if T == "identity":
                    continue
                candidate_specs.append((x, y, T))

            for xi, yi, T in candidate_specs:
                x_arr = cache.col(xi)
                y_raw = cache.col(yi)

                # build domain mask and numeric transformed column
                dom = np.isfinite(t_arr) & np.isfinite(x_arr) & np.isfinite(y_raw)
                if T == "sqrt":
                    dom &= (y_raw > 0.0)  # sqrt domain; numeric uses cache.sqrt_col
                    y_arr = cache.sqrt_col(yi)
                elif T == "square":
                    y_arr = cache.sq_col(yi)
                elif T == "log":
                    # Domain: >0 unless epsilon>0 (clamps internally in symbolic op).
                    if log_epsilon <= 0.0:
                        dom &= (y_raw > 0.0)
                    y_arr = _log_num(np.maximum(y_raw, log_epsilon))  # mirror symbolic clamp
                else:  # identity
                    y_arr = y_raw

                if require_finite:
                    dom &= np.isfinite(y_arr)
                if require_min_rows and np.count_nonzero(dom) < int(require_min_rows):
                    continue

                X = np.stack([x_arr[dom], y_arr[dom]], axis=1)
                yv = t_arr[dom]

                # Fit both directions
                try:
                    a_up, b_up = _solve_sum_slack_lp(X, yv, sense="upper")
                    a_lo, b_lo = _solve_sum_slack_lp(X, yv, sense="lower")
                except RuntimeError:
                    continue

                if nonneg_coeffs and (np.any(a_up < 0) or np.any(a_lo < 0)):
                    continue

                # Symbolic RHS for chosen transform
                xE = to_expr(xi)
                yE_base = to_expr(yi)
                if T == "identity":
                    yE = yE_base
                elif T == "sqrt":
                    yE = SYM_XFORM["sqrt"](yE_base)
                elif T == "square":
                    yE = SYM_XFORM["square"](yE_base)
                elif T == "log":
                    yE = SYM_XFORM["log"](yE_base)
                else:
                    continue  # shouldn’t happen

                def _rhs(a1: float, a2: float, b: float):
                    if rationalize:
                        a1C = to_frac_const(a1, config.max_denom)
                        a2C = to_frac_const(a2, config.max_denom)
                        bC  = to_frac_const(b,  config.max_denom)
                        return a1C * xE + a2C * yE + bC
                    else:
                        return Const(a1) * xE + Const(a2) * yE + Const(b)

                # Base symbolic RHS
                rhs_up = _rhs(a_up[0], a_up[1], b_up)
                rhs_lo = _rhs(a_lo[0], a_lo[1], b_lo)

                # Emit base conjectures
                uppers.append(Conjecture(Le(target_expr, rhs_up), H))
                lowers.append(Conjecture(Ge(target_expr, rhs_lo), H))

                # Whole-expression ceil/floor variants (global validity check on slice)
                if config.use_floor_ceil_if_true:
                    rhs_up_num = a_up[0] * X[:, 0] + a_up[1] * X[:, 1] + b_up
                    rhs_lo_num = a_lo[0] * X[:, 0] + a_lo[1] * X[:, 1] + b_lo
                    if np.all(yv <= np.floor(rhs_up_num)):
                        uppers.append(Conjecture(Le(target_expr, floor(rhs_up)), H))
                    if np.all(yv >= np.ceil(rhs_lo_num)):
                        lowers.append(Conjecture(Ge(target_expr, ceil(rhs_lo)), H))

    return lowers, uppers
