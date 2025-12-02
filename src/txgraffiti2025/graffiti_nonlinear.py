# src/txgraffiti2025/graffiti_nonlinear.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union, Optional, Dict, List, Iterable, Callable
from itertools import combinations, product

import numpy as np
import pandas as pd
import pulp
import shutil

from txgraffiti2025.graffiti_relations import GraffitiClassRelations
from txgraffiti2025.forms.utils import Expr, ColumnTerm, Const
from txgraffiti2025.forms.predicates import Predicate, Where
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE

# ─────────────────────────── Touch & array helpers ─────────────────────────── #

def _expr_array(e: Expr, df: pd.DataFrame) -> np.ndarray:
    return e.eval(df).to_numpy(dtype=float, copy=False)

def _touch_count_on_mask(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> tuple[int, float, int]:
    if mask is None:
        m = np.ones_like(lhs, dtype=bool)
    else:
        m = mask
    m = m & np.isfinite(lhs) & np.isfinite(rhs)
    n = int(m.sum())
    if n == 0:
        return 0, 0.0, 0
    diff = np.abs(lhs[m] - rhs[m])
    tol = atol + rtol * np.abs(rhs[m])
    tc = int((diff <= tol).sum())
    return tc, (tc / n), n

def _attach_touch_and_sort(conjs: list[Conjecture]) -> list[Conjecture]:
    for cj in conjs:
        if not hasattr(cj, "touch_count"):
            setattr(cj, "touch_count", 0)
        if not hasattr(cj, "touch_rate"):
            setattr(cj, "touch_rate", 0.0)
        if not hasattr(cj, "support_n"):
            setattr(cj, "support_n", 0)
    conjs.sort(key=lambda c: (getattr(c, "touch_count", 0), getattr(c, "touch_rate", 0.0)), reverse=True)
    return conjs

def _recompute_touch_for(cj: Conjecture, df: pd.DataFrame, mask: np.ndarray, *, atol: float, rtol: float) -> None:
    rel = cj.relation
    lhs = getattr(rel, "lhs", None)
    rhs = getattr(rel, "rhs", None)
    if lhs is None or rhs is None:
        setattr(cj, "touch_count", 0)
        setattr(cj, "touch_rate", 0.0)
        setattr(cj, "support_n", int(mask.sum()))
        return
    lhs_arr = _expr_array(lhs, df)
    rhs_arr = _expr_array(rhs, df)
    tc, tr, n = _touch_count_on_mask(lhs_arr, rhs_arr, mask, atol=atol, rtol=rtol)
    setattr(cj, "touch_count", tc)
    setattr(cj, "touch_rate", tr)
    setattr(cj, "support_n", n)

# ─────────────────────────── Solver discovery ─────────────────────────── #

def _get_available_solver():
    cbc = shutil.which("cbc")
    if cbc:
        return pulp.COIN_CMD(path=cbc, msg=False)
    glpk = shutil.which("glpsol")
    if glpk:
        pulp.LpSolverDefault.msg = 0
        return pulp.GLPK_CMD(path=glpk, msg=False)
    raise RuntimeError("No LP solver found (install CBC or GLPK)")

# ───────────────────────── LP: sum-of-slacks fit ───────────────────────── #

def _solve_sum_slack_lp(X: np.ndarray, y: np.ndarray, *, sense: str) -> Tuple[np.ndarray, float]:
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
            prob += lhs - yi == s[i]  # y <= lhs
        elif sense == "lower":
            prob += yi - lhs == s[i]  # y >= lhs
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

# ───────────────────────── Config & utilities ─────────────────────────── #

@dataclass
class LPConfig:
    features: Sequence[Union[str, Expr]]
    target: Union[str, Expr]
    direction: str = "both"
    max_denominator: int = 50
    tol: float = 1e-9
    min_support: int = 3

def _rational_const(x: float, max_den: int, tol: float) -> Const:
    from fractions import Fraction
    if not np.isfinite(x) or abs(x) <= tol:
        return Const(0.0)
    fr = Fraction(x).limit_denominator(max_den)
    return Const(fr)

def _to_expr(e: Union[str, Expr], exprs: Dict[str, Expr]) -> Expr:
    if isinstance(e, Expr):
        return e
    if isinstance(e, str):
        return exprs.get(e, ColumnTerm(e))
    raise TypeError(f"Unsupported feature/target type: {type(e)}")

def _finite_mask(arrs: Sequence[np.ndarray]) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def _affine_expr(a: np.ndarray, feats: Sequence[Expr], b: float, max_den: int, tol: float) -> Expr:
    expr: Optional[Expr] = None
    for coef, fj in zip(a, feats):
        if abs(coef) <= tol:
            continue
        term = _rational_const(coef, max_den, tol) * fj
        expr = term if expr is None else (expr + term)
    if abs(b) > tol:
        c = _rational_const(b, max_den, tol)
        expr = c if expr is None else (expr + c)
    return expr if expr is not None else Const(0.0)

# ─────────────────────── Transform registry (NEW) ─────────────────────── #

@dataclass(frozen=True)
class UnaryTransform:
    name: str
    fn: Callable[[Expr], Expr]
    # Optional: row-wise domain guard to build a finiteness mask quickly
    guard: Optional[Callable[[np.ndarray], np.ndarray]] = None

def _sqrt_expr(x: Expr) -> Expr:
    from txgraffiti2025.forms.utils import sqrt
    return sqrt(x)

def _log1p_expr(x: Expr) -> Expr:
    from txgraffiti2025.forms.utils import log
    # log1p(t) := log(1 + t). Keep it explicit so Expr prints nicely.
    return log(Const(1.0) + x)

def _square_expr(x: Expr) -> Expr:
    return x * x

DEFAULT_TRANSFORMS: tuple[UnaryTransform, ...] = (
    UnaryTransform("id",     lambda z: z,                     guard=None),
    UnaryTransform("sqrt",   _sqrt_expr,                      guard=lambda a: (a >= 0.0)),
    UnaryTransform("log1p",  _log1p_expr,                     guard=lambda a: (a > -1.0)),
    UnaryTransform("square", _square_expr,                    guard=None),
)

# ───────────────────── NonlinearConfig (extended) ─────────────────────── #

@dataclass(frozen=True)
class NonlinearConfig:
    require_finite: bool = True
    rationalize_constants: bool = True
    rationalize_exponents: bool = True
    max_denom: int = 30
    touch_atol: float = 0.0
    touch_rtol: float = 0.0

    # NEW: nonlinear search knobs
    transforms: tuple[UnaryTransform, ...] = DEFAULT_TRANSFORMS
    per_pair_transform_limit: int = 8     # safety cap on (#transform combos kept)
    hypotheses_limit: Optional[int] = 10  # default H cap for k2 routine
    min_touch: int = 3                    # default min touch for keeping a bound

# ────────────────────────── GraffitiNonlinear ─────────────────────────── #

class GraffitiNonlinear:
    """
    Affine-now, nonlinear-later miner that *adopts* the GraffitiClassRelations
    context and optionally applies unary transforms to RHS features before fitting.
    """

    def __init__(
        self,
        gcr: GraffitiClassRelations,
        *,
        config: Optional[NonlinearConfig] = None,
    ) -> None:
        if not isinstance(gcr, GraffitiClassRelations):
            raise TypeError("gcr must be an instance of GraffitiClassRelations")

        self.gcr: GraffitiClassRelations = gcr
        self.df: pd.DataFrame = gcr.df
        self.base_hypothesis: Predicate = gcr.base_hypothesis
        self.base_hypothesis_name: str = gcr.base_hypothesis_name
        self.nonredundant_conjunctions_ = gcr.nonredundant_conjunctions_
        self.hypotheses: List[Tuple[str, Predicate]] = gcr.sort_conjunctions_by_generality()

        self.exprs: Dict[str, Expr] = gcr.get_exprs()
        self.expr_columns: List[str] = gcr.get_expr_columns()
        self.base_predicates: Dict[str, Predicate] = gcr.get_base_predicates()
        self.invariants = list(self.exprs.values())

        self.config: NonlinearConfig = config or NonlinearConfig()

        self._mask = self._mask_via_gcr

    @classmethod
    def from_gcr(cls, gcr: GraffitiClassRelations, *, config: Optional[NonlinearConfig] = None) -> "GraffitiNonlinear":
        return cls(gcr, config=config)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, config: Optional[NonlinearConfig] = None) -> "GraffitiNonlinear":
        gcr = GraffitiClassRelations(df)
        return cls(gcr, config=config)

    # ------------------------------ Internals ---------------------------------

    def _mask_via_gcr(self, pred: Predicate) -> np.ndarray:
        return self.gcr._mask_cached(pred)

    def condition_or_base(self, cond: Optional[Predicate]) -> Predicate:
        return cond or self.base_hypothesis or TRUE

    def get_condition_mask(self, cond: Optional[Predicate]) -> np.ndarray:
        target = self.condition_or_base(cond)
        if target is TRUE:
            return np.ones(len(self.df), dtype=bool)
        return self._mask(target)

    def _support_size(self, pred: Optional[Predicate]) -> int:
        m = self.get_condition_mask(pred or self.base_hypothesis)
        return int(m.sum())

    def _dedup_by_hash(self, conjs: list[Conjecture]) -> list[Conjecture]:
        seen = set(); out = []
        for c in conjs:
            key = str(c)
            if key in seen:
                continue
            seen.add(key); out.append(c)
        return out

    def _sorted_invariants_excluding(self, target_name: str) -> list[Expr]:
        return [inv for inv in getattr(self, "invariants", []) or [] if repr(inv) != target_name]

    # ------------------------------ Linear fits -------------------------------

    def fit_linear_bounds(
        self,
        cfg: LPConfig,
        *,
        condition: Optional[Predicate] = None,
        compress: bool = True,            # kept for API compatibility; unused here
        touch_atol: Optional[float] = None,
        touch_rtol: Optional[float] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture], List[Conjecture]]:
        """
        Fit y ≲ a·x + b and/or y ≳ a·x + b under a condition with touch metadata.
        """
        t_atol = float(self.config.touch_atol if touch_atol is None else touch_atol)
        t_rtol = float(self.config.touch_rtol if touch_rtol is None else touch_rtol)

        y_expr = _to_expr(cfg.target, self.exprs)
        feat_exprs = [_to_expr(f, self.exprs) for f in cfg.features]

        cond = self.condition_or_base(condition)
        mask = self.get_condition_mask(cond)

        y_arr_full = _expr_array(y_expr, self.df)
        X_cols_full = [_expr_array(fx, self.df) for fx in feat_exprs]

        fin = _finite_mask([y_arr_full, *X_cols_full]) & mask
        if int(fin.sum()) < cfg.min_support:
            return [], [], []

        y = y_arr_full[fin]
        X = np.column_stack([c[fin] for c in X_cols_full])

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []
        equals: List[Conjecture] = []

        want_lower = cfg.direction in ("both", "lower")
        want_upper = cfg.direction in ("both", "upper")

        a_lo = b_lo = a_up = b_up = None

        if want_lower:
            a_lo, b_lo = _solve_sum_slack_lp(X, y, sense="lower")
            rhs_lo = _affine_expr(a_lo, feat_exprs, b_lo, cfg.max_denominator, cfg.tol)
            cj_lo = Conjecture(
                relation=Ge(y_expr, rhs_lo),
                condition=(None if cond is TRUE else cond),
                name=f"{repr(cond)} | lower_affine",
            )
            _recompute_touch_for(cj_lo, self.df, mask, atol=t_atol, rtol=t_rtol)
            lowers.append(cj_lo)

        if want_upper:
            a_up, b_up = _solve_sum_slack_lp(X, y, sense="upper")
            rhs_up = _affine_expr(a_up, feat_exprs, b_up, cfg.max_denominator, cfg.tol)
            cj_up = Conjecture(
                relation=Le(y_expr, rhs_up),
                condition=(None if cond is TRUE else cond),
                name=f"{repr(cond)} | upper_affine",
            )
            _recompute_touch_for(cj_up, self.df, mask, atol=t_atol, rtol=t_rtol)
            uppers.append(cj_up)

        if want_lower and want_upper and (a_lo is not None) and (a_up is not None):
            same_a = np.allclose(a_lo, a_up, atol=cfg.tol, rtol=0.0)
            b_lo_v = (b_lo if b_lo is not None else 0.0)
            b_up_v = (b_up if b_up is not None else 0.0)
            same_b = abs(b_lo_v - b_up_v) <= cfg.tol
            if same_a and same_b:
                rhs_eq = _affine_expr(a_lo, feat_exprs, b_lo_v, cfg.max_denominator, cfg.tol)
                cj_eq = Conjecture(
                    relation=Eq(y_expr, rhs_eq, tol=cfg.tol),
                    condition=(None if cond is TRUE else cond),
                    name=f"{repr(cond)} | eq_affine",
                )
                _recompute_touch_for(cj_eq, self.df, mask, atol=t_atol, rtol=t_rtol)
                equals.append(cj_eq)

        _attach_touch_and_sort(lowers)
        _attach_touch_and_sort(uppers)
        _attach_touch_and_sort(equals)
        return lowers, uppers, equals

    # ---------------- k=2 bounds with transform search (NEW) ------------------

    def generate_k2_affine_bounds(
        self,
        *,
        target: Union[str, Expr],
        hypotheses_limit: Optional[int] = None,
        min_touch: Optional[int] = None,
        max_denominator: Optional[int] = None,
        tol: float = 1e-9,
        touch_atol: Optional[float] = None,
        touch_rtol: Optional[float] = None,
        apply_morgan_filter: bool = True,
        transforms: Optional[Sequence[UnaryTransform]] = None,
    ) -> dict[str, list[Conjecture]]:
        """
        Enumerate all pairs of invariants (excluding target) and, for each active
        hypothesis H, fit LOWER/UPPER affine bounds over *transformed* features:
            lower: y ≥ a1*T1(x1) + a2*T2(x2) + b
            upper: y ≤ a1*T1(x1) + a2*T2(x2) + b

        For each pair, we grid over transform combos (id/sqrt/log1p/square by default),
        compute touches on the H-support, keep the best‐touching results, and promote
        to equality if all rows under H touch.
        """
        # Optional Morgan post-filter
        try:
            from txgraffiti2025.processing.post import morgan_filter
        except Exception:
            morgan_filter = None
            apply_morgan_filter = False

        cfg = self.config
        tset: tuple[UnaryTransform, ...] = tuple(transforms) if transforms is not None else cfg.transforms

        # knobs
        H_cap = cfg.hypotheses_limit if hypotheses_limit is None else hypotheses_limit
        keep_min_touch = cfg.min_touch if min_touch is None else min_touch
        max_den = cfg.max_denom if max_denominator is None else max_denominator
        t_atol = cfg.touch_atol if touch_atol is None else float(touch_atol)
        t_rtol = cfg.touch_rtol if touch_rtol is None else float(touch_rtol)

        # Resolve target & RHS candidates
        if isinstance(target, str):
            target_name = target
            y_expr = ColumnTerm(target)
        else:
            y_expr = target
            target_name = repr(target)

        feats = self._sorted_invariants_excluding(target_name)
        if not feats:
            return {"lowers": [], "uppers": [], "equals": []}

        # Hypotheses to iterate
        H_iter = list(self.hypotheses)
        if H_cap is not None:
            H_iter = H_iter[:H_cap]

        df = self.df
        y_full = y_expr.eval(df).to_numpy(dtype=float, copy=False)

        lowers: list[Conjecture] = []
        uppers: list[Conjecture] = []
        equals: list[Conjecture] = []

        # Pre-evaluate raw arrays for all invariants for speed
        arr_cache: Dict[str, np.ndarray] = {}
        for inv in feats:
            arr_cache[repr(inv)] = inv.eval(df).to_numpy(dtype=float, copy=False)

        # iterate pairs
        for x1, x2 in combinations(feats, 2):
            x1_name, x2_name = repr(x1), repr(x2)
            x1_full = arr_cache[x1_name]
            x2_full = arr_cache[x2_name]

            finite_base = np.isfinite(y_full) & np.isfinite(x1_full) & np.isfinite(x2_full)
            if int(finite_base.sum()) < 3:
                continue

            # try each hypothesis
            for H_name, H in H_iter:
                H_mask = self.get_condition_mask(H)
                sup_mask = finite_base & H_mask
                sup_n = int(sup_mask.sum())
                if sup_n < 3:
                    continue

                # local helpers for transform guards
                def _guard_ok(a: np.ndarray, t: UnaryTransform) -> np.ndarray:
                    if t.guard is None:
                        return np.ones_like(a, dtype=bool)
                    try:
                        g = t.guard(a)
                        # ensure dense bool array
                        return np.asarray(g, dtype=bool)
                    except Exception:
                        return np.zeros_like(a, dtype=bool)

                # search over transform pairs
                # cap the number of transform combos we *keep* per pair&hypothesis to avoid blowups
                kept_for_this_pairH = 0

                for t1, t2 in product(tset, repeat=2):
                    # domain guards refine support
                    guard_mask = sup_mask & _guard_ok(x1_full, t1) & _guard_ok(x2_full, t2)
                    n_guard = int(guard_mask.sum())
                    if n_guard < 3:
                        continue

                    # build transformed Exprs (not arrays) for RHS
                    tx1_expr = t1.fn(x1)
                    tx2_expr = t2.fn(x2)

                    # arrays restricted to guard
                    y = y_full[guard_mask]
                    tx1 = tx1_expr.eval(df).to_numpy(dtype=float, copy=False)[guard_mask]
                    tx2 = tx2_expr.eval(df).to_numpy(dtype=float, copy=False)[guard_mask]

                    # fit LOWER
                    X = np.column_stack([tx1, tx2])
                    try:
                        a_lo, b_lo = _solve_sum_slack_lp(X, y, sense="lower")
                    except Exception:
                        continue
                    rhs_lo = _affine_expr(a_lo, [tx1_expr, tx2_expr], b_lo, max_den, tol)
                    cj_lo = Conjecture(relation=Ge(y_expr, rhs_lo), condition=H,
                                       name=f"{H_name} | lower_affine[{t1.name},{t2.name}]")
                    # touch on H-support (re-evaluate RHS on full df, then restrict)
                    rhs_lo_full = rhs_lo.eval(df).to_numpy(dtype=float, copy=False)
                    tc, tr, n = _touch_count_on_mask(y_full, rhs_lo_full, H_mask, atol=t_atol, rtol=t_rtol)
                    setattr(cj_lo, "touch_count", tc); setattr(cj_lo, "touch_rate", tr); setattr(cj_lo, "support_n", n)

                    if tc == sup_n:
                        cj_eq = Conjecture(relation=Eq(y_expr, rhs_lo, tol=tol), condition=H,
                                           name=f"{H_name} | eq_affine[{t1.name},{t2.name}]")
                        setattr(cj_eq, "touch_count", tc); setattr(cj_eq, "touch_rate", tr); setattr(cj_eq, "support_n", n)
                        equals.append(cj_eq)
                    elif tc > keep_min_touch:
                        lowers.append(cj_lo)

                    # fit UPPER
                    try:
                        a_up, b_up = _solve_sum_slack_lp(X, y, sense="upper")
                    except Exception:
                        continue
                    rhs_up = _affine_expr(a_up, [tx1_expr, tx2_expr], b_up, max_den, tol)
                    cj_up = Conjecture(relation=Le(y_expr, rhs_up), condition=H,
                                       name=f"{H_name} | upper_affine[{t1.name},{t2.name}]")
                    rhs_up_full = rhs_up.eval(df).to_numpy(dtype=float, copy=False)
                    tc, tr, n = _touch_count_on_mask(y_full, rhs_up_full, H_mask, atol=t_atol, rtol=t_rtol)
                    setattr(cj_up, "touch_count", tc); setattr(cj_up, "touch_rate", tr); setattr(cj_up, "support_n", n)

                    if tc == sup_n:
                        cj_eq = Conjecture(relation=Eq(y_expr, rhs_up, tol=tol), condition=H,
                                           name=f"{H_name} | eq_affine[{t1.name},{t2.name}]")
                        setattr(cj_eq, "touch_count", tc); setattr(cj_eq, "touch_rate", tr); setattr(cj_eq, "support_n", n)
                        equals.append(cj_eq)
                    elif tc > keep_min_touch:
                        uppers.append(cj_up)

                    kept_for_this_pairH += 1
                    if kept_for_this_pairH >= self.config.per_pair_transform_limit:
                        break  # prevent combinatorial blow-up

        # De-dup and sort by touches
        for bucket in (lowers, uppers, equals):
            bucket[:] = self._dedup_by_hash(bucket)
            _attach_touch_and_sort(bucket)

        if apply_morgan_filter and morgan_filter is not None:
            lowers = list(morgan_filter(self.df, lowers).kept)
            uppers = list(morgan_filter(self.df, uppers).kept)
            equals = list(morgan_filter(self.df, equals).kept)

        return {"lowers": lowers, "uppers": uppers, "equals": equals}
