# txg_mini.py
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# TxGraffitiMini additions: constants → conjectures + quick constant checks
# ─────────────────────────────────────────────────────────────────────────────
from fractions import Fraction
import numpy as np
import pandas as pd
import pulp
import shutil
from dataclasses import dataclass
try:
    import polars as pl
except Exception:
    pl = None

from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq
from txgraffiti2025.forms.predicates import Predicate
# If you prefer to emit class relations for boolean implications:
from txgraffiti2025.forms.class_relations import ClassInclusion  # adjust import path if different


def _is_pl(df) -> bool:
    return (pl is not None) and isinstance(df, pl.DataFrame)

def _is_numeric_dtype_pandas(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_numeric_dtype_polars(dt) -> bool:
    # Polars ≥0.20 has pl.datatypes.is_numeric
    try:
        from polars.datatypes import is_numeric
        return bool(is_numeric(dt))
    except Exception:
        NUM = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }
        return dt in NUM

from txgraffiti2025.forms.generic_conjecture import Eq, Ge, Le
from txgraffiti2025.forms.utils import to_expr

# External (keep these as deps to your forms/processing packages)
from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
    detect_base_hypothesis,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_and_dedup_hypotheses,
)
from txgraffiti2025.processing.post import morgan_filter

from txgraffiti2025.forms.utils import (
    to_expr, Expr, Const, floor, ceil, sqrt, safe_sqrt_series, BinOp, Const as _ConstClass
)
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE
from txgraffiti2025.forms.predicates import Predicate, Where
from txgraffiti2025.forms.qualitative import MonotoneRelation, Method as CorrMethod
from txgraffiti2025.forms.class_relations import ClassInclusion, ClassEquivalence

from txgraffiti2025.processing.post.generalize_linear import PGFC, generalize_linear_bounds as _glb



@dataclass
class QualResult:
    relation: MonotoneRelation
    condition: Predicate
    rho: float
    n: int
    support: int
    x: str
    y: str

    def pretty(self) -> str:
        arrow = "↑" if (self.rho >= 0) else "↓"
        rho_tag = "ρₛ" if self.relation.method == "spearman" else "ρₚ"
        return (f"{self.y} {arrow} {self.x} "
                f"({rho_tag}={self.rho:+.3f}, n={self.n}, support={self.support}) "
                f"under {getattr(self.condition,'pretty',lambda:repr(self.condition))()}")


import shutil, pulp
from typing import Literal

def _get_available_solver():
    """
    Return a silent CBC/GLPK pulp solver instance if available.
    """
    cbc = shutil.which("cbc")
    if cbc:
        return pulp.COIN_CMD(path=cbc, msg=False)
    glpk = shutil.which("glpsol")
    if glpk:
        pulp.LpSolverDefault.msg = 0
        return pulp.GLPK_CMD(path=glpk, msg=False)
    raise RuntimeError("No LP solver found (install CBC or GLPK)")

def _safe_bounds_for_ab(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Heuristic, data-driven bounds for |a| and |b| to keep MILP well-posed.
    """
    # Use finite subsets
    fx = np.isfinite(x); fy = np.isfinite(y)
    m = fx & fy
    if not np.any(m):
        # fallback
        return 1e3, 1e3
    xv = x[m]; yv = y[m]
    xr = float(np.nanmax(np.abs(xv))) if xv.size else 1.0
    yr = float(np.nanmax(np.abs(yv))) if yv.size else 1.0
    # Sane, not too huge:
    A = max(10.0 * (yr / max(xr, 1e-9)), 10.0)
    B = max(10.0 * yr, 10.0)
    return A, B

def _bigM_for_rows(x: np.ndarray, y: np.ndarray, A: float, B: float) -> float:
    """
    Big-M that safely dominates |y_i - (a x_i + b)| when |a|<=A, |b|<=B, |x_i|<=X.
    """
    fx = np.isfinite(x); fy = np.isfinite(y)
    m = fx & fy
    if not np.any(m):
        return 1e6
    xv = np.abs(x[m])
    yv = np.abs(y[m])
    X = float(np.nanmax(xv)) if xv.size else 1.0
    Y = float(np.nanmax(yv)) if yv.size else 1.0
    # Worst-case residual magnitude:
    return 2.0 * (Y + A * X + B) + 1.0  # a little slack

def _solve_affine_bound_milp(
    *,
    x: np.ndarray,
    y: np.ndarray,
    direction: Literal["upper","lower"],
    eps_eq: float = 1e-9,
) -> tuple[float, float, int]:
    """
    Solve MILP:
      - For 'upper':  y_i <= a x_i + b  for all i
      - For 'lower':  y_i >= a x_i + b  for all i
    Maximize count of rows that are 'tight' within eps_eq:
      - 'upper':  y_i >= a x_i + b - eps_eq  when z_i = 1
      - 'lower':  y_i <= a x_i + b + eps_eq  when z_i = 1
    Returns (a, b, touch_count). Raises if infeasible or no solver.
    """
    n = len(y)
    # Variable bounds
    A, B = _safe_bounds_for_ab(x, y)
    M = _bigM_for_rows(x, y, A, B)

    prob = pulp.LpProblem("affine_bound_touchmax", pulp.LpMaximize)

    # Variables
    a = pulp.LpVariable("a", lowBound=-A, upBound=A, cat=pulp.LpContinuous)
    b = pulp.LpVariable("b", lowBound=-B, upBound=B, cat=pulp.LpContinuous)
    z = [pulp.LpVariable(f"z_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in range(n)]

    # Objective: maximize number of tight rows
    prob += pulp.lpSum(z)

    # Constraints
    for i in range(n):
        xi = float(x[i]); yi = float(y[i])
        # Skip non-finite data rows completely (they can't constrain or be tight)
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue

        if direction == "upper":
            # Inequality must hold:
            #   yi <= a*xi + b
            prob += yi <= a*xi + b
            # Tightness indicator (z_i = 1) enforces closeness:
            #   yi >= a*xi + b - eps_eq  (within eps on the other side), but let go if z_i = 0
            #   => (a*xi + b) - yi <= eps_eq + M*(1 - z_i)
            prob += (a*xi + b) - yi <= eps_eq + M*(1 - z[i])
        else:
            # 'lower'
            #   yi >= a*xi + b
            prob += yi >= a*xi + b
            # Tightness: yi <= a*xi + b + eps_eq, relaxed if z_i = 0
            #   => yi - (a*xi + b) <= eps_eq + M*(1 - z_i)
            prob += yi - (a*xi + b) <= eps_eq + M*(1 - z[i])

    # Solve
    solver = _get_available_solver()
    res = prob.solve(solver)
    if pulp.LpStatus[res] not in ("Optimal", "Optimal Infeasible", "Not Solved", "Feasible"):
        # CBC returns 'Optimal' on success; GLPK 'Optimal'. 'Feasible' sometimes with tolerances.
        raise RuntimeError(f"MILP solve failed: {pulp.LpStatus[res]}")

    a_val = float(a.varValue)
    b_val = float(b.varValue)
    touch = int(round(pulp.value(pulp.lpSum(z))))  # sum of binaries

    return a_val, b_val, touch

# ---- affine parsing (y ? a*x + b) ----
from txgraffiti2025.forms.utils import Const, ColumnTerm, BinOp, UnaryOp
import numpy as np

def _parse_affine_rhs(rhs_expr) -> Optional[tuple[Const, Expr, float | Const]]:
    """
    Try to parse rhs_expr as a*x + b where:
      - a is Const
      - x is ColumnTerm (or any Expr we accept as 'feature'; here we expect ColumnTerm)
      - b is Const
    Returns (a_const, x_expr, b_const_or_float) or None if not matched.
    """
    e = rhs_expr
    # Reject rounded forms for now
    if isinstance(e, UnaryOp) and e.fn in (np.floor, np.ceil):
        return None
    if not (isinstance(e, BinOp) and e.fn is np.add):
        return None

    # e = (a*x) + b   or   b + (a*x)
    L, R = e.left, e.right
    def _as_const(c):
        if isinstance(c, Const): return c
        if isinstance(c, (int, float, np.integer, np.floating)): return Const(float(c))
        return None

    # Case 1: (a*x) + b
    if isinstance(L, BinOp) and L.fn is np.multiply and isinstance(L.left, Const) and _as_const(R) is not None:
        aC = L.left
        xE = L.right
        bC = _as_const(R)
        return (aC, xE, bC)

    # Case 2: b + (a*x)
    if isinstance(R, BinOp) and R.fn is np.multiply and isinstance(R.left, Const) and _as_const(L) is not None:
        aC = R.left
        xE = R.right
        bC = _as_const(L)
        return (aC, xE, bC)

    return None

def _normalize_affine_rhs(rhs_expr: Expr) -> Expr:
    """
    Remove obvious +0 in a*x + b and internal '+0' inside ratio a if any.
    """
    # normalize a if it's a composite expr (ratio) and drop '+0' inside it
    if isinstance(rhs_expr, BinOp) and rhs_expr.fn is np.add:
        L, R = rhs_expr.left, rhs_expr.right
        # normalize multiply side
        if isinstance(L, BinOp) and L.fn is np.multiply and not isinstance(L.left, Const):
            newL = BinOp(np.multiply, normalize_ratio_expr(L.left), L.right)
        elif isinstance(L, BinOp) and L.fn is np.multiply:
            newL = BinOp(np.multiply, L.left, L.right)
        else:
            newL = L
        # drop +0 (R == 0)
        if isinstance(R, Const) and float(R.value) == 0.0:
            return newL
        return BinOp(np.add, newL, R)
    return rhs_expr

# ───────────────────────────── imports ───────────────────────────── #

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from typing import Iterable, List, Tuple, Callable, Optional, Sequence, Set
import numpy as np
import pandas as pd
import re
import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Eq, Ge, Le
from txgraffiti2025.forms.utils import to_expr

# External (keep these as deps to your forms/processing packages)
from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
    detect_base_hypothesis,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_and_dedup_hypotheses,
)
from txgraffiti2025.processing.post import morgan_filter

from txgraffiti2025.forms.utils import (
    to_expr, Expr, Const, floor, ceil, sqrt, safe_sqrt_series, BinOp, Const as _ConstClass
)
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE
from txgraffiti2025.forms.predicates import Predicate, Where

# New -------------
@dataclass
class RatioStats:
    support: int
    nfinite: int
    rmin: float
    rmax: float
    rmed: float
    rstd: float
    is_flat: bool

@dataclass
class RatioCandidate:
    """A mined ratio R = (a + s_num)/(b + s_den) + s_post witnessed under hypothesis H."""
    expr: Expr          # symbolic expression for R
    a: str              # numerator column name
    b: str              # denominator column name
    s_num: int          # numerator shift (0 or 1)
    s_den: int          # denominator shift (0 or 1)
    s_post: int         # post-add shift (0 or 1)
    H: Predicate        # witnessing hypothesis
    H_name: str         # pretty/if-then name for printing
    stats: "RatioStats" # summary stats on R under H



# ───────────────────────────── bound parsing helpers ───────────────────────────── #


# ───────────────────────────── bound parsing helpers (fixed for your Expr types) ───────────────────────────── #
from txgraffiti2025.forms.utils import Const, ColumnTerm, BinOp, UnaryOp  # ensure these are imported
import numpy as np

def _is_const(e) -> bool:
    return isinstance(e, Const)

def _is_symbol_col(e) -> bool:
    return isinstance(e, ColumnTerm)

def _symbol_name(e) -> Optional[str]:
    return e.col if isinstance(e, ColumnTerm) else None

def _unwrap_no_rounding(e):
    """
    Reject floor/ceil wrapped expressions for now (we’ll add support later).
    Return None if rounded; otherwise return e unchanged.
    """
    if isinstance(e, UnaryOp) and e.fn in (np.floor, np.ceil):
        return None
    return e

def _extract_const_times_symbol(rhs_expr) -> Optional[Tuple[Const, Expr, str, Expr]]:
    """
    Parse rhs_expr as k * x where k is Const and x is a ColumnTerm.
    Accepts:
      • pure symbol (ColumnTerm)          -> interpret as 1 * symbol
      • BinOp(np.multiply, Const, ColumnTerm) or (ColumnTerm, Const)
    Rejects:
      • floor/ceil wrapped expressions (for now)
    Returns (k_const, k_expr, x_colname, x_expr) or None.
    """
    e = _unwrap_no_rounding(rhs_expr)
    if e is None:
        return None

    # 1) Pure symbol => 1 * symbol
    if isinstance(e, ColumnTerm):
        xname = e.col
        return (Const(1), Const(1), xname, e)

    # 2) Multiply node
    if isinstance(e, BinOp) and e.fn is np.multiply:
        L, R = e.left, e.right

        # Const * Symbol
        if isinstance(L, Const) and isinstance(R, ColumnTerm):
            return (L, L, R.col, R)

        # Symbol * Const
        if isinstance(L, ColumnTerm) and isinstance(R, Const):
            return (R, R, L.col, L)

    # Not a recognized k*x
    return None

def _affine_rhs(
    a: float,
    x_expr: Expr,
    b: float,
    *,
    zero_tol: float = 1e-9,
    max_denom: int = 30,
    rationalize: bool = True,
) -> Expr:
    """
    Build a*x (+ b if nonzero), rationalizing small fractions.
    Returns:
        - Const(0) if |a|,|b| ≤ zero_tol (degenerate zero expression)
        - Const(b) if a ≈ 0 but b ≠ 0
        - a*x       if b ≈ 0 but a ≠ 0
        - a*x + b   otherwise
    """
    # --- both nearly zero ---
    if abs(a) <= zero_tol and abs(b) <= zero_tol:
        return Const(0)

    # --- only coefficient zero ---
    if abs(a) <= zero_tol:
        B = to_frac_const(b, max_denom) if rationalize else Const(b)
        return B

    # --- only constant zero ---
    A = to_frac_const(a, max_denom) if rationalize else Const(a)
    ax = BinOp(np.multiply, A, x_expr)

    if abs(b) <= zero_tol:
        return ax

    # --- both nonzero ---
    B = to_frac_const(b, max_denom) if rationalize else Const(b)
    return BinOp(np.add, ax, B)


from txgraffiti2025.forms.utils import BinOp, Const

def _strip_zero_add(expr: Expr) -> Expr:
    """Recursively remove + 0 or + Const(0) additions inside an Expr tree."""
    # (x + 0) → x
    if isinstance(expr, BinOp) and expr.fn is np.add:
        L, R = expr.left, expr.right
        if isinstance(L, Const) and float(L.value) == 0:
            return _strip_zero_add(R)
        if isinstance(R, Const) and float(R.value) == 0:
            return _strip_zero_add(L)
        return BinOp(np.add, _strip_zero_add(L), _strip_zero_add(R))
    # Recursively strip inside BinOps
    if isinstance(expr, BinOp):
        return BinOp(expr.fn, _strip_zero_add(expr.left), _strip_zero_add(expr.right))
    return expr

def normalize_ratio_expr(expr: Expr) -> Expr:
    """
    Simplify ratio expressions like:
        (a + 0) / (b + 0) + 0
    →     a / b
    """
    e = _strip_zero_add(expr)
    # handle trailing "+ 0" after the fraction
    if isinstance(e, BinOp) and e.fn is np.add:
        L, R = e.left, e.right
        if isinstance(R, Const) and float(R.value) == 0:
            e = _strip_zero_add(L)
    return e

def _is_unrounded_simple_linear(conj: Conjecture) -> bool:
    """
    Return True iff conj is Le/Ge with RHS 'k * x' (no floor/ceil).
    Uses the same parser as ratio lifting. Rejects Eq and rounded forms.
    """
    rel = conj.relation
    if not (isinstance(rel, Le) or isinstance(rel, Ge)):
        return False
    parsed = _extract_const_times_symbol(rel.right)
    return parsed is not None

def _normalize_ratio_in_conjecture(c: Conjecture) -> Conjecture:
    """
    If RHS is of the form R * x (with R possibly containing +0), strip the +0 inside R.
    Leaves constants (k*x) untouched (they don't come from ratio lifting).
    """
    rel = c.relation
    if not isinstance(rel, (Le, Ge)):
        return c

    # If RHS is multiply, try to normalize only the left factor (R) when it's not a Const.
    if isinstance(rel.right, BinOp) and rel.right.fn is np.multiply:
        L, R_ = rel.right.left, rel.right.right
        # Normalize if L is not a pure Const (i.e., it's a ratio Expr we created).
        if not isinstance(L, Const):
            new_L = normalize_ratio_expr(L)
            new_right = BinOp(np.multiply, new_L, R_)
            new_rel = Le(rel.left, new_right) if isinstance(rel, Le) else Ge(rel.left, new_right)
            return Conjecture(new_rel, c.condition)
        # Or if right is the non-const and left is const (x*k), swap logic:
        if not isinstance(R_, Const) and isinstance(L, Const):
            new_R = normalize_ratio_expr(R_)
            new_right = BinOp(np.multiply, L, new_R)
            new_rel = Le(rel.left, new_right) if isinstance(rel, Le) else Ge(rel.left, new_right)
            return Conjecture(new_rel, c.condition)
    return c

def _dedup_conjectures(conjs: list[Conjecture]) -> list[Conjecture]:
    """
    Deduplicate by (condition, left, right) after normalizing any ratio part.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[Conjecture] = []
    for c in conjs:
        cn = c
        try:
            cn = _normalize_ratio_in_conjecture(c)
        except Exception:
            pass
        Hk = _pred_cache_key(cn.condition) if cn.condition is not None else "TRUE"
        Ls = repr(cn.relation.left)
        Rs = repr(cn.relation.right)
        key = (Hk, Ls, Rs)
        if key not in seen:
            seen.add(key)
            out.append(cn)
    return out

# ----------------

# ---- 1) Stable keys for caching ----
def _pred_cache_key(p) -> str:
    # cheap, stable, no pretty() inside hot loops
    n = getattr(p, "name", None)
    if n:
        return f"name:{n}"
    return f"repr:{repr(p)}"

def _expr_cache_key(e) -> str:
    # rely on repr; avoids expensive pretty() calls repeatedly
    return repr(e)

# ---- 2) Mask cache (per DF id + predicate) ----
class _MaskCache:
    __slots__ = ("df_id", "n", "cache")
    def __init__(self, df):
        self.df_id = id(df)
        self.n = len(df)
        self.cache: dict[str, np.ndarray] = {}

    def get(self, df, pred) -> np.ndarray:
        key = _pred_cache_key(pred)
        m = self.cache.get(key)
        if m is not None:
            return m
        s = pred.mask(df)
        # align & coerce once
        s = s.reindex(df.index, fill_value=False)
        if s.dtype != bool:
            s = s.fillna(False).astype(bool, copy=False)
        a = s.to_numpy(copy=False)
        self.cache[key] = a
        return a

# ---- 3) Expr eval cache per hypothesis mask ----
class _ExprEvalCache:
    """
    Evaluate expressions once per hypothesis and reuse.
    Stores arrays aligned to full df index, not sliced frames.
    Use boolean masks to pick rows.
    """
    __slots__ = ("df", "array_cache")
    def __init__(self, df):
        self.df = df
        self.array_cache: dict[str, np.ndarray] = {}

    def arr(self, expr) -> np.ndarray:
        k = _expr_cache_key(expr)
        a = self.array_cache.get(k)
        if a is not None:
            return a
        # Eval returns a Series aligned to df.index
        s = expr.eval(self.df)
        if not hasattr(s, "to_numpy"):
            # allow scalars/arrays; broadcast scalar
            if np.isscalar(s):
                a = np.full(len(self.df), float(s), dtype=float)
            else:
                a = np.asarray(s, dtype=float)
        else:
            a = s.to_numpy(dtype=float, copy=False)
        self.array_cache[k] = a
        return a

# ---- 4) Batch touch counts ----
def _batch_touch_counts(
    df,
    conjs: list,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> dict[int, int]:
    """
    Returns {idx_in_input_list: touch_count}.
    Groups by (condition, left_expr, right_expr) so each array is computed once.
    """
    if not conjs:
        return {}
    mcache = _MaskCache(df)
    ecache = _ExprEvalCache(df)

    # Group
    groups = {}
    for i, c in enumerate(conjs):
        Hk = _pred_cache_key(c.condition) if c.condition is not None else "TRUE"
        Lk = _expr_cache_key(c.relation.left)
        Rk = _expr_cache_key(c.relation.right)
        groups.setdefault((Hk, Lk, Rk), []).append(i)

    # Precompute masks and arrays once per group
    out: dict[int, int] = {}
    mask_by_Hk: dict[str, np.ndarray] = {"TRUE": np.ones(len(df), dtype=bool)}
    for (Hk, Lk, Rk), idxs in groups.items():
        if Hk != "TRUE" and Hk not in mask_by_Hk:
            # recover a predicate to compute mask (any representative works)
            # pick the first conj in this group to get the actual object:
            rep = conjs[idxs[0]]
            mask_by_Hk[Hk] = _MaskCache(df).get(df, rep.condition)
        m = mask_by_Hk[Hk]
        if not np.any(m):
            for j in idxs:
                out[j] = 0
            continue

        L = ecache.arr(conjs[idxs[0]].relation.left)
        R = ecache.arr(conjs[idxs[0]].relation.right)
        # tolerance per-row to avoid re-allocs
        # (vectorized np.isclose on masked slices is already fast)
        eq = np.isclose(L[m], R[m], rtol=rtol, atol=atol)
        cnt = int(eq.sum())

        # same L/R/m for all members of this group
        for j in idxs:
            out[j] = cnt
    return out

# ---- 5) Faster rank_and_filter using batch counts ----
def rank_and_filter_fast(df, conjs: list, min_touch: int) -> list:
    if not conjs:
        return []
    counts = _batch_touch_counts(df, conjs)
    # sort once by cached counts; avoid recomputing
    order = sorted(range(len(conjs)), key=lambda i: counts.get(i, 0), reverse=True)
    kept = [conjs[i] for i in order if counts.get(i, 0) >= int(min_touch)]
    # Morgan filter expects to re-check truth; keep as is.
    from txgraffiti2025.processing.post import morgan_filter
    return list(morgan_filter(df, kept).kept)

# ---- 6) Optional: memoize hypothesis masks & supports up front ----
def _prime_hypothesis_cache(df, hyps):
    mcache = _MaskCache(df)
    pre = {}
    for H in hyps:
        mk = _pred_cache_key(H)
        m = mcache.get(df, H)
        pre[mk] = (H, m, int(m.sum()))
    return pre

# Matches names like: "[ (hypothesis) ] :: conclusion"
_IFTHEN_RE = re.compile(r"^\[\s*(?P<hyp>.+?)\s*\]\s*::\s*(?P<phi>.+)$")

def _strip_outer_parens_once(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        inner = s[1:-1].strip()
        return inner if (inner.startswith("(") and inner.endswith(")")) else f"({inner})"
    return s

def predicate_to_conjunction(p: Predicate, *, ascii_ops: bool = False) -> str:
    """Render Predicate as conjunction text; derived “[H]::phi” → “(H ∧ phi)”."""
    land = r"\land" if ascii_ops else "∧"
    raw = getattr(p, "name", None)
    if raw:
        m = _IFTHEN_RE.match(raw)
        if m:
            hyp = _strip_outer_parens_once(m.group("hyp"))
            phi = m.group("phi").strip()
            if not (hyp.startswith("(") and hyp.endswith(")")): hyp = f"({hyp})"
            if not (phi.startswith("(") and phi.endswith(")")): phi = f"({phi})"
            return f"({hyp} {land} {phi})"
    s = p.pretty() if hasattr(p, "pretty") else repr(p)
    s = _strip_outer_parens_once(s)
    if not (s.startswith("(") and s.endswith(")")): s = f"({s})"
    return s

def predicate_to_if_then(p: Predicate) -> str:
    """Derived “[H]::phi” → “If (H) ⇒ phi”; else pretty/repr."""
    name = getattr(p, "name", None)
    if not name:
        return p.pretty() if hasattr(p, "pretty") else repr(p)
    m = _IFTHEN_RE.match(name)
    if not m:
        return name
    hyp = _strip_outer_parens_once(m.group("hyp"))
    phi = m.group("phi").strip()
    if not (hyp.startswith("(") and hyp.endswith(")")): hyp = f"({hyp})"
    return f"If {hyp} ⇒ {phi}"


# ───────────────────────────── mask & eval helpers ───────────────────────────── #

def _mask_from_pred(df: pd.DataFrame, p: Predicate) -> np.ndarray:
    s = p.mask(df) if hasattr(p, "mask") else p(df)
    return np.asarray(s, dtype=bool)

def _support(mask: np.ndarray) -> int:
    return int(mask.sum())

# ───────────────────────────── small utilities ───────────────────────────── #

def to_frac_const(val: float, max_denom: int = 30) -> Const:
    return Const(Fraction(val).limit_denominator(max_denom))

def _finite_mask(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)

def _constant_value_on_mask(vals: np.ndarray, eps: float) -> tuple[bool, float]:
    f = _finite_mask(vals)
    if not np.any(f): return False, np.nan
    v = float(np.median(vals[f].astype(float)))
    dev = float(np.max(np.abs(vals[f] - v)))
    return (dev <= eps * (1.0 + abs(v))), v

def _mask_includes(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape: return False
    return bool(np.all(~a | b))

# ───────────────────────────── Config ───────────────────────────── #

@dataclass
class GenerationConfig:
    min_touch_keep: int = 3
    max_denom: int = 30
    use_floor_ceil_if_true: bool = True

    # const bank / reciprocal
    min_support_const: int = 5
    eps_const: float = 1e-12
    coeff_eps_match: float = 1e-3
    shifts: tuple[int, ...] = (-2, -1, 0, 1, 2)
    numerators: tuple[int, ...] = (1, 2, 3, 4)
    coeff_sources: str = "allow_symbolic_columns"  # or "constants_only"
    symbolic_col_limit: int = 10

# ───────────────────────────── TxGraffitiMini ───────────────────────────── #

class TxGraffitiMini:
    """
    Minimal, single-module orchestrator:
      • hypothesis discovery (boolean)
      • single-feature bounds (with ceil/floor strengthening)
      • two-feature mixes (sqrt/square)
      • targeted product bounds (with triviality & cancellation)
      • reciprocal/generalized coefficients via constant banks
      • qualitative monotone mining
      • class relations (R₁) with structural/mask guards
    """

    def __init__(self, df: pd.DataFrame, *, config: Optional[GenerationConfig] = None):
        self.df = df
        self.config = config or GenerationConfig()

        self.base_hyp = detect_base_hypothesis(df)
        self.hyps_all = enumerate_boolean_hypotheses(
            df, treat_binary_ints=True, include_base=True, include_pairs=True, skip_always_false=True,
        )
        self.hyps_kept, _ = simplify_and_dedup_hypotheses(
            df, self.hyps_all, min_support=10, treat_binary_ints=True,
        )
        self._mask_cache = _MaskCache(self.df)
        self._hyp_cache  = _prime_hypothesis_cache(self.df, self.hyps_kept)
        self.bool_columns, self.numeric_columns = self._split_columns(df)

    # --- inside TxGraffitiMini ----------------------------------------------------

    def _mask_from_pred(self, df: pd.DataFrame, H) -> np.ndarray:
        """Boolean mask for hypothesis/predicate H (supports your Predicate/Where/TRUE)."""
        # If H already provides a mask(df)
        if hasattr(H, "mask"):
            m = H.mask(df)
            if isinstance(m, np.ndarray):
                return m.astype(bool, copy=False)
            if hasattr(m, "to_numpy"):
                return m.to_numpy(dtype=bool, na_value=False)
            return np.asarray(m, dtype=bool)
        # If H is callable
        if callable(H):
            m = H(df)
            return np.asarray(m, dtype=bool)
        # Fallback: treat anything else as TRUE
        return np.ones(len(df), dtype=bool)

    @staticmethod
    def _support(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    @staticmethod
    def _mask_includes(A: np.ndarray, B: np.ndarray) -> bool:
        """Return True iff rows(B) ⊆ rows(A)."""
        # B implies A: whenever B[i] is True, A[i] must be True
        return bool(np.all(~B | A))

    def _is_true(self, H) -> bool:
        txt = ""
        if hasattr(H, "pretty"):
            txt = str(H.pretty())
        else:
            txt = str(H)
        txt = txt.strip().lower()
        return txt in {"true", "(true)", ""}

    # NEW: broader hyps but never broader than self.base_hyp
    def _broader_hypotheses(self, base_H) -> list:
        """
        Return hypotheses that are >= base_H (inclusion by mask) but ⩽ self.base_hyp.
        Ordered by decreasing support. Never returns anything strictly broader than self.base_hyp.
        """
        df = self.df

        base_mask      = self._mask_from_pred(df, base_H)
        cap_mask       = self._mask_from_pred(df, self.base_hyp)     # cap at base_hyp
        # If base_H already exceeds cap (shouldn't happen), clamp it to cap
        if not self._mask_includes(cap_mask, base_mask):
            # Intersect: keep only rows allowed by base_hyp
            base_mask = base_mask & cap_mask

        cands = []
        hyps_pool = list(getattr(self, "hyps_kept", []))
        # Ensure base_hyp itself is present
        if all(h is not self.base_hyp for h in hyps_pool):
            hyps_pool.append(self.base_hyp)

        for Hp in hyps_pool:
            pmask = self._mask_from_pred(df, Hp)

            # 1) Hp must include base_H: rows(base_H) ⊆ rows(Hp)
            if not self._mask_includes(pmask, base_mask):
                continue

            # 2) Hp must be within the cap: rows(Hp) ⊆ rows(self.base_hyp)
            if not self._mask_includes(cap_mask, pmask):
                continue

            # 3) Keep only sufficiently supported hypotheses
            if self._support(pmask) >= getattr(self.config, "min_support_const", 1):
                cands.append((Hp, self._support(pmask)))

        # sort by decreasing support, widest first (but still ≤ base_hyp)
        cands.sort(key=lambda t: -t[1])
        return [h for h, _ in cands]

    # --- small helper to show base_hyp explicitly when desired (optional) ---------
    def pretty_conjecture(self, conj, *, show_base_if_global=True) -> str:
        H = conj.hypothesis
        rel = getattr(conj, "relation", conj)  # in case you pass a relation directly
        Hs = H.pretty() if hasattr(H, "pretty") else str(H)
        if show_base_if_global and (self._is_true(H) or Hs.strip() in {"", "TRUE", "(TRUE)"}):
            # If something ever slipped through as TRUE, show base_hyp instead
            Hs = self.base_hyp.pretty() if hasattr(self.base_hyp, "pretty") else str(self.base_hyp)
        return f"{Hs} ⇒ {rel}"


    @staticmethod
    def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        bool_cols: List[str] = []
        for c in df.columns:
            s = df[c]
            if s.dtype == bool:
                bool_cols.append(c)
            elif pd.api.types.is_integer_dtype(s):
                vals = pd.unique(s.dropna())
                try: ints = set(int(v) for v in vals)
                except Exception: continue
                if len(ints) <= 2 and ints.issubset({0, 1}):
                    bool_cols.append(c)
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in bool_cols]
        return bool_cols, num_cols

    @staticmethod
    def _mask_for(df: pd.DataFrame, hypothesis) -> np.ndarray:
        return np.asarray(hypothesis.mask(df), dtype=bool) if hasattr(hypothesis, "mask") else np.asarray(hypothesis(df), dtype=bool)

    def generate_single_feature_bounds(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable] = None,
        candidates: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        Produce single-feature bounds:
        • Lower:  target ≥ c_min * x
        • Upper:  target ≤ c_max * x

        Enhancement (hard-coded here):
        If on hypothesis H there exist constant columns (or simple ratios/shifted ratios
        of constant columns) whose value equals c_min or c_max, replace the scalar with
        that *Expr* and attempt to generalize the inequality to broader hypotheses Hp ⊃ H.
        Among all successful (coeff, Hp) candidates:
        • prefer the *broadest* Hp (first returned by _broader_hypotheses / _broader_hypotheses_with_base_first)
        • on that Hp, prefer the *rounded* version (ceil for Ge, floor for Le) if it also holds
        • if multiple candidates succeed on the same Hp, keep only the *strongest*:
                Ge (lower): maximal RHS on Hp
                Le (upper): minimal RHS on Hp
        If no generalized candidate succeeds, fall back to numeric bounds on H (with optional ceil/floor).
        """
        import numpy as np
        import pandas as pd
        from fractions import Fraction

        target = to_expr(target_col)
        hyps_iter = (hyps or self.hyps_kept)
        cand_cols = list(candidates or self.numeric_columns)

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []

        # helper: robust mask
        def _mask(df, H):
            m = H.mask(df)
            return np.asarray(m.to_numpy() if hasattr(m, "to_numpy") else m, dtype=bool)

        # pick broader list (base_hyp first if you have it)
        def _broader_list(H):
            fn = getattr(self, "_broader_hypotheses_with_base_first", None) or getattr(self, "_broader_hypotheses", None)
            if fn is None:
                return [H]
            lst = fn(H)
            # ensure it's non-empty and Hp are unique-ish by repr
            seen = set()
            out = []
            for Hp in lst:
                r = repr(Hp)
                if r not in seen:
                    out.append(Hp); seen.add(r)
            if not out:
                out = [H]
            return out

        for H in hyps_iter:
            mask_H = self._mask_for(self.df, H)
            if not np.any(mask_H):
                continue

            # single cache shared by all x for this hypothesis
            ecache = _ExprEvalCache(self.df)
            t_full = ecache.arr(target)
            t_arr = t_full[mask_H]

            for xname in cand_cols:
                if xname == target_col:
                    continue

                x_expr = to_expr(xname)
                x_full = ecache.arr(x_expr)
                x_arr = x_full[mask_H]

                # positivity / finiteness guard
                if x_arr.size == 0 or np.nanmin(x_arr) <= 0:
                    continue

                rx = t_arr / x_arr
                f = np.isfinite(rx)
                if not np.any(f):
                    continue

                cmin_f = float(np.min(rx[f]))
                cmax_f = float(np.max(rx[f]))

                cmin = to_frac_const(cmin_f, self.config.max_denom)
                cmax = to_frac_const(cmax_f, self.config.max_denom)

                # ---------- detect symbolic coefficient matches on H ----------
                consts = self.constants_on_hypothesis(H, exclude_booleans=True)
                coeff_hits_cmin: list[Expr] = []
                coeff_hits_cmax: list[Expr] = []

                # direct matches: constant == cmin/cmax
                for cname, val in consts.items():
                    if np.isclose(val, cmin_f, atol=1e-12, rtol=1e-9):
                        coeff_hits_cmin.append(to_expr(cname))
                    if np.isclose(val, cmax_f, atol=1e-12, rtol=1e-9):
                        coeff_hits_cmax.append(to_expr(cname))

                # ratio / shifted-ratio matches between pairs
                cols = list(consts.keys())
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        A, B = cols[i], cols[j]
                        a, b = consts[A], consts[B]
                        if not (np.isfinite(a) and np.isfinite(b)):
                            continue
                        Aexpr, Bexpr = to_expr(A), to_expr(B)

                        # a/b
                        if b != 0:
                            r = a / b
                            rexpr = Aexpr / Bexpr
                            if np.isclose(r, cmin_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmin.append(rexpr)
                            if np.isclose(r, cmax_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmax.append(rexpr)

                            r = (a + 1.0) / b
                            rexpr = (Aexpr + Const(1)) / Bexpr
                            if np.isclose(r, cmin_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmin.append(rexpr)
                            if np.isclose(r, cmax_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmax.append(rexpr)

                            r = (b + 1.0) / b
                            rexpr = (Bexpr + Const(1)) / Bexpr
                            if np.isclose(r, cmin_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmin.append(rexpr)
                            if np.isclose(r, cmax_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmax.append(rexpr)

                        # b/a
                        if a != 0:
                            r = b / a
                            rexpr = Bexpr / Aexpr
                            if np.isclose(r, cmin_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmin.append(rexpr)
                            if np.isclose(r, cmax_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmax.append(rexpr)

                            r = (b + 1.0) / a
                            rexpr = (Bexpr + Const(1)) / Aexpr
                            if np.isclose(r, cmin_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmin.append(rexpr)
                            if np.isclose(r, cmax_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmax.append(rexpr)

                            r = (a + 1.0) / a
                            rexpr = (Aexpr + Const(1)) / Aexpr
                            if np.isclose(r, cmin_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmin.append(rexpr)
                            if np.isclose(r, cmax_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmax.append(rexpr)

                        # a/(b+1)
                        if (b + 1.0) != 0.0:
                            r = a / (b + 1.0)
                            rexpr = Aexpr / (Bexpr + Const(1))
                            if np.isclose(r, cmin_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmin.append(rexpr)
                            if np.isclose(r, cmax_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmax.append(rexpr)

                            r = 1.0 / (b + 1.0)
                            rexpr = Const(1) / (Bexpr + Const(1))
                            if np.isclose(r, cmin_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmin.append(rexpr)
                            if np.isclose(r, cmax_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmax.append(rexpr)

                        # b/(a+1)
                        if (a + 1.0) != 0.0:
                            r = b / (a + 1.0)
                            rexpr = Bexpr / (Aexpr + Const(1))
                            if np.isclose(r, cmin_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmin.append(rexpr)
                            if np.isclose(r, cmax_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmax.append(rexpr)

                            r = 1.0 / (a + 1.0)
                            rexpr = Const(1) / (Aexpr + Const(1))
                            if np.isclose(r, cmin_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmin.append(rexpr)
                            if np.isclose(r, cmax_f, atol=1e-12, rtol=1e-9):
                                coeff_hits_cmax.append(rexpr)

                # ---------- generalize + (ceil/floor) + choose strongest & most general ----------
                def _pick_best(side: str, coeff_exprs: list[Expr]):
                    """
                    Returns best Conjecture or None for this side ('lower'|'upper'):
                    - broadest Hp that passes Truth,
                    - on that Hp prefer rounded whole RHS if also True,
                    - among multiple coeffs on same Hp, keep strongest (max RHS for Ge / min RHS for Le).
                    """
                    if not coeff_exprs:
                        return None
                    broader_list = _broader_list(H)

                    best = None  # tuple(order_idx, strength_score, Conjecture)
                    for order_idx, Hp in enumerate(broader_list):
                        maskHp = _mask(self.df, Hp)
                        if not maskHp.any():
                            continue

                        for coeffE in coeff_exprs:
                            rhs_expr = coeffE * x_expr
                            if side == "lower":
                                raw_rel = Ge(target, rhs_expr)
                                rnd_rel = Ge(target, ceil(rhs_expr))
                            else:
                                raw_rel = Le(target, rhs_expr)
                                rnd_rel = Le(target, floor(rhs_expr))

                            # check raw
                            try:
                                ok_raw = raw_rel.evaluate(self.df)[maskHp].all()
                            except Exception:
                                ok_raw = False

                            # if raw fails, no reason to test rounded; but rounded might still pass
                            # (e.g., ceil makes it larger for Ge / floor makes it smaller for Le)
                            try:
                                ok_rnd = rnd_rel.evaluate(self.df)[maskHp].all()
                            except Exception:
                                ok_rnd = False

                            # choose candidate relation for this (coeffE, Hp)
                            chosen_rel = None
                            if ok_rnd:
                                chosen_rel = rnd_rel
                            elif ok_raw:
                                chosen_rel = raw_rel

                            if chosen_rel is None:
                                continue

                            # strength score for tie-breaking on same Hp:
                            #   Ge → larger RHS is stronger  (score = +mean(RHS))
                            #   Le → smaller RHS is stronger (score = -mean(RHS))
                            rhs_vals = chosen_rel.right.eval(self.df)
                            rhs_arr = np.asarray(rhs_vals.to_numpy() if hasattr(rhs_vals, "to_numpy") else rhs_vals, dtype=float)[maskHp]
                            if rhs_arr.size == 0 or not np.isfinite(rhs_arr).any():
                                continue
                            mean_rhs = float(np.nanmean(rhs_arr))
                            score = mean_rhs if isinstance(chosen_rel, Ge) else -mean_rhs

                            cand = Conjecture(chosen_rel, Hp)

                            if best is None:
                                best = (order_idx, score, cand)
                            else:
                                # prefer smaller order_idx (broader Hp)
                                if order_idx < best[0]:
                                    best = (order_idx, score, cand)
                                elif order_idx == best[0] and score > best[1]:
                                    best = (order_idx, score, cand)

                        # If we found at least one on the broadest Hp so far, we can keep scanning to
                        # see if another coeff is stronger on the same Hp, but we do not jump to *narrower* Hps.
                        # The loop naturally enforces that.

                    return None if best is None else best[2]

                best_lower = _pick_best("lower", coeff_hits_cmin)
                best_upper = _pick_best("upper", coeff_hits_cmax)

                if (best_lower is not None) or (best_upper is not None):
                    if best_lower is not None:
                        lowers.append(best_lower)
                    if best_upper is not None:
                        uppers.append(best_upper)
                    # skip numeric fallback for this x
                    continue

                # ---------- Fallback: numeric bounds on H (with optional ceil/floor) ----------
                lb_expr = cmin * x_expr
                ub_expr = cmax * x_expr

                lowers.append(Conjecture(Ge(target, lb_expr), H))
                uppers.append(Conjecture(Le(target, ub_expr), H))

                if self.config.use_floor_ceil_if_true:
                    # strengthen lower with ceil when valid on H
                    ceil_arr = np.ceil(cmin_f * x_arr)
                    if np.all(t_arr[f] >= ceil_arr[f]) and getattr(cmin, "value", cmin).denominator > 1:
                        lowers.append(Conjecture(Ge(target, ceil(lb_expr)), H))

                    # strengthen upper with floor when valid on H
                    floor_arr = np.floor(cmax_f * x_arr)
                    if np.all(t_arr[f] <= floor_arr[f]) and getattr(cmax, "value", cmax).denominator > 1:
                        uppers.append(Conjecture(Le(target, floor(ub_expr)), H))

        # light ranking/filtering as before
        lowers = self.rank_and_filter(lowers, min_touch=3)
        uppers = self.rank_and_filter(uppers, min_touch=3)
        return lowers, uppers

    def rank_and_filter(self, conjs: List[Conjecture], *, min_touch: Optional[int]=None) -> List[Conjecture]:
        m = int(min_touch if min_touch is not None else self.config.min_touch_keep)
        return rank_and_filter_fast(self.df, conjs, m)

    def run_pipeline(self, target_col: str):
        lows, ups = self.generate_single_feature_bounds(target_col)
        lows, ups = self.rank_and_filter(lows), self.rank_and_filter(ups)
        # two_lows, two_ups = self.generate_intricate_mixed_bounds(target_col)
        # two_lows, two_ups = self.rank_and_filter(two_lows), self.rank_and_filter(two_ups)
        affine_lows, affine_ups = self.generate_lp_affine_bounds(target_col)
        affine_lows, affine_ups = self.rank_and_filter(affine_lows), self.rank_and_filter(affine_ups)
        return self.rank_and_filter(lows+affine_lows), self.rank_and_filter(ups+affine_ups)

        # ── pretty helpers ── #
    @staticmethod
    def pretty_block(title: str, conjs: List[Conjecture], max_items: int = 100) -> None:
        print(f"\n=== {title} ===")
        for i, c in enumerate(conjs[:max_items], start=1):
            print(f"{i:3d}. {c.pretty(arrow='⇒')}")



    def _relation_mode(self, conj: Conjecture) -> str:
        """Return 'upper' for Le, 'lower' for Ge; raise on other types."""
        rel = conj.relation
        if isinstance(rel, Le):
            return "upper"
        if isinstance(rel, Ge):
            return "lower"
        raise ValueError("Only simple Le/Ge bounds are supported for ratio lifting.")

    def _parse_simple_linear_bound(self, conj: Conjecture) -> Tuple[Expr, str, Const, Expr]:
        """
        Extract (y_expr, x_col_name, k_const, x_expr) from a simple bound y ≤/≥ k*x.
        Rejects floor/ceil rounded forms for now.
        """
        rel = conj.relation
        y_expr = rel.left
        rhs = rel.right
        parsed = _extract_const_times_symbol(rhs)
        if parsed is None:
            raise ValueError("Expected simple linear RHS of the form k * x (no floor/ceil).")
        k_const, k_expr, xname, x_expr = parsed
        if not isinstance(k_const, Const):
            k_const = Const(k_const)  # normalize
        return y_expr, xname, k_const, x_expr


    # ---------- 1) Inspect: which columns are constant on H? ----------
    def constants_on_hypothesis(
        self,
        H,
        *,
        atol: float = 1e-12,
        rtol: float = 1e-9,
        exclude_booleans: bool = True,
    ) -> dict[str, float]:
        """
        Return {col: constant_value} for all columns that are constant over rows satisfying H.
        By default excludes boolean-like columns.
        """
        df = self.df
        m = H.mask(df)

        consts: dict[str, float] = {}

        if _is_pl(df):
            # Polars branch
            if not isinstance(m, pl.Series):
                m = pl.Series(np.asarray(m, dtype=bool))
            sub = df.filter(m)

            for c, dt in zip(sub.columns, sub.dtypes):
                if exclude_booleans and dt == pl.Boolean:
                    continue
                if not _is_numeric_dtype_polars(dt):
                    continue
                vals = np.asarray(sub[c].to_numpy(), dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                if np.allclose(vals, vals[0], atol=atol, rtol=rtol):
                    consts[c] = float(vals[0])
        else:
            # Pandas branch
            if not isinstance(m, pd.Series):
                m = pd.Series(m, index=df.index, dtype=bool)
            m = m.fillna(False)

            sub = df.loc[m]
            for c in sub.columns:
                s = sub[c]
                if exclude_booleans:
                    # also treat {0,1} integer columns as boolean-like
                    if s.dtype == bool:
                        continue
                    if pd.api.types.is_integer_dtype(s):
                        uniq = pd.unique(s.dropna())
                        try:
                            ints = set(int(v) for v in uniq)
                            if len(ints) <= 2 and ints.issubset({0, 1}):
                                continue
                        except Exception:
                            pass
                if not _is_numeric_dtype_pandas(s):
                    continue
                arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, copy=False)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                if np.allclose(arr, arr[0], atol=atol, rtol=rtol):
                    consts[c] = float(arr[0])

        return consts

    def generate_intricate_mixed_bounds(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable] = None,
        primary: Optional[Iterable[str]] = None,   # x
        secondary: Optional[Iterable[str]] = None, # y
        weight: float = 0.5,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        Mixed 'intricate inequalities' with hard-coded cmin/cmax promotion:

        √-mix:
            lower: y ≥ w*(cmin_x·x + cmin_sqrt·sqrt(y))   best of {base, ceil-whole, ceil-split}
            upper: y ≤ w*(cmax_x·x + cmax_sqrt·sqrt(y))   best of {base, floor-whole, floor-split}

        square-mix:
            lower: y ≥ w*(cmin_x·x + q_cmin·(y^2))        best of {base, ceil-whole}
            upper: y ≤ w*(cmax_x·x + q_cmax·(y^2))        best of {base, floor-whole}

        Generalization step (per H and per (x,y,mix)):
        • detect Expr matches for the scalar coefficients (for x, sqrt(y), y^2)
        • try promoted coefficients on broader hypotheses Hp (base_hyp first)
        • prefer whole-RHS rounding (ceil for lower, floor for upper) if also true
        • keep a single conjecture per side: broadest Hp, then strongest RHS

        Falls back to legacy intricate selection on H if no promoted candidate succeeds.
        """
        import numpy as np

        assert 0 < weight <= 1.0
        target = to_expr(target_col)
        hyps_iter = (hyps or self.hyps_kept)
        prim_cols = list(primary or self.numeric_columns)
        sec_cols  = list(secondary or self.numeric_columns)

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []

        w = float(weight)
        w_const = to_frac_const(weight, self.config.max_denom)

        # ---------- helpers ----------
        def _broader_list(H):
            fn = getattr(self, "_broader_hypotheses_with_base_first", None) or getattr(self, "_broader_hypotheses", None)
            if fn is None:
                return [H]
            lst = fn(H)
            # de-dup by repr
            seen, out = set(), []
            for Hp in lst:
                r = repr(Hp)
                if r not in seen:
                    out.append(Hp); seen.add(r)
            return out or [H]

        def _mask(df, H):
            m = H.mask(df)
            return np.asarray(m.to_numpy() if hasattr(m, "to_numpy") else m, dtype=bool)

        def _coeff_hits_for_value(H, val_f: float) -> List[Expr]:
            """
            Find Expr coefficients whose value on H equals val_f (≈ within tolerance),
            using constants_on_hypothesis + simple ratio/shifted-ratio constructions.
            """
            consts = self.constants_on_hypothesis(H, exclude_booleans=True)
            hits: List[Expr] = []
            # direct
            for cname, v in consts.items():
                if np.isclose(v, val_f, atol=1e-12, rtol=1e-9):
                    hits.append(to_expr(cname))
            cols = list(consts.keys())
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    A, B = cols[i], cols[j]
                    a, b = consts[A], consts[B]
                    if not (np.isfinite(a) and np.isfinite(b)):
                        continue
                    Aexpr, Bexpr = to_expr(A), to_expr(B)
                    # a/b
                    if b != 0:
                        r = a / b
                        if np.isclose(r, val_f, atol=1e-12, rtol=1e-9):
                            hits.append(Aexpr / Bexpr)
                        r = (a + 1.0) / b
                        if np.isclose(r, val_f, atol=1e-12, rtol=1e-9):
                            hits.append((Aexpr + Const(1)) / Bexpr)
                        r = (b + 1.0) / b
                        if np.isclose(r, val_f, atol=1e-12, rtol=1e-9):
                            hits.append((Bexpr + Const(1)) / Bexpr)
                    # b/a
                    if a != 0:
                        r = b / a
                        if np.isclose(r, val_f, atol=1e-12, rtol=1e-9):
                            hits.append(Bexpr / Aexpr)
                        r = (b + 1.0) / a
                        if np.isclose(r, val_f, atol=1e-12, rtol=1e-9):
                            hits.append((Bexpr + Const(1)) / Aexpr)
                        r = (a + 1.0) / a
                        if np.isclose(r, val_f, atol=1e-12, rtol=1e-9):
                            hits.append((Aexpr + Const(1)) / Aexpr)
                    # a/(b+1)
                    if (b + 1.0) != 0.0:
                        r = a / (b + 1.0)
                        if np.isclose(r, val_f, atol=1e-12, rtol=1e-9):
                            hits.append(Aexpr / (Bexpr + Const(1)))
                        r = 1.0 / (b + 1.0)
                        if np.isclose(r, val_f, atol=1e-12, rtol=1e-9):
                            hits.append(Const(1) / (Bexpr + Const(1)))
                    # b/(a+1)
                    if (a + 1.0) != 0.0:
                        r = b / (a + 1.0)
                        if np.isclose(r, val_f, atol=1e-12, rtol=1e-9):
                            hits.append(Bexpr / (Aexpr + Const(1)))
                        r = 1.0 / (a + 1.0)
                        if np.isclose(r, val_f, atol=1e-12, rtol=1e-9):
                            hits.append(Const(1) / (Aexpr + Const(1)))
            # dedupe by repr
            uniq, out = set(), []
            for e in hits:
                r = repr(e)
                if r not in uniq:
                    out.append(e); uniq.add(r)
            return out

        def _strength_on_mask(side: str, rhs_arr: np.ndarray, t_arr: np.ndarray, mask: np.ndarray) -> float:
            ok = np.isfinite(rhs_arr[mask]) & np.isfinite(t_arr[mask])
            if not np.any(ok):
                return -np.inf if side == "lower" else np.inf
            if side == "lower":
                if not np.all(t_arr[mask][ok] >= rhs_arr[mask][ok]):
                    return -np.inf
                return float(np.mean(rhs_arr[mask][ok]))  # larger better
            else:
                if not np.all(t_arr[mask][ok] <= rhs_arr[mask][ok]):
                    return np.inf
                return float(np.mean(rhs_arr[mask][ok]))  # smaller better

        def _emit_best_promoted(side: str, best: Tuple[float, Predicate, Expr]):
            # best = (score, Hp, rhs_expr) and rounding already chosen
            if best is None:
                return
            _, Hp, rhs = best
            if side == "lower":
                lowers.append(Conjecture(Ge(target, rhs), Hp))
            else:
                uppers.append(Conjecture(Le(target, rhs), Hp))

        # ---------- legacy pickers for fallback ----------
        def _pick_best_ge(t_arr, rhs_variants):
            best = None; best_score = -np.inf
            for _, rhs, make_expr in rhs_variants:
                ok_mask = np.isfinite(t_arr) & np.isfinite(rhs)
                if not np.any(ok_mask):
                    continue
                if np.all(t_arr[ok_mask] >= rhs[ok_mask]):
                    score = float(np.mean(rhs[ok_mask]))
                    if score > best_score:
                        best = make_expr; best_score = score
            return best

        def _pick_best_le(t_arr, rhs_variants):
            best = None; best_score = np.inf
            for _, rhs, make_expr in rhs_variants:
                ok_mask = np.isfinite(t_arr) & np.isfinite(rhs)
                if not np.any(ok_mask):
                    continue
                if np.all(t_arr[ok_mask] <= rhs[ok_mask]):
                    score = float(np.mean(rhs[ok_mask]))
                    if score < best_score:
                        best = make_expr; best_score = score
            return best

        # ---------- main ----------
        for H in hyps_iter:
            maskH = self._mask_for(self.df, H)
            if not np.any(maskH):
                continue

            ecache = _ExprEvalCache(self.df)
            t_arr = ecache.arr(target)[maskH]

            for xname in prim_cols:
                if xname == target_col:
                    continue

                xE = to_expr(xname)
                x_arr = ecache.arr(xE)[maskH]
                if x_arr.size == 0 or np.nanmin(x_arr) <= 0:
                    continue

                # linear ratios wrt x
                rx = t_arr / x_arr
                f_rx = np.isfinite(rx)
                if not np.any(f_rx):
                    continue
                cmin_f = float(np.min(rx[f_rx]))
                cmax_f = float(np.max(rx[f_rx]))

                for yname in sec_cols:
                    if yname == target_col:
                        continue

                    yE = to_expr(yname)
                    y_arr = ecache.arr(yE)[maskH]
                    if y_arr.size == 0 or np.nanmin(y_arr) <= 0:
                        continue

                    # ---------- sqrt mix ----------
                    sqrt_y_arr = np.sqrt(y_arr, dtype=float)
                    r_sqrt = t_arr / sqrt_y_arr
                    f_sq = np.isfinite(r_sqrt)
                    s_have = np.any(f_sq)
                    if s_have:
                        s_cmin_f = float(np.min(r_sqrt[f_sq]))
                        s_cmax_f = float(np.max(r_sqrt[f_sq]))

                    # ---------- square mix ----------
                    y_sq_arr = np.square(y_arr, dtype=float)
                    r_sq = t_arr / y_sq_arr
                    f_rsq = np.isfinite(r_sq)
                    q_have = np.any(f_rsq)
                    if q_have:
                        q_cmin_f = float(np.min(r_sq[f_rsq]))
                        q_cmax_f = float(np.max(r_sq[f_rsq]))

                    # ===== Promotion phase over broader hypotheses =====
                    # Precompute coefficient hit lists on H (for reuse across Hp)
                    hits_x_cmin  = _coeff_hits_for_value(H, cmin_f)
                    hits_x_cmax  = _coeff_hits_for_value(H, cmax_f)
                    hits_s_cmin  = _coeff_hits_for_value(H, s_cmin_f) if s_have else []
                    hits_s_cmax  = _coeff_hits_for_value(H, s_cmax_f) if s_have else []
                    hits_q_cmin  = _coeff_hits_for_value(H, q_cmin_f) if q_have else []
                    hits_q_cmax  = _coeff_hits_for_value(H, q_cmax_f) if q_have else []

                    best_lower: Optional[Tuple[float, Predicate, Expr]] = None
                    best_upper: Optional[Tuple[float, Predicate, Expr]] = None

                    for Hp in _broader_list(H):
                        maskHp = _mask(self.df, Hp)
                        if not np.any(maskHp):
                            continue
                        # arrays on Hp for scoring/promoted checks
                        xHp = xE.eval(self.df).to_numpy()[maskHp].astype(float)
                        yHp = yE.eval(self.df).to_numpy()[maskHp].astype(float)
                        tHp = target.eval(self.df).to_numpy()[maskHp].astype(float)
                        finHp = np.isfinite(xHp) & np.isfinite(yHp) & np.isfinite(tHp) & (xHp > 0)

                        # ——— sqrt mix promotion ———
                        if s_have:
                            sqrt_yHp = np.sqrt(np.maximum(yHp, 0.0))
                            finS = finHp & (sqrt_yHp > 0) & np.isfinite(sqrt_yHp)

                            # LOWER: w*(coeff_x*x + coeff_s*sqrt(y))
                            for cx in ([to_frac_const(cmin_f)] + hits_x_cmin):
                                for cs in ([to_frac_const(s_cmin_f)] + hits_s_cmin):
                                    rhs_expr = (w_const*cx*xE) + (w_const*cs*sqrt(yE))
                                    # choose whole ceil if also valid and stronger
                                    rhs_arr = rhs_expr.eval(self.df).to_numpy()[maskHp].astype(float)
                                    score_base = _strength_on_mask("lower", rhs_arr, tHp, finS)
                                    if score_base != -np.inf:
                                        rhs_best_expr = rhs_expr
                                        rhs_best_score = score_base
                                        rhs_ceil_expr = ceil(rhs_expr)
                                        rhs_ceil_arr  = np.ceil(rhs_arr)
                                        score_ceil = _strength_on_mask("lower", rhs_ceil_arr, tHp, finS)
                                        if score_ceil > rhs_best_score:
                                            rhs_best_expr = rhs_ceil_expr
                                            rhs_best_score = score_ceil
                                        cand = (rhs_best_score, Hp, rhs_best_expr)
                                        if (best_lower is None) or (Hp is best_lower[1] and rhs_best_score > best_lower[0]) or (best_lower is not None and Hp is not best_lower[1] and repr(Hp) == repr(self.base_hyp)) or (best_lower is not None and repr(best_lower[1]) != repr(self.base_hyp) and repr(Hp) == repr(self.base_hyp)):
                                            # prefer base_hyp, then higher score
                                            best_lower = cand

                            # UPPER: w*(coeff_x*x + coeff_s*sqrt(y))
                            for cx in ([to_frac_const(cmax_f)] + hits_x_cmax):
                                for cs in ([to_frac_const(s_cmax_f)] + hits_s_cmax):
                                    rhs_expr = (w_const*cx*xE) + (w_const*cs*sqrt(yE))
                                    rhs_arr = rhs_expr.eval(self.df).to_numpy()[maskHp].astype(float)
                                    score_base = _strength_on_mask("upper", rhs_arr, tHp, finS)
                                    if score_base != np.inf:
                                        rhs_best_expr = rhs_expr
                                        rhs_best_score = score_base
                                        rhs_floor_expr = floor(rhs_expr)
                                        rhs_floor_arr  = np.floor(rhs_arr)
                                        score_floor = _strength_on_mask("upper", rhs_floor_arr, tHp, finS)
                                        if score_floor < rhs_best_score:
                                            rhs_best_expr = rhs_floor_expr
                                            rhs_best_score = score_floor
                                        cand = (rhs_best_score, Hp, rhs_best_expr)
                                        if (best_upper is None) or (Hp is best_upper[1] and rhs_best_score < best_upper[0]) or (best_upper is not None and Hp is not best_upper[1] and repr(Hp) == repr(self.base_hyp)) or (best_upper is not None and repr(best_upper[1]) != repr(self.base_hyp) and repr(Hp) == repr(self.base_hyp)):
                                            best_upper = cand

                        # ——— square mix promotion ———
                        if q_have:
                            y2Hp = np.square(yHp, dtype=float)
                            finQ = finHp & (y2Hp > 0) & np.isfinite(y2Hp)

                            # LOWER: w*(coeff_x*x + coeff_q*y^2)
                            for cx in ([to_frac_const(cmin_f)] + hits_x_cmin):
                                for cq in ([to_frac_const(q_cmin_f)] + hits_q_cmin):
                                    rhs_expr = (w_const*cx*xE) + (w_const*cq*(yE ** to_frac_const(2)))
                                    rhs_arr = rhs_expr.eval(self.df).to_numpy()[maskHp].astype(float)
                                    score_base = _strength_on_mask("lower", rhs_arr, tHp, finQ)
                                    if score_base != -np.inf:
                                        rhs_best_expr = rhs_expr
                                        rhs_best_score = score_base
                                        rhs_ceil_expr = ceil(rhs_expr)
                                        rhs_ceil_arr  = np.ceil(rhs_arr)
                                        score_ceil = _strength_on_mask("lower", rhs_ceil_arr, tHp, finQ)
                                        if score_ceil > rhs_best_score:
                                            rhs_best_expr = rhs_ceil_expr
                                            rhs_best_score = score_ceil
                                        cand = (rhs_best_score, Hp, rhs_best_expr)
                                        if (best_lower is None) or (Hp is best_lower[1] and rhs_best_score > best_lower[0]) or (best_lower is not None and repr(best_lower[1]) != repr(self.base_hyp) and repr(Hp) == repr(self.base_hyp)):
                                            best_lower = cand

                            # UPPER: w*(coeff_x*x + coeff_q*y^2)
                            for cx in ([to_frac_const(cmax_f)] + hits_x_cmax):
                                for cq in ([to_frac_const(q_cmax_f)] + hits_q_cmax):
                                    rhs_expr = (w_const*cx*xE) + (w_const*cq*(yE ** to_frac_const(2)))
                                    rhs_arr = rhs_expr.eval(self.df).to_numpy()[maskHp].astype(float)
                                    score_base = _strength_on_mask("upper", rhs_arr, tHp, finQ)
                                    if score_base != np.inf:
                                        rhs_best_expr = rhs_expr
                                        rhs_best_score = score_base
                                        rhs_floor_expr = floor(rhs_expr)
                                        rhs_floor_arr  = np.floor(rhs_arr)
                                        score_floor = _strength_on_mask("upper", rhs_floor_arr, tHp, finQ)
                                        if score_floor < rhs_best_score:
                                            rhs_best_expr = rhs_floor_expr
                                            rhs_best_score = score_floor
                                        cand = (rhs_best_score, Hp, rhs_best_expr)
                                        if (best_upper is None) or (Hp is best_upper[1] and rhs_best_score < best_upper[0]) or (best_upper is not None and repr(best_upper[1]) != repr(self.base_hyp) and repr(Hp) == repr(self.base_hyp)):
                                            best_upper = cand

                    # Emit promoted (if any)
                    _emit_best_promoted("lower", best_lower)
                    _emit_best_promoted("upper", best_upper)

                    # ===== Fallback legacy selection on H if needed =====
                    need_lower_fallback = (best_lower is None)
                    need_upper_fallback = (best_upper is None)

                    if need_lower_fallback or need_upper_fallback:
                        # build legacy candidates on H only
                        # sqrt mix legacy
                        if s_have:
                            base_lo   = w * (cmin_f * x_arr + s_cmin_f * sqrt_y_arr)
                            base_up   = w * (cmax_f * x_arr + s_cmax_f * sqrt_y_arr)
                            ceil_whole = np.ceil(base_lo)
                            floor_whole = np.floor(base_up)
                            ceil_split = np.ceil(w * cmin_f * x_arr) + np.ceil(w * s_cmin_f * sqrt_y_arr) - 1.0
                            floor_split = np.floor(w * cmax_f * x_arr) + np.floor(w * s_cmax_f * sqrt_y_arr)

                            def _lo_base():
                                return (w_const * to_frac_const(cmin_f) * xE
                                    + w_const * to_frac_const(s_cmin_f) * sqrt(yE))
                            def _lo_ceil_whole():
                                return ceil(_lo_base())
                            def _lo_ceil_split():
                                return (ceil(w_const * to_frac_const(cmin_f) * xE)
                                    + ceil(w_const * to_frac_const(s_cmin_f) * sqrt(yE))
                                    - Const(1))

                            def _up_base():
                                return (w_const * to_frac_const(cmax_f) * xE
                                    + w_const * to_frac_const(s_cmax_f) * sqrt(yE))
                            def _up_floor_whole():
                                return floor(_up_base())
                            def _up_floor_split():
                                return (floor(w_const * to_frac_const(cmax_f) * xE)
                                    + floor(w_const * to_frac_const(s_cmax_f) * sqrt(yE)))

                            if need_lower_fallback:
                                lo_choice = _pick_best_ge(t_arr, [
                                    ("base",        base_lo,     _lo_base),
                                    ("ceil_whole",  ceil_whole,  _lo_ceil_whole),
                                    ("ceil_split",  ceil_split,  _lo_ceil_split),
                                ])
                                if lo_choice is not None:
                                    lowers.append(Conjecture(Ge(target, lo_choice()), H))

                            if need_upper_fallback:
                                up_choice = _pick_best_le(t_arr, [
                                    ("base",        base_up,      _up_base),
                                    ("floor_whole", floor_whole,  _up_floor_whole),
                                    ("floor_split", floor_split,  _up_floor_split),
                                ])
                                if up_choice is not None:
                                    uppers.append(Conjecture(Le(target, up_choice()), H))

                        # square mix legacy
                        if q_have:
                            base_lo_sq   = w * (cmin_f * x_arr + q_cmin_f * y_sq_arr)
                            base_up_sq   = w * (cmax_f * x_arr + q_cmax_f * y_sq_arr)
                            ceil_whole_sq  = np.ceil(base_lo_sq)
                            floor_whole_sq = np.floor(base_up_sq)

                            def _lo_sq_base():
                                return (w_const * to_frac_const(cmin_f) * xE
                                    + w_const * to_frac_const(q_cmin_f) * (yE ** to_frac_const(2)))
                            def _lo_sq_ceil_whole():
                                return ceil(_lo_sq_base())

                            def _up_sq_base():
                                return (w_const * to_frac_const(cmax_f) * xE
                                    + w_const * to_frac_const(q_cmax_f) * (yE ** to_frac_const(2)))
                            def _up_sq_floor_whole():
                                return floor(_up_sq_base())

                            if need_lower_fallback:
                                lo_sq_choice = _pick_best_ge(t_arr, [
                                    ("base",       base_lo_sq,     _lo_sq_base),
                                    ("ceil_whole", ceil_whole_sq,  _lo_sq_ceil_whole),
                                ])
                                if lo_sq_choice is not None:
                                    lowers.append(Conjecture(Ge(target, lo_sq_choice()), H))

                            if need_upper_fallback:
                                up_sq_choice = _pick_best_le(t_arr, [
                                    ("base",        base_up_sq,     _up_sq_base),
                                    ("floor_whole", floor_whole_sq, _up_sq_floor_whole),
                                ])
                                if up_sq_choice is not None:
                                    uppers.append(Conjecture(Le(target, up_sq_choice()), H))

        # final pass: rank/filter if you want (optional)
        lowers = self.rank_and_filter(lowers, min_touch=3)
        uppers = self.rank_and_filter(uppers, min_touch=3)
        return lowers, uppers

    def generate_lp_affine_bounds_for_feature(
        self,
        target_col: str,
        xname: str,
        *,
        hyps: Optional[Iterable[Predicate]] = None,
        eps_eq: float = 1e-9,
        min_support: Optional[int] = None,
    ) -> tuple[list[Conjecture], list[Conjecture]]:
        """
        For a fixed feature x, and for each hypothesis H (broad to specific),
        compute MILP-based affine bounds y ≤ a x + b and y ≥ a x + b that
        are valid on all rows in H and maximize the number of 'touch' rows.

        Returns (lowers, uppers) as lists of Conjecture (no floor/ceil rounding).
        """
        y_expr = to_expr(target_col)
        x_expr = to_expr(xname)

        hyps_iter = list(hyps or self.hyps_kept)
        hyps_iter.append(self.base_hyp)

        min_support = int(min_support if min_support is not None else self.config.min_support_const)

        # Pre-evaluate arrays once
        ecache = _ExprEvalCache(self.df)
        y_full = ecache.arr(y_expr)
        x_full = ecache.arr(x_expr)

        lowers: list[Conjecture] = []
        uppers: list[Conjecture] = []

        for H in hyps_iter:
            m = self._mask_for(self.df, H)
            if int(np.sum(m)) < min_support:
                continue

            y = y_full[m]
            x = x_full[m]

            # Need enough finite, variable data
            fin = np.isfinite(y) & np.isfinite(x)
            if np.sum(fin) < max(5, min_support):  # heuristic
                continue
            xv = x[fin]; yv = y[fin]
            if np.nanstd(xv) <= 0:
                continue  # no variation in x ⇒ skip

            # LOWER: y ≥ a x + b
            try:
                aL, bL, _touchL = _solve_affine_bound_milp(x=xv, y=yv, direction="lower", eps_eq=eps_eq)
                rhsL = _affine_rhs(aL, x_expr, bL, zero_tol=1e-8)
                lowers.append(Conjecture(Ge(y_expr, rhsL), H))
            except Exception:
                pass

            # UPPER: y ≤ a x + b
            try:
                aU, bU, _touchU = _solve_affine_bound_milp(x=xv, y=yv, direction="upper", eps_eq=eps_eq)
                rhsU = _affine_rhs(aU, x_expr, bU, zero_tol=1e-8)
                uppers.append(Conjecture(Le(y_expr, rhsU), H))
            except Exception:
                pass

        return lowers, uppers

    def generate_lp_affine_bounds(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable[Predicate]] = None,
        candidates: Optional[Iterable[str]] = None,
        eps_eq: float = 1e-9,
        rationalize: bool = True,
    ) -> tuple[list[Conjecture], list[Conjecture]]:
        """
        Run the MILP affine bound search for all candidate features x and hypotheses,
        returning ranked lower and upper bounds before any floor/ceil strengthening.

        If `rationalize` is True, convert a,b to small Fractions (via to_frac_const).
        """
        hyps_iter = list(hyps or self.hyps_kept)
        if TRUE not in hyps_iter:
            hyps_iter.append(TRUE)

        cand_cols = list(candidates or self.numeric_columns)
        # Remove the target from candidates
        cand_cols = [c for c in cand_cols if c != target_col]

        conjs_lo: list[Conjecture] = []
        conjs_up: list[Conjecture] = []

        for xname in cand_cols:
            lo, up = self.generate_lp_affine_bounds_for_feature(
                target_col, xname, hyps=hyps_iter, eps_eq=eps_eq,
            )
            conjs_lo.extend(lo)
            conjs_up.extend(up)

        # Optional: rationalize a and b to pretty fractions
        if rationalize:
            def _rat(c: Conjecture) -> Conjecture:
                rel = c.relation
                if not (isinstance(rel, (Le, Ge)) and isinstance(rel.right, BinOp) and rel.right.fn is np.add):
                    return c
                # right = a*x + b (as BinOp add)
                addL, addR = rel.right.left, rel.right.right
                # addL should be a*x (BinOp multiply)
                if isinstance(addL, BinOp) and addL.fn is np.multiply and isinstance(addL.left, Const):
                    a = float(addL.left.value); b = float(addR.value) if isinstance(addR, Const) else None
                    if b is None:
                        return c
                    aC = to_frac_const(a, self.config.max_denom)
                    bC = to_frac_const(b, self.config.max_denom)
                    new_right = BinOp(np.add, BinOp(np.multiply, aC, addL.right), bC)
                    new_rel = Le(rel.left, new_right) if isinstance(rel, Le) else Ge(rel.left, new_right)
                    return Conjecture(new_rel, c.condition)
                return c
            conjs_lo = [_rat(c) for c in conjs_lo]
            conjs_up = [_rat(c) for c in conjs_up]

        # Rank & filter
        kept_lo = self.rank_and_filter(conjs_lo)
        kept_up = self.rank_and_filter(conjs_up)
        return kept_lo, kept_up

    def generate_qualitative_relations(
        self,
        *,
        y_targets: Optional[Iterable[str]] = None,
        x_candidates: Optional[Iterable[str]] = None,
        hyps: Optional[Iterable[Predicate]] = None,
        method: CorrMethod = "spearman",
        min_abs_rho: float = 0.35,
        min_n: int = 12,
        drop_constant: bool = True,
        top_k_per_hyp: Optional[int] = None,
    ) -> List[QualResult]:
        y_cols = list(y_targets or self.numeric_columns)
        x_cols = list(x_candidates or self.numeric_columns)
        hyps_iter = list(hyps or self.hyps_kept)
        results: List[QualResult] = []

        for H in hyps_iter:
            mask = _mask_from_pred(self.df, H)
            support = int(mask.sum())
            if support < min_n: continue
            dfH = self.df.loc[mask]
            num_df = dfH.apply(pd.to_numeric, errors="coerce")
            hyp_results: List[QualResult] = []

            for y in y_cols:
                ys = num_df.get(y)
                if ys is None: continue
                if drop_constant and ys.nunique(dropna=True) <= 1: continue

                for x in x_cols:
                    if x == y: continue
                    xs = num_df.get(x)
                    if xs is None: continue
                    if drop_constant and xs.nunique(dropna=True) <= 1: continue

                    valid = xs.notna() & ys.notna()
                    n = int(valid.sum())
                    if n < min_n: continue

                    mr = MonotoneRelation(x=x, y=y, direction="increasing",
                                          method=method, min_abs_rho=min_abs_rho, min_n=min_n)
                    rho = mr._corr(xs[valid].to_numpy(), ys[valid].to_numpy())
                    if not np.isfinite(rho): continue
                    if abs(rho) < float(min_abs_rho): continue

                    mr.direction = "increasing" if rho >= 0 else "decreasing"
                    hyp_results.append(QualResult(relation=mr, condition=H, rho=float(rho),
                                                  n=n, support=support, x=x, y=y))

            if top_k_per_hyp is not None and top_k_per_hyp > 0:
                hyp_results.sort(key=lambda r: (abs(r.rho), r.support), reverse=True)
                hyp_results = hyp_results[:top_k_per_hyp]
            results.extend(hyp_results)

        results.sort(key=lambda r: (abs(r.rho), r.support), reverse=True)
        return results


    @staticmethod
    def pretty_ratios(ratios: List[RatioCandidate], k: int = 20, only_flat: bool = False) -> None:
        """
        Print a compact table of mined ratios.
        """
        rows = ratios
        if only_flat:
            rows = [r for r in rows if r.stats.is_flat]
        rows = sorted(rows, key=lambda r: (-r.stats.support, r.stats.rstd))
        print("\n=== Mined Ratios (top {}) ===".format(min(k, len(rows))))
        for i, r in enumerate(rows[:k], start=1):
            flat_flag = "≈const" if r.stats.is_flat else ""
            rng = f"[{r.stats.rmin:.6g}, {r.stats.rmax:.6g}]"
            med_std = f"med={r.stats.rmed:.6g}, σ={r.stats.rstd:.3g}"
            print(
                f"{i:3d}. H: {r.H_name} | R = {r.expr.pretty() if hasattr(r.expr,'pretty') else repr(r.expr)} | "
                f"support={r.stats.support}, nfin={r.stats.nfinite}, {rng}; {med_std} {flat_flag}"
            )



def constant_numeric_columns(df, H, *, atol=1e-12, rtol=1e-9):
    """
    Return dict of numeric columns whose values are constant (within tolerance)
    on all rows satisfying the hypothesis H.
    """
    # Boolean mask of rows satisfying H
    m = H.mask(df)
    if not isinstance(m, pd.Series):
        m = pd.Series(m, index=df.index, dtype=bool)
    m = m.fillna(False)

    sub = df.loc[m]
    consts = {}
    for c in sub.columns:
        if pd.api.types.is_numeric_dtype(sub[c]):
            arr = pd.to_numeric(sub[c], errors="coerce").dropna().to_numpy()
            if len(arr) == 0:
                continue
            if np.allclose(arr, arr[0], atol=atol, rtol=rtol):
                consts[c] = arr[0]
    return consts






# lifted = ai.postprocess_affine_bounds_with_ratios("independence_number", require_near_constant=False)
# TxGraffitiMini.pretty_block("Affine (ratio-lifted, pre-rounding)", lifted[:20])
# for H in ai.hyps_kept:
#     consts = constant_numeric_columns(ai.df, H)
#     if consts:
#         print(f"Hypothesis {H} has constant values:")
#         for k, v in consts.items():
#             print(f"  {k} = {v:g}")


# # Pick a hypothesis
# H = next(h for h in ai.hyps_kept if "regular" in repr(h))

# # 1) Inspect constants
# consts = ai.constants_on_hypothesis(H)
# print("Constants on H:", consts)

# # 2) Turn constants into conjectures
# eq_conjs, bool_incls = ai.constant_conjectures_from_hypothesis(
#     H,
#     include_boolean_implications=True,
#     include_numeric_equalities=True,
#     include_ratio_equalities=True,
#     max_denom=30,
# )

# # Pretty print / rank as you like
# TxGraffitiMini.pretty_block("Constant equalities on H", eq_conjs[:20])
# for i, inc in enumerate(bool_incls[:20], 1):
#     print(f"{i:3d}. {inc.pretty()}")  # or your class-relations pretty

# # 3) Quick test: is clique_number/chromatic_number = 1 on H?
# ok = ai.is_constant_value_on(H, "clique_number", 2, return_detail=True)
# print("clique_number==2 on H?", ok)

# ok_ratio = ai.is_constant_value_on(H, "chromatic_number", 2, return_detail=True)
# print("chromatic_number==2 on H?", ok_ratio)
# ───────────────────────────── LP/MILP affine bound generator (with equality maximization) ─────────────────────────────
# Dependencies: pulp, numpy; uses your existing Expr/Conjecture/Predicate utilities.



# start here
# from txgraffiti.example_data import graph_data as df

# df = df.drop(columns=['cograph', 'eulerian', 'chordal', 'subcubic'])

# numerical_cols = df.select_dtypes(include=np.number).columns.to_list()

df = pd.read_csv('polytope_data.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df['p6>0'] = df['p6']>0
df['p5>0'] = df['p5']>0
df['|p5 - p6|'] = np.abs(df['p5'] - df['p6'])
df['|p3 - p4|'] = np.abs(df['p3'] - df['p4'])
df['|p4 - p5|'] = np.abs(df['p4'] - df['p5'])
df['|p3 - p6|'] = np.abs(df['p3'] - df['p6'])
df['|p4 - p6|'] = np.abs(df['p4'] - df['p6'])
df['|p6 - p7|'] = np.abs(df['p6'] - df['p7'])
df['log(n)'] = df['n'].apply(np.log)
df['log(independence_number)'] = df['independence_number'].apply(np.log)
df['vertex_cover_number'] = df['n'] - df['independence_number']
df['log(vertex_cover_number)'] = df['vertex_cover_number'].apply(np.log)

# numerical_cols = [x for x in numerical_cols if x != 'independence_number']
ai = TxGraffitiMini(df)

# # # Single-feature bounds

low1, up1 = ai.run_pipeline("p6")
TxGraffitiMini.pretty_block("Lower", low1[:85])
TxGraffitiMini.pretty_block("Upper", up1[:85])

low1, up1 = ai.run_pipeline("temperature(p6)")
TxGraffitiMini.pretty_block("Lower", low1[:85])
TxGraffitiMini.pretty_block("Upper", up1[:85])

low1, up1 = ai.run_pipeline('|p3 - p6|')
TxGraffitiMini.pretty_block("Lower", low1[:85])
TxGraffitiMini.pretty_block("Upper", up1[:85])

low1, up1 = ai.run_pipeline('|p4 - p6|')
TxGraffitiMini.pretty_block("Lower", low1[:85])
TxGraffitiMini.pretty_block("Upper", up1[:85])

low1, up1 = ai.run_pipeline('|p5 - p6|')
TxGraffitiMini.pretty_block("Lower", low1[:85])
TxGraffitiMini.pretty_block("Upper", up1[:85])

