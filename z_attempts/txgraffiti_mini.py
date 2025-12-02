# txg_mini.py
from __future__ import annotations

# ───────────────────────────── imports ───────────────────────────── #

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from typing import Iterable, List, Tuple, Callable, Optional, Sequence, Set
import numpy as np
import pandas as pd
import re
import shutil
import pulp
import numpy as np
import pandas as pd
from functools import lru_cache

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



# --- fallback (safe if already defined elsewhere) ---
try:
    _ExprEvalCache
except NameError:
    class _ExprEvalCache:
        __slots__ = ("df", "array_cache")
        def __init__(self, df):
            self.df = df
            self.array_cache = {}
        def arr(self, expr):
            k = repr(expr)
            a = self.array_cache.get(k)
            if a is not None:
                return a
            s = expr.eval(self.df)
            if hasattr(s, "to_numpy"):
                a = s.to_numpy(dtype=float, copy=False)
            else:
                a = np.asarray(s, dtype=float)
            self.array_cache[k] = a
            return a

def _get_available_solver():
    cbc = shutil.which("cbc")
    if cbc:
        return pulp.COIN_CMD(path=cbc, msg=False)
    glpk = shutil.which("glpsol")
    if glpk:
        pulp.LpSolverDefault.msg = 0
        return pulp.GLPK_CMD(path=glpk, msg=False)
    raise RuntimeError("No LP solver found (install CBC or GLPK)")


def _solve_sum_slack_lp_strict(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sense: str,            # "upper" makes y <= a·x + b ; "lower" makes y >= a·x + b
    l1_penalty: float = 0, # optional L1 on a's for stability (set 0 to disable)
    var_bounds: float | None = None,  # optional |a_j| <= var_bounds
) -> Tuple[np.ndarray, float]:
    """
    Find (a, b) s.t. inequality holds for *all* rows (global bound), while minimizing
    the sum of nonnegative tightness variables.

    • Upper:  enforce  (a·x_i + b) - y_i >= 0  for all i,  minimize Σ t_i, with t_i >= (a·x_i + b) - y_i and t_i >= 0
    • Lower:  enforce  y_i - (a·x_i + b) >= 0  for all i,  minimize Σ t_i, with t_i >= y_i - (a·x_i + b) and t_i >= 0

    Optional extras:
      - l1_penalty * Σ |a_j|  (via standard split: a_j = a_j^+ - a_j^- with nonnegatives)
      - |a_j| <= var_bounds   (simple box constraints on the split vars)
    """
    import pulp

    n, k = X.shape
    prob = pulp.LpProblem("sum_slack_strict", pulp.LpMinimize)

    # Split variables for L1 regularization: a_j = a_pos - a_neg, both >= 0
    a_pos = [pulp.LpVariable(f"a_pos_{j}", lowBound=0) for j in range(k)]
    a_neg = [pulp.LpVariable(f"a_neg_{j}", lowBound=0) for j in range(k)]
    b = pulp.LpVariable("b", lowBound=None)

    if var_bounds is not None and var_bounds > 0:
        for j in range(k):
            a_pos[j].upBound = var_bounds
            a_neg[j].upBound = var_bounds

    # helper: linear form a·x + b
    def _lhs(i: int):
        return pulp.lpSum((a_pos[j] - a_neg[j]) * float(X[i, j]) for j in range(k)) + b

    # per-row tightness variables t_i >= 0, and dominate the margin
    t = [pulp.LpVariable(f"t_{i}", lowBound=0) for i in range(n)]

    if sense == "upper":
        # enforce (a·x_i + b) - y_i >= 0  AND  t_i >= (a·x_i + b) - y_i
        for i in range(n):
            margin = _lhs(i) - float(y[i])
            prob += margin >= 0
            prob += t[i] >= margin
    elif sense == "lower":
        # enforce y_i - (a·x_i + b) >= 0  AND  t_i >= y_i - (a·x_i + b)
        for i in range(n):
            margin = float(y[i]) - _lhs(i)
            prob += margin >= 0
            prob += t[i] >= margin
    else:
        raise ValueError("sense must be 'upper' or 'lower'")

    # objective: minimize sum(t_i) + l1_penalty * sum(|a_j|)
    obj = pulp.lpSum(t)
    if l1_penalty and l1_penalty > 0:
        obj += float(l1_penalty) * pulp.lpSum(a_pos[j] + a_neg[j] for j in range(k))
    prob += obj

    solver = _get_available_solver()
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP did not solve optimally: {pulp.LpStatus[status]}")

    def _val(v):
        vv = v.value()
        return float(vv) if vv is not None else float("nan")

    a_sol = np.array([_val(a_pos[j]) - _val(a_neg[j]) for j in range(k)], dtype=float)
    b_sol = _val(b)
    return a_sol, b_sol

# ───────────────────────── helpers to build RHS Expr ───────────────────────── #

def _affine_expr_from_coef(names: list[str], a: np.ndarray, b: float, max_denom: int):
    """
    Build Expr:  sum_j (a_j * col_j) + b   with rationalized constants.
    """
    terms = []
    for name, aj in zip(names, a):
        if aj == 0 or not np.isfinite(aj):
            continue
        terms.append(to_frac_const(float(aj), max_denom) * to_expr(name))
    rhs = None
    for t in terms:
        rhs = t if rhs is None else (rhs + t)
    b_const = to_frac_const(float(b), max_denom)
    rhs = b_const if rhs is None else (rhs + b_const)
    return rhs


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

def _split_top_level_conj(s: str) -> list[str]:
    """Split on top-level '∧' or '\\land' while respecting parentheses."""
    s = s.strip()
    parts, buf, depth = [], [], 0
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "(":
            depth += 1; buf.append(ch); i += 1; continue
        if ch == ")":
            if depth > 0: depth -= 1
            buf.append(ch); i += 1; continue
        if depth == 0:
            if ch == "∧":
                part = "".join(buf).strip()
                if part: parts.append(part)
                buf = []; i += 1; continue
            if s.startswith(r"\land", i):
                part = "".join(buf).strip()
                if part: parts.append(part)
                buf = []; i += len(r"\land"); continue
        buf.append(ch); i += 1
    tail = "".join(buf).strip()
    if tail: parts.append(tail)
    return parts or [s]

def _canon_atom(s: str) -> str:
    s = " ".join(_strip_outer_parens_once(s).split())
    if not (s.startswith("(") and s.endswith(")")):
        s = f"({s})"
    return s

def _pred_key(p: Predicate) -> str:
    return p.pretty() if hasattr(p, "pretty") else repr(p)

def _atoms_for_pred(p: Predicate) -> Set[str]:
    """Canonical atom set for a predicate p."""
    # Structured path (optional metadata on Where)
    hyp_obj = getattr(p, "_derived_hypothesis", None)
    phi_obj = getattr(p, "_derived_conclusion", None)
    if hyp_obj is not None and phi_obj is not None:
        hyp_txt = getattr(hyp_obj, "pretty", lambda: repr(hyp_obj))()
        phi_txt = getattr(phi_obj, "pretty", lambda: repr(phi_obj))()
        return {_canon_atom(x) for x in _split_top_level_conj(hyp_txt)} | {_canon_atom(phi_txt)}

    # Parse derived name
    name = getattr(p, "name", None)
    if name:
        m = _IFTHEN_RE.match(name)
        if m:
            hyp_txt = m.group("hyp").strip()
            phi_txt = m.group("phi").strip()
            return {_canon_atom(x) for x in _split_top_level_conj(hyp_txt)} | {_canon_atom(phi_txt)}

    # Plain
    s = (p.pretty() if hasattr(p, "pretty") else repr(p)).strip()
    return {_canon_atom(x) for x in _split_top_level_conj(s)}

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

def _mask_bool(df: pd.DataFrame, p: Predicate) -> np.ndarray:
    s = p.mask(df).reindex(df.index, fill_value=False)
    if s.dtype != bool:
        s = s.fillna(False).astype(bool, copy=False)
    return s.to_numpy()

def _mask_from_pred(df: pd.DataFrame, p: Predicate) -> np.ndarray:
    s = p.mask(df) if hasattr(p, "mask") else p(df)
    return np.asarray(s, dtype=bool)

def _support(mask: np.ndarray) -> int:
    return int(mask.sum())

def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))

def _as_series(x, index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(index)
    return pd.Series(x, index=index)

def _touch_mask(conj: Conjecture, df: pd.DataFrame, rtol=1e-8, atol=1e-8) -> np.ndarray:
    lhs = _as_series(conj.relation.left.eval(df), df.index).astype(float)
    rhs = _as_series(conj.relation.right.eval(df), df.index).astype(float)
    Hm  = _mask_from_pred(df, conj.condition) if conj.condition is not None else np.ones(len(df), dtype=bool)
    eq  = np.isclose(lhs.values, rhs.values, rtol=rtol, atol=atol)
    return Hm & eq

def _strict_mask(conj: Conjecture, df: pd.DataFrame, side: str, rtol=1e-8, atol=1e-8) -> np.ndarray:
    lhs = _as_series(conj.relation.left.eval(df), df.index).astype(float)
    rhs = _as_series(conj.relation.right.eval(df), df.index).astype(float)
    Hm  = _mask_from_pred(df, conj.condition) if conj.condition is not None else np.ones(len(df), dtype=bool)
    tol = np.maximum(atol, rtol * np.maximum(1.0, np.abs(rhs.values)))
    if isinstance(conj.relation, Le):
        strict = (lhs.values < rhs.values - tol) if side == "lt" else (lhs.values > rhs.values + tol)
    elif isinstance(conj.relation, Ge):
        strict = (lhs.values > rhs.values + tol) if side == "gt" else (lhs.values < rhs.values - tol)
    else:
        raise ValueError("Conjecture relation must be Le or Ge")
    return Hm & strict

from numbers import Number
from typing import Union
# ───────────────────────────── small utilities ───────────────────────────── #

def to_frac_const(val: float, max_denom: int = 300) -> Const:
    return Const(Fraction(val).limit_denominator(max_denom))


def _finite_mask(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)

def _constant_value_on_mask(vals: np.ndarray, eps: float) -> tuple[bool, float]:
    f = _finite_mask(vals)
    if not np.any(f): return False, np.nan
    v = float(np.median(vals[f].astype(float)))
    dev = float(np.max(np.abs(vals[f] - v)))
    return (dev <= eps * (1.0 + abs(v))), v

def _values_on_mask(df: pd.DataFrame, mask: np.ndarray, col: str) -> np.ndarray:
    return df.loc[mask, col].to_numpy(dtype=float, copy=False)

def _finite_nonzero_scalar(x: float) -> bool:
    return np.isfinite(x) and x != 0.0

def _finite_scalar(x: float) -> bool:
    return np.isfinite(x)

def _mask_includes(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape: return False
    return bool(np.all(~a | b))

def _close(a: float, b: float, eps: float) -> bool:
    return abs(a - b) <= eps * (1.0 + max(abs(a), abs(b)))

def _hyp_key(H) -> str:
    if hasattr(H, "pretty"):
        try: return H.pretty()
        except Exception: pass
    return repr(H)


# ───────────────────────────── predicate builders from conjectures ───────────────────────────── #

def _mk_name(kind: str, conj: Conjecture) -> str:
    Hname = conj.condition.pretty() if hasattr(conj.condition, "pretty") else repr(conj.condition)
    lhs   = conj.relation.left.pretty()
    rhs   = conj.relation.right.pretty()
    tag = {"onbound":"==","strict_lt":"<","strict_gt":">"}.get(kind,"?")
    return f"[{Hname}] :: {lhs} {tag} {rhs}"

def predicates_from_conjecture(
    conj: Conjecture,
    *,
    make_eq: bool = True,
    make_strict: bool = True,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> List[Predicate]:
    preds: List[Predicate] = []
    if make_eq:
        def fn_eq(df, _conj=conj, _rtol=rtol, _atol=atol):
            return _touch_mask(_conj, df, rtol=_rtol, atol=_atol)
        p_eq = Where(fn=fn_eq, name=_mk_name("onbound", conj))
        p_eq._derived_hypothesis = conj.condition
        preds.append(p_eq)

    if make_strict:
        if isinstance(conj.relation, Le):
            def fn_lt(df, _conj=conj, _rtol=rtol, _atol=atol):
                return _strict_mask(_conj, df, side="lt", rtol=_rtol, atol=_atol)
            p_lt = Where(fn=fn_lt, name=_mk_name("strict_lt", conj))
            p_lt._derived_hypothesis = conj.condition
            preds.append(p_lt)
        elif isinstance(conj.relation, Ge):
            def fn_gt(df, _conj=conj, _rtol=rtol, _atol=atol):
                return _strict_mask(_conj, df, side="gt", rtol=_rtol, atol=_atol)
            p_gt = Where(fn=fn_gt, name=_mk_name("strict_gt", conj))
            p_gt._derived_hypothesis = conj.condition
            preds.append(p_gt)
    return preds


# ───────────────────────────── Eval cache ───────────────────────────── #

class _EvalCache:
    def __init__(self, df_temp: pd.DataFrame):
        self.df = df_temp
        self._col: dict[str, np.ndarray] = {}
        self._sqrt: dict[str, np.ndarray] = {}
        self._sq: dict[str, np.ndarray] = {}

    def col(self, name: str) -> np.ndarray:
        if name not in self._col:
            a = to_expr(name).eval(self.df).values.astype(float, copy=False)
            self._col[name] = a
        return self._col[name]

    def sqrt_col(self, name: str) -> np.ndarray:
        a = self._sqrt.get(name)
        if a is None:
            x = self.col(name)
            a = safe_sqrt_series(x)
            self._sqrt[name] = a
        return a

    def sq_col(self, name: str) -> np.ndarray:
        if name not in self._sq:
            x = self.col(name)
            self._sq[name] = np.square(x, dtype=float)
        return self._sq[name]


# ───────────────────────────── Qualitative results ───────────────────────────── #

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


# ───────────────────────────── Class relations helpers ───────────────────────────── #

def _flatten_and_conjuncts(pred: Predicate) -> list[Predicate]:
    base = getattr(pred, "_derived_hypothesis", None)
    if base is not None:
        pred = base
    try:
        from txgraffiti2025.forms.predicates import AndPred
        out: list[Predicate] = []
        stack = [pred]
        while stack:
            q = stack.pop()
            if isinstance(q, AndPred):
                if hasattr(q, "left") and hasattr(q, "right"):
                    stack.append(q.left); stack.append(q.right)
                elif hasattr(q, "args"):
                    stack.extend(list(q.args))
                else:
                    out.append(q)
            else:
                out.append(q)
        return out
    except Exception:
        return [pred]

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

        # constant banks & ratio equalities
        self.const_bank: dict[str, dict[str, float]] = {}
        self.constant_ratio_equalities: list[Conjecture] = []
        self._build_constant_banks_and_ratio_equalities()

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
        Produce:
        • Lower bounds:  target ≥ c_min * x
        • Upper bounds:  target ≤ c_max * x
        Reuses a per-hypothesis _ExprEvalCache to avoid repeated expr.eval().
        """
        target = to_expr(target_col)
        hyps_iter = (hyps or self.hyps_kept)
        cand_cols = list(candidates or self.numeric_columns)

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []

        for H in hyps_iter:
            mask = self._mask_for(self.df, H)
            if not np.any(mask):
                continue

            # single cache shared by all x for this hypothesis
            ecache = _ExprEvalCache(self.df)
            t_full = ecache.arr(target)
            t_arr = t_full[mask]

            for xname in cand_cols:
                if xname == target_col:
                    continue

                x_full = ecache.arr(to_expr(xname))
                x_arr = x_full[mask]

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

                x_expr = to_expr(xname)

                # y ≥ c_min x
                lb_expr = cmin * x_expr
                lowers.append(Conjecture(Ge(target, lb_expr), H))

                # y ≤ c_max x
                ub_expr = cmax * x_expr
                uppers.append(Conjecture(Le(target, ub_expr), H))

                if not self.config.use_floor_ceil_if_true:
                    continue

                # strengthening when true (use arrays already in memory)
                ceil_arr = np.ceil(cmin_f * x_arr)
                if np.all(t_arr[f] >= ceil_arr[f]) and cmin.value.denominator > 1:
                    lowers.append(Conjecture(Ge(target, ceil(lb_expr)), H))

                floor_arr = np.floor(cmax_f * x_arr)
                if np.all(t_arr[f] <= floor_arr[f]) and cmax.value.denominator > 1:
                    uppers.append(Conjecture(Le(target, floor(ub_expr)), H))

        return lowers, uppers


    def generalize_linear_bounds(self, base_conjectures: Sequence[Conjecture], **kw) -> list[Conjecture]:
        """
        Constant-bank linear generalizer (batch).
        Uses self.const_bank (built in _build_constant_banks_and_ratio_equalities).
        """
        return _glb(
            self.df,
            base_conjectures,
            const_bank=self.const_bank,
            max_denom=self.config.max_denom,
            use_intercept=True,
            allow_fractional=True,
            try_floor_ceil=self.config.use_floor_ceil_if_true,
            **kw,
        )

    def run_linear_generalizer(self, base_conjectures: Sequence[Conjecture], **kw) -> list[Conjecture]:
        """Convenience pipeline step + ranking."""
        props = self.generalize_linear_bounds(base_conjectures, **kw)
        return self.rank_and_filter(props)

    # ── constant banks & ratio equalities ── #
    def _build_constant_banks_and_ratio_equalities(self):
        eps = self.config.eps_const
        min_sup = self.config.min_support_const
        max_denom = self.config.max_denom

        def _make_const_bank_for_H(df: pd.DataFrame, mask: np.ndarray, num_cols: list[str]) -> dict[str, float]:
            bank: dict[str, float] = {}
            if _support(mask) < min_sup:
                return bank
            for c in num_cols:
                vals = _values_on_mask(df, mask, c)
                is_const, val = _constant_value_on_mask(vals, eps)
                if is_const: bank[c] = val
            return bank

        def _const_ratio_conjecture(H, A: str, a_val: float, B: str, b_val: float) -> Conjecture | None:
            if not np.isfinite(a_val) or not np.isfinite(b_val) or b_val == 0.0:
                return None
            C = Fraction(a_val / b_val).limit_denominator(max_denom)
            return Conjecture(Eq(to_expr(A), to_expr(B) * Const(C)), H)

        def _iter_ratio_eqs_from_bank(H, bank: dict[str, float]):
            cols = sorted(bank.keys())
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    A, B = cols[i], cols[j]
                    a_val, b_val = bank[A], bank[B]
                    if _finite_nonzero_scalar(b_val):
                        conj = _const_ratio_conjecture(H, A, a_val, B, b_val)
                        if conj is not None:
                            yield conj

        ratio_eqs: list[Conjecture] = []
        for H in self.hyps_kept:
            mask = _mask_from_pred(self.df, H)
            bank = _make_const_bank_for_H(self.df, mask, self.numeric_columns)
            self.const_bank[_hyp_key(H)] = bank
            for conj in _iter_ratio_eqs_from_bank(H, bank):
                try:
                    if conj.is_true(self.df):
                        ratio_eqs.append(conj)
                except Exception:
                    pass

        # dedupe
        seen = set(); uniq = []
        for c in ratio_eqs:
            key = (repr(c.relation), c.condition.pretty() if hasattr(c.condition, "pretty") else str(c.condition))
            if key in seen: continue
            seen.add(key); uniq.append(c)
        self.constant_ratio_equalities = uniq

    # ── mixed two-feature bounds (sqrt and square) ── #
    def generate_mixed_bounds(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable] = None,
        primary: Optional[Iterable[str]] = None,
        secondary: Optional[Iterable[str]] = None,
        weight: float = 0.5,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        assert 0 < weight <= 1.0
        target = to_expr(target_col)
        hyps_iter = (hyps or self.hyps_kept)
        prim_cols = list(primary or self.numeric_columns)
        sec_cols  = list(secondary or self.numeric_columns)

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []

        w = float(weight)
        w_const = to_frac_const(weight, self.config.max_denom)

        def _pick_best_ge(t_arr, rhs_variants):
            best = None; best_score = -np.inf
            for lab, rhs, make_expr in rhs_variants:
                if np.all(t_arr >= rhs):
                    score = float(np.mean(rhs))
                    if score > best_score:
                        best = (lab, make_expr); best_score = score
            return best

        def _pick_best_le(t_arr, rhs_variants):
            best = None; best_score = np.inf
            for lab, rhs, make_expr in rhs_variants:
                if np.all(t_arr <= rhs):
                    score = float(np.mean(rhs))
                    if score < best_score:
                        best = (lab, make_expr); best_score = score
            return best

        for H in hyps_iter:
            mask = self._mask_for(self.df, H)
            if not np.any(mask): continue
            dfH = self.df.loc[mask]
            cache = _EvalCache(dfH)
            t_arr = target.eval(dfH).values.astype(float, copy=False)

            for xname in prim_cols:
                if xname == target_col: continue
                x_arr = cache.col(xname)
                if np.min(x_arr) <= 0: continue

                rx = t_arr / x_arr
                cmin_f, cmax_f = float(np.min(rx)), float(np.max(rx))

                for yname in sec_cols:
                    if yname == target_col: continue
                    y_arr = cache.col(yname)
                    if np.min(y_arr) <= 0: continue

                    # sqrt mix
                    sqrt_y_arr = cache.sqrt_col(yname)
                    r_sqrt = t_arr / sqrt_y_arr
                    s_cmin_f, s_cmax_f = float(np.min(r_sqrt)), float(np.max(r_sqrt))

                    mix_lower_arr = w * (cmin_f * x_arr + s_cmin_f * sqrt_y_arr)
                    mix_upper_arr = w * (cmax_f * x_arr + s_cmax_f * sqrt_y_arr)

                    lower_mix_variants = [
                        ("base", mix_lower_arr,
                         lambda: w_const * to_frac_const(cmin_f) * to_expr(xname)
                               + w_const * to_frac_const(s_cmin_f) * sqrt(to_expr(yname))),
                        ("ceil whole", np.ceil(mix_lower_arr),
                         lambda: ceil(w_const * to_frac_const(cmin_f) * to_expr(xname)
                                   + w_const * to_frac_const(s_cmin_f) * sqrt(to_expr(yname)))),
                        ("ceil-split", np.ceil(w * cmin_f * x_arr) + np.ceil(w * s_cmin_f * sqrt_y_arr) - 1.0,
                         lambda: ceil(w_const * to_frac_const(cmin_f) * to_expr(xname))
                               + ceil(w_const * to_frac_const(s_cmin_f) * sqrt(to_expr(yname))) - Const(1)),
                    ]
                    choice = _pick_best_ge(t_arr, lower_mix_variants)
                    if choice is not None:
                        _, make_expr = choice
                        lowers.append(Conjecture(Ge(target, make_expr()), H))

                    upper_mix_variants = [
                        ("base", mix_upper_arr,
                         lambda: w_const * to_frac_const(cmax_f) * to_expr(xname)
                               + w_const * to_frac_const(s_cmax_f) * sqrt(to_expr(yname))),
                        ("floor whole", np.floor(mix_upper_arr),
                         lambda: floor(w_const * to_frac_const(cmax_f) * to_expr(xname)
                                     + w_const * to_frac_const(s_cmax_f) * sqrt(to_expr(yname)))),
                        ("floor-split", np.floor(w * cmax_f * x_arr) + np.floor(w * s_cmax_f * sqrt_y_arr),
                         lambda: floor(w_const * to_frac_const(cmax_f) * to_expr(xname))
                               + floor(w_const * to_frac_const(s_cmax_f) * sqrt(to_expr(yname)))),
                    ]
                    choice = _pick_best_le(t_arr, upper_mix_variants)
                    if choice is not None:
                        _, make_expr = choice
                        uppers.append(Conjecture(Le(target, make_expr()), H))

                    # square mix
                    y_sq_arr = cache.sq_col(yname)
                    r_sq = t_arr / y_sq_arr
                    q_cmin_f, q_cmax_f = float(np.min(r_sq)), float(np.max(r_sq))

                    lower_sq_variants = [
                        ("base", w * (cmin_f * x_arr + q_cmin_f * y_sq_arr),
                         lambda: w_const * to_frac_const(cmin_f) * to_expr(xname)
                               + w_const * to_frac_const(q_cmin_f) * (to_expr(yname) ** to_frac_const(2))),
                        ("ceil whole", np.ceil(w * (cmin_f * x_arr + q_cmin_f * y_sq_arr)),
                         lambda: ceil(w_const * to_frac_const(cmin_f) * to_expr(xname)
                                    + w_const * to_frac_const(q_cmin_f) * (to_expr(yname) ** to_frac_const(2)))),
                    ]
                    choice = _pick_best_ge(t_arr, lower_sq_variants)
                    if choice is not None:
                        _, make_expr = choice
                        lowers.append(Conjecture(Ge(target, make_expr()), H))

                    upper_sq_variants = [
                        ("base", w * (cmax_f * x_arr + q_cmax_f * y_sq_arr),
                         lambda: w_const * to_frac_const(cmax_f) * to_expr(xname)
                               + w_const * to_frac_const(q_cmax_f) * (to_expr(yname) ** to_frac_const(2))),
                        ("floor whole", np.floor(w * (cmax_f * x_arr + q_cmax_f * y_sq_arr)),
                         lambda: floor(w_const * to_frac_const(cmax_f) * to_expr(xname)
                                      + w_const * to_frac_const(q_cmax_f) * (to_expr(yname) ** to_frac_const(2)))),
                    ]
                    choice = _pick_best_le(t_arr, upper_sq_variants)
                    if choice is not None:
                        _, make_expr = choice
                        uppers.append(Conjecture(Le(target, make_expr()), H))

        return lowers, uppers

    # ── targeted product bounds ── #
    def generate_targeted_product_bounds(
        self,
        target_col: str,
        *,
        hyps=None,
        x_candidates=None,
        yz_candidates=None,
        require_pos: bool = True,
        enable_cancellation: bool = True,
        allow_x_equal_yz: bool = True,
    ):
        target_expr = to_expr(target_col)
        hyps_iter = hyps or self.hyps_kept

        x_cands = list(x_candidates or [c for c in self.numeric_columns if c != target_col])
        yz_cands = list(yz_candidates or list(self.numeric_columns))

        uppers: List[Conjecture] = []
        lowers: List[Conjecture] = []

        def _all_le_on_mask(a, b, m):
            m = m & np.isfinite(a) & np.isfinite(b)
            return np.any(m) and bool(np.all(a[m] <= b[m]))

        def _strictly_pos_on(a, m):
            m = m & np.isfinite(a)
            return np.any(m) and bool(np.all(a[m] > 0.0))

        def _caro_trivial(T, x, y, z, m, upper: bool):
            m = m & np.isfinite(T) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if not np.any(m): return False
            if upper:
                c1 = _all_le_on_mask(T, y, m) and _all_le_on_mask(x, z, m)
                if c1: return True
                c2 = _all_le_on_mask(T, z, m) and _all_le_on_mask(x, y, m)
                return c2
            else:
                c1 = _all_le_on_mask(y, T, m) and _all_le_on_mask(z, x, m)
                if c1: return True
                c2 = _all_le_on_mask(z, T, m) and _all_le_on_mask(y, x, m)
                return c2

        for H in hyps_iter:
            mask = self._mask_for(self.df, H)
            if not np.any(mask): continue
            dfH = self.df.loc[mask]
            cache = _EvalCache(dfH)

            T_arr = target_expr.eval(dfH).values.astype(float, copy=False)
            arrays = {c: cache.col(c) for c in set(x_cands) | set(yz_cands)}
            finite = {c: np.isfinite(arrays[c]) for c in arrays}

            for x in x_cands:
                x_arr = arrays[x]
                x_ok = finite[x] & (x_arr > 0.0 if require_pos else np.isfinite(x_arr))
                if not np.any(x_ok): continue

                for (y, z) in combinations_with_replacement(yz_cands, 2):
                    if not allow_x_equal_yz and (x == y or x == z):
                        continue
                    y_arr, z_arr = arrays[y], arrays[z]
                    base_valid = finite[y] & finite[z] & finite[x] & np.isfinite(T_arr)
                    if require_pos:
                        base_valid &= (T_arr > 0.0) & (y_arr > 0.0) & (z_arr > 0.0) & (x_arr > 0.0)
                    if not np.any(base_valid): continue

                    canceled_upper = False
                    canceled_lower = False

                    if enable_cancellation:
                        if x == y and _strictly_pos_on(x_arr, base_valid):
                            if _all_le_on_mask(T_arr, z_arr, base_valid):
                                uppers.append(Conjecture(Le(target_expr, to_expr(z)), H)); canceled_upper = True
                            if _all_le_on_mask(z_arr, T_arr, base_valid):
                                lowers.append(Conjecture(Ge(target_expr, to_expr(z)), H)); canceled_lower = True
                        if x == z and _strictly_pos_on(x_arr, base_valid):
                            if _all_le_on_mask(T_arr, y_arr, base_valid):
                                uppers.append(Conjecture(Le(target_expr, to_expr(y)), H)); canceled_upper = True
                            if _all_le_on_mask(y_arr, T_arr, base_valid):
                                lowers.append(Conjecture(Ge(target_expr, to_expr(y)), H)); canceled_lower = True

                    L = T_arr * x_arr
                    R = y_arr * z_arr

                    if not canceled_upper:
                        if not (require_pos and _caro_trivial(T_arr, x_arr, y_arr, z_arr, base_valid, upper=True)):
                            if _all_le_on_mask(L, R, base_valid):
                                uppers.append(Conjecture(Le(target_expr * to_expr(x), to_expr(y) * to_expr(z)), H))

                    if not canceled_lower:
                        if not (require_pos and _caro_trivial(T_arr, x_arr, y_arr, z_arr, base_valid, upper=False)):
                            if _all_le_on_mask(R, L, base_valid):
                                lowers.append(Conjecture(Ge(target_expr * to_expr(x), to_expr(y) * to_expr(z)), H))

        return lowers, uppers

    # ── qualitative mining ── #
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
        Reintroduces the classic 'intricate inequalities' technique:
        y ≥ w*(cmin_x·x + cmin_sqrt·sqrt(y)) with best-of {base, ceil-whole, ceil-split-1}
        y ≤ w*(cmax_x·x + cmax_sqrt·sqrt(y)) with best-of {base, floor-whole, floor-split}
        and the analogous square-mix using y^2.
        Optimized: single _ExprEvalCache per hypothesis; no repeated expr.eval.
        """
        assert 0 < weight <= 1.0
        target = to_expr(target_col)
        hyps_iter = (hyps or self.hyps_kept)
        prim_cols = list(primary or self.numeric_columns)
        sec_cols  = list(secondary or self.numeric_columns)

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []

        w = float(weight)
        w_const = to_frac_const(weight, self.config.max_denom)

        def _pick_best_ge(t_arr, rhs_variants):
            # choose the valid variant (all t ≥ rhs) with largest mean(rhs)
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
            # choose the valid variant (all t ≤ rhs) with smallest mean(rhs)
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

        for H in hyps_iter:
            mask = self._mask_for(self.df, H)
            if not np.any(mask):
                continue

            ecache = _ExprEvalCache(self.df)
            t_arr = ecache.arr(target)[mask]

            for xname in prim_cols:
                if xname == target_col:
                    continue

                x_arr = ecache.arr(to_expr(xname))[mask]
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

                    y_arr = ecache.arr(to_expr(yname))[mask]
                    if y_arr.size == 0 or np.nanmin(y_arr) <= 0:
                        continue

                    # ---------- sqrt mix ----------
                    sqrt_y_arr = np.sqrt(y_arr, dtype=float)
                    r_sqrt = t_arr / sqrt_y_arr
                    f_sq = np.isfinite(r_sqrt)
                    if np.any(f_sq):
                        s_cmin_f = float(np.min(r_sqrt[f_sq]))
                        s_cmax_f = float(np.max(r_sqrt[f_sq]))

                        # arrays for candidate variants
                        base_lo   = w * (cmin_f * x_arr + s_cmin_f * sqrt_y_arr)
                        base_up   = w * (cmax_f * x_arr + s_cmax_f * sqrt_y_arr)
                        ceil_whole = np.ceil(base_lo)
                        floor_whole = np.floor(base_up)
                        ceil_split = np.ceil(w * cmin_f * x_arr) + np.ceil(w * s_cmin_f * sqrt_y_arr) - 1.0
                        floor_split = np.floor(w * cmax_f * x_arr) + np.floor(w * s_cmax_f * sqrt_y_arr)

                        # symbolic builders (match the arrays above)
                        def _lo_base():
                            return (w_const * to_frac_const(cmin_f) * to_expr(xname)
                                + w_const * to_frac_const(s_cmin_f) * sqrt(to_expr(yname)))

                        def _lo_ceil_whole():
                            return ceil(_lo_base())

                        def _lo_ceil_split():
                            return (ceil(w_const * to_frac_const(cmin_f) * to_expr(xname))
                                + ceil(w_const * to_frac_const(s_cmin_f) * sqrt(to_expr(yname)))
                                - Const(1))

                        def _up_base():
                            return (w_const * to_frac_const(cmax_f) * to_expr(xname)
                                + w_const * to_frac_const(s_cmax_f) * sqrt(to_expr(yname)))

                        def _up_floor_whole():
                            return floor(_up_base())

                        def _up_floor_split():
                            return (floor(w_const * to_frac_const(cmax_f) * to_expr(xname))
                                + floor(w_const * to_frac_const(s_cmax_f) * sqrt(to_expr(yname))))

                        lo_choice = _pick_best_ge(t_arr, [
                            ("base",        base_lo,     _lo_base),
                            ("ceil_whole",  ceil_whole,  _lo_ceil_whole),
                            ("ceil_split",  ceil_split,  _lo_ceil_split),
                        ])
                        if lo_choice is not None:
                            lowers.append(Conjecture(Ge(target, lo_choice()), H))

                        up_choice = _pick_best_le(t_arr, [
                            ("base",        base_up,      _up_base),
                            ("floor_whole", floor_whole,  _up_floor_whole),
                            ("floor_split", floor_split,  _up_floor_split),
                        ])
                        if up_choice is not None:
                            uppers.append(Conjecture(Le(target, up_choice()), H))

                    # ---------- square mix ----------
                    y_sq_arr = np.square(y_arr, dtype=float)
                    r_sq = t_arr / y_sq_arr
                    f_rsq = np.isfinite(r_sq)
                    if not np.any(f_rsq):
                        continue
                    q_cmin_f = float(np.min(r_sq[f_rsq]))
                    q_cmax_f = float(np.max(r_sq[f_rsq]))

                    base_lo_sq   = w * (cmin_f * x_arr + q_cmin_f * y_sq_arr)
                    base_up_sq   = w * (cmax_f * x_arr + q_cmax_f * y_sq_arr)
                    ceil_whole_sq  = np.ceil(base_lo_sq)
                    floor_whole_sq = np.floor(base_up_sq)

                    def _lo_sq_base():
                        return (w_const * to_frac_const(cmin_f) * to_expr(xname)
                            + w_const * to_frac_const(q_cmin_f) * (to_expr(yname) ** to_frac_const(2)))

                    def _lo_sq_ceil_whole():
                        return ceil(_lo_sq_base())

                    def _up_sq_base():
                        return (w_const * to_frac_const(cmax_f) * to_expr(xname)
                            + w_const * to_frac_const(q_cmax_f) * (to_expr(yname) ** to_frac_const(2)))

                    def _up_sq_floor_whole():
                        return floor(_up_sq_base())

                    lo_sq_choice = _pick_best_ge(t_arr, [
                        ("base",       base_lo_sq,     _lo_sq_base),
                        ("ceil_whole", ceil_whole_sq,  _lo_sq_ceil_whole),
                    ])
                    if lo_sq_choice is not None:
                        lowers.append(Conjecture(Ge(target, lo_sq_choice()), H))

                    up_sq_choice = _pick_best_le(t_arr, [
                        ("base",        base_up_sq,     _up_sq_base),
                        ("floor_whole", floor_whole_sq, _up_sq_floor_whole),
                    ])
                    if up_sq_choice is not None:
                        uppers.append(Conjecture(Le(target, up_sq_choice()), H))

        return lowers, uppers

    def _broader_hypotheses(self, base_H) -> list:
        base_mask = _mask_from_pred(self.df, base_H)
        cands = []
        hyps = list(self.hyps_kept)
        if TRUE not in hyps:
            hyps.append(TRUE)
        for Hp in hyps:
            pmask = _mask_from_pred(self.df, Hp)
            if _mask_includes(base_mask, pmask) and (_support(pmask) >= self.config.min_support_const):
                cands.append((Hp, _support(pmask)))
        return [h for h, _ in sorted(cands, key=lambda t: -t[1])]

    def rank_and_filter(self, conjs: List[Conjecture], *, min_touch: Optional[int]=None) -> List[Conjecture]:
        m = int(min_touch if min_touch is not None else self.config.min_touch_keep)
        return rank_and_filter_fast(self.df, conjs, m)

    def run_single_feature_pipeline(self, target_col: str):
        lows, ups = self.generate_single_feature_bounds(target_col)
        return self.rank_and_filter(lows), self.rank_and_filter(ups)

    def run_mixed_pipeline(self, target_col: str, *, weight: float = 0.5,
                           primary: Optional[Iterable[str]] = None,
                           secondary: Optional[Iterable[str]] = None):
        lows, ups = self.generate_mixed_bounds(target_col, weight=weight, primary=primary, secondary=secondary)
        return self.rank_and_filter(lows), self.rank_and_filter(ups)

    def run_targeted_product_pipeline(self, target_col: str, *,
                                      require_pos: bool = True,
                                      enable_cancellation: bool = True,
                                      allow_x_equal_yz: bool = True,
                                      x_candidates=None, yz_candidates=None):
        lowers, uppers = self.generate_targeted_product_bounds(
            target_col, require_pos=require_pos, enable_cancellation=enable_cancellation,
            allow_x_equal_yz=allow_x_equal_yz, x_candidates=x_candidates, yz_candidates=yz_candidates,
        )
        return self.rank_and_filter(lowers), self.rank_and_filter(uppers)


    # ── R₁ class relations ── #
    def _transitive_reduce_inclusions(self, inclusions: List[ClassInclusion], mask_cache: dict[str, np.ndarray]) -> List[ClassInclusion]:
        nodes = sorted({_pred_key(inc.A) for inc in inclusions} | {_pred_key(inc.B) for inc in inclusions})
        idx = {k: i for i, k in enumerate(nodes)}
        adj = [[False]*len(nodes) for _ in nodes]
        edge_obj: dict[tuple[int,int], ClassInclusion] = {}
        for inc in inclusions:
            i = idx[_pred_key(inc.A)]; j = idx[_pred_key(inc.B)]
            adj[i][j] = True; edge_obj[(i, j)] = inc
        keep = set(edge_obj.keys())
        for i in range(len(nodes)):
            for k in range(len(nodes)):
                if not adj[i][k]: continue
                if any(adj[i][j] and adj[j][k] for j in range(len(nodes)) if j != i and j != k):
                    keep.discard((i, k))
        out, seen = [], set()
        for (i, j) in keep:
            inc = edge_obj[(i, j)]
            sig = inc.signature()
            if sig in seen: continue
            seen.add(sig); out.append(inc)
        return out

    def generate_lp_linear_with_intercept(
        self,
        target_col: str,
        *,
        features: list[str] | None = None,
        hyps: Iterable | None = None,
        max_denom: int = 30,
        tol: float = 1e-8,
        try_int_strengthen: bool = True,
    ) -> tuple[list[Conjecture], list[Conjecture]]:
        target = to_expr(target_col)
        hyps_iter = list(hyps or self.hyps_kept)
        feat_cols = [c for c in (features or self.numeric_columns) if c != target_col]

        lowers: list[Conjecture] = []
        uppers: list[Conjecture] = []
        if not feat_cols:
            return lowers, uppers

        # Pre-materialize numeric block once
        numdf = self.df[feat_cols + [target_col]].apply(pd.to_numeric, errors="coerce")

        def _rhs_from_coeffs(names: list[str], a: np.ndarray, b: float, *, rational: bool):
            rhs = Const(0.0)
            if rational:
                from fractions import Fraction
                rhs = Const(Fraction(b).limit_denominator(max_denom))
                for aj, name in zip(a, names):
                    if abs(aj) < tol:
                        continue
                    rhs = rhs + Const(Fraction(aj).limit_denominator(max_denom)) * to_expr(name)
            else:
                rhs = Const(float(b))
                for aj, name in zip(a, names):
                    if abs(aj) < tol:
                        continue
                    rhs = rhs + Const(float(aj)) * to_expr(name)
            return rhs

        def _valid_on_rows(df_slice: pd.DataFrame, lhs_e, rhs_e, *, upper: bool, eps: float = 1e-9) -> bool:
            L = lhs_e.eval(df_slice).to_numpy(dtype=float, copy=False)
            R = rhs_e.eval(df_slice).to_numpy(dtype=float, copy=False)
            ok = np.isfinite(L) & np.isfinite(R)
            if not np.any(ok):
                return False
            if upper:
                return bool(np.all(L[ok] <= R[ok] + eps))
            else:
                return bool(np.all(L[ok] + eps >= R[ok]))

        for H in hyps_iter:
            mask = self._mask_for(self.df, H)
            if not np.any(mask):
                continue

            sub = numdf.loc[mask]
            X_full = sub[feat_cols].to_numpy(dtype=float, copy=False)
            y_full = sub[target_col].to_numpy(dtype=float, copy=False)

            finite_rows = np.isfinite(y_full)
            for j in range(X_full.shape[1]):
                finite_rows &= np.isfinite(X_full[:, j])

            if not np.any(finite_rows):
                continue

            # Use the exact rows the LP will see for BOTH fitting and validation
            X = X_full[finite_rows, :]
            y = y_full[finite_rows]
            df_fit = sub.iloc[np.flatnonzero(finite_rows)]
            if X.shape[0] == 0:
                continue

            for sense in ("upper", "lower"):
                try:
                    a, b = self._lp_solve_sum_slack(X, y, sense=sense)
                except Exception:
                    continue

                # Guard against pathological solver outputs
                if not np.all(np.isfinite(a)) or not np.isfinite(b):
                    continue

                # Float fit (guaranteed by LP on df_fit)
                rhs_float = _rhs_from_coeffs(feat_cols, a, b, rational=False)
                rel_float = Le(target, rhs_float) if sense == "upper" else Ge(target, rhs_float)
                if not _valid_on_rows(df_fit, target, rhs_float, upper=(sense == "upper")):
                    continue  # shouldn’t happen, but be safe
                keep = Conjecture(rel_float, H)

                # Try rationalized coefficients—keep only if still valid on same rows
                rhs_rat = _rhs_from_coeffs(feat_cols, a, b, rational=True)
                if _valid_on_rows(df_fit, target, rhs_rat, upper=(sense == "upper")):
                    keep = Conjecture(Le(target, rhs_rat), H) if sense == "upper" else Conjecture(Ge(target, rhs_rat), H)

                # Optional integer strengthening (ceil for lowers, floor for uppers)
                if try_int_strengthen:
                    if sense == "upper":
                        rhs_try = floor(keep.relation.right)
                        if _valid_on_rows(df_fit, target, rhs_try, upper=True):
                            keep = Conjecture(Le(target, rhs_try), H)
                    else:
                        rhs_try = ceil(keep.relation.right)
                        if _valid_on_rows(df_fit, target, rhs_try, upper=False):
                            keep = Conjecture(Ge(target, rhs_try), H)

                (uppers if sense == "upper" else lowers).append(keep)

        return lowers, uppers

    def discover_class_relations(
        self,
        *,
        predicates: Optional[Iterable[Predicate]] = None,
        include_bool_columns: bool = False,
        min_support_A: int = 3,
        skip_trivial_equiv: bool = True,
        disallow_shared_atoms: bool = True,
        ambient_atoms: Optional[Set[str]] = None,
    ) -> Tuple[List[ClassEquivalence], List[ClassInclusion]]:
        cand: List[Predicate] = list(predicates or self.hyps_kept)
        if include_bool_columns:
            from txgraffiti2025.forms.predicates import Predicate as _P
            for c in getattr(self, "bool_columns", []):
                try: cand.append(_P.from_column(c))
                except Exception: pass

        mask_cache: dict[str, np.ndarray] = {}
        obj_cache:  dict[str, Predicate]  = {}
        atoms_cache: dict[str, Set[str]]  = {}

        for p in cand:
            k = _pred_key(p)
            if k in mask_cache: continue
            try: m = _mask_bool(self.df, p)
            except Exception: continue
            mask_cache[k] = m; obj_cache[k] = p; atoms_cache[k] = _atoms_for_pred(p)

        keys = list(mask_cache.keys())
        n = len(self.df)
        all_true  = np.ones(n, dtype=bool)
        all_false = np.zeros(n, dtype=bool)

        ambient_atoms = set() if ambient_atoms is None else {_canon_atom(a) for a in ambient_atoms}

        equivalences: List[ClassEquivalence] = []
        inclusions:   List[ClassInclusion]   = []

        # Equivalences
        for i, ki in enumerate(keys):
            mi = mask_cache[ki]; Ai = obj_cache[ki]; atoms_i = atoms_cache[ki]
            for kj in keys[i+1:]:
                mj = mask_cache[kj]; Aj = obj_cache[kj]; atoms_j = atoms_cache[kj]
                if not _same_mask(mi, mj): continue
                if skip_trivial_equiv and (np.array_equal(mi, all_true) or np.array_equal(mi, all_false)):
                    continue
                if (atoms_i.issubset(atoms_j) and atoms_i != atoms_j) or (atoms_j.issubset(atoms_i) and atoms_i != atoms_j):
                    continue
                equivalences.append(ClassEquivalence(Ai, Aj))

        # Inclusions
        for ki in keys:
            Ai = obj_cache[ki]; mi = mask_cache[ki]; atoms_i = atoms_cache[ki]
            suppA = _support(mi)
            if suppA < int(min_support_A) or np.array_equal(mi, all_false): continue

            lhs_conj_masks = [_mask_bool(self.df, c) for c in _flatten_and_conjuncts(Ai)]

            for kj in keys:
                if kj == ki: continue
                Aj = obj_cache[kj]; mj = mask_cache[kj]; atoms_j = atoms_cache[kj]
                if np.array_equal(mj, all_true): continue
                if np.any(mi & ~mj): continue
                if atoms_j.issubset(atoms_i): continue
                if disallow_shared_atoms:
                    inter = (atoms_i & atoms_j) - ambient_atoms
                    if inter: continue
                if any(_same_mask(cm, mj) for cm in lhs_conj_masks):  # RHS already a conjunct
                    continue
                inclusions.append(ClassInclusion(Ai, Aj))

        # de-dup
        def _uniq_by_sig(items):
            seen = set(); out = []
            for x in items:
                s = x.signature()
                if s in seen: continue
                seen.add(s); out.append(x)
            return out

        equivalences = _uniq_by_sig(equivalences)
        inclusions   = _uniq_by_sig(inclusions)

        inclusions = self._transitive_reduce_inclusions(inclusions, mask_cache)

        def _equiv_support(eq: ClassEquivalence) -> int:
            return _support(mask_cache[_pred_key(eq.A)])
        equivalences.sort(key=lambda e: (_equiv_support(e), e.pretty()), reverse=True)
        inclusions.sort(key=lambda inc: (_support(mask_cache[_pred_key(inc.A)]), inc.pretty()), reverse=True)

        return equivalences, inclusions

    def generate_lp_affine_bounds(
        self,
        target_col: str,
        *,
        features: Optional[Iterable[str]] = None,
        hyps: Optional[Iterable] = None,
        sense: str = "both",
        l1_penalty: float = 0.0,
        var_bounds: float | None = None,
        strengthen_int: bool = True,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        Global LP bounds with an intercept. Returns valid conjectures even if
        rationalization would break validity (falls back to float coefficients).
        """
        target = to_expr(target_col)
        hyps_iter = (hyps or self.hyps_kept)
        feat_cols = [c for c in (features or self.numeric_columns) if c != target_col]

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []

        # materialize once
        numdf = self.df[feat_cols + [target_col]].apply(pd.to_numeric, errors="coerce")
        T_all = numdf[target_col].to_numpy(dtype=float, copy=False)
        X_all = numdf[feat_cols].to_numpy(dtype=float, copy=False)

        def _rhs_expr_float(names, a, b):
            # build RHS using float constants (no rationalization)
            rhs = None
            for name, aj in zip(names, a):
                if not np.isfinite(aj) or aj == 0.0:
                    continue
                term = Const(float(aj)) * to_expr(name)
                rhs = term if rhs is None else (rhs + term)
            bterm = Const(float(b))
            rhs = bterm if rhs is None else (rhs + bterm)
            return rhs

        def _valid_everywhere(mask, lhs_expr, rhs_expr, upper: bool) -> bool:
            dfH = self.df.loc[mask]
            lhs = lhs_expr.eval(dfH).to_numpy(dtype=float, copy=False)
            rhs = rhs_expr.eval(dfH).to_numpy(dtype=float, copy=False)
            ok = np.isfinite(lhs) & np.isfinite(rhs)
            if not np.any(ok):
                return False
            return bool(np.all(lhs[ok] <= rhs[ok])) if upper else bool(np.all(lhs[ok] >= rhs[ok]))

        senses = []
        if sense in ("upper", "both"): senses.append("upper")
        if sense in ("lower", "both"): senses.append("lower")

        for H in hyps_iter:
            mask = self._mask_for(self.df, H)
            if not np.any(mask):
                continue

            X = X_all[mask, :]
            y = T_all[mask]

            finite_rows = np.isfinite(y)
            for j in range(X.shape[1]):
                finite_rows &= np.isfinite(X[:, j])
            if not np.any(finite_rows):
                continue
            X = X[finite_rows, :]
            y = y[finite_rows]

            if X.shape[0] == 0:
                continue

            for sn in senses:
                try:
                    a, b = _solve_sum_slack_lp_strict(X, y, sense=sn, l1_penalty=l1_penalty, var_bounds=var_bounds)
                except Exception:
                    continue

                # 1) float version (always check this first)
                rhs_float = _rhs_expr_float(feat_cols, a, b)
                rel = Le(target, rhs_float) if sn == "upper" else Ge(target, rhs_float)
                conj = Conjecture(rel, H)
                if not _valid_everywhere(mask, target, rhs_float, upper=(sn == "upper")):
                    # should basically never happen; LP enforces global feasibility
                    continue

                final_conj = conj

                # 2) try rationalized version; keep only if still valid
                rhs_rat = _affine_expr_from_coef(feat_cols, a, b, self.config.max_denom)
                if _valid_everywhere(mask, target, rhs_rat, upper=(sn == "upper")):
                    final_conj = Conjecture(Le(target, rhs_rat), H) if sn == "upper" else Conjecture(Ge(target, rhs_rat), H)

                # 3) optional strengthening on the version we currently keep
                if strengthen_int:
                    if sn == "upper":
                        rhs_try = floor(final_conj.relation.right)
                        if _valid_everywhere(mask, target, rhs_try, upper=True):
                            final_conj = Conjecture(Le(target, rhs_try), H)
                    else:
                        rhs_try = ceil(final_conj.relation.right)
                        if _valid_everywhere(mask, target, rhs_try, upper=False):
                            final_conj = Conjecture(Ge(target, rhs_try), H)

                if sn == "upper":
                    uppers.append(final_conj)
                else:
                    lowers.append(final_conj)

        return lowers, uppers

    def generate_lp_two_feature_bounds_affine(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable] = None,
        features: Optional[Sequence[Union[str, Expr]]] = None,
        transforms: Sequence[str] = ("identity", "sqrt", "square"),
        rationalize: bool = True,
        require_min_rows: int = 3,
        require_finite: bool = True,
        nonneg_coeffs: bool = False,
        max_denom: int = 30,
        coef_eps: float = 1e-12,
        try_whole_rounds: bool = True,
        try_split_rounds: bool = True,
    ) -> tuple[list[Conjecture], list[Conjecture]]:

        def _get_solver():
            cbc = shutil.which("cbc")
            if cbc:
                return pulp.COIN_CMD(path=cbc, msg=False)
            glpk = shutil.which("glpsol")
            if glpk:
                pulp.LpSolverDefault.msg = 0
                return pulp.GLPK_CMD(path=glpk, msg=False)
            raise RuntimeError("No LP solver found (install CBC or GLPK)")

        def _solve_sum_slack_lp(X: np.ndarray, y: np.ndarray, *, sense: str) -> tuple[np.ndarray, float]:
            n, k = X.shape
            prob = pulp.LpProblem("sum_slack", pulp.LpMinimize)
            a = [pulp.LpVariable(f"a_{j}", lowBound=None) for j in range(k)]
            b = pulp.LpVariable("b", lowBound=None)
            s = [pulp.LpVariable(f"s_{i}", lowBound=0) for i in range(n)]
            prob += pulp.lpSum(s)
            for i in range(n):
                lhs = pulp.lpSum(a[j]*float(X[i, j]) for j in range(k)) + b
                yi  = float(y[i])
                if sense == "upper":
                    prob += lhs - yi == s[i]
                elif sense == "lower":
                    prob += yi - lhs == s[i]
                else:
                    raise ValueError("sense must be 'upper' or 'lower'")
            status = prob.solve(_get_solver())
            if pulp.LpStatus[status] != "Optimal":
                raise RuntimeError(f"LP did not solve optimally: {pulp.LpStatus[status]}")
            a_sol = np.array([v.value() for v in a], dtype=float)
            b_sol = float(b.value())
            return a_sol, b_sol

        lp_solve = getattr(self, "_lp_solve_sum_slack", None) or _solve_sum_slack_lp

        targetE = to_expr(target_col)
        Hs = list(hyps or self.hyps_kept)

        if features is None:
            cols = [c for c in self.numeric_columns if c != target_col]
            feat_exprs = [to_expr(c) for c in cols]
        else:
            feat_exprs = []
            for f in features:
                e = to_expr(f)
                if getattr(e, "pretty", lambda: "")() != target_col:
                    feat_exprs.append(e)
        if not feat_exprs:
            return [], []

        def _to_float_np(s: pd.Series) -> np.ndarray:
            return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, copy=False)

        t_full = _to_float_np(targetE.eval(self.df))

        x_full_map: dict[str, np.ndarray] = {}
        y_full_map: dict[tuple[str, str], np.ndarray] = {}
        y_sym_map:  dict[tuple[str, str], Expr] = {}

        for e in feat_exprs:
            key = e.pretty()
            s = e.eval(self.df)
            x_full_map[key] = np.asarray(pd.to_numeric(s, errors="coerce").to_numpy(), dtype=float, order="C")

        allowed_T = [t for t in transforms if t in ("identity", "sqrt", "square")] or ["identity"]

        for e in feat_exprs:
            key = e.pretty()
            arr = x_full_map[key]
            for T in allowed_T:
                tup = (key, T)
                if T == "identity":
                    y_full_map[tup] = arr
                    y_sym_map[tup]  = e
                elif T == "sqrt":
                    y_full_map[tup] = np.sqrt(np.maximum(arr, 0.0))
                    y_sym_map[tup]  = sqrt(e)
                elif T == "square":
                    y_full_map[tup] = np.square(arr, dtype=float)
                    y_sym_map[tup]  = e ** to_frac_const(2, max_denom)

        pairs = [(feat_exprs[i], feat_exprs[j]) for i in range(len(feat_exprs)) for j in range(i, len(feat_exprs))]

        lowers: list[Conjecture] = []
        uppers: list[Conjecture] = []

        # === ZERO-DROP HELPERS (post-rationalization) ============================
        def _nz_const_from_float(x: float):
            """Return a Const for x after rationalization, or None if it’s exactly 0."""
            if abs(x) <= coef_eps:
                return None
            c = to_frac_const(x, max_denom) if rationalize else Const(float(x))
            try:
                # when rationalized, ensure it’s not exactly zero
                if hasattr(c, "value"):
                    # Fraction-like zero check
                    if float(c.value) == 0.0:
                        return None
            except Exception:
                pass
            return c

        def _rhs_expr(a1: float, a2: float, b: float, xE: Expr, ySym: Expr) -> Optional[Expr]:
            terms: list[Expr] = []
            c1 = _nz_const_from_float(a1)
            if c1 is not None:
                terms.append(c1 * xE)
            c2 = _nz_const_from_float(a2)
            if c2 is not None:
                terms.append(c2 * ySym)
            cb = _nz_const_from_float(b)
            if cb is not None:
                terms.append(cb)
            if not terms:
                return None
            out = terms[0]
            for t in terms[1:]:
                out = out + t
            return out

        def _numeric_terms(a1, a2, b, X2) -> list[np.ndarray]:
            """Per-term numeric arrays (only for truly nonzero terms, pre-split rounding)."""
            pieces = []
            if _nz_const_from_float(a1) is not None:
                pieces.append(a1 * X2[:, 0])
            if _nz_const_from_float(a2) is not None:
                pieces.append(a2 * X2[:, 1])
            if _nz_const_from_float(b) is not None:
                pieces.append(np.full(X2.shape[0], b, dtype=float))
            return pieces

        def _symbolic_terms(a1, a2, b, xE: Expr, ySym: Expr) -> list[Expr]:
            parts: list[Expr] = []
            c1 = _nz_const_from_float(a1)
            if c1 is not None:
                parts.append(c1 * xE)
            c2 = _nz_const_from_float(a2)
            if c2 is not None:
                parts.append(c2 * ySym)
            cb = _nz_const_from_float(b)
            if cb is not None:
                parts.append(cb)
            return parts

        # === MAIN LOOP ===========================================================
        for H in Hs:
            m = self._mask_for(self.df, H)
            if not np.any(m):
                continue

            tH = t_full[m]
            dom_t = np.isfinite(tH)

            for xE, yE in pairs:
                xk, yk = xE.pretty(), yE.pretty()
                xH = x_full_map[xk][m]
                dom_x = np.isfinite(xH)

                for T in allowed_T:
                    yH = y_full_map[(yk, T)][m]
                    dom = dom_t & dom_x & np.isfinite(yH)
                    if T == "sqrt":
                        raw = x_full_map[yk][m]
                        dom &= (raw >= 0.0)
                    if require_finite:
                        dom &= np.isfinite(tH) & np.isfinite(xH) & np.isfinite(yH)

                    if np.count_nonzero(dom) < int(require_min_rows):
                        continue

                    X2 = np.stack([xH[dom], yH[dom]], axis=1)
                    yv = tH[dom]
                    if X2.shape[0] == 0:
                        continue

                    try:
                        a_up, b_up = lp_solve(X2, yv, sense="upper")
                        a_lo, b_lo = lp_solve(X2, yv, sense="lower")
                    except Exception:
                        continue

                    if nonneg_coeffs and (np.any(a_up < -coef_eps) or np.any(a_lo < -coef_eps)):
                        continue

                    symY = y_sym_map[(yk, T)]

                    # UPPER
                    rhsU = _rhs_expr(a_up[0], a_up[1], b_up, xE, symY)
                    if rhsU is not None:
                        uppers.append(Conjecture(Le(targetE, rhsU), H))

                        if try_whole_rounds:
                            rhs_num = a_up[0]*X2[:,0] + a_up[1]*X2[:,1] + b_up
                            if np.all(yv <= np.floor(rhs_num)):
                                uppers.append(Conjecture(Le(targetE, floor(rhsU)), H))

                        if try_split_rounds:
                            num_terms = _numeric_terms(a_up[0], a_up[1], b_up, X2)
                            if num_terms:
                                split_num = np.zeros_like(yv, dtype=float)
                                for arr in num_terms:
                                    split_num += np.floor(arr)
                                if np.all(yv <= split_num):
                                    parts = _symbolic_terms(a_up[0], a_up[1], b_up, xE, symY)
                                    if parts:
                                        sym = floor(parts[0])
                                        for p in parts[1:]:
                                            sym = sym + floor(p)
                                        uppers.append(Conjecture(Le(targetE, sym), H))

                    # LOWER
                    rhsL = _rhs_expr(a_lo[0], a_lo[1], b_lo, xE, symY)
                    if rhsL is not None:
                        lowers.append(Conjecture(Ge(targetE, rhsL), H))

                        if try_whole_rounds:
                            rhs_num = a_lo[0]*X2[:,0] + a_lo[1]*X2[:,1] + b_lo
                            if np.all(yv >= np.ceil(rhs_num)):
                                lowers.append(Conjecture(Ge(targetE, ceil(rhsL)), H))

                        if try_split_rounds:
                            num_terms = _numeric_terms(a_lo[0], a_lo[1], b_lo, X2)
                            m_terms = len(num_terms)
                            if m_terms > 0:
                                split_num = np.zeros_like(yv, dtype=float)
                                for arr in num_terms:
                                    split_num += np.ceil(arr)
                                adj = (m_terms - 1)
                                split_num_adj = split_num - adj
                                if np.all(yv >= split_num_adj):
                                    parts = _symbolic_terms(a_lo[0], a_lo[1], b_lo, xE, symY)
                                    if parts:
                                        sym = ceil(parts[0])
                                        for p in parts[1:]:
                                            sym = sym + ceil(p)
                                        if adj:
                                            sym = sym - Const(adj)
                                        lowers.append(Conjecture(Ge(targetE, sym), H))

        return lowers, uppers

    def promote_conclusions_to_hypotheses(
        self,
        conjs: list[Conjecture],
        *,
        min_support: int = 5,
        make_strict: bool = True,
        make_equality: bool = True,
        strict_epsilon: float = 1e-9,
        eq_tol: float = 1e-9,
        simplify: bool = True,
        treat_binary_ints: bool = True,
    ) -> list:
        """
        Turn conclusions (Ge/Le) into new hypothesis predicates by adding either:
        • strict versions:  left > right (resp. <) evaluated with a hidden epsilon
        • equalities:       left = right with tolerance eq_tol
        Returns a list of predicate-like objects that implement .mask(df) and .pretty().
        """
        # -- tiny predicate wrapper -------------------------------------------------
        class _PredWrap:
            def __init__(self, name: str, mask_fn):
                self._name = name
                self._mask_fn = mask_fn
            def mask(self, df: pd.DataFrame) -> pd.Series:
                m = self._mask_fn(df)
                if isinstance(m, pd.Series):
                    return m.astype(bool, copy=False)
                return pd.Series(np.asarray(m, dtype=bool), index=df.index)
            def pretty(self) -> str:
                return self._name
            def __repr__(self) -> str:
                return self._name

        def _and(H, Q):
            # Try native &, else wrap conjunction manually.
            try:
                return H & Q
            except Exception:
                def _mask(df):
                    m1 = H.mask(df).to_numpy(dtype=bool, copy=False)
                    m2 = Q.mask(df).to_numpy(dtype=bool, copy=False)
                    return pd.Series(m1 & m2, index=df.index, dtype=bool)
                name = f"({getattr(H, 'pretty', lambda: repr(H))()}) ∧ ({getattr(Q, 'pretty', lambda: repr(Q))()})"
                return _PredWrap(name, _mask)

        def _support(mask_like) -> int:
            if isinstance(mask_like, pd.Series):
                arr = mask_like.fillna(False).to_numpy(dtype=bool, copy=False)
            else:
                arr = np.asarray(mask_like, dtype=bool)
            return int(np.count_nonzero(arr))

        def _wrap_equality(leftE, rightE) -> _PredWrap:
            rel = Eq(leftE, rightE, tol=eq_tol)
            name = rel.pretty()
            def _mask(df):
                return rel.evaluate(df)
            return _PredWrap(name, _mask)

        def _wrap_strict(leftE, rightE, *, sense: str) -> _PredWrap:
            # evaluate with epsilon, pretty-print as strict > / <
            if sense == "Ge":  # left > right
                name = f"{repr(leftE)} > {repr(rightE)}"
                def _mask(df):
                    l = leftE.eval(df).to_numpy(dtype=float, copy=False)
                    r = rightE.eval(df).to_numpy(dtype=float, copy=False)
                    return pd.Series(l >= r + strict_epsilon, index=df.index, dtype=bool)
            else:              # "Le": left < right
                name = f"{repr(leftE)} < {repr(rightE)}"
                def _mask(df):
                    l = leftE.eval(df).to_numpy(dtype=float, copy=False)
                    r = rightE.eval(df).to_numpy(dtype=float, copy=False)
                    return pd.Series(l <= r - strict_epsilon, index=df.index, dtype=bool)
            return _PredWrap(name, _mask)

        # -- build candidates -------------------------------------------------------
        promoted = []
        for c in conjs:
            H = getattr(c, "condition", None)
            rel = getattr(c, "relation", None)
            if H is None or rel is None:
                continue
            leftE, rightE = to_expr(rel.left), to_expr(rel.right)

            if make_equality:
                eq_pred = _wrap_equality(leftE, rightE)
                promoted.append(eq_pred)

            if make_strict:
                if isinstance(rel, Ge):
                    promoted.append(_wrap_strict(leftE, rightE, sense="Ge"))
                elif isinstance(rel, Le):
                    promoted.append(_wrap_strict(leftE, rightE, sense="Le"))

        # -- filter by support + de-dup by pretty() --------------------------------
        kept, seen = [], set()
        for p in promoted:
            try:
                m = p.mask(self.df)
            except Exception:
                continue
            if _support(m) < int(min_support):
                continue
            key = getattr(p, "pretty", lambda: repr(p))()
            if key in seen:
                continue
            seen.add(key)
            kept.append(p)

        # -- optional simplification ------------------------------------------------
        if kept and simplify:
            try:
                from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses
                kept, _ = simplify_and_dedup_hypotheses(
                    self.df, kept,
                    min_support=min_support,
                    treat_binary_ints=treat_binary_ints,
                )
            except Exception:
                pass

        return kept

    # ── pretty helpers ── #
    @staticmethod
    def pretty_block(title: str, conjs: List[Conjecture], max_items: int = 100) -> None:
        print(f"\n=== {title} ===")
        for i, c in enumerate(conjs[:max_items], start=1):
            print(f"{i:3d}. {c.pretty(arrow='⇒')}")

    @staticmethod
    def pretty_qualitative_block_descriptive(title: str, items: list, max_items: int = 40):
        print(f"\n=== {title} ===")
        for i, r in enumerate(items[:max_items], 1):
            Hs = getattr(r.condition, "pretty", lambda: repr(r.condition))()
            rho_tag = "ρₛ" if r.relation.method == "spearman" else "ρₚ"
            dir_text = "tends to increase" if r.rho >= 0 else "tends to decrease"
            print(f"\n{i:3d}. Hypothesis: {Hs}")
            print(f"     Response: {r.y}")
            print(f"     Predictor: {r.x}")
            print(f"     Observation: As {r.x} increases, {r.y} {dir_text}.")
            print(f"     Correlation: {rho_tag} = {r.rho:+.3f} (n={r.n}, support={r.support})")

    @staticmethod
    def pretty_class_relations_ifthen(title: str, eqs, incs, df, max_items: int = 60):
        print(f"\n=== {title} ===")
        print("\n-- Equivalences --")
        for i, e in enumerate(eqs[:max_items], 1):
            left  = predicate_to_if_then(e.A)
            right = predicate_to_if_then(e.B)
            viol  = e.violation_count(df)
            print(f"{i:3d}. {left}  ⇔  {right}    [violations={viol}]")
        print("\n-- Inclusions --")
        for i, inc in enumerate(incs[:max_items], 1):
            left   = predicate_to_if_then(inc.A)
            right  = predicate_to_if_then(inc.B)
            suppA  = int(inc.A.mask(df).sum())
            viol   = inc.violation_count(df)
            print(f"{i:3d}. {left}  ⇒  {right}    [support(A)={suppA}, violations={viol}]")

    @staticmethod
    def pretty_class_relations_conj(title: str, eqs, incs, df, *, max_items: int = 60, ascii_ops: bool = False):
        impl = r"\Rightarrow" if ascii_ops else "⇒"
        equiv = r"\Leftrightarrow" if ascii_ops else "⇔"
        print(f"\n=== {title} ===")
        print("\n-- Equivalences --")
        for i, e in enumerate(eqs[:max_items], 1):
            left  = predicate_to_conjunction(e.A, ascii_ops=ascii_ops)
            right = predicate_to_conjunction(e.B, ascii_ops=ascii_ops)
            print(f"{i:3d}. {left} {equiv} {right}")
        print("\n-- Inclusions --")
        for i, inc in enumerate(incs[:max_items], 1):
            left   = predicate_to_conjunction(inc.A, ascii_ops=ascii_ops)
            right  = predicate_to_conjunction(inc.B, ascii_ops=ascii_ops)
            suppA  = int(inc.A.mask(df).sum())
            print(f"{i:3d}. {left} {impl} {right}  [support(A)={suppA}]")

# from txgraffiti.example_data import graph_data as df

# df = df.drop(columns=['cograph', 'eulerian', 'chordal', 'vertex_cover_number'])

# numerical_cols = df.select_dtypes(include=np.number).columns.to_list()

# numerical_cols = [x for x in numerical_cols if x != 'independence_number']

df = pd.read_csv('polytope_data_full.csv')
df.drop(columns=[
    'Unnamed: 0',
    'sum(pk_for_k>7)<(p3+p4+p5+p6+p7)',
    'fullerene', 'perfect_matching',
    'p5>p6', 'p6<p7', 'p4>p5', 'p3>0', 'p4<p5', 'p4>0', 'p5<p6', 'p3<p4'], inplace=True)

# df.drop(columns=['Unnamed: 0', 'p3>0', 'p4>0', 'p5>0', 'p6>0', 'fullerene'], inplace=True)

df.dropna(inplace=True)
# df['p6>0'] = df['p6']>0
# df['p5>0'] = df['p5']>0
# df['|p5 - p6|'] = np.abs(df['p5'] - df['p6'])
# df['|p3 - p4|'] = np.abs(df['p3'] - df['p4'])
# df['|p4 - p5|'] = np.abs(df['p4'] - df['p5'])
# df['|p3 - p6|'] = np.abs(df['p3'] - df['p6'])
# df['|p4 - p6|'] = np.abs(df['p4'] - df['p6'])
# df['|p6 - p7|'] = np.abs(df['p6'] - df['p7'])
df['log(n)'] = df['n'].apply(np.log)
df['log(independence_number)'] = df['independence_number'].apply(np.log)
df['vertex_cover_number'] = df['n'] - df['independence_number']
df['log(vertex_cover_number)'] = df['vertex_cover_number'].apply(np.log)
# df['p6=0'] = df['p6']==0

# numerical_cols = [x for x in numerical_cols if x != 'independence_number']
ai = TxGraffitiMini(df)

# # # Single-feature bounds
# low1, up1 = ai.run_pipeline("temperature(p6)")
# TxGraffitiMini.pretty_block("Lower", low1[:85])
# TxGraffitiMini.pretty_block("Upper", up1[:85])


# ai = TxGraffitiMini(df)

# Single-feature bounds
low1, up1 = ai.run_single_feature_pipeline('temperature(p6)')
TxGraffitiMini.pretty_block("temperature(p6) Lower", low1[:20])
TxGraffitiMini.pretty_block("temperature(p6) Upper", up1[:20])

low2, up2 = ai.run_single_feature_pipeline('p6')
TxGraffitiMini.pretty_block("p6 Lower", low1[:20])
TxGraffitiMini.pretty_block("p6 Upper", up1[:20])

low3, up3 = ai.run_single_feature_pipeline('|p5 - p6|')
TxGraffitiMini.pretty_block("|p5 - p6| Lower", low1[:20])
TxGraffitiMini.pretty_block("|p5 - p6| Upper", up1[:20])

# Mixed (sqrt / square) bounds
# low_mix, up_mix = ai.run_mixed_pipeline("independence_number", weight=0.5)

# Targeted product bounds
low_prod, up_prod = ai.run_targeted_product_pipeline('temperature(p6)')

TxGraffitiMini.pretty_block("Lower", low_prod[:40])
TxGraffitiMini.pretty_block("Upper", up_prod[:40])

low_prod, up_prod = ai.run_targeted_product_pipeline('p6')

TxGraffitiMini.pretty_block("Lower", low_prod[:40])
TxGraffitiMini.pretty_block("Upper", up_prod[:40])

# low_prod, up_prod = ai.run_targeted_product_pipeline('|p3 - p6|')

# TxGraffitiMini.pretty_block("Lower", low_prod[:40])
# TxGraffitiMini.pretty_block("Upper", up_prod[:40])

low_prod, up_prod = ai.run_targeted_product_pipeline('|p4 - p6|')

TxGraffitiMini.pretty_block("Lower", low_prod[:40])
TxGraffitiMini.pretty_block("Upper", up_prod[:40])

low_prod, up_prod = ai.run_targeted_product_pipeline('|p5 - p6|')

TxGraffitiMini.pretty_block("Lower", low_prod[:40])
TxGraffitiMini.pretty_block("Upper", up_prod[:40])

low_prod, up_prod = ai.run_targeted_product_pipeline('|p6 - p7|')

TxGraffitiMini.pretty_block("Lower", low_prod[:40])
TxGraffitiMini.pretty_block("Upper", up_prod[:40])

# # === Linear generalizer (constant-bank) ===
# gen_lin_up  = ai.run_linear_generalizer(up1[:50])     # generalize top-N uppers
# gen_lin_low = ai.run_linear_generalizer(low1[:50])    # (optional) generalize lowers

# TxGraffitiMini.pretty_block("Linear-generalized (from uppers)", gen_lin_up[:20])
# TxGraffitiMini.pretty_block("Linear-generalized (from lowers)", gen_lin_low[:20])

# Reciprocal generalizer (unchanged)
# recip = ai.run_reciprocal_generalizer(up1[:20])
# TxGraffitiMini.pretty_block("Reciprocal-generalized (from uppers)", recip[:20])

# # (Optional) Combine and re-rank unique proposals
# combined = ai.rank_and_filter(recip, min_touch=ai.config.min_touch_keep)
# TxGraffitiMini.pretty_block("Generalized — combined & ranked", combined[:30])

# # Qualitative
quals = ai.generate_qualitative_relations(
    y_targets=["p6", "temperature(p6)", "|p5 - p6|"],
    method="spearman", min_abs_rho=0.35, min_n=12, top_k_per_hyp=10,
)
TxGraffitiMini.pretty_qualitative_block_descriptive("Monotone tendencies", quals, max_items=60)

# # Class relations (R₁)
eqs, incs = ai.discover_class_relations(include_bool_columns=True, min_support_A=5)
TxGraffitiMini.pretty_class_relations_conj("Class relations (R₁)", eqs, incs, df)

# |p5 - p6|

# low_intricate, up_intricate = ai.generate_intricate_mixed_bounds(
#     'temperature(p6)',
#     weight=0.5,                 # matches your 1/2 blend
#     primary=None,               # default: all numeric columns as x
#     secondary=None              # default: all numeric columns as y
# )

# # (optional) rank & keep
# low_intricate = ai.rank_and_filter(low_intricate)
# up_intricate  = ai.rank_and_filter(up_intricate)

# # TxGraffitiMini.pretty_block("Intricate lowers", low_intricate)
# TxGraffitiMini.pretty_block("Intricate uppers", up_intricate[:100])


# low_intricate, up_intricate = ai.generate_intricate_mixed_bounds(
#     '|p5 - p6|',
#     weight=0.5,                 # matches your 1/2 blend
#     primary=None,               # default: all numeric columns as x
#     secondary=None              # default: all numeric columns as y
# )

# # (optional) rank & keep
# low_intricate = ai.rank_and_filter(low_intricate)
# up_intricate  = ai.rank_and_filter(up_intricate)

# # TxGraffitiMini.pretty_block("Intricate lowers", low_intricate)
# TxGraffitiMini.pretty_block("Intricate uppers", up_intricate[:100])


# low_intricate, up_intricate = ai.generate_intricate_mixed_bounds(
#     'p6',
#     weight=0.5,                 # matches your 1/2 blend
#     primary=None,               # default: all numeric columns as x
#     secondary=None              # default: all numeric columns as y
# )

# # (optional) rank & keep
# low_intricate = ai.rank_and_filter(low_intricate)
# up_intricate  = ai.rank_and_filter(up_intricate)

# # TxGraffitiMini.pretty_block("Intricate lowers", low_intricate)
# TxGraffitiMini.pretty_block("Intricate uppers", up_intricate[:100])

numerical_cols = df.select_dtypes(include=np.number).columns.to_list()

# numerical_cols = [x for x in numerical_cols if x != 'p6']

# lp2_low, lp2_up = ai.generate_lp_two_feature_bounds_affine(
#     '|p3 - p6|',
#     features=numerical_cols,  # or None for all numeric cols
#     transforms=("identity", "sqrt", "square"),                     # matches your legacy mixes
#     rationalize=True,
#     max_denom=45,
# )

# TxGraffitiMini.pretty_block("LP 2-feature lowers", ai.rank_and_filter(lp2_low)[:100])
# TxGraffitiMini.pretty_block("LP 2-feature uppers", ai.rank_and_filter(lp2_up)[:100])


lp2_low, lp2_up = ai.generate_lp_two_feature_bounds_affine(
    '|p4 - p6|',
    features=numerical_cols,  # or None for all numeric cols
    transforms=("identity", "sqrt", "square"),                     # matches your legacy mixes
    rationalize=True,
    max_denom=45,
)

TxGraffitiMini.pretty_block("LP 2-feature lowers", ai.rank_and_filter(lp2_low)[:100])
TxGraffitiMini.pretty_block("LP 2-feature uppers", ai.rank_and_filter(lp2_up)[:100])

lp2_low, lp2_up = ai.generate_lp_two_feature_bounds_affine(
    '|p5 - p6|',
    features=numerical_cols,  # or None for all numeric cols
    transforms=("identity", "sqrt", "square"),                     # matches your legacy mixes
    rationalize=True,
    max_denom=45,
)

TxGraffitiMini.pretty_block("LP 2-feature lowers", ai.rank_and_filter(lp2_low)[:100])
TxGraffitiMini.pretty_block("LP 2-feature uppers", ai.rank_and_filter(lp2_up)[:100])


lp2_low, lp2_up = ai.generate_lp_two_feature_bounds_affine(
    '|p6 - p7|',
    features=numerical_cols,  # or None for all numeric cols
    transforms=("identity", "sqrt", "square"),                     # matches your legacy mixes
    rationalize=True,
    max_denom=45,
)

TxGraffitiMini.pretty_block("LP 2-feature lowers", ai.rank_and_filter(lp2_low)[:100])
TxGraffitiMini.pretty_block("LP 2-feature uppers", ai.rank_and_filter(lp2_up)[:100])

# # 1) Run any generators you want first
low1, up1 = ai.run_single_feature_pipeline('|p5 - p6|')
low1 = ai.rank_and_filter(low1)
up1  = ai.rank_and_filter(up1)
# low_mix, up_mix = ai.run_mixed_pipeline("independence_number", weight=0.5)
# lp_lows, lp_ups = ai.generate_lp_two_feature_bounds_affine("independence_number")

# 2) Promote their conclusions to new hypotheses
new_hyps = ai.promote_conclusions_to_hypotheses(
    conjs = low1[:15] + up1[:15] + low2[:15] + up2[:15] + low3[:15] + up3[:15],
    min_support = 5,          # tweak as you like
    make_strict = True,
    make_equality = False,
    simplify = True,
)

# 3) Append to your hypothesis pool and rerun pipelines
ai.hyps_kept.extend(new_hyps)

eqs, incs = ai.discover_class_relations(include_bool_columns=True, min_support_A=5)
TxGraffitiMini.pretty_class_relations_conj("Class relations (R₁)", eqs, incs, df)


# (Optionally dedupe ai.hyps_kept again with your simplify function)
from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses
ai.hyps_kept, _ = simplify_and_dedup_hypotheses(ai.df, ai.hyps_kept, min_support=5)

# 4) Rerun whatever pipelines you want, now using augmented hypotheses
# low2, up2 = ai.run_single_feature_pipeline('|p3 - p6|')

# low2 = ai.rank_and_filter(low2)
# up2  = ai.rank_and_filter(up2)

# TxGraffitiMini.pretty_block("NEW UPPER", up2[:100])
# TxGraffitiMini.pretty_block("NEW LOWER", low2[:100])

low2, up2 = ai.run_single_feature_pipeline('|p4 - p6|')

low2 = ai.rank_and_filter(low2)
up2  = ai.rank_and_filter(up2)

TxGraffitiMini.pretty_block("NEW UPPER", up2[:100])
TxGraffitiMini.pretty_block("NEW LOWER", low2[:100])

low2, up2 = ai.run_single_feature_pipeline('|p5 - p6|')

low2 = ai.rank_and_filter(low2)
up2  = ai.rank_and_filter(up2)

TxGraffitiMini.pretty_block("NEW UPPER", up2[:100])
TxGraffitiMini.pretty_block("NEW LOWER", low2[:100])

low_intricate, up_intricate = ai.generate_intricate_mixed_bounds(
    'temperature(p6)',
    weight=0.5,                 # matches your 1/2 blend
    primary=None,               # default: all numeric columns as x
    secondary=None              # default: all numeric columns as y
)

# (optional) rank & keep
low_intricate = ai.rank_and_filter(low_intricate)
up_intricate  = ai.rank_and_filter(up_intricate)

# TxGraffitiMini.pretty_block("Intricate lowers", low_intricate)
TxGraffitiMini.pretty_block("Intricate uppers", up_intricate[:100])


low_intricate, up_intricate = ai.generate_intricate_mixed_bounds(
    '|p5 - p6|',
    weight=0.5,                 # matches your 1/2 blend
    primary=None,               # default: all numeric columns as x
    secondary=None              # default: all numeric columns as y
)

# (optional) rank & keep
low_intricate = ai.rank_and_filter(low_intricate)
up_intricate  = ai.rank_and_filter(up_intricate)

# TxGraffitiMini.pretty_block("Intricate lowers", low_intricate)
TxGraffitiMini.pretty_block("Intricate uppers", up_intricate[:100])


low_intricate, up_intricate = ai.generate_intricate_mixed_bounds(
    'p6',
    weight=0.5,                 # matches your 1/2 blend
    primary=None,               # default: all numeric columns as x
    secondary=None              # default: all numeric columns as y
)

# (optional) rank & keep
low_intricate = ai.rank_and_filter(low_intricate)
up_intricate  = ai.rank_and_filter(up_intricate)

# TxGraffitiMini.pretty_block("Intricate lowers", low_intricate)
TxGraffitiMini.pretty_block("Intricate uppers", up_intricate[:100])

numerical_cols = df.select_dtypes(include=np.number).columns.to_list()

# numerical_cols = [x for x in numerical_cols if x != 'p6']

# lp2_low, lp2_up = ai.generate_lp_two_feature_bounds_affine(
#     '|p3 - p6|',
#     features=numerical_cols,  # or None for all numeric cols
#     transforms=("identity", "sqrt", "square"),                     # matches your legacy mixes
#     rationalize=True,
#     max_denom=45,
# )

# TxGraffitiMini.pretty_block("LP 2-feature lowers", ai.rank_and_filter(lp2_low)[:100])
# TxGraffitiMini.pretty_block("LP 2-feature uppers", ai.rank_and_filter(lp2_up)[:100])


lp2_low, lp2_up = ai.generate_lp_two_feature_bounds_affine(
    '|p4 - p6|',
    features=numerical_cols,  # or None for all numeric cols
    transforms=("identity", "sqrt", "square"),                     # matches your legacy mixes
    rationalize=True,
    max_denom=45,
)

TxGraffitiMini.pretty_block("LP 2-feature lowers", ai.rank_and_filter(lp2_low)[:100])
TxGraffitiMini.pretty_block("LP 2-feature uppers", ai.rank_and_filter(lp2_up)[:100])

lp2_low, lp2_up = ai.generate_lp_two_feature_bounds_affine(
    '|p5 - p6|',
    features=numerical_cols,  # or None for all numeric cols
    transforms=("identity", "sqrt", "square"),                     # matches your legacy mixes
    rationalize=True,
    max_denom=45,
)

TxGraffitiMini.pretty_block("LP 2-feature lowers", ai.rank_and_filter(lp2_low)[:100])
TxGraffitiMini.pretty_block("LP 2-feature uppers", ai.rank_and_filter(lp2_up)[:100])


lp2_low, lp2_up = ai.generate_lp_two_feature_bounds_affine(
    '|p6 - p7|',
    features=numerical_cols,  # or None for all numeric cols
    transforms=("identity", "sqrt", "square"),                     # matches your legacy mixes
    rationalize=True,
    max_denom=45,
)

TxGraffitiMini.pretty_block("LP 2-feature lowers", ai.rank_and_filter(lp2_low)[:100])
TxGraffitiMini.pretty_block("LP 2-feature uppers", ai.rank_and_filter(lp2_up)[:100])

# # 1) Run any generators you want first
low1, up1 = ai.run_single_feature_pipeline('|p5 - p6|')
low1 = ai.rank_and_filter(low1)
up1  = ai.rank_and_filter(up1)
