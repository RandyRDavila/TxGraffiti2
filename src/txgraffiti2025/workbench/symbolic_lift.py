# symbolic_lift.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import (
    Expr, Const, to_expr, BinOp, UnaryOp, LogOp,
)
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le


# put near other helpers in symbolic_lift.py
def _conjecture_struct_key(c) -> tuple:
    """
    Build a stable, hashable key for a Conjecture:
    (condition_key, relation_type, left_pretty, right_pretty)
    """
    H = getattr(c, "condition", None)
    cond_key = _pred_key(H) if H is not None else "NONE"
    rel = c.relation
    rel_type = rel.__class__.__name__
    left_s = rel.left.pretty() if hasattr(rel.left, "pretty") else repr(rel.left)
    right_s = rel.right.pretty() if hasattr(rel.right, "pretty") else repr(rel.right)
    return (cond_key, rel_type, left_s, right_s)

def _pred_key(H: Predicate) -> str:
    """Stable string key for a Predicate used in dicts."""
    # Prefer pretty(), then explicit name, then repr()
    if hasattr(H, "pretty"):
        try:
            return f"pretty::{H.pretty()}"
        except Exception:
            pass
    name = getattr(H, "name", None)
    if name:
        return f"name::{name}"
    return f"repr::{repr(H)}"

# --- Add (or keep) this helper near the top of symbolic_lift.py ---
from txgraffiti2025.forms.utils import Expr, BinOp, UnaryOp, LogOp, Const, ColumnTerm

def columns_in_expr(e: Expr) -> set[str]:
    """Collect column names referenced by an Expr (recursively)."""
    out: set[str] = set()

    def walk(node: Expr):
        if isinstance(node, ColumnTerm):
            out.add(node.col)
        elif isinstance(node, BinOp):
            walk(node.left); walk(node.right)
        elif isinstance(node, UnaryOp):
            walk(node.arg)
        elif isinstance(node, LogOp):
            walk(node.arg)
        # Const / LinearForm etc.: no nested column names

    walk(e)
    return out

# ---------- utils ----------
def _mask(df: pd.DataFrame, H: Optional[Predicate]) -> np.ndarray:
    if H is None:
        return np.ones(len(df), dtype=bool)
    return H.mask(df).astype(bool).to_numpy()

def _support(m: np.ndarray) -> int:
    return int(np.asarray(m, dtype=bool).sum())

def _finite(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)

def _const_like_on_slice(vals: np.ndarray, m: np.ndarray, *, rel_eps: float, abs_eps: float) -> tuple[bool, float]:
    """Is vals ~ constant on mask m? Returns (is_const, numeric_constant)."""
    mm = (m & _finite(vals))
    if not np.any(mm):
        return False, np.nan
    v = float(np.median(vals[mm]))
    dev = float(np.max(np.abs(vals[mm] - v)))
    tol = abs_eps + rel_eps * (1.0 + abs(v))
    return (dev <= tol), v

def _is_const_equal(node: Expr, value: float, atol: float = 1e-12) -> bool:
    return isinstance(node, Const) and abs(float(node.value) - float(value)) <= atol

def _replace_scalar_coeff(expr: Expr, old_val: float, new_coeff: Expr) -> Expr:
    """Replace multiplicative factor Const(old_val) anywhere in expr by new_coeff."""
    if isinstance(expr, BinOp):
        if expr.fn is np.multiply:
            L, R = expr.left, expr.right
            if _is_const_equal(L, old_val):
                return BinOp(np.multiply, new_coeff, _replace_scalar_coeff(R, old_val, new_coeff))
            if _is_const_equal(R, old_val):
                return BinOp(np.multiply, _replace_scalar_coeff(L, old_val, new_coeff), new_coeff)
        return BinOp(expr.fn,
                     _replace_scalar_coeff(expr.left,  old_val, new_coeff),
                     _replace_scalar_coeff(expr.right, old_val, new_coeff))
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.fn, _replace_scalar_coeff(expr.arg, old_val, new_coeff))
    if isinstance(expr, LogOp):
        return LogOp(_replace_scalar_coeff(expr.arg, old_val, new_coeff), base=expr.base, epsilon=expr.epsilon)
    if _is_const_equal(expr, old_val):
        return new_coeff
    return expr

def _supersets_of(df: pd.DataFrame, H0: Predicate, pool: Iterable[Predicate], *, min_support: int) -> List[Predicate]:
    m0 = _mask(df, H0)
    outs: List[Tuple[Predicate, int]] = []
    for H in pool:
        m = _mask(df, H)
        # H is superset of H0 if every H0 row is allowed by H
        if ((~m0) | m).all():
            s = _support(m)
            if s >= min_support:
                outs.append((H, s))
    outs.sort(key=lambda t: -t[1])
    return [H for H, _ in outs]


# ---------- symbolic-constant bank ----------

@dataclass(frozen=True)
class SymConst:
    """One symbolic constant found on a hypothesis."""
    value: float          # numeric value it equals on H
    expr: Expr           # symbolic form (e.g., 1/(Δ+1), Δ/δ, u/v)
    columns: Tuple[str, ...]  # columns used
    support: int         # support of H where this was established

# --- update the type alias (old: Dict[Predicate, List[SymConst]]) ---
SymConstBank = Dict[str, List[SymConst]]

# symbolic_lift.py (patch)
from typing import Sequence


def build_symbolic_constant_bank(
    df: pd.DataFrame,
    *,
    hypotheses: Sequence[Predicate],
    numeric_cols: Sequence[str],
    reciprocal_shifts: Sequence[int] = (1,),
    ratio_shifts: Sequence[int] = (0,),
    min_support: int = 5,
    rel_eps: float = 1e-9,
    abs_eps: float = 1e-9,
    avoid_div_zero_eps: float = 1e-12,
    exclude_cols: Sequence[str] = (),          # <-- NEW
) -> SymConstBank:
    """
    For each hypothesis H, scan candidate expressions that behave like constants on H,
    and store them keyed by a stable string key for H.

    Parameters
    ----------
    exclude_cols : Sequence[str]
        Columns to never use when constructing symbolic constants (e.g., boolean class columns).
    """
    bank: SymConstBank = {}
    excl = set(exclude_cols)

    for H in hypotheses:
        key = _pred_key(H)
        m = _mask(df, H)
        s = _support(m)
        if s < min_support:
            bank[key] = []
            continue

        entries: List[SymConst] = []

        # --- reciprocals: 1 / (col + shift)
        for col in numeric_cols:
            if col in excl:
                continue
            base = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, copy=False)
            if not np.any(m & _finite(base)):
                continue
            for sh in reciprocal_shifts:
                denom = base + float(sh)
                denom = np.where(np.abs(denom) < avoid_div_zero_eps, np.nan, denom)
                cand = 1.0 / denom
                ok, val = _const_like_on_slice(cand, m, rel_eps=rel_eps, abs_eps=abs_eps)
                if not ok or not np.isfinite(val):
                    continue
                expr = Const(1) / (to_expr(col) + Const(sh))
                entries.append(SymConst(value=float(val), expr=expr, columns=(col,), support=s))

        # --- ratios: u / (v + shift)
        cols = [c for c in numeric_cols if c not in excl]
        for u in cols:
            uvals = pd.to_numeric(df[u], errors="coerce").to_numpy(dtype=float, copy=False)
            if not np.any(m & _finite(uvals)):
                continue
            for v in cols:
                vvals = pd.to_numeric(df[v], errors="coerce").to_numpy(dtype=float, copy=False)
                if not np.any(m & _finite(vvals)):
                    continue
                for sh in ratio_shifts:
                    denom = vvals + float(sh)
                    denom = np.where(np.abs(denom) < avoid_div_zero_eps, np.nan, denom)
                    cand = uvals / denom
                    ok, val = _const_like_on_slice(cand, m, rel_eps=rel_eps, abs_eps=abs_eps)
                    if not ok or not np.isfinite(val):
                        continue
                    expr = to_expr(u) / (to_expr(v) + Const(sh))
                    entries.append(SymConst(value=float(val), expr=expr, columns=(u, v), support=s))

        bank[key] = entries

    return bank


# --- REPLACE your existing lifter with this guarded version ---

from typing import Iterable, List, Sequence, Tuple, Dict
import numpy as np
import pandas as pd
from txgraffiti2025.forms.generic_conjecture import Conjecture

# SymConstBank is Dict[str, List[SymConst]] (string-keyed by predicate via _pred_key)
# SymConstBank = Dict[str, List[SymConst]]

def lift_conjectures_with_bank(
    df: pd.DataFrame,
    base_conjectures: Iterable[Conjecture],
    *,
    bank: SymConstBank,
    broader_hypotheses_pool: Sequence,
    min_support_broader: int = 5,
    require_strict_growth: bool = True,
    value_match_tol: float = 1e-9,
    target_columns_exclude: Sequence[str] = (),
    forbid_rhs_dependency: bool = False,
) -> List[Tuple[Conjecture, Conjecture, int, int]]:
    """
    Lift numeric constants in each base conjecture using symbolic constants from a per-hypothesis bank.

    Deduplication: identical generalized conjectures are merged (hash-based).
    If duplicates occur, the version with the largest generalized-support is kept.
    """
    out_records: List[Tuple[Conjecture, Conjecture, int, int]] = []
    target_excl = set(target_columns_exclude)

    def _constants_to_try(c: float, H0) -> List[SymConst]:
        key = _pred_key(H0)
        items = bank.get(key, [])
        matched: List[SymConst] = []
        for sc in items:
            val = sc.value
            if not np.isfinite(val):
                continue
            if abs(val - c) <= value_match_tol * (1.0 + abs(c)):
                matched.append(sc)
        return matched

    for base in base_conjectures:
        H0 = getattr(base, "condition", None)
        if H0 is None:
            continue
        m0 = _mask(df, H0)
        base_support = _support(m0)
        if base_support < min_support_broader:
            continue

        rel = base.relation
        lhs_cols = columns_in_expr(rel.left)
        rhs_cols = columns_in_expr(rel.right)

        # detect numeric constants in repr
        candidate_vals: List[float] = []
        import re
        r = repr(rel)
        for a_str, b_str in re.findall(r"\b(\d+)\s*/\s*(\d+)\b", r):
            a, b = float(a_str), float(b_str)
            if b != 0.0:
                candidate_vals.append(a / b)
        for num in re.findall(r"\b-?\d+(?:\.\d+)?\b", r):
            candidate_vals.append(float(num))
        uniq_vals: List[float] = []
        for v in candidate_vals:
            if not any(abs(v - u) <= 1e-12 for u in uniq_vals):
                uniq_vals.append(v)

        for c in uniq_vals:
            matches = _constants_to_try(c, H0)
            if not matches:
                continue
            for sc in matches:
                sc_cols = set(sc.columns)
                if sc_cols & lhs_cols:
                    continue
                if sc_cols & target_excl:
                    continue
                if forbid_rhs_dependency and (sc_cols & rhs_cols):
                    continue

                new_left  = _replace_scalar_coeff(rel.left,  c, sc.expr)
                new_right = _replace_scalar_coeff(rel.right, c, sc.expr)
                RelCls = type(rel)

                base_check = Conjecture(RelCls(new_left, new_right), H0)
                applicable, holds, _ = base_check.check(df, auto_base=False)
                if not bool(holds[applicable].all()):
                    continue

                supers = _supersets_of(df, H0, broader_hypotheses_pool, min_support=min_support_broader)
                for Hb in supers:
                    gen = Conjecture(RelCls(new_left, new_right), Hb)
                    appl_b, holds_b, _ = gen.check(df, auto_base=False)
                    if not bool(holds_b[appl_b].all()):
                        continue
                    gen_support = int(appl_b.sum())
                    if require_strict_growth and gen_support <= base_support:
                        continue
                    if (not require_strict_growth) and gen_support < base_support:
                        continue

                    out_records.append((base, gen, base_support, gen_support))
                    break  # stop after first valid broader generalization

    # --- Deduplicate identical generalized conjectures (structure-based) ---
    dedup: Dict[tuple, Tuple[Conjecture, Conjecture, int, int]] = {}
    for base, gen, bsupp, gsupp in out_records:
        k = _conjecture_struct_key(gen)
        prev = dedup.get(k)
        # keep the version with the largest generalized support
        if prev is None or gsupp > prev[3]:
            dedup[k] = (base, gen, bsupp, gsupp)

    return list(dedup.values())
