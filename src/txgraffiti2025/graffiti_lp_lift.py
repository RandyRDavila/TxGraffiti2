# src/txgraffiti2025/graffiti_lp_lift.py
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE
from txgraffiti2025.forms.utils import Expr, Const

# --- optional DSL ops: floor/ceil/round ----
def _get_unary(name: str) -> Optional[Callable[[Expr], Expr]]:
    # Try to find a DSL unary function by name in utils first, then builtins on Expr.
    import txgraffiti2025.forms.utils as U
    fn = getattr(U, name, None)
    if callable(fn):
        return fn
    # fallback: some Expr implementations may expose .floor()/.ceil() etc.
    def _method_fallback(e: Expr):
        if hasattr(e, name) and callable(getattr(e, name)):
            return getattr(e, name)()
        raise AttributeError(f"{name} not available for Expr")
    return _method_fallback

_FLOOR = _get_unary("floor")
_CEIL  = _get_unary("ceil")
_ROUND = _get_unary("round")  # optional; if absent we won't use it


# ─────────────────────────── helpers ─────────────────────────── #

def _arr(e: Expr, df: pd.DataFrame) -> np.ndarray:
    return e.eval(df).to_numpy(dtype=float, copy=False)

def _touch(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> Tuple[int, float, int]:
    m = mask & np.isfinite(lhs) & np.isfinite(rhs)
    n = int(m.sum())
    if n == 0:
        return 0, 0.0, 0
    d = np.abs(lhs[m] - rhs[m])
    t = atol + rtol * np.abs(rhs[m])
    tc = int((d <= t).sum())
    return tc, (tc / n), n

def _mask_for(cond, gcr, nrows: int) -> np.ndarray:
    if cond is None or cond is TRUE:
        return np.ones(nrows, dtype=bool)
    return gcr._mask_cached(cond)

def _get_sides(rel) -> Tuple[Expr, Expr]:
    # Works for Ge/Le/Eq (they all have .left/.right)
    left = getattr(rel, "left", None)
    right = getattr(rel, "right", None)
    if left is None or right is None:
        raise TypeError(f"Unsupported relation for lifting: {type(rel)}")
    return left, right

def _valid(rel, lhs_vals: np.ndarray, rhs_vals: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> bool:
    m = mask & np.isfinite(lhs_vals) & np.isfinite(rhs_vals)
    if not m.any():
        return False
    if isinstance(rel, Ge):
        return bool(np.all(lhs_vals[m] + (atol + rtol * np.abs(rhs_vals[m])) >= rhs_vals[m]))
    if isinstance(rel, Le):
        return bool(np.all(lhs_vals[m] - (atol + rtol * np.abs(rhs_vals[m])) <= rhs_vals[m]))
    if isinstance(rel, Eq):
        d = np.abs(lhs_vals[m] - rhs_vals[m])
        t = atol + rtol * np.abs(rhs_vals[m])
        return bool(np.all(d <= t))
    return False

def _dominance_gain(rel, rhs_old: np.ndarray, rhs_new: np.ndarray, mask: np.ndarray, lhs: np.ndarray) -> float:
    # Positive gain means strictly tighter on average.
    m = mask & np.isfinite(rhs_old) & np.isfinite(rhs_new) & np.isfinite(lhs)
    if not m.any():
        return -np.inf
    if isinstance(rel, Ge):
        # larger RHS ⇒ tighter lower bound
        gain = np.maximum(0.0, rhs_new[m] - rhs_old[m])
        return float(gain.mean()) if gain.size else -np.inf
    if isinstance(rel, Le):
        # smaller RHS ⇒ tighter upper bound
        gain = np.maximum(0.0, rhs_old[m] - rhs_new[m])
        return float(gain.mean()) if gain.size else -np.inf
    # For Eq, stick to touches improvement only.
    return -np.inf

def _lift_rhs_monotone(name: str, op: Callable[[Expr], Expr], *, increasing: bool):
    """
    Return a function that, given a Conjecture (Ge/Le/Eq), applies a monotone transform to the RHS.
    For monotone increasing ops:
      - Ge:   y ≥ g(rhs)  (tightens if g(rhs) ≥ rhs)
      - Le:   y ≤ g(rhs)  (tightens if g(rhs) ≤ rhs)
      - Eq:   y = g(rhs)  only accepted if touches increase (handled by caller)
    """
    def _lift(cj: Conjecture, df: pd.DataFrame, touch_atol: float, touch_rtol: float) -> Optional[Conjecture]:
        rel = cj.relation
        if not isinstance(rel, (Ge, Le, Eq)):
            return None
        left, right = _get_sides(rel)
        try:
            new_rhs = op(right)
        except Exception:
            return None

        if isinstance(rel, Ge):
            new_rel = Ge(left, new_rhs)
        elif isinstance(rel, Le):
            new_rel = Le(left, new_rhs)
        else:
            new_rel = Eq(left, new_rhs, tol=getattr(rel, "tol", 1e-9))

        cand = Conjecture(relation=new_rel, condition=cj.condition)
        return cand
    _lift.__name__ = f"lift_{name}"
    return _lift


# Assemble the available lifts based on the DSL functions we found.
LIFTS: List[Callable[[Conjecture, pd.DataFrame, object, float, float], Optional[Conjecture]]] = []
if _CEIL is not None:
    LIFTS.append(_lift_rhs_monotone("ceil", _CEIL, increasing=True))
if _FLOOR is not None:
    LIFTS.append(_lift_rhs_monotone("floor", _FLOOR, increasing=True))
if _ROUND is not None:
    LIFTS.append(_lift_rhs_monotone("round", _ROUND, increasing=True))


# ─────────────────────────── public API ─────────────────────────── #

def lift_conjectures(
    *,
    df: pd.DataFrame,
    gcr,
    conjectures: Iterable[Conjecture],
    touch_atol: float = 0.0,
    touch_rtol: float = 0.0,
    eps_tight: float = 1e-12,
) -> List[Conjecture]:
    """
    Try safe RHS lifts (ceil/floor/round) for Ge/Le/Eq conjectures.

    Accept a lift if:
      (A) It remains valid on the same H-support (no new violations), and
      (B) EITHER it increases touches, OR (touches tie and) it strictly tightens
          on average by at least eps_tight (dominance gain).

    Notes
    -----
    • For Ge: larger RHS is tighter; for Le: smaller RHS is tighter.
    • For Eq: we only keep a lift if touches increase (dominance not meaningful).
    """
    out: List[Conjecture] = []

    for cj in conjectures:
        rel0 = cj.relation
        if not isinstance(rel0, (Ge, Le, Eq)):
            out.append(cj)
            continue

        left0, right0 = _get_sides(rel0)
        mask = _mask_for(cj.condition, gcr, len(df))

        lhs0 = _arr(left0, df)
        rhs0 = _arr(right0, df)
        tc0, tr0, n0 = _touch(lhs0, rhs0, mask, atol=touch_atol, rtol=touch_rtol)

        best = cj
        best_key = (tc0, tr0, 0.0)  # touches, touch_rate, dominance_gain

        for lift in LIFTS:
            cand = None
            try:
                cand = lift(cj, df, gcr, touch_atol, touch_rtol)
            except Exception:
                cand = None
            if cand is None:
                continue

            rel1 = cand.relation
            l1, r1 = _get_sides(rel1)

            lhs1 = _arr(l1, df)
            rhs1 = _arr(r1, df)

            # Must remain valid
            if not _valid(rel1, lhs1, rhs1, mask, atol=touch_atol, rtol=touch_rtol):
                continue

            # Score improvement
            tc1, tr1, _ = _touch(lhs1, rhs1, mask, atol=touch_atol, rtol=touch_rtol)

            if isinstance(rel0, Eq):
                # For Eq keep only if touches increased
                if tc1 > best_key[0] or (tc1 == best_key[0] and tr1 > best_key[1]):
                    setattr(cand, "touch_count", tc1)
                    setattr(cand, "touch_rate", tr1)
                    setattr(cand, "support_n", int(mask.sum()))
                    best = cand
                    best_key = (tc1, tr1, 0.0)
                continue

            # For Ge/Le, also allow strict dominance at tie of touches/rate
            gain = _dominance_gain(rel0, rhs0, rhs1, mask, lhs0)
            if (tc1 > best_key[0]) or \
               (tc1 == best_key[0] and tr1 > best_key[1]) or \
               (tc1 == best_key[0] and abs(tr1 - best_key[1]) <= 1e-15 and gain >= eps_tight):
                setattr(cand, "touch_count", tc1)
                setattr(cand, "touch_rate", tr1)
                setattr(cand, "support_n", int(mask.sum()))
                best = cand
                best_key = (tc1, tr1, gain)

        out.append(best)

    return out
