# txgraffiti2025/processing/post/generalize_linear.py
from __future__ import annotations
from typing import Iterable, Sequence, Optional, List
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq
from txgraffiti2025.forms.utils import Const, to_expr, floor, ceil

# --- core helper that works on a single conjecture ---
def propose_generalizations_from_constants(
    df: pd.DataFrame,
    conj: Conjecture,
    *,
    const_bank_for_h: dict[str, float],
    max_denom: int = 64,
    use_intercept: bool = True,
    allow_fractional: bool = True,
    try_floor_ceil: bool = True,
) -> list[Conjecture]:
    """
    Replace a numeric coefficient on the RHS with symbolic expressions
    formed from constants in `const_bank_for_h` (the constant bank for the conjecture's hypothesis),
    optionally add an intercept, and strengthen with floor/ceil when exactly true.
    """
    rel = conj.relation
    right = rel.right

    # Only handle simple c * X forms for now.
    if not hasattr(right, "op") or right.op != "*" or not hasattr(right, "left") or not hasattr(right, "right"):
        return []

    # Pull C * var
    C, var = right.left, right.right
    try:
        c_float = float(getattr(C, "value", C))
    except Exception:
        return []
    out: list[Conjecture] = []

    # 1) Try replacing C with a/(K+s), K/(a+s), a/sqrt(K), etc. using the constant bank for this H
    #    (Keep it minimal; you can expand later.)
    def _emit(coeff_expr):
        new_rhs = coeff_expr * var
        new_rel = type(rel)(rel.left, new_rhs)  # Ge/Le/Eq preserved
        cand = Conjecture(new_rel, conj.condition)
        try:
            if cand.is_true(df):
                out.append(cand)
        except Exception:
            pass

    for K, kval in const_bank_for_h.items():
        # a/(K+s)
        for a in (1, 2, 3):
            for s in (0, 1, -1):
                denom = kval + s
                if denom == 0:
                    continue
                # numeric check first
                if abs((a / denom) - c_float) < 1e-9:
                    coeff_expr = Const(a) / (to_expr(K) + Const(s))
                    _emit(coeff_expr)

        # a/sqrt(K)
        if kval > 0:
            import numpy as np
            root = np.sqrt(kval)
            for a in (1, 2, 3):
                if abs((a / root) - c_float) < 1e-9:
                    coeff_expr = Const(a) / (to_expr(K) ** Const(1/2))
                    _emit(coeff_expr)

        # K/(a+s)
        for a in (1, 2, 3):
            for s in (0, 1, -1):
                denom = a + s
                if denom == 0:
                    continue
                if abs((kval / denom) - c_float) < 1e-9:
                    coeff_expr = to_expr(K) / Const(denom)
                    _emit(coeff_expr)

    # Optionally try floor/ceil strengthening if perfectly tight everywhere
    if try_floor_ceil and isinstance(rel, Ge):
        strengthened = Conjecture(Ge(rel.left, ceil(right)), conj.condition)
        try:
            if strengthened.is_true(df):
                out.append(strengthened)
        except Exception:
            pass
    if try_floor_ceil and isinstance(rel, Le):
        strengthened = Conjecture(Le(rel.left, floor(right)), conj.condition)
        try:
            if strengthened.is_true(df):
                out.append(strengthened)
        except Exception:
            pass

    # Dedup
    seen = set(); uniq = []
    for c in out:
        k = (repr(c.relation), getattr(c.condition, "pretty", lambda: repr(c.condition))())
        if k in seen: continue
        seen.add(k); uniq.append(c)
    return uniq


# --- wrapper kept for backward compatibility (old drivers call PGFC(DF, conj, consts, **kw)) ---
def PGFC(
    df: pd.DataFrame,
    conj: Conjecture,
    const_bank: dict[str, dict[str, float]] | None = None,
    **kw,
) -> list[Conjecture]:
    bank = const_bank or {}
    H_key = getattr(conj.condition, "pretty", lambda: repr(conj.condition))()
    return propose_generalizations_from_constants(
        df,
        conj,
        const_bank_for_h=bank.get(H_key, {}),
        **kw,
    )


# --- batch version for a list of conjectures ---
def generalize_linear_bounds(
    df: pd.DataFrame,
    base_conjectures: Sequence[Conjecture],
    *,
    const_bank: dict[str, dict[str, float]],
    max_denom: int = 64,
    use_intercept: bool = True,
    allow_fractional: bool = True,
    try_floor_ceil: bool = True,
) -> list[Conjecture]:
    out: list[Conjecture] = []
    for c in base_conjectures:
        out.extend(
            PGFC(
                df,
                c,
                const_bank=const_bank,
                max_denom=max_denom,
                use_intercept=use_intercept,
                allow_fractional=allow_fractional,
                try_floor_ceil=try_floor_ceil,
            )
        )
    # de-dup
    seen = set(); uniq = []
    for c in out:
        k = (repr(c.relation), getattr(c.condition, "pretty", lambda: repr(c.condition))())
        if k in seen: continue
        seen.add(k); uniq.append(c)
    return uniq
