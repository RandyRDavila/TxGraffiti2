from __future__ import annotations
from typing import Iterable, List, Sequence, Dict, Optional
import numpy as np
import pandas as pd
from fractions import Fraction

from txgraffiti2025.forms.utils import to_expr, Const, BinOp
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from .config import GenerationConfig
from .arrays import includes as mask_includes

def _split_const_times_expr(expr):
    if not isinstance(expr, BinOp) or getattr(expr, "op", None) not in ("*",):
        return None, None
    L, R = expr.left, expr.right
    from txgraffiti2025.forms.utils import Const as _ConstClass
    if isinstance(L, _ConstClass): return L, R
    if isinstance(R, _ConstClass): return R, L
    return None, None

def _to_frac_const(val: float, max_denom: int) -> Const:
    return Const(Fraction(val).limit_denominator(max_denom))

def _close(a: float, b: float, eps: float) -> bool:
    return abs(a - b) <= eps * (1.0 + max(abs(a), abs(b)))

def _replace_coeff_and_build(base_rel, coeff_expr, var_expr, condition) -> Optional[Conjecture]:
    rhs = coeff_expr * var_expr
    if isinstance(base_rel, Ge): return Conjecture(Ge(base_rel.left, rhs), condition)
    if isinstance(base_rel, Le): return Conjecture(Le(base_rel.left, rhs), condition)
    return None

def generalize_from_reciprocal_patterns(
    df: pd.DataFrame,
    base_conjectures: Sequence[Conjecture],
    *,
    const_banks: Dict[str, Dict[str, float]],
    hyps_sorted_broadest_first,
    config: GenerationConfig,
) -> List[Conjecture]:
    """
    Replace numeric constant C in T ∘ (C * X) with symbolic coeffs built from constants on broader H'.
    """
    proposals: List[Conjecture] = []
    for base in base_conjectures:
        C_const, var_expr = _split_const_times_expr(base.relation.right)
        if C_const is None: continue
        C_float = float(C_const.value)
        base_mask = base.condition.mask(df).astype(bool).to_numpy()
        base_sup  = int(base_mask.sum())

        for Hp in hyps_sorted_broadest_first(base.condition):
            p_mask = Hp.mask(df).astype(bool).to_numpy()
            if int(p_mask.sum()) < config.min_support_const: continue
            # key name for bank storage
            Hkey = getattr(Hp, "pretty", lambda: repr(Hp))()
            bank = const_banks.get(Hkey, {})

            # constants-only path
            for K, kval in bank.items():
                for a in config.numerators:
                    for s in config.shifts:
                        denom = kval + s
                        if not (np.isfinite(denom) and denom != 0.0): continue
                        coeff_val = a / denom
                        if not _close(coeff_val, C_float, config.coeff_eps_match): continue
                        from txgraffiti2025.forms.utils import Const as _Const
                        coeff_expr = _Const(a) / (to_expr(K) + _Const(s))
                        prop = _replace_coeff_and_build(base.relation, coeff_expr, var_expr, Hp)
                        if prop is None: continue
                        if not prop.is_true(df): continue
                        # keep base H unless Hp strictly broader
                        if mask_includes(base_mask, p_mask) and (int(p_mask.sum()) > base_sup):
                            pass
                        else:
                            prop.condition = base.condition
                        proposals.append(prop)

    # de-dup by (relation, condition)
    seen, out = set(), []
    for p in proposals:
        key = (repr(p.relation), getattr(p.condition, "pretty", lambda: repr(p.condition))())
        if key in seen: continue
        seen.add(key); out.append(p)
    return out
