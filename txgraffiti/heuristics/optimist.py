
from typing import List
from txgraffiti.logic import *

__all__ = [
    'morgan',
]

def normalize_inequality_key(ineq: Inequality) -> tuple[str,str,str]:
    """
    Return a canonical key for any Inequality so that
      lhs <= rhs
    is always represented as (lhs.name, "<=", rhs.name),
    even if the original op was >= or > (in which case
    we swap lhs and rhs).
    """
    lhs, op, rhs = ineq.lhs, ineq.op, ineq.rhs

    # If it’s ≥ or >, flip it
    if op in (">=", ">", "≥"):
        return (rhs.name, "<=", lhs.name)
    else:
        # ≤, <, or "≤"
        return (lhs.name, "<=", rhs.name)


def same_conclusion(a: Conjecture, b: Conjecture) -> bool:
    # reuse our normalization logic so <= and >= flip to the same key
    return normalize_inequality_key(a.conclusion) == normalize_inequality_key(b.conclusion)

def is_strict_subset(m1: pd.Series, m2: pd.Series) -> bool:
    """m1 ⊂ m2? True iff every True in m1 is True in m2, and m2 has strictly more Trues."""
    return (m1 & ~m2).sum() == 0 and (m2.sum() > m1.sum())

def morgan(
    new_conj: Conjecture,
    existing: List[Conjecture],
    df:        pd.DataFrame
) -> bool:
    """
    Accept `new_conj` only if _no_ existing conjecture with the same logical
    conclusion has a hypothesis‐mask that strictly contains new_conj’s mask.
    """
    new_mask = new_conj.hypothesis(df)
    for old in existing:
        if same_conclusion(old, new_conj):
            old_mask = old.hypothesis(df)
            if is_strict_subset(new_mask, old_mask):
                # an old one is strictly more general → reject
                return False
    return True
