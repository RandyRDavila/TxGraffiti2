from __future__ import annotations
from typing import Iterable, List, Tuple
import pandas as pd
from txgraffiti2025.forms.predicates import Where, AndPred, NotPred
from txgraffiti2025.forms.generic_conjecture import TRUE

__all__ = [
    'list_boolean_columns',
    'detect_base_hypothesis',
    'enumerate_boolean_hypotheses'
]




def _pred_from_col(col: str) -> Where:
    # Pretty name: "(col)"
    return Where(lambda d, c=col: d[c], name=f"({col})")

def _is_bool_col(s: pd.Series, treat_binary_ints: bool) -> bool:
    if s.dtype == bool:
        return True
    if treat_binary_ints and s.dtype.kind in ("i", "u"):
        vals = pd.unique(s.dropna())
        return len(vals) <= 2 and set(map(int, vals)).issubset({0, 1})
    return False

def _bool_series(s: pd.Series, treat_binary_ints: bool) -> pd.Series:
    """Coerce a boolean-like column to strictly boolean Series."""
    if s.dtype == bool:
        return s
    if treat_binary_ints and s.dtype.kind in ("i", "u"):
        return s.astype(bool)
    # Fallback: try generic truthiness
    return s.astype(bool)


# --- helpers ---------------------------------------------------------------

def _pred_from_col(col: str) -> Where:
    # Pretty name: "(col)"
    return Where(lambda d, c=col: d[c], name=f"({col})")

def _and_all(preds: Iterable[Where], names: Iterable[str]) -> AndPred:
    it = iter(preds)
    try:
        acc = next(it)
    except StopIteration:
        return _pred_from_col("•empty•")  # should never happen here
    for p in it:
        acc = AndPred(acc, p)
    # Pretty name: "((a) ∧ (b) ∧ ...)"
    acc.name = "(" + " ∧ ".join(f"({n})" for n in names) + ")"
    return acc


# --- public API ------------------------------------------------------------

def list_boolean_columns(
    df: pd.DataFrame,
    *,
    treat_binary_ints: bool = True,
) -> List[str]:
    """
    Return all column names that are boolean-like:
    - dtype == bool, or (if treat_binary_ints) integer columns with values in {0,1}.
    """
    cols: List[str] = []
    for col in df.columns:
        s = df[col]
        if _is_bool_col(s, treat_binary_ints=treat_binary_ints):
            cols.append(col)
    return cols

def detect_base_hypothesis(df: pd.DataFrame):
    """
    Existing function from earlier message (kept here for completeness):

    - If one or more boolean columns are True for all rows, return their
      conjunction: ((c1) ∧ (c2) ∧ ...).
    - If exactly one exists, return (c).
    - If none, return TRUE.
    """
    bool_cols = list_boolean_columns(df)
    always_true = [c for c in bool_cols if df[c].astype(bool).all()]

    if not always_true:
        return TRUE

    preds = [_pred_from_col(c) for c in always_true]
    if len(preds) == 1:
        return preds[0]
    return _and_all(preds, always_true)

def enumerate_boolean_hypotheses(
    df: pd.DataFrame,
    *,
    treat_binary_ints: bool = True,
    include_base: bool = True,
    include_pairs: bool = True,
    skip_always_false: bool = True,
) -> List:
    """
    Enumerate hypothesis predicates anchored at the detected base.

    Output order:
      [base?] + [base ∧ (c)] + [base ∧ (c_i) ∧ (c_j)]

    Pairwise constraints are checked UNDER THE BASE:
      - neither (c_i) nor (c_j) is a strict subset of the other
      - (c_i ∧ c_j) is non-empty
      - (c_i) and (c_j) themselves are non-empty under base

    Parameters
    ----------
    df : pd.DataFrame
    treat_binary_ints : bool, default True
        Treat {0,1} integer columns as boolean.
    include_base : bool, default True
        Include the base predicate itself as the first element.
    include_pairs : bool, default True
        If True, include pairwise (c_i, c_j) combos meeting the constraints.
    skip_always_false : bool, default True
        Skip candidates that are entirely False under base.

    Returns
    -------
    list of Predicate
    """
    base = detect_base_hypothesis(df)
    base_mask = (base.mask(df) if base is not TRUE else pd.Series(True, index=df.index))

    # Boolean-like candidates
    bool_cols = list_boolean_columns(df, treat_binary_ints=treat_binary_ints)

    # Columns already absorbed into base (always True globally)
    base_cols = [c for c in bool_cols if _bool_series(df[c], treat_binary_ints).all()]
    cand_cols = [c for c in bool_cols if c not in base_cols]

    # Consider truth under base
    def under_base(col: str) -> pd.Series:
        return _bool_series(df[col], treat_binary_ints) & base_mask

    # Filter empty under base if requested
    if skip_always_false:
        cand_cols = [c for c in cand_cols if under_base(c).any()]

    # Build output
    out = []
    if include_base:
        out.append(base)

    # Singles: base ∧ (c)  (or just (c) if base is TRUE)
    def AND(a, b):
        return AndPred(a, b) if a is not TRUE else b

    singles = []
    for c in sorted(cand_cols):
        P = _pred_from_col(c)
        singles.append(AND(base, P))
    out.extend(singles)

    if not include_pairs:
        return out

    # Pairs: base ∧ (c_i) ∧ (c_j), with constraints under base
    pairs = []
    for i in range(len(cand_cols)):
        c1 = cand_cols[i]
        s1b = under_base(c1)
        if not s1b.any():
            continue
        for j in range(i + 1, len(cand_cols)):
            c2 = cand_cols[j]
            s2b = under_base(c2)
            if not s2b.any():
                continue

            inter = s1b & s2b
            if not inter.any():
                # empty intersection under base
                continue

            # strict subset checks under base
            s1_outside_s2 = (s1b & ~s2b).any()
            s2_outside_s1 = (s2b & ~s1b).any()

            # if either is strict subset of the other, skip
            if not (s1_outside_s2 and s2_outside_s1):
                continue

            P1 = _pred_from_col(c1)
            P2 = _pred_from_col(c2)
            pair_pred = AND(AND(base, P1), P2)
            pairs.append(((c1, c2), pair_pred))

    # Deterministic order: sort by (col_i, col_j)
    pairs.sort(key=lambda t: t[0])
    out.extend(pred for _, pred in pairs)

    return out

