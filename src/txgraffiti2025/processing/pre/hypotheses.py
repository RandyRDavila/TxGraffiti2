# # src/txgraffiti2025/processing/pre/hypotheses.py

# from __future__ import annotations
# from typing import Iterable, List
# import pandas as pd
# from pandas.api.types import is_bool_dtype, is_integer_dtype

# from txgraffiti2025.forms.predicates import Predicate
# from txgraffiti2025.forms.generic_conjecture import TRUE

# __all__ = [
#     "list_boolean_columns",
#     "detect_base_hypothesis",
#     "enumerate_boolean_hypotheses",
# ]

# # ─────────────────────────────────────────────────────────────────────────────
# # Column classification & predicate factory
# # ─────────────────────────────────────────────────────────────────────────────

# def _is_bool_like_col(s: pd.Series, *, treat_binary_ints: bool) -> bool:
#     """Discovery-time test: accept True/False dtype, and optionally {0,1} Int dtypes."""
#     if is_bool_dtype(s):
#         return True
#     if treat_binary_ints and is_integer_dtype(s):
#         vals = pd.unique(s.dropna())
#         try:
#             ints = {int(v) for v in vals}
#         except Exception:
#             return False
#         return len(ints) <= 2 and ints.issubset({0, 1})
#     return False

# def _pred_from_col(col: str) -> Predicate:
#     """
#     NA-safe predicate that reads a column and interprets truthiness.
#     Uses the forms layer: Predicate.from_column(col, truthy_only=True).
#     """
#     return Predicate.from_column(col, truthy_only=True)

# def list_boolean_columns(
#     df: pd.DataFrame,
#     *,
#     treat_binary_ints: bool = True,
# ) -> List[str]:
#     """Return names of boolean-like columns (nullable bools and, optionally, {0,1} Int)."""
#     return [c for c in df.columns if _is_bool_like_col(df[c], treat_binary_ints=treat_binary_ints)]

# # ─────────────────────────────────────────────────────────────────────────────
# # Naming helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _strip_outer_parens(s: str) -> str:
#     s = s.strip()
#     return s[1:-1] if s.startswith("(") and s.endswith(")") else s

# def _and_name(base: Predicate, cols: list[str]) -> str:
#     """
#     Produce names like:
#       - "(c)" when base is TRUE and one column
#       - "((base) ∧ (c))" for single under base
#       - "((c1) ∧ (c2))" for pair when base is TRUE
#       - "((base) ∧ (c1) ∧ (c2))" for pair under base
#     """
#     parts: list[str] = []
#     if base is not TRUE:
#         base_name = getattr(base, "name", repr(base))
#         parts.append(_strip_outer_parens(str(base_name)))
#     parts.extend([f"({c})" for c in cols])
#     return "(" + " ∧ ".join(parts) + ")"

# # ─────────────────────────────────────────────────────────────────────────────
# # Base detection
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_base_hypothesis(df: pd.DataFrame) -> Predicate:
#     """
#     Detect a base hypothesis from columns that are all True.
#     - If none, return TRUE.
#     - If one, return that column predicate "(col)".
#     - If multiple, return their ∧-conjunction with name "((c1) ∧ (c2) ∧ ...)".
#     """
#     bool_cols = list_boolean_columns(df, treat_binary_ints=True)

#     always_true: list[str] = []
#     for c in bool_cols:
#         # Build mask via forms to keep NA-safe behavior consistent
#         m = _pred_from_col(c).mask(df)
#         if bool(m.all()):
#             always_true.append(c)

#     if not always_true:
#         return TRUE
#     always_true.sort()

#     preds = [_pred_from_col(c) for c in always_true]
#     it = iter(preds)
#     acc: Predicate = next(it, TRUE)
#     for p in it:
#         acc = acc & p
#     acc.name = "(" + " ∧ ".join(f"({c})" for c in always_true) + ")"
#     return acc

# # ─────────────────────────────────────────────────────────────────────────────
# # Hypothesis enumeration
# # ─────────────────────────────────────────────────────────────────────────────

# def enumerate_boolean_hypotheses(
#     df: pd.DataFrame,
#     *,
#     treat_binary_ints: bool = True,
#     include_base: bool = True,
#     include_pairs: bool = True,
#     skip_always_false: bool = True,
# ) -> List[Predicate]:
#     """
#     Enumerate hypothesis predicates anchored at the detected base.

#     Output order:
#       [base?] + [base ∧ (c)] + [base ∧ (c_i) ∧ (c_j)]   (pairs sorted lexicographically)
#     """
#     base = detect_base_hypothesis(df)
#     base_mask = (base.mask(df) if base is not TRUE else pd.Series(True, index=df.index)).astype(bool)

#     # Discover candidates
#     bool_cols = list_boolean_columns(df, treat_binary_ints=treat_binary_ints)

#     # Columns subsumed by base (globally True)
#     base_cols = []
#     for c in bool_cols:
#         if _pred_from_col(c).mask(df).all():
#             base_cols.append(c)

#     cand_cols = [c for c in bool_cols if c not in base_cols]

#     # Cache masks under base to avoid recomputation
#     def mask_under_base(col: str) -> pd.Series:
#         return (_pred_from_col(col).mask(df) & base_mask)

#     ub: dict[str, pd.Series] = {c: mask_under_base(c) for c in cand_cols}

#     if skip_always_false:
#         cand_cols = [c for c in cand_cols if ub[c].any()]

#     out: List[Predicate] = []
#     if include_base:
#         out.append(base)

#     # Singles
#     for c in sorted(cand_cols):
#         P = _pred_from_col(c)
#         S = (base & P) if base is not TRUE else P
#         S.name = _and_name(base, [c])
#         out.append(S)

#     if not include_pairs:
#         return out

#     # Pairs: enforce non-empty overlap under base and mutual contribution
#     pairs: list[tuple[tuple[str, str], Predicate]] = []
#     for i in range(len(cand_cols)):
#         c1 = cand_cols[i]
#         s1 = ub[c1]
#         if not s1.any():
#             continue
#         for j in range(i + 1, len(cand_cols)):
#             c2 = cand_cols[j]
#             s2 = ub[c2]
#             if not s2.any():
#                 continue
#             inter = s1 & s2
#             if not inter.any():
#                 continue
#             # each adds something under base
#             if not ( (s1 & ~s2).any() and (s2 & ~s1).any() ):
#                 continue

#             P1, P2 = _pred_from_col(c1), _pred_from_col(c2)
#             pair = (base & P1 & P2) if base is not TRUE else (P1 & P2)
#             cols_sorted = sorted([c1, c2])
#             pair.name = "(" + " ∧ ".join(f"({c})" for c in cols_sorted) + ")"
#             pairs.append(((cols_sorted[0], cols_sorted[1]), pair))

#     pairs.sort(key=lambda t: t[0])
#     out.extend(pred for _, pred in pairs)
#     return out

# src/txgraffiti2025/processing/pre/hypotheses.py
from __future__ import annotations

"""
Enumerate boolean-style hypothesis predicates from a DataFrame.

Exports
-------
- list_boolean_columns(df, treat_binary_ints=True)
- detect_base_hypothesis(df)
- enumerate_boolean_hypotheses(df, ...)

Conventions
-----------
- NA → False when interpreting boolean columns as predicates.
- Base hypothesis is the ∧ of all columns that are *always True* across df.
- Column predicates are built via forms: Predicate.from_column(col, truthy_only=True)
"""

from typing import List
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.generic_conjecture import TRUE

__all__ = [
    "list_boolean_columns",
    "detect_base_hypothesis",
    "enumerate_boolean_hypotheses",
]


# ──────────────────────────────────────────────────────────────────────
# Column classification & predicate factory
# ──────────────────────────────────────────────────────────────────────

def _is_bool_like_col(
    s: pd.Series,
    *,
    treat_binary_ints: bool,
    include_binary_numeric: bool = False,
) -> bool:
    """
    Discovery-time check for boolean-like columns.

    True if:
      - dtype is a bool/nullable-Boolean, or
      - (treat_binary_ints=True) and integer dtype limited to {0,1}, or
      - (include_binary_numeric=True) and *any* numeric dtype limited to {0,1}.
    """
    if is_bool_dtype(s):
        return True

    if treat_binary_ints and is_integer_dtype(s):
        vals = pd.unique(pd.Series(s).dropna())
        try:
            ints = {int(v) for v in vals}
        except Exception:
            return False
        return len(ints) <= 2 and ints.issubset({0, 1})

    if include_binary_numeric and is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        vals = set(pd.unique(v.dropna()))
        return vals.issubset({0, 1}) and len(vals) <= 2

    return False


def _pred_from_col(col: str) -> Predicate:
    """
    NA-safe predicate that reads a column and interprets truthiness.
    Delegates to the forms layer for consistent NA→False semantics.
    """
    return Predicate.from_column(col, truthy_only=True)


def list_boolean_columns(
    df: pd.DataFrame,
    *,
    treat_binary_ints: bool = True,
) -> List[str]:
    """
    Return names of boolean-like columns:
      - (nullable) boolean dtypes,
      - and (optionally) integer dtypes that are subset of {0,1}.
    """
    return [
        c for c in df.columns
        if _is_bool_like_col(df[c], treat_binary_ints=treat_binary_ints, include_binary_numeric=False)
    ]


# ──────────────────────────────────────────────────────────────────────
# Naming helpers
# ──────────────────────────────────────────────────────────────────────

def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    return s[1:-1] if (len(s) >= 2 and s[0] == "(" and s[-1] == ")") else s


def _and_name(base: Predicate, cols: list[str]) -> str:
    """
    Produce consistent names:

      base = TRUE:
        "(c)" or "((c1) ∧ (c2))"
      base ≠ TRUE:
        "((base) ∧ (c))" or "((base) ∧ (c1) ∧ (c2))"
    """
    parts: list[str] = []
    if base is not TRUE:
        base_name = getattr(base, "name", repr(base))
        parts.append(_strip_outer_parens(str(base_name)))
    parts.extend([f"({c})" for c in cols])
    return "(" + " ∧ ".join(parts) + ")"


# ──────────────────────────────────────────────────────────────────────
# Base detection
# ──────────────────────────────────────────────────────────────────────

def detect_base_hypothesis(df: pd.DataFrame) -> Predicate:
    """
    Detect a base hypothesis from columns that are always True.

    Returns
    -------
    - TRUE if none are always-True,
    - predicate "(c)" if one column c is always-True,
    - conjunction "((c1) ∧ (c2) ∧ ...)" otherwise.
    """
    bool_cols = list_boolean_columns(df, treat_binary_ints=True)

    always_true: list[str] = []
    for c in bool_cols:
        m = _pred_from_col(c).mask(df)
        # NA→False already handled by forms; .all() over empty Series is True, which is fine
        if bool(m.all()):
            always_true.append(c)

    if not always_true:
        return TRUE

    always_true.sort()
    preds = [_pred_from_col(c) for c in always_true]
    it = iter(preds)
    acc: Predicate = next(it, TRUE)
    for p in it:
        acc = acc & p
    acc.name = "(" + " ∧ ".join(f"({c})" for c in always_true) + ")"
    return acc


# ──────────────────────────────────────────────────────────────────────
# Hypothesis enumeration
# ──────────────────────────────────────────────────────────────────────

def enumerate_boolean_hypotheses(
    df: pd.DataFrame,
    *,
    treat_binary_ints: bool = True,
    include_base: bool = True,
    include_pairs: bool = True,
    skip_always_false: bool = True,
) -> List[Predicate]:
    """
    Enumerate hypothesis predicates anchored at the detected base.

    Output order:
      [base?] + [base ∧ (c)] + [base ∧ (c_i) ∧ (c_j)]   (pairs sorted lexicographically)

    Notes
    -----
    - NA semantics: NA→False when reading columns as predicates.
    - Pair generation requires: non-empty intersection under base AND each column
      contributes uniquely under base ((s1 & ~s2).any() and (s2 & ~s1).any()).
    """
    base = detect_base_hypothesis(df)
    base_mask = (base.mask(df) if base is not TRUE else pd.Series(True, index=df.index)).astype(bool)

    # Discover candidates
    bool_cols = list_boolean_columns(df, treat_binary_ints=treat_binary_ints)

    # Columns subsumed by base (globally True)
    base_cols: list[str] = []
    for c in bool_cols:
        if _pred_from_col(c).mask(df).all():
            base_cols.append(c)

    cand_cols = [c for c in bool_cols if c not in base_cols]

    # Cache masks under base to avoid recomputation
    def mask_under_base(col: str) -> pd.Series:
        return _pred_from_col(col).mask(df).astype(bool) & base_mask

    ub = {c: mask_under_base(c) for c in cand_cols}

    if skip_always_false:
        cand_cols = [c for c in cand_cols if ub[c].any()]

    out: List[Predicate] = []
    if include_base:
        out.append(base)

    # Singles
    for c in sorted(cand_cols):
        P = _pred_from_col(c)
        S = (base & P) if base is not TRUE else P
        S.name = _and_name(base, [c])
        out.append(S)

    if not include_pairs:
        return out

    # Pairs: enforce non-empty overlap under base and mutual contribution
    pairs: list[tuple[tuple[str, str], Predicate]] = []
    for i, c1 in enumerate(cand_cols):
        s1 = ub[c1]
        if not s1.any():
            continue
        for c2 in cand_cols[i + 1 :]:
            s2 = ub[c2]
            if not s2.any():
                continue
            inter = s1 & s2
            if not inter.any():
                continue
            # each adds something under base
            if not ((s1 & ~s2).any() and (s2 & ~s1).any()):
                continue

            P1, P2 = _pred_from_col(c1), _pred_from_col(c2)
            pair = (base & P1 & P2) if base is not TRUE else (P1 & P2)
            cols_sorted = sorted([c1, c2])
            # include base in the pair name for consistency with singles
            pair.name = _and_name(base, cols_sorted)
            pairs.append(((cols_sorted[0], cols_sorted[1]), pair))

    pairs.sort(key=lambda t: t[0])
    out.extend(pred for _, pred in pairs)
    return out
