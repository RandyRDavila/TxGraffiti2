# # src/txgraffiti2025/processing/pre/simplify_hypotheses.py

# from __future__ import annotations
# from typing import List, Tuple, Iterable, Dict, Optional
# import re
# import pandas as pd
# from pandas.api.types import is_bool_dtype, is_integer_dtype

# from txgraffiti2025.forms.predicates import Predicate
# from txgraffiti2025.forms.class_relations import ClassEquivalence
# from txgraffiti2025.forms.generic_conjecture import TRUE
# from txgraffiti2025.processing.pre.hypotheses import (
#     list_boolean_columns,
# )

# # ───────── discovery-time boolean-like test ─────────

# def _is_bool_like_col(s: pd.Series, *, treat_binary_ints: bool) -> bool:
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

# # ───────── mask helpers ─────────

# def _mask(h: Predicate, df: pd.DataFrame) -> pd.Series:
#     m = h.mask(df)
#     if not isinstance(m, pd.Series):
#         m = pd.Series(m, index=df.index, dtype=bool)
#     return m.astype(bool).reindex(df.index, fill_value=False)

# def _mask_key(h: Predicate, df: pd.DataFrame) -> bytes:
#     return _mask(h, df).to_numpy(dtype=bool).tobytes()

# # ───────── parse atoms from names like "(col)" inside ∧ trees ─────────

# _ATOM_RE = re.compile(r"^\((?P<col>[A-Za-z0-9_]+)\)$")

# def _extract_atoms(
#     h: Predicate,
#     df: pd.DataFrame,
#     *,
#     treat_binary_ints: bool,
# ) -> List[str]:
#     from txgraffiti2025.forms.predicates import AndPred  # local import to avoid cycle
#     atoms: List[str] = []

#     def walk(p: Predicate):
#         if isinstance(p, AndPred):
#             walk(p.a); walk(p.b); return
#         name = getattr(p, "name", None)
#         if isinstance(name, str):
#             m = _ATOM_RE.match(name.strip())
#             if m:
#                 col = m.group("col")
#                 if col in df.columns and _is_bool_like_col(df[col], treat_binary_ints=treat_binary_ints):
#                     atoms.append(col)

#     walk(h)
#     atoms.sort()
#     return atoms

# # ───────── compute ∧-mask for a list of columns ─────────

# def _conj_mask_from_cols(
#     df: pd.DataFrame,
#     cols: List[str],
# ) -> pd.Series:
#     if not cols:
#         return pd.Series(True, index=df.index, dtype=bool)
#     m = pd.Series(True, index=df.index, dtype=bool)
#     for c in cols:
#         # Use forms layer for masks (truthy_only=True ensures NA-safe boolean casting)
#         m &= Predicate.from_column(c, truthy_only=True).mask(df)
#     return m

# # ───────── minimal conjunction search ─────────

# def _minimal_conjunction_cols(
#     df: pd.DataFrame,
#     target: pd.Series,
#     bool_cols: List[str],
# ) -> List[str]:
#     """
#     Find a minimal set S ⊆ bool_cols such that ∧_{c∈S} c ≡ target on df,
#     using greedy deletion for minimality.
#     """
#     target = target.astype(bool)
#     if not target.any():
#         return []

#     # Any column usable must cover all target rows
#     candidates: List[str] = []
#     for c in bool_cols:
#         cm = Predicate.from_column(c, truthy_only=True).mask(df)
#         if cm.loc[target].all():
#             candidates.append(c)

#     # Greedy deletion to minimality
#     S = candidates[:]
#     for c in list(S):
#         trial = [x for x in S if x != c]
#         if _conj_mask_from_cols(df, trial).equals(target):
#             S.remove(c)
#     return S

# # ───────── public API ─────────

# def simplify_predicate_via_df(
#     h: Predicate,
#     df: pd.DataFrame,
#     *,
#     treat_binary_ints: bool = True,
# ) -> Tuple[Predicate, Optional[ClassEquivalence]]:
#     """
#     Return a simplified hypothesis Hs equivalent to h on df, and optionally a ClassEquivalence.
#     - Use boolean-like columns discovered via `list_boolean_columns`.
#     - Build masks via `Predicate.from_column(col, truthy_only=True)`.
#     - Anchor the always-true columns (if any), then add a minimal conjunction for h's mask.
#     """
#     bool_cols = list_boolean_columns(df, treat_binary_ints=treat_binary_ints)
#     target = _mask(h, df)

#     # Always-true anchors
#     base_cols: List[str] = []
#     for c in bool_cols:
#         if Predicate.from_column(c, truthy_only=True).mask(df).all():
#             base_cols.append(c)

#     cols_min = _minimal_conjunction_cols(df, target, bool_cols)

#     # Stable order: base first (original DF order preserved within each group)
#     anchored_cols = list(dict.fromkeys([*base_cols, *cols_min]))  # stable de-dup

#     # Build simplified predicate
#     if not anchored_cols:
#         Hs = TRUE
#     else:
#         it = iter(anchored_cols)
#         acc = Predicate.from_column(next(it), truthy_only=True)
#         for c in it:
#             acc = acc & Predicate.from_column(c, truthy_only=True)
#         acc.name = "(" + " ∧ ".join(f"({c})" for c in anchored_cols) + ")"
#         Hs = acc

#     # Emit equivalence witness only if the atom lists differ
#     lhs_atoms = _extract_atoms(h, df, treat_binary_ints=treat_binary_ints)
#     rhs_atoms = anchored_cols[:]
#     if sorted(lhs_atoms) == sorted(rhs_atoms):
#         return Hs, None
#     return Hs, ClassEquivalence(h, Hs)

# def _eq_key(A: Predicate, B: Predicate, df: pd.DataFrame) -> tuple[bytes, bytes]:
#     kA, kB = _mask_key(A, df), _mask_key(B, df)
#     return (kA, kB) if kA <= kB else (kB, kA)

# def simplify_and_dedup_hypotheses(
#     df: pd.DataFrame,
#     hyps: Iterable[Predicate],
#     *,
#     min_support: Optional[int] = None,
#     treat_binary_ints: bool = True,
# ) -> Tuple[List[Predicate], List[ClassEquivalence]]:
#     """
#     Simplify each predicate against df, drop exact duplicates by mask,
#     enforce min_support, and produce ClassEquivalence witnesses for renamed forms.
#     """
#     simplified: List[Tuple[Predicate, Optional[ClassEquivalence]]] = []
#     for h in hyps:
#         Hs, eq = simplify_predicate_via_df(h, df, treat_binary_ints=treat_binary_ints)
#         if (min_support is not None) and (_mask(Hs, df).sum() < min_support):
#             continue
#         simplified.append((Hs, eq))

#     # Dedup by mask; prefer shorter textual representative
#     by_key: Dict[bytes, Predicate] = {}
#     for Hs, _ in simplified:
#         k = _mask_key(Hs, df)
#         best = by_key.get(k)
#         if best is None or len(repr(Hs)) < len(repr(best)):
#             by_key[k] = Hs

#     kept_hyps = list(by_key.values())
#     kept_hyps.sort(key=lambda p: (repr(p).count("∧"), repr(p)))

#     # Unique equivalence witnesses (unordered pair key)
#     seen_eq: set[tuple[bytes, bytes]] = set()
#     eq_conjs: List[ClassEquivalence] = []
#     for _, eq in simplified:
#         if eq is None:
#             continue
#         key = _eq_key(eq.A, eq.B, df)
#         if key in seen_eq:
#             continue
#         seen_eq.add(key)
#         eq_conjs.append(eq)

#     # Optional: sort witnesses by simplified RHS size then LHS size
#     def _size(p: Predicate) -> int:
#         return repr(p).count("∧") + 1 if "(" in repr(p) else 1

#     eq_conjs.sort(key=lambda e: (_size(e.B), repr(e.B), _size(e.A), repr(e.A)))
#     return kept_hyps, eq_conjs


# src/txgraffiti2025/processing/pre/simplify_hypotheses.py
from __future__ import annotations
from typing import List, Tuple, Iterable, Dict, Optional

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype
import re

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.class_relations import ClassEquivalence
from txgraffiti2025.forms.generic_conjecture import TRUE
from txgraffiti2025.processing.pre.hypotheses import list_boolean_columns

__all__ = [
    "simplify_predicate_via_df",
    "simplify_and_dedup_hypotheses",
]

_PSPACE = re.compile(r"\s+")
def _canon_text(p: Predicate) -> str:
    """
    Canonicalize predicate text to avoid cosmetic-only differences triggering equivalences.
    Effects:
      - collapse duplicate parentheses: '((A))' -> '(A)'
      - normalize spaces and '∧' spacing
      - strip a single pair of outer parens when safe
    """
    s = repr(p)

    # collapse (( ... )) repeatedly
    prev = None
    while prev != s:
        prev = s
        s = s.replace("((", "(").replace("))", ")")

    # normalize spaces around '∧' and parentheses
    s = s.replace(") ∧ (", ") ∧ (")
    s = s.replace("( ", "(").replace(" )", ")")
    s = _PSPACE.sub(" ", s).strip()

    # strip one outer pair if it wraps the whole thing
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        s_inner = s[1:-1].strip()
        # only strip if parentheses count looks balanced and no loss of grouping
        if s_inner.count("(") == s_inner.count(")"):
            s = s_inner
    return s


# ──────────────────────────────────────────────────────────────────────
# Mask helpers (NA→False; aligned to df.index)
# ──────────────────────────────────────────────────────────────────────

def _mask(h: Predicate, df: pd.DataFrame) -> pd.Series:
    m = h.mask(df)
    if not isinstance(m, pd.Series):
        m = pd.Series(bool(m), index=df.index, dtype=bool)
    return m.reindex(df.index, fill_value=False).fillna(False).astype(bool, copy=False)

def _mask_key(h: Predicate, df: pd.DataFrame) -> bytes:
    return _mask(h, df).to_numpy(dtype=bool, copy=False).tobytes()

# ──────────────────────────────────────────────────────────────────────
# Conjunction builder for a list of column names (via forms)
# ──────────────────────────────────────────────────────────────────────

def _conj_from_cols(cols: List[str]) -> Optional[Predicate]:
    if not cols:
        return None
    it = iter(cols)
    acc = Predicate.from_column(next(it), truthy_only=True)
    for c in it:
        acc = acc & Predicate.from_column(c, truthy_only=True)
    acc.name = "(" + " ∧ ".join(f"({c})" for c in cols) + ")"
    return acc

def _conj_mask_from_cols(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    if not cols:
        return pd.Series(True, index=df.index, dtype=bool)
    m = pd.Series(True, index=df.index, dtype=bool)
    for c in cols:
        m &= Predicate.from_column(c, truthy_only=True).mask(df)
    return m.reindex(df.index, fill_value=False).fillna(False).astype(bool, copy=False)

# ──────────────────────────────────────────────────────────────────────
# Minimal conjunction via set cover over non-target rows
# ──────────────────────────────────────────────────────────────────────

def _minimal_conjunction_cols(
    df: pd.DataFrame,
    target: pd.Series,
    bool_cols: List[str],
) -> Optional[List[str]]:
    """
    Find a small set S ⊆ bool_cols such that ⋂_{c∈S} c ≡ target on df.
    If impossible (no exact cover), return None.

    Method:
      1) Eligibility: a column c is eligible iff c is True on all target rows.
      2) Let U be the set of non-target rows. Each eligible column c “covers”
         some subset of U, namely rows where c is False. We need columns whose
         False-union covers all U (classic set cover).
      3) Greedy set cover + backward pruning for minimality.
    """
    target = target.astype(bool)
    U = (~target)  # non-target rows

    # Precompute masks for all bool-like columns
    cm: Dict[str, pd.Series] = {c: Predicate.from_column(c, truthy_only=True).mask(df).astype(bool) for c in bool_cols}

    # Eligible columns: never false on any target row
    eligible = [c for c in bool_cols if target.empty or cm[c].loc[target].all()]
    if not eligible:
        # Only trivial case works: target must be all True
        return [] if target.all() else None

    # Universe to cover and per-column coverage sets (False on non-target)
    uncovered = set(df.index[U])
    covers: Dict[str, set] = {c: set(df.index[U & (~cm[c])]) for c in eligible}

    # Greedy set cover
    chosen: List[str] = []
    while uncovered:
        best_col = None
        best_gain = 0
        for c in eligible:
            gain = len(covers[c] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_col = c
        if best_col is None or best_gain == 0:
            # Cannot cover all non-target rows ⇒ exact conjunction not representable
            return None
        chosen.append(best_col)
        uncovered -= covers[best_col]

    # Backward pruning: remove redundant columns
    for c in list(chosen):
        trial = [x for x in chosen if x != c]
        if _conj_mask_from_cols(df, trial).equals(target):
            chosen.remove(c)

    # Stable order: preserve original dataframe column order among chosen
    order = {c: i for i, c in enumerate(df.columns)}
    chosen.sort(key=lambda c: order.get(c, 10**9))
    return chosen

# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def simplify_predicate_via_df(
    h: Predicate,
    df: pd.DataFrame,
    *,
    treat_binary_ints: bool = True,
) -> Tuple[Predicate, Optional[ClassEquivalence]]:
    """
    Simplify a hypothesis `h` against `df` by expressing its mask as an
    intersection of boolean-like columns, if possible.

    Returns (Hs, eq):
      - Hs: simplified predicate (TRUE or a tidy ∧-conjunction) if representable,
            otherwise the original `h`.
      - eq: ClassEquivalence witness if Hs and h are mask-equivalent but textually different.
    """
    bool_cols = list_boolean_columns(df, treat_binary_ints=treat_binary_ints)
    target = _mask(h, df)

    # Always-true anchors (optional cosmetic prefix)
    base_cols = [c for c in bool_cols if Predicate.from_column(c, truthy_only=True).mask(df).all()]

    cols_min = _minimal_conjunction_cols(df, target, bool_cols)
    if cols_min is None:
        # Not representable as intersection of column predicates; return original
        return h, None

    # Compose base + minimal cover; base doesn't change the mask but improves readability
    anchored_cols = list(dict.fromkeys([*base_cols, *cols_min]))

    Hs = TRUE if not anchored_cols else _conj_from_cols(anchored_cols)
    if Hs is None:
        Hs = TRUE

    # Emit equivalence only if masks match and canonical text actually differs
    if _mask(Hs, df).equals(target) and _canon_text(Hs) != _canon_text(h):
        return Hs, ClassEquivalence(h, Hs)
    return Hs, None


def _eq_key(A: Predicate, B: Predicate, df: pd.DataFrame) -> tuple[bytes, bytes]:
    kA, kB = _mask_key(A, df), _mask_key(B, df)
    return (kA, kB) if kA <= kB else (kB, kA)

def simplify_and_dedup_hypotheses(
    df: pd.DataFrame,
    hyps: Iterable[Predicate],
    *,
    min_support: Optional[int] = None,
    treat_binary_ints: bool = True,
) -> Tuple[List[Predicate], List[ClassEquivalence]]:
    """
    Simplify each predicate via `simplify_predicate_via_df`, drop duplicates by mask,
    enforce `min_support`, and return any `ClassEquivalence` witnesses produced.
    """
    simplified: List[Tuple[Predicate, Optional[ClassEquivalence]]] = []
    for h in hyps:
        Hs, eq = simplify_predicate_via_df(h, df, treat_binary_ints=treat_binary_ints)
        if (min_support is not None) and (_mask(Hs, df).sum() < int(min_support)):
            continue
        simplified.append((Hs, eq))

    # Dedup by mask; prefer the shortest textual representative
    by_key: Dict[bytes, Predicate] = {}
    for Hs, _ in simplified:
        k = _mask_key(Hs, df)
        best = by_key.get(k)
        if best is None or len(repr(Hs)) < len(repr(best)):
            by_key[k] = Hs

    kept_hyps = list(by_key.values())
    kept_hyps.sort(key=lambda p: (repr(p).count("∧"), repr(p)))

    # Unique equivalence witnesses (unordered mask pair key)
    seen_eq: set[tuple[bytes, bytes]] = set()
    eq_conjs: List[ClassEquivalence] = []
    for _, eq in simplified:
        if eq is None:
            continue
        key = _eq_key(eq.A, eq.B, df)
        if key in seen_eq:
            continue
        seen_eq.add(key)
        eq_conjs.append(eq)

    # Sort witnesses: smaller RHS first for readability
    def _size(p: Predicate) -> int:
        # Heuristic: number of ∧ plus 1, if any parens are present
        r = repr(p)
        return (r.count("∧") + 1) if "(" in r else 1

    eq_conjs.sort(key=lambda e: (_size(e.B), repr(e.B), _size(e.A), repr(e.A)))
    return kept_hyps, eq_conjs
