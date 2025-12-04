# src/txgraffiti/graffiti3/class_relations.py

from __future__ import annotations

from collections import defaultdict
from functools import reduce
from itertools import combinations
import operator
from typing import Dict, List, Tuple
from fractions import Fraction

import numpy as np
import pandas as pd

from .exprs import Expr, ColumnTerm, Const, abs_, min_, max_
from .predicates import Predicate, Where
from .relations import (
    TRUE,
    Conjecture,
    Eq, AllOf,
    Relation,
    Ge,
    Le,
    ClassEquivalence,
    ClassInclusion,
)

__all__ = [
    'GraffitiClassRelations'
]

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _const_for_expr(x: float, *, rationalize: bool, max_denom: int) -> Const:
    """
    Return a Const whose internal value is a Fraction (pretty as p/q) if rationalize,
    otherwise a numeric Const. Evaluation should convert to float internally.
    """
    if rationalize and np.isfinite(x):
        fr = Fraction(x).limit_denominator(max_denom)
        # if the rational is a clean match, keep Fraction; else fall back to float
        if abs(float(fr) - x) <= 1e-12:
            return Const(fr)
    return Const(float(x))

def _isclose_vec(a: np.ndarray, b: np.ndarray, *, atol: float = 0.0, rtol: float = 0.0) -> np.ndarray:
    """Vectorized closeness: |a-b| <= atol + rtol*|b|."""
    return np.abs(a - b) <= (atol + rtol * np.abs(b))

def _is_boolean_like(s: pd.Series) -> bool:
    """Return True if a pandas Series is boolean-like.

    Boolean-like means:
    - dtype is a pandas Boolean type (`bool` or nullable 'boolean'), or
    - numeric with all non-NA values in {0, 1}.

    Parameters
    ----------
    s : pd.Series
        Series to test.

    Returns
    -------
    bool
        True if boolean-like, False otherwise.
    """
    if pd.api.types.is_bool_dtype(s) or str(s.dtype).lower().startswith("boolean"):
        return True
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        nonna = s.dropna()
        if nonna.empty:
            return False
        vals = set(pd.unique(nonna.astype(float)))
        return vals.issubset({0.0, 1.0})
    return False


    # """Discover class relations and constant-equality characterizations.

    # This container inspects a table of objects (rows) annotated with:
    # - Boolean-like columns (treated as class predicates);
    # - Other columns (treated as numeric expressions via `Expr/ColumnTerm`).

    # It provides:
    # 1) A **base hypothesis**: conjunction of all universally-true Boolean predicates.
    # 2) Enumeration of **base conjunctions**: base ∧ p_i (1-way) and base ∧ p_i ∧ p_j (2-way), etc.
    # 3) Redundancy checks among conjunctions (dominance by strict subsets with same mask).
    # 4) Constant detection: columns that are constant (within tolerance) under each hypothesis.
    # 5) Construction of **constant-equality conjectures** and **class characterizations**:
    #    equivalence and inclusion between “equalities hold” and the hypothesis class.

    # Parameters
    # ----------
    # df : pd.DataFrame
    #     Input table of objects and invariants.

    # Notes
    # -----
    # - Boolean-like detection treats 0/1-coded numeric columns as predicates.
    # - All mask computations are cached per-predicate for speed.
    # - Printing/usage is kept behind `if __name__ == "__main__"` to keep the module import-clean.
    # """
# ──────────────────────────────────────────────────────────────────────────────
# Core Container
# ──────────────────────────────────────────────────────────────────────────────

class GraffitiClassRelations:
    """
    Discover and relate hypothesis classes from boolean-like columns, and
    synthesize constant/equality and ratio-based conjectures over numeric
    invariants—using masks, equivalence/dominance tests, and cached evaluation.

    The workflow is:

    1. **Partition columns** into boolean-like (classes) and numeric (expressions).
    2. **Form the base hypothesis** as the conjunction of all universally true
       boolean columns (or `TRUE` if none).
    3. **Enumerate conjunctions** of non-base boolean predicates with the base.
    4. **Deduplicate** conjunctions by mask-equivalence and mark redundant ones
       (dominated by strict subsets).
    5. **Mine constants** for numeric columns within each nonredundant conjunction
       and emit equality conjectures, optionally grouped per hypothesis.
    6. **Characterize classes** by comparing “equalities hold” vs “class predicate”
       via `(A ≡ B)`, `(A ⊆ B)`, `(B ⊆ A)`.
    7. **Mine ratio bounds** on the base hypothesis: for each ordered pair
       `(inv1, inv2)`, compute `cmin ≤ inv1/inv2 ≤ cmax`, record touch counts
       (equality witnesses), and emit lower/upper conjectures.
    8. **Prune** and **compress** conjecture lists by merging equivalent conditions,
       deduplicating relations by truth masks, and keeping sharpest one-sided bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input table. Boolean-like columns (dtype bool/nullable-boolean or
        numeric with all values in {0,1}) are treated as class predicates.
        All other columns are wrapped as `Expr` via `ColumnTerm`.

    Attributes
    ----------
    df : pd.DataFrame
        Current working DataFrame.
    boolean_cols : list[str]
        Names of boolean-like columns.
    expr_cols : list[str]
        Names of non-boolean (numeric) columns used as expressions.
    exprs : dict[str, Expr]
        Mapping `column -> Expr` wrapper (`ColumnTerm`).
    predicates : dict[str, Predicate]
        Mapping `boolean column -> Predicate` (truthy-only).
    base_hypothesis : Predicate
        Conjunction of universally true boolean columns, or `TRUE` if none.
    base_hypothesis_name : str
        Pretty name of `base_hypothesis` (e.g., `"connected ∧ planar"` or `"TRUE"`).
    base_predicates : dict[str, Predicate]
        Non-base boolean predicates (i.e., boolean columns minus always-true set).
    base_conjunctions : list[tuple[str, Predicate]]
        All enumerated conjunctions of base with up to `max_arity` non-base predicates.
    nonredundant_conjunctions_ : list[tuple[str, Predicate]]
        Conjunctions that are not dominated by strict subsets (plus explicit base).
    redundant_conjunctions_ : list[tuple[str, Predicate]]
        Conjunctions dominated by a strict subset with identical mask.
    equivalent_conjunction_groups_ : list[list[tuple[str, Predicate]]]
        Groups of conjunctions that are mask-equivalent and of equal arity.
    constant_exprs_ : dict[str, list[tuple[str, float]]]
        For each nonredundant hypothesis, (column, constant) pairs discovered.
    class_equivalences_ : list[ClassEquivalence]
        Results where “equalities hold” ≡ (hypothesis class).
    class_inclusions_AB_ : list[ClassInclusion]
        Results where (equalities ⇒ class).
    class_inclusions_BA_ : list[ClassInclusion]
        Results where (class ⇒ equalities).
    class_non_equivalences_ : list[dict]
        Diagnostics for cases where equivalence fails (with violation counts).
    class_cond_implies_equalities_ : list[Conjecture]
        Conjectures of the form (condition ⇒ all equalities).
    simple_ratio_bounds : list[Conjecture]
        Lower/upper conjectures emitted by `generate_ratio_bounds_on_base`.
    sorted_conjunctions_ : list[tuple[str, Predicate]]
        Nonredundant conjunctions sorted by support size (generality).
    _mask_cache : dict[int, np.ndarray]
        Cache of predicate masks keyed by `id(predicate)`.
    _mask_cache_version : int
        Cache invalidation token bound to `id(self.df)`.

    Notes
    -----
    - **Boolean-like detection**: a column is considered boolean-like if it is
      a pandas boolean dtype or numeric with all observed non-NA values in {0,1}.
    - **Caching**: Masks are cached as dense `np.ndarray[bool]`. Cache is cleared
      automatically when `df` identity changes (e.g., `refresh` or new frame).
    - **Stability**: Set/combination orderings are made deterministic by sorting
      predicate names lexicographically when enumerating conjunctions.

    See Also
    --------
    Predicate, Where
        Boolean DSL for row-wise masking.
    Expr, ColumnTerm, Const
        Expression DSL for numeric columns.
    ClassEquivalence, ClassInclusion
        Class-relation checkers over masks.
    Conjecture, Eq, Ge, Le, AllOf, Relation, TRUE
        Conjecture and relations machinery.

    """
    # ------------------------------ Lifecycle ---------------------------------

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame and compute baseline state:
        column partition, base hypothesis, initial enumeration/analysis, and
        default ratio bounds & sorting.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset. See class docstring for boolean-like detection.

        Raises
        ------
        TypeError
            If `df` is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df
        self._reset_parameters()

    def refresh(self, df: pd.DataFrame | None = None) -> None:
        """
        Refresh internal state with a new DataFrame (if provided) or the current one.

        This clears the mask cache (if `df` identity changes), recomputes boolean/
        numeric partitions, base hypothesis, and re-runs:
        - `enumerate_conjunctions()`
        - `find_redundant_conjunctions()`
        - `characterize_constant_classes()`
        - `generate_ratio_bounds_on_base()`
        - `sort_conjunctions_by_generality()`

        Parameters
        ----------
        df : pd.DataFrame or None
            If provided, replaces the existing DataFrame before recomputing.
        """
        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")
            self.df = df
        self._reset_parameters()

    # ------------------------------ Introspection -----------------------------

    def get_expr_columns(self) -> List[str]:
        """
        Return the names of non-boolean (expression) columns.

        Returns
        -------
        list[str]
            Column names detected as non-boolean.
        """
        return list(self.expr_cols)

    def get_exprs(self) -> Dict[str, Expr]:
        """
        Mapping from column name to `Expr` for non-boolean columns.

        Returns
        -------
        dict[str, Expr]
            `{name: ColumnTerm(name)}` for each non-boolean column.
        """
        return dict(self.exprs)

    def get_base_predicates(self) -> Dict[str, Predicate]:
        """
        Mapping from boolean-like column to `Predicate`, excluding always-true columns.

        Returns
        -------
        dict[str, Predicate]
            Non-base boolean predicates available for conjunction enumeration.
        """
        return dict(self.base_predicates)

    def expr(self, name: str) -> Expr:
        """
        Return the `Expr` wrapper for a non-boolean column.

        Parameters
        ----------
        name : str
            Column name in `expr_cols`.

        Returns
        -------
        Expr
            `ColumnTerm(name)`.

        Raises
        ------
        KeyError
            If `name` is not a recognized non-boolean column.
        """
        return self.exprs[name]

    def pred(self, name: str) -> Predicate:
        """
        Return the `Predicate` wrapper for a boolean-like column (non-base).

        Parameters
        ----------
        name : str
            Column name in `base_predicates`.

        Returns
        -------
        Predicate
            Boolean predicate for the column.

        Raises
        ------
        KeyError
            If `name` is not a recognized non-base boolean column.
        """
        return self.base_predicates[name]

    # ------------------------------ Enumeration -------------------------------

    def enumerate_conjunctions(self, *, max_arity: int = 2) -> List[Tuple[str, Predicate]]:
        """
        Enumerate `(base ∧ p1 ∧ … ∧ pk)` over non-base predicates for `1 ≤ k ≤ max_arity`.

        Predicate names are sorted for deterministic pretty names and combinations.

        Parameters
        ----------
        max_arity : int, default=2
            Maximum number of non-base predicates to conjoin with the base.

        Returns
        -------
        list[tuple[str, Predicate]]
            Pairs of `(pretty_name, predicate_conjunction)`.

        Raises
        ------
        ValueError
            If `max_arity < 1`.
        """
        if max_arity < 1:
            raise ValueError("max_arity must be >= 1")

        items = list(self.base_predicates.items())
        conjs: List[Tuple[str, Predicate]] = []

        # order predicates for deterministic names
        items.sort(key=lambda t: t[0])

        # arity = 1..max_arity
        base_parts = () if self.base_hypothesis_name == "TRUE" else tuple(self.base_hypothesis_name.split(" ∧ "))
        base_and = (lambda p: p) if self.base_hypothesis is TRUE else (lambda p: self.base_hypothesis & p)

        # 1-way
        if max_arity >= 1:
            for n1, p1 in items:
                name = " ∧ ".join((*base_parts, n1)) if base_parts else n1
                conjs.append((name, base_and(p1)))

        # higher arities
        for k in range(2, max_arity + 1):
            for combo in combinations(items, k):
                names = [n for (n, _) in combo]
                preds = [p for (_, p) in combo]
                name = " ∧ ".join((*base_parts, *names)) if base_parts else " ∧ ".join(names)
                conj = reduce(operator.and_, preds, self.base_hypothesis)
                conjs.append((name, conj))

        self.base_conjunctions = conjs
        return conjs

    def find_redundant_conjunctions(self):
        """
        Partition enumerated conjunctions into **nonredundant**, **redundant**, and **equivalent groups**.

        Definitions
        -----------
        - Redundant: A conjunction whose mask is identical to that of a
          strict-subset of its (non-base) predicates (dominated).
        - Equivalent: Conjunctions with identical masks and equal arity.

        Returns
        -------
        tuple
            `(nonredundant, redundant, equivalence_groups)` where
            - `nonredundant : list[(name, Predicate)]`
            - `redundant : list[(name, Predicate)]`
            - `equivalence_groups : list[list[(name, Predicate)]]`

        Raises
        ------
        AttributeError
            If `enumerate_conjunctions()` has not been called.
        """
        if not hasattr(self, "base_conjunctions"):
            raise AttributeError("Call enumerate_conjunctions() first.")

        base_parts = tuple(self.base_hypothesis_name.split(" ∧ ")) if self.base_hypothesis_name != "TRUE" else tuple()

        # Extract records with extras and signature
        records = []
        for idx, (name, pred) in enumerate(self.base_conjunctions):
            mask = self._mask_cached(pred)  # cached mask
            sig = mask.tobytes()            # byte signature of boolean mask
            parts = tuple(name.split(" ∧ "))
            extras = parts[len(base_parts):]
            records.append({
                "idx": idx, "name": name, "pred": pred,
                "sig": sig, "extras": frozenset(extras), "size": len(extras)
            })

        # Group by (signature, size) to detect equivalences of the same length
        by_sig_size = defaultdict(list)
        for r in records:
            by_sig_size[(r["sig"], r["size"])].append(r)

        equivalence_groups = []
        for (_, _), group in by_sig_size.items():
            if len(group) > 1:
                equivalence_groups.append([(g["name"], g["pred"]) for g in group])

        # Redundancy: dominated by a STRICT subset with the same signature
        redundant_ids = set()
        by_sig = defaultdict(list)
        for r in records:
            by_sig[r["sig"]].append(r)

        for sig, group in by_sig.items():
            group_sorted = sorted(group, key=lambda r: r["size"])
            minimal_extras: List[frozenset] = []
            for r in group_sorted:
                if any(me < r["extras"] for me in minimal_extras):  # strict subset check
                    redundant_ids.add(r["idx"])
                else:
                    minimal_extras.append(r["extras"])

        nonredundant = [(records[i]["name"], records[i]["pred"])
                        for i in range(len(records)) if records[i]["idx"] not in redundant_ids]
        redundant = [(r["name"], r["pred"]) for r in records if r["idx"] in redundant_ids]

        # include pure base hypothesis explicitly
        nonredundant.append((self.base_hypothesis_name, self.base_hypothesis))

        self.nonredundant_conjunctions_ = nonredundant
        self.redundant_conjunctions_ = redundant
        self.equivalent_conjunction_groups_ = equivalence_groups
        return nonredundant, redundant, equivalence_groups

    def sort_conjunctions_by_generality(self, *, ascending: bool = False):
        """
        Sort nonredundant conjunctions by *generality* using support size (row count where mask is True).

        A is more general than B if `mask(A)` ⊇ `mask(B)`. We approximate generality
        by support size, tie-breaking by name for reproducibility.

        Parameters
        ----------
        ascending : bool, default=False
            If False, sorts from most general (largest support) to most specific.

        Returns
        -------
        list[tuple[str, Predicate]]
            Sorted `(name, Predicate)` pairs.

        Raises
        ------
        AttributeError
            If `find_redundant_conjunctions()` has not been called.
        """
        if not hasattr(self, "nonredundant_conjunctions_"):
            raise AttributeError("Call find_redundant_conjunctions() first.")

        scored = []
        for name, pred in self.nonredundant_conjunctions_:
            support = int(self._mask_cached(pred).sum())
            scored.append((name, pred, support))

        scored.sort(key=lambda t: (t[2], t[0]), reverse=not ascending)
        self.sorted_conjunctions_ = [(n, p) for (n, p, _) in scored]
        return self.sorted_conjunctions_

    # ------------------------------ Constants & Conjectures --------------------

    def find_constant_exprs(self, *, tol: float = 0.0, require_finite: bool = True):
        """
        Detect numeric columns that are (approximately) constant within each nonredundant hypothesis.

        Parameters
        ----------
        tol : float, default=0.0
            If 0, requires exact equality; otherwise marks constant when `max-min ≤ tol`.
        require_finite : bool, default=True
            If True, ignores ±inf and NaN before testing constancy.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            Mapping `hypothesis_name -> [(column, constant_value), ...]`.
            Hypotheses with no constants map to an empty list.

        Notes
        -----
        Uses `Expr.eval(df)` for each numeric column and the cached predicate masks.
        If derived attributes are missing, this method triggers enumeration and redundancy passes.
        """
        if not hasattr(self, "nonredundant_conjunctions_"):
            self.enumerate_conjunctions()
            _ = self.find_redundant_conjunctions()

        def _constant_value(s: pd.Series) -> tuple[bool, float | None]:
            s = s.replace([np.inf, -np.inf], np.nan) if require_finite else s
            s = s.dropna()
            if s.empty:
                return False, None
            a = s.to_numpy(dtype=float, copy=False)
            if tol == 0.0:
                v0 = a[0]
                return (True, float(v0)) if np.all(a == v0) else (False, None)
            amin, amax = float(np.min(a)), float(np.max(a))
            return (True, (amin + amax) / 2.0) if (amax - amin) <= tol else (False, None)

        out: dict[str, list[tuple[str, float]]] = {}
        for name, pred in self.nonredundant_conjunctions_:
            mask = self._mask_cached(pred)
            consts: list[tuple[str, float]] = []
            for col in self.expr_cols:
                s = self.exprs[col].eval(self.df)[mask]
                ok, val = _constant_value(s)
                if ok and val is not None:
                    consts.append((col, float(val)))
            out[name] = consts

        self.constant_exprs_ = out
        return out

    def build_constant_conjectures(self, *, tol: float = 0.0, group_per_hypothesis: bool = False) -> List[Conjecture]:
        """
        Create conjectures from constants found in `find_constant_exprs`.

        Parameters
        ----------
        tol : float, default=0.0
            Tolerance used when forming `Eq(expr(col), constant, tol=tol)`.
        group_per_hypothesis : bool, default=False
            If True: one conjecture per hypothesis with `AllOf([...equalities...])`.
            If False: one conjecture per (hypothesis, column).

        Returns
        -------
        list[Conjecture]
            Equality conjectures conditioned on each hypothesis. The result is
            passed through `_compress_conjectures` to merge identical conditions.

        See Also
        --------
        characterize_constant_classes : Compares “equalities hold” vs class predicates.
        """
        if not hasattr(self, "nonredundant_conjunctions_"):
            self.enumerate_conjunctions()
            self.find_redundant_conjunctions()

        const_map = self.find_constant_exprs(tol=tol)
        hyp_lookup = {name: pred for name, pred in self.nonredundant_conjunctions_}

        conjs: List[Conjecture] = []
        for hyp_name, pairs in const_map.items():
            if not pairs:
                continue
            cond = hyp_lookup[hyp_name]
            if group_per_hypothesis:
                parts = [Eq(self.exprs[col], val, tol=tol) for (col, val) in pairs]
                conjs.append(Conjecture(relation=AllOf(parts), condition=cond, name=f"Const[{hyp_name}]"))
            else:
                for col, val in pairs:
                    conjs.append(Conjecture(relation=Eq(self.exprs[col], val, tol=tol),
                                            condition=cond, name=f"{col} const | {hyp_name}"))
        return self._compress_conjectures(conjs)

    # ------------------------------ Class Characterization ---------------------

    def characterize_constant_classes(
        self,
        *,
        tol: float = 0.0,
        group_per_hypothesis: bool = True,
        limit: int | None = None,
        recompute: bool = False,
    ) -> dict:
        """
        Compare “equalities hold” (A) with the hypothesis class predicate (B) for grouped constants.

        For each grouped constant-equality conjecture `C: AllOf([...]) | cond`:
        define
            A := Where(row-wise test that all equalities in `C.relation` hold),
            B := `cond` (the class predicate).
        We record:
            - equivalences:  A ≡ B
            - inclusions_AB: A ⊆ B
            - inclusions_BA: B ⊆ A
            - non_equivalences with violation diagnostics
        and also emit `(cond ⇒ equalities)` as a Conjecture.

        Parameters
        ----------
        tol : float, default=0.0
            Tolerance used to build equalities when grouping.
        group_per_hypothesis : bool, default=True
            Ensures one `AllOf` per hypothesis; required for A/B comparison.
        limit : int or None, default=None
            If provided, only analyze the first `limit` grouped conjectures.
        recompute : bool, default=False
            If True, re-run enumeration + redundancy before characterization.

        Returns
        -------
        dict
            Keys:
            - 'equivalences' : list[ClassEquivalence]
            - 'inclusions_AB' : list[ClassInclusion]
            - 'inclusions_BA' : list[ClassInclusion]
            - 'non_equivalences' : list[dict]
            - 'cond_implies_equalities_conjectures' : list[Conjecture]
        """
        if recompute or not hasattr(self, "nonredundant_conjunctions_"):
            self.enumerate_conjunctions()
            self.find_redundant_conjunctions()

        grouped = self.build_constant_conjectures(tol=tol, group_per_hypothesis=group_per_hypothesis)
        if limit is not None:
            grouped = grouped[:max(0, int(limit))]

        equivalences: List[ClassEquivalence] = []
        inclusions_AB: List[ClassInclusion] = []
        inclusions_BA: List[ClassInclusion] = []
        cond_implies_equalities_conjs: List[Conjecture] = []
        non_equivalences: List[dict] = []

        for cj in grouped:
            cond = cj.condition or TRUE
            rel = cj.relation
            rel_pred = self._relation_to_predicate(rel)

            A_in_B = ClassInclusion(rel_pred, cond)  # equalities ⇒ class
            B_in_A = ClassInclusion(cond, rel_pred)  # class ⇒ equalities
            EQ     = ClassEquivalence(rel_pred, cond)

            eq_holds = EQ.holds_all(self.df)
            a_holds  = A_in_B.holds_all(self.df)
            b_holds  = B_in_A.holds_all(self.df)

            if eq_holds:
                equivalences.append(EQ)
            else:
                if a_holds:
                    inclusions_AB.append(A_in_B)
                if b_holds:
                    inclusions_BA.append(B_in_A)
                non_equivalences.append({
                    "equivalence": EQ,
                    "A_subset_B": {
                        "holds": a_holds,
                        "violations": A_in_B.violation_count(self.df),
                        "obj": A_in_B,
                    },
                    "B_subset_A": {
                        "holds": b_holds,
                        "violations": B_in_A.violation_count(self.df),
                        "obj": B_in_A,
                    },
                })

            cond_implies_equalities_conjs.append(
                self._conjecture_cond_implies_equalities(cond, rel, tol=tol)
            )

        self.class_equivalences_ = equivalences
        self.class_inclusions_AB_ = inclusions_AB
        self.class_inclusions_BA_ = inclusions_BA
        self.class_non_equivalences_ = non_equivalences
        self.class_cond_implies_equalities_ = cond_implies_equalities_conjs

        return {
            "equivalences": equivalences,
            "inclusions_AB": inclusions_AB,
            "inclusions_BA": inclusions_BA,
            "non_equivalences": non_equivalences,
            "cond_implies_equalities_conjectures": cond_implies_equalities_conjs,
        }

    def generate_ratio_bounds_on_base(
        self,
        *,
        min_support: float = 0.01,
        positive_denominator: bool = True,
        require_finite: bool = True,
        rationalize: bool = True,
        max_denom: int = 30,
    ) -> tuple[pd.DataFrame, list[Conjecture]]:
        """
        Compute ratio bounds over the base hypothesis for all ordered pairs (inv1, inv2).

        For each pair with sufficient support:
            `cmin ≤ inv1/inv2 ≤ cmax`  (optionally with `inv2 > 0`),
        emit conjectures:
            base ⇒ `inv1 ≥ cmin·inv2` and base ⇒ `inv1 ≤ cmax·inv2`.

        Parameters
        ----------
        min_support : float, default=0.01
            Minimum fraction of base rows required to accept a pair.
        positive_denominator : bool, default=True
            If True, restrict to rows with `inv2 > 0`.
        require_finite : bool, default=True
            If True, ignore rows where `inv1` or `inv2` is NaN/±inf.
        rationalize : bool, default=True
            Snap `cmin/cmax` to nearby small-denominator rationals for display.
        max_denom : int, default=30
            Maximum denominator used when rationalizing constants.

        Returns
        -------
        summary : pd.DataFrame
            Columns: `inv1, inv2, support, cmin, cmax, n_rows`, sorted by support.
        conjectures : list[Conjecture]
            Two conjectures (lower/upper) per accepted pair, compressed by condition.

        Notes
        -----
        - Uses `Expr.eval(df)` and cached base-mask. Stores `simple_ratio_bounds`.
        - If `base_hypothesis` is `TRUE`, emitted conjectures carry `condition=None`.
        """
        # Base mask (TRUE allowed)
        base_mask = self._mask_cached(self.base_hypothesis if self.base_hypothesis is not TRUE else Where(lambda d: np.ones(len(d), dtype=bool)))

        # Collect numeric columns from the expr side
        num_cols = list(self.expr_cols)

        rows = []
        conjs: list[Conjecture] = []

        n_base = int(base_mask.sum())
        if n_base == 0:
            return pd.DataFrame(columns=["inv1","inv2","support","cmin","cmax","n_rows"]), conjs

        # Pre-evaluate all expressions once on df for speed
        eval_cache: Dict[str, pd.Series] = {c: self.exprs[c].eval(self.df) for c in num_cols}

        for i, inv1 in enumerate(num_cols):
            s1 = eval_cache[inv1]
            if require_finite:
                s1 = s1.replace([np.inf, -np.inf], np.nan)
            for inv2 in num_cols:
                if inv1 == inv2:
                    continue
                s2 = eval_cache[inv2]
                if require_finite:
                    s2 = s2.replace([np.inf, -np.inf], np.nan)

                # Build mask for this pair under base hypothesis
                m = base_mask.copy()
                # finite/nonnull
                m &= s1.notna().to_numpy(dtype=bool, copy=False)
                m &= s2.notna().to_numpy(dtype=bool, copy=False)
                if positive_denominator:
                    m &= (s2.to_numpy(dtype=float, copy=False) > 0.0)

                n = int(m.sum())
                if n == 0:
                    continue
                support = n / n_base
                if support < min_support:
                    continue

                # Compute ratio stats
                r = (s1.to_numpy(dtype=float, copy=False)[m]) / (s2.to_numpy(dtype=float, copy=False)[m])
                cmin = np.min(r)
                cmax = np.max(r)

                cmin_nice = _const_for_expr(cmin, rationalize=rationalize, max_denom=max_denom)
                cmax_nice = _const_for_expr(cmax, rationalize=rationalize, max_denom=max_denom)

                rows.append({
                    "inv1": inv1, "inv2": inv2,
                    "support": support, "cmin": cmin_nice, "cmax": cmax_nice, "n_rows": n
                })

                # Build conjectures: base ⇒ inv1 >= cmin·inv2  and base ⇒ inv1 <= cmax·inv2
                lhs = self.exprs[inv1]
                rhs_lo = self.exprs[inv2] * cmin_nice  # Expr math
                rhs_hi = self.exprs[inv2] * cmax_nice

                # lower and upper
                cj_lo = Conjecture(
                    relation=Ge(lhs, rhs_lo),
                    condition=self.base_hypothesis if self.base_hypothesis is not TRUE else None,
                    name=f"{inv1} >= {cmin_nice}·{inv2} | {self.base_hypothesis_name}"
                )
                cj_hi = Conjecture(
                    relation=Le(lhs, rhs_hi),
                    condition=self.base_hypothesis if self.base_hypothesis is not TRUE else None,
                    name=f"{inv1} <= {cmax_nice}·{inv2} | {self.base_hypothesis_name}"
                )
                conjs.extend([cj_lo, cj_hi])

        summary = pd.DataFrame(rows, columns=["inv1","inv2","support","cmin","cmax","n_rows"]).sort_values(
            ["support","inv1","inv2"], ascending=[False, True, True]
        ).reset_index(drop=True)

        # NEW: keep raw (atomic) list for reporting
        self.simple_ratio_bounds_raw = list(conjs)
        # Optional: compress conjectures that share identical (TRUE) condition
        conjs = self._compress_conjectures(conjs)
        self.simple_ratio_bounds = conjs
        return summary, conjs

    def analyze_ratio_bounds_on_base(
        self,
        *,
        min_support: float = 0.01,
        positive_denominator: bool = True,
        require_finite: bool = True,
        rationalize: bool = True,
        max_denom: int = 30,
        # tightness tolerance (use small atol for integer invariants; rtol=0 keeps it exact)
        touch_atol: float = 0.0,
        touch_rtol: float = 0.0,
        emit_conjectures: bool = True,
    ) -> tuple[pd.DataFrame, list[Conjecture]]:
        """
        Like `generate_ratio_bounds_on_base`, but also report **touch counts/rates**
        for equality at both extremes and (optionally) emit the lower/upper conjectures.

        Touch definition
        ----------------
        For accepted rows of a pair (inv1, inv2),
            - lower touch: `inv1 ≈ cmin * inv2`,
            - upper touch: `inv1 ≈ cmax * inv2`,
        with closeness tested by `|a-b| ≤ atol + rtol*|b|`.

        Parameters
        ----------
        min_support, positive_denominator, require_finite, rationalize, max_denom :
            Passed through as in `generate_ratio_bounds_on_base`.
        touch_atol : float, default=0.0
            Absolute tolerance for touch tests (good to use a small value for integer invariants).
        touch_rtol : float, default=0.0
            Relative tolerance for touch tests.
        emit_conjectures : bool, default=True
            If True, emits (compressed) conjectures for the accepted pairs.

        Returns
        -------
        summary : pd.DataFrame
            Columns: `inv1, inv2, support, n_rows, cmin, cmax, touch_lower, touch_upper,
            touch_lower_rate, touch_upper_rate`, sorted by touch rates then support.
        conjectures : list[Conjecture]
            (Possibly empty) list of emitted conjectures if `emit_conjectures=True`.
        """
        # Base mask
        base_mask = self._mask_cached(self.base_hypothesis if self.base_hypothesis is not TRUE else Where(lambda d: np.ones(len(d), dtype=bool)))
        n_base = int(base_mask.sum())
        if n_base == 0:
            cols = ["inv1","inv2","support","n_rows","cmin","cmax","touch_lower","touch_upper","touch_lower_rate","touch_upper_rate"]
            return pd.DataFrame(columns=cols), []

        # Numeric expr cols
        num_cols = list(self.expr_cols)
        eval_cache: Dict[str, pd.Series] = {c: self.exprs[c].eval(self.df) for c in num_cols}

        rows = []
        conjs: list[Conjecture] = []

        def _nice_const(x: float) -> float:
            if not rationalize or not np.isfinite(x):
                return float(x)
            try:
                from fractions import Fraction
                fr = Fraction(x).limit_denominator(max_denom)
                if abs(float(fr) - x) <= 1e-12:
                    return float(fr)
            except Exception:
                pass
            return float(x)

        for inv1 in num_cols:
            s1 = eval_cache[inv1]
            if require_finite:
                s1 = s1.replace([np.inf, -np.inf], np.nan)
            a1 = s1.to_numpy(dtype=float, copy=False)

            for inv2 in num_cols:
                if inv1 == inv2:
                    continue
                s2 = eval_cache[inv2]
                if require_finite:
                    s2 = s2.replace([np.inf, -np.inf], np.nan)
                a2 = s2.to_numpy(dtype=float, copy=False)

                m = base_mask.copy()
                m &= np.isfinite(a1)
                m &= np.isfinite(a2)
                if positive_denominator:
                    m &= (a2 > 0.0)

                n = int(m.sum())
                if n == 0:
                    continue
                support = n / n_base
                if support < min_support:
                    continue

                r = a1[m] / a2[m]
                cmin = float(np.min(r))
                cmax = float(np.max(r))
                cmin_n = _nice_const(cmin)
                cmax_n = _nice_const(cmax)

                # touches
                tgt_lo = cmin * a2[m]
                tgt_hi = cmax * a2[m]
                tl_mask = _isclose_vec(a1[m], tgt_lo, atol=touch_atol, rtol=touch_rtol)
                tu_mask = _isclose_vec(a1[m], tgt_hi, atol=touch_atol, rtol=touch_rtol)
                touch_lower = int(tl_mask.sum())
                touch_upper = int(tu_mask.sum())

                rows.append({
                    "inv1": inv1,
                    "inv2": inv2,
                    "support": support,
                    "n_rows": n,
                    "cmin": cmin_n,
                    "cmax": cmax_n,
                    "touch_lower": touch_lower,
                    "touch_upper": touch_upper,
                    "touch_lower_rate": touch_lower / n if n else 0.0,
                    "touch_upper_rate": touch_upper / n if n else 0.0,
                })

                if emit_conjectures:
                    cond = self.base_hypothesis if self.base_hypothesis is not TRUE else None
                    lhs = self.exprs[inv1]
                    # keep numeric so Expr math composes; your printers handle prettiness
                    cj_lo = Conjecture(relation=Le(self.exprs[inv2] * cmin_n, lhs),
                                    condition=cond, name=f"{inv1} ≥ {cmin_n}·{inv2} | {self.base_hypothesis_name}")
                    cj_hi = Conjecture(relation=Le(lhs, self.exprs[inv2] * cmax_n),
                                    condition=cond, name=f"{inv1} ≤ {cmax_n}·{inv2} | {self.base_hypothesis_name}")
                    conjs.extend([cj_lo, cj_hi])

        cols = ["inv1","inv2","support","n_rows","cmin","cmax","touch_lower","touch_upper","touch_lower_rate","touch_upper_rate"]
        summary = pd.DataFrame(rows, columns=cols).sort_values(
            ["touch_lower_rate","touch_upper_rate","support"], ascending=[False, False, False]
        ).reset_index(drop=True)

        if emit_conjectures:
            conjs = self._compress_conjectures(conjs)

        return summary, conjs

    def spawn_equality_classes_from_ratio_row(
        self,
        row: pd.Series,
        *,
        which: str = "auto",   # "lower" | "upper" | "auto" (choose the better touch rate)
        tol: float = 0.0,
    ) -> tuple[str, Predicate]:
        """
        Convert a summary row from `analyze_ratio_bounds_on_base` into an equality-class predicate.

        The class is either `inv1 == cmin·inv2` (lower) or `inv1 == cmax·inv2` (upper).
        If `which="auto"`, choose the side with higher touch rate.

        Parameters
        ----------
        row : pd.Series
            One row from the summary returned by `analyze_ratio_bounds_on_base`.
        which : {"lower", "upper", "auto"}, default="auto"
            Which equality side to use.
        tol : float, default=0.0
            Tolerance forwarded to `Eq(..., tol=tol)`.

        Returns
        -------
        (name, predicate) : tuple[str, Predicate]
            Pretty name and the corresponding equality-class predicate.

        Raises
        ------
        ValueError
            If `which` is not one of {"lower","upper","auto"}.
        """
        inv1, inv2 = row["inv1"], row["inv2"]
        lo_rate, up_rate = float(row["touch_lower_rate"]), float(row["touch_upper_rate"])
        side = which
        if which == "auto":
            side = "lower" if lo_rate >= up_rate else "upper"

        if side == "lower":
            const = float(row["cmin"])
            rel = Eq(self.exprs[inv1], self.exprs[inv2] * const, tol=tol)
            name = f"{inv1} = {const}·{inv2}"
        elif side == "upper":
            const = float(row["cmax"])
            rel = Eq(self.exprs[inv1], self.exprs[inv2] * const, tol=tol)
            name = f"{inv1} = {const}·{inv2}"
        else:
            raise ValueError("which must be 'lower', 'upper', or 'auto'.")

        pred = self._relation_to_predicate(rel, name=name)
        return name, pred

    def analyze_ratio_bounds_on_condition(
        self,
        condition: Predicate,
        *,
        min_support: float = 0.01,
        positive_denominator: bool = True,
        require_finite: bool = True,
        rationalize: bool = True,
        max_denom: int = 30,
        touch_atol: float = 0.0,
        touch_rtol: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute ratio-bound statistics exactly as in `analyze_ratio_bounds_on_base`,
        but restricted to a user-specified `condition` predicate. Does not emit conjectures.

        Parameters
        ----------
        condition : Predicate
            Predicate restricting the domain for the analysis.
        min_support, positive_denominator, require_finite, rationalize, max_denom :
            Same semantics as in `analyze_ratio_bounds_on_base`.
        touch_atol : float, default=0.0
            Absolute tolerance for touch tests.
        touch_rtol : float, default=0.0
            Relative tolerance for touch tests.

        Returns
        -------
        pd.DataFrame
            Summary with columns:
            `inv1, inv2, support, n_rows, cmin, cmax, touch_lower, touch_upper,
             touch_lower_rate, touch_upper_rate`, sorted by touch rates then support.
        """
        cond_mask = self._mask_cached(condition)
        n_cond = int(cond_mask.sum())
        if n_cond == 0:
            return pd.DataFrame(columns=["inv1","inv2","support","n_rows","cmin","cmax","touch_lower","touch_upper","touch_lower_rate","touch_upper_rate"])

        num_cols = list(self.expr_cols)
        eval_cache: Dict[str, pd.Series] = {c: self.exprs[c].eval(self.df) for c in num_cols}
        rows = []

        def _nice_const(x: float) -> float:
            if not rationalize or not np.isfinite(x):
                return float(x)
            try:
                from fractions import Fraction
                fr = Fraction(x).limit_denominator(max_denom)
                if abs(float(fr) - x) <= 1e-12:
                    return float(fr)
            except Exception:
                pass
            return float(x)

        for inv1 in num_cols:
            a1 = eval_cache[inv1].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float, copy=False) if require_finite else eval_cache[inv1].to_numpy(dtype=float, copy=False)
            for inv2 in num_cols:
                if inv1 == inv2:
                    continue
                a2 = eval_cache[inv2].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float, copy=False) if require_finite else eval_cache[inv2].to_numpy(dtype=float, copy=False)

                m = cond_mask.copy()
                m &= np.isfinite(a1)
                m &= np.isfinite(a2)
                if positive_denominator:
                    m &= (a2 > 0.0)

                n = int(m.sum())
                if n == 0:
                    continue
                support = n / n_cond
                if support < min_support:
                    continue

                r = a1[m] / a2[m]
                cmin = float(np.min(r)); cmax = float(np.max(r))
                cmin_n = _nice_const(cmin); cmax_n = _nice_const(cmax)

                tl_mask = _isclose_vec(a1[m], cmin * a2[m], atol=touch_atol, rtol=touch_rtol)
                tu_mask = _isclose_vec(a1[m], cmax * a2[m], atol=touch_atol, rtol=touch_rtol)

                rows.append({
                    "inv1": inv1, "inv2": inv2, "support": support, "n_rows": n,
                    "cmin": cmin_n, "cmax": cmax_n,
                    "touch_lower": int(tl_mask.sum()), "touch_upper": int(tu_mask.sum()),
                    "touch_lower_rate": float(int(tl_mask.sum())/n), "touch_upper_rate": float(int(tu_mask.sum())/n),
                })

        return pd.DataFrame(rows).sort_values(
            ["touch_lower_rate","touch_upper_rate","support"], ascending=[False, False, False]
        ).reset_index(drop=True)

    def register_hypotheses(self, hyps: list[tuple[str, Predicate]]):
        """
        Register a batch of `(name, predicate)` hypotheses and normalize them by
        mask-equivalence and dominance.

        Process
        -------
        1) **Equivalence merging**: hypotheses with identical masks are grouped,
           and a representative is chosen (max support, then shortest name).
        2) **Dominance pruning**: if `mask(A) ⊂ mask(B)`, drop A (recorded as dominated).

        Parameters
        ----------
        hyps : list[tuple[str, Predicate]]
            Hypotheses to normalize.

        Returns
        -------
        kept : list[tuple[str, Predicate]]
            Representatives after equivalence and dominance pruning.
        merged : list[list[tuple[str, Predicate]]]
            Groups of equivalent hypotheses that were merged.
        dominated : list[tuple[str, Predicate, str]]
            Triples `(name, predicate, by_name)` for dominated items.

        Notes
        -----
        This does not mutate internal hypothesis stores; it is a pure helper
        to canonicalize an external set of hypotheses.
        """
        # compute mask signatures
        recs = []
        for name, pred in hyps:
            m = self._mask_cached(pred)
            recs.append({"name": name, "pred": pred, "mask": m, "sig": m.tobytes(), "support": int(m.sum())})

        # group equivalents by signature
        by_sig = {}
        merged_groups = []
        for r in recs:
            if r["sig"] not in by_sig:
                by_sig[r["sig"]] = [r]
            else:
                by_sig[r["sig"]].append(r)

        kept = []
        for sig, group in by_sig.items():
            # choose a stable representative: max support (tie-break by shortest name)
            rep = sorted(group, key=lambda t: (-t["support"], len(t["name"]), t["name"]))[0]
            kept.append((rep["name"], rep["pred"]))
            if len(group) > 1:
                merged_groups.append([(g["name"], g["pred"]) for g in group])

        # dominance: A dominated by B if mask(A) ⊆ mask(B) and not equal
        dominated = []
        kept_recs = [{"name": n, "pred": p, "mask": self._mask_cached(p)} for (n, p) in kept]
        for i, ri in enumerate(kept_recs):
            for j, rj in enumerate(kept_recs):
                if i == j:
                    continue
                mi, mj = ri["mask"], rj["mask"]
                if mi is mj:
                    continue
                # strict subset?
                if mi.all() or (mi.sum() == 0 and mj.sum() == 0):
                    # handled by signature already
                    continue
                if (mi & (~mj)).sum() == 0 and (mj & (~mi)).sum() > 0:
                    dominated.append((ri["name"], ri["pred"], rj["name"]))

        # remove dominated from kept
        dominated_names = {n for (n, _, __) in dominated}
        kept = [(n, p) for (n, p) in kept if n not in dominated_names]

        return kept, merged_groups, dominated

    def use_top_equality_classes_as_hypotheses(
        self,
        *,
        # mining controls
        min_support: float = 0.05,
        min_touch_count: int = 1,
        min_touch_rate: float = 0.10,
        which: str = "auto",          # "lower" | "upper" | "auto"
        touch_atol: float = 0.0,
        touch_rtol: float = 0.0,
        rationalize: bool = True,
        max_denom: int = 30,
        top_k: int | None = 20,
        # class construction
        conjoin_base: bool = True,
        # follow-on analysis for each derived class
        analyze_bounds_per_class: bool = False,
        per_class_min_support: float = 0.05,
    ) -> dict:
        """
        Promote high-touch ratio equalities into hypothesis classes, normalize them,
        and (optionally) analyze ratio bounds within each new class.

        Pipeline
        --------
        1) Get ratio summary on the base hypothesis (no conjecture emission).
        2) Filter rows by support / touch thresholds.
        3) Convert each selected row into an equality-class predicate via
        `spawn_equality_classes_from_ratio_row(...)`.
        4) Optionally conjoin with the base hypothesis.
        5) Canonicalize with `register_hypotheses(...)` (equivalence merge + dominance).
        6) (Optional) For each kept class, run `analyze_ratio_bounds_on_condition(...)`.

        Parameters
        ----------
        min_support : float, default=0.05
            Minimum fraction of base rows required to consider a pair.
        min_touch_count : int, default=1
            Minimum # of exact/equal-within-tol hits on either side to be eligible.
        min_touch_rate : float, default=0.10
            Minimum max(touch_lower_rate, touch_upper_rate) to be eligible.
        which : {"lower","upper","auto"}, default="auto"
            Which equality side to instantiate for each row ("auto" picks the better rate).
        touch_atol, touch_rtol : float
            Touch tolerances forwarded to the base analysis.
        rationalize : bool, default=True
            Rationalize cmin/cmax in the summary for readability.
        max_denom : int, default=30
            Max denominator for rationalization.
        top_k : int or None, default=20
            Keep at most this many candidate equality-classes after sorting by
            (max touch rate, support, n_rows). Use None to keep all.
        conjoin_base : bool, default=True
            If True and base != TRUE, conjoin each equality-class with the base.
        analyze_bounds_per_class : bool, default=False
            If True, compute per-class ratio summaries with
            `analyze_ratio_bounds_on_condition(...)`.
        per_class_min_support : float, default=0.05
            Minimum support used in the per-class analyses.

        Returns
        -------
        dict with keys:
            "candidates_df" : pd.DataFrame
                The filtered + sorted rows from the base ratio summary used to spawn classes.
            "kept" : list[(name, Predicate)]
                Representatives after equivalence/dominance normalization.
            "merged" : list[list[(name, Predicate)]]
                Groups that were merged as equivalent.
            "dominated" : list[(name, Predicate, by_name)]
                Items removed by dominance.
            "per_class_summaries" : list[tuple[str, pd.DataFrame]]
                (Only if analyze_bounds_per_class=True) list of (hyp_name, summary_df).

        Notes
        -----
        - This does *not* mutate your existing nonredundant conjunctions; it returns a
        clean set you can register elsewhere if you like.
        - Names include a ' | base' suffix when conjoining with a non-TRUE base.
        """
        # 1) ratio summary on base (no conjecture emission)
        summary, _ = self.analyze_ratio_bounds_on_base(
            min_support=min_support,
            positive_denominator=True,
            require_finite=True,
            rationalize=rationalize,
            max_denom=max_denom,
            touch_atol=touch_atol,
            touch_rtol=touch_rtol,
            emit_conjectures=False,
        )

        if summary.empty:
            return {
                "candidates_df": summary,
                "kept": [],
                "merged": [],
                "dominated": [],
                "per_class_summaries": [],
            }

        # 2) filter by touch stats & support
        # compute a single "best" touch rate per row
        best_rate = np.maximum(summary["touch_lower_rate"].astype(float),
                            summary["touch_upper_rate"].astype(float))
        best_count = np.maximum(summary["touch_lower"].astype(int),
                                summary["touch_upper"].astype(int))

        m = (
            (summary["support"].astype(float) >= float(min_support)) &
            (best_rate >= float(min_touch_rate)) &
            (best_count >= int(min_touch_count))
        )
        cand = summary.loc[m].copy()

        if cand.empty:
            return {
                "candidates_df": cand,
                "kept": [],
                "merged": [],
                "dominated": [],
                "per_class_summaries": [],
            }

        # sort by (best touch rate, support, n_rows)
        cand["_best_rate"] = np.maximum(cand["touch_lower_rate"].astype(float),
                                        cand["touch_upper_rate"].astype(float))
        cand.sort_values(
            ["_best_rate", "support", "n_rows"],
            ascending=[False, False, False],
            inplace=True
        )
        if top_k is not None:
            cand = cand.head(int(top_k)).copy()

        # 3) & 4) spawn equality predicates (and conjoin with base if requested)
        hyps: list[tuple[str, Predicate]] = []
        for _, row in cand.iterrows():
            name, pred = self.spawn_equality_classes_from_ratio_row(
                row, which=which, tol=0.0
            )
            if conjoin_base and (self.base_hypothesis is not TRUE):
                pred = self.base_hypothesis & pred
                name = f"{name} | {self.base_hypothesis_name}"
            hyps.append((name, pred))

        if not hyps:
            return {
                "candidates_df": cand.drop(columns=["_best_rate"], errors="ignore"),
                "kept": [],
                "merged": [],
                "dominated": [],
                "per_class_summaries": [],
            }

        # 5) normalize (equivalence + dominance)
        kept, merged, dominated = self.register_hypotheses(hyps)

        # 6) optional per-class ratio analysis
        per_class_summaries: list[tuple[str, pd.DataFrame]] = []
        if analyze_bounds_per_class and kept:
            for (hname, hpred) in kept:
                s = self.analyze_ratio_bounds_on_condition(
                    hpred,
                    min_support=per_class_min_support,
                    positive_denominator=True,
                    require_finite=True,
                    rationalize=rationalize,
                    max_denom=max_denom,
                    touch_atol=touch_atol,
                    touch_rtol=touch_rtol,
                )
                per_class_summaries.append((hname, s))

        return {
            "candidates_df": cand.drop(columns=["_best_rate"], errors="ignore"),
            "kept": kept,
            "merged": merged,
            "dominated": dominated,
            "per_class_summaries": per_class_summaries,
        }

    def _integrate_derived_hypotheses(self, kept: list[tuple[str, Predicate]]) -> None:
        """
        Merge a list of (name, Predicate) into self.nonredundant_conjunctions_ by
        mask-equivalence and dominance, then refresh the sorted view.
        """
        if not kept:
            # still record an empty list for visibility
            self.derived_conjunctions_ = []
            return

        # Keep a copy of what's there + derived
        existing = getattr(self, "nonredundant_conjunctions_", [])
        merged_in = list(existing) + list(kept)

        # Normalize using the same rule as external hypotheses
        kept_all, _merged_groups, _dominated = self.register_hypotheses(merged_in)

        # Store (for inspection) both the raw derived set and the final merged set
        self.derived_conjunctions_ = list(kept)
        self.nonredundant_conjunctions_ = kept_all

        # Rebuild the sorted list by generality (support size)
        self.sort_conjunctions_by_generality(ascending=False)

    def add_absdiff_exprs_for_meaningfully_incomparable(
        self,
        condition: Predicate | None = None,
        *,
        use_base_if_none: bool = True,
        require_finite: bool = True,
        # incomparability thresholds
        min_support: float = 0.10,      # fraction of domain rows the pair covers
        min_side_rate: float = 0.10,    # each side’s rate ≥ this
        min_side_count: int = 5,        # and each side’s count ≥ this
        # gap-size thresholds
        min_median_gap: float = 0.5,    # median(|x-y|) ≥ this  OR
        min_mean_gap: float = 0.5,      # mean(|x-y|) ≥ this
        # naming
        name_style: str = "abs_x_minus_y",   # "abs_x_minus_y" or "pipe_abs(x-y)"
        prefix: str = "abs_",
        suffix: str = "",
        # registry behavior
        overwrite_existing: bool = False,    # if False, make a unique name if collision
    ) -> pd.DataFrame:
        """
        Identify unordered numeric-invariant pairs (x,y) that are *meaningfully incomparable*
        and register derived DSL Exprs for

            |x - y|,   min(x, y),   max(x, y)

        in `self.exprs` (no DataFrame columns added).

        Returns a summary DataFrame with selection diagnostics and the
        registered expr names.
        """

        # ---- domain mask ----
        if condition is not None:
            domain_mask = self._mask_cached(condition)
        elif use_base_if_none and (self.base_hypothesis is not TRUE):
            domain_mask = self._mask_cached(self.base_hypothesis)
        else:
            domain_mask = np.ones(len(self.df), dtype=bool)

        domain_rows = int(domain_mask.sum())
        if domain_rows == 0:
            return pd.DataFrame(columns=[
                "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
                "rate_lt","rate_gt","rate_eq","support",
                "median_gap","mean_gap","selected",
                "abs_expr_name","min_expr_name","max_expr_name",
            ])

        # ---- numeric exprs & cached eval ----
        num_cols = list(self.expr_cols)  # NOTE: only raw df-backed columns here
        eval_cache: Dict[str, pd.Series] = {c: self.exprs[c].eval(self.df) for c in num_cols}

        def _propose_abs_name(x: str, y: str) -> str:
            if name_style == "pipe_abs(x-y)":
                # Pretty display name; OK to use as key (Expr is what matters)
                return f"| {x} - {y} |"
            return f"{prefix}{x}_minus_{y}{suffix}"

        def _propose_min_name(x: str, y: str) -> str:
            return f"min_{x}_{y}"

        def _propose_max_name(x: str, y: str) -> str:
            return f"max_{x}_{y}"

        def _unique_name(base: str) -> str:
            if overwrite_existing or (base not in self.exprs):
                return base
            i = 2
            while f"{base}__{i}" in self.exprs:
                i += 1
            return f"{base}__{i}"

        rows = []

        # preconvert to numpy floats once
        arrays: Dict[str, np.ndarray] = {}
        finite_masks: Dict[str, np.ndarray] = {}
        for c in num_cols:
            s = eval_cache[c]
            if require_finite:
                s = s.replace([np.inf, -np.inf], np.nan)
            a = s.to_numpy(dtype=float, copy=False)
            arrays[c] = a
            finite_masks[c] = np.isfinite(a) if require_finite else np.ones_like(a, dtype=bool)

        for i in range(len(num_cols)):
            x = num_cols[i]; ax = arrays[x]; fx = finite_masks[x]
            for j in range(i+1, len(num_cols)):
                y = num_cols[j]; ay = arrays[y]; fy = finite_masks[y]

                m = domain_mask & fx & fy
                n = int(m.sum())
                if n == 0:
                    continue

                axm = ax[m]; aym = ay[m]
                lt = axm < aym
                gt = axm > aym
                eq = ~(lt | gt)

                n_lt = int(lt.sum())
                n_gt = int(gt.sum())
                n_eq = int(eq.sum())

                rate_lt = n_lt / n
                rate_gt = n_gt / n
                rate_eq = n_eq / n
                support = n / domain_rows

                both_sides = (n_lt > 0) and (n_gt > 0)
                side_ok = (
                    min(rate_lt, rate_gt) >= float(min_side_rate)
                    and min(n_lt, n_gt) >= int(min_side_count)
                )
                support_ok = (support >= float(min_support))

                gaps = np.abs(axm - aym)
                median_gap = float(np.nanmedian(gaps)) if n > 0 else 0.0
                mean_gap   = float(np.nanmean(gaps))   if n > 0 else 0.0
                gap_ok = (median_gap >= float(min_median_gap)) or (mean_gap >= float(min_mean_gap))

                selected = bool(both_sides and side_ok and support_ok and gap_ok)

                abs_expr_name = None
                min_expr_name = None
                max_expr_name = None

                if selected:
                    base_x = self.exprs[x]
                    base_y = self.exprs[y]

                    # --- 1. |x - y| ---
                    abs_expr = abs_(base_x - base_y)
                    proposed_abs = _propose_abs_name(x, y)
                    key_abs = proposed_abs if overwrite_existing else _unique_name(proposed_abs)
                    self.exprs[key_abs] = abs_expr

                    # Track synthetic exprs
                    synthetic = getattr(self, "synthetic_expr_names_", set())
                    synthetic.add(key_abs)

                    abs_expr_name = key_abs

                    # --- 2. min(x, y) ---
                    min_expr = min_(base_x, base_y)
                    proposed_min = _propose_min_name(x, y)
                    key_min = proposed_min if overwrite_existing else _unique_name(proposed_min)
                    self.exprs[key_min] = min_expr
                    synthetic.add(key_min)
                    min_expr_name = key_min

                    # --- 3. max(x, y) ---
                    max_expr = max_(base_x, base_y)
                    proposed_max = _propose_max_name(x, y)
                    key_max = proposed_max if overwrite_existing else _unique_name(proposed_max)
                    self.exprs[key_max] = max_expr
                    synthetic.add(key_max)
                    max_expr_name = key_max

                    self.synthetic_expr_names_ = synthetic

                rows.append({
                    "inv1": x, "inv2": y,
                    "n_rows": n,
                    "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
                    "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                    "support": support,
                    "median_gap": median_gap, "mean_gap": mean_gap,
                    "selected": selected,
                    "abs_expr_name": abs_expr_name,
                    "min_expr_name": min_expr_name,
                    "max_expr_name": max_expr_name,
                })

        out = pd.DataFrame(rows, columns=[
            "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
            "rate_lt","rate_gt","rate_eq","support",
            "median_gap","mean_gap","selected",
            "abs_expr_name","min_expr_name","max_expr_name",
        ])

        # ── Rank by mean gap and store the top abs-diff Exprs ──────────────────────
        out_sorted = out.sort_values("mean_gap", ascending=False).reset_index(drop=True)

        # Keep only abs Exprs actually registered (selected==True and abs_expr_name not None)
        selected_abs_exprs = [
            (row["abs_expr_name"], self.exprs[row["abs_expr_name"]])
            for _, row in out_sorted.iterrows()
            if row["selected"] and isinstance(row["abs_expr_name"], str)
        ]

        # Save the top-N (e.g., all or top 20) to self.abs_exprs (backwards-compatible)
        top_n = min(20, len(selected_abs_exprs))
        self.abs_exprs = selected_abs_exprs[:top_n]

        # Optionally: you could also stash min/max somewhere if you want, e.g.:
        # self.minmax_exprs = [
        #     (row["min_expr_name"], self.exprs[row["min_expr_name"]],
        #      row["max_expr_name"], self.exprs[row["max_expr_name"]])
        #     for _, row in out_sorted.iterrows()
        #     if row["selected"]
        #        and isinstance(row["min_expr_name"], str)
        #        and isinstance(row["max_expr_name"], str)
        # ]

        # For inspection
        self.last_meaningful_incomparability_exprs_ = out_sorted
        return out_sorted

    # def add_absdiff_exprs_for_meaningfully_incomparable(
    #     self,
    #     condition: Predicate | None = None,
    #     *,
    #     use_base_if_none: bool = True,
    #     require_finite: bool = True,
    #     # incomparability thresholds
    #     min_support: float = 0.10,      # fraction of domain rows the pair covers
    #     min_side_rate: float = 0.10,    # each side’s rate ≥ this
    #     min_side_count: int = 5,        # and each side’s count ≥ this
    #     # gap-size thresholds
    #     min_median_gap: float = 0.5,    # median(|x-y|) ≥ this  OR
    #     min_mean_gap: float = 0.5,      # mean(|x-y|) ≥ this
    #     # naming
    #     name_style: str = "abs_x_minus_y",   # "abs_x_minus_y" or "pipe_abs(x-y)"
    #     prefix: str = "abs_",
    #     suffix: str = "",
    #     # registry behavior
    #     overwrite_existing: bool = False,    # if False, make a unique name if collision
    # ) -> pd.DataFrame:
    #     """
    #     Identify unordered numeric-invariant pairs (x,y) that are *meaningfully incomparable*
    #     and register a derived DSL Expr for |x - y| in `self.exprs` (no DataFrame columns added).

    #     Returns a summary DataFrame with selection diagnostics and the registered expr names.
    #     """

    #     # ---- domain mask ----
    #     if condition is not None:
    #         domain_mask = self._mask_cached(condition)
    #     elif use_base_if_none and (self.base_hypothesis is not TRUE):
    #         domain_mask = self._mask_cached(self.base_hypothesis)
    #     else:
    #         domain_mask = np.ones(len(self.df), dtype=bool)

    #     domain_rows = int(domain_mask.sum())
    #     if domain_rows == 0:
    #         return pd.DataFrame(columns=[
    #             "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
    #             "rate_lt","rate_gt","rate_eq","support",
    #             "median_gap","mean_gap","selected","expr_name"
    #         ])

    #     # ---- numeric exprs & cached eval ----
    #     num_cols = list(self.expr_cols)  # NOTE: only raw df-backed columns here
    #     eval_cache: Dict[str, pd.Series] = {c: self.exprs[c].eval(self.df) for c in num_cols}

    #     def _propose_name(x: str, y: str) -> str:
    #         if name_style == "pipe_abs(x-y)":
    #             # Pretty display name; OK to use as key (Expr.eval ignores presence in df)
    #             return f"| {x} - {y} |"
    #         return f"{prefix}{x}_minus_{y}{suffix}"

    #     def _unique_name(base: str) -> str:
    #         if overwrite_existing or (base not in self.exprs):
    #             return base
    #         i = 2
    #         while f"{base}__{i}" in self.exprs:
    #             i += 1
    #         return f"{base}__{i}"

    #     rows = []

    #     # preconvert to numpy floats once
    #     arrays: Dict[str, np.ndarray] = {}
    #     finite_masks: Dict[str, np.ndarray] = {}
    #     for c in num_cols:
    #         s = eval_cache[c]
    #         if require_finite:
    #             s = s.replace([np.inf, -np.inf], np.nan)
    #         a = s.to_numpy(dtype=float, copy=False)
    #         arrays[c] = a
    #         finite_masks[c] = np.isfinite(a) if require_finite else np.ones_like(a, dtype=bool)

    #     for i in range(len(num_cols)):
    #         x = num_cols[i]; ax = arrays[x]; fx = finite_masks[x]
    #         for j in range(i+1, len(num_cols)):
    #             y = num_cols[j]; ay = arrays[y]; fy = finite_masks[y]

    #             m = domain_mask & fx & fy
    #             n = int(m.sum())
    #             if n == 0:
    #                 continue

    #             axm = ax[m]; aym = ay[m]
    #             lt = axm < aym
    #             gt = axm > aym
    #             eq = ~(lt | gt)

    #             n_lt = int(lt.sum())
    #             n_gt = int(gt.sum())
    #             n_eq = int(eq.sum())

    #             rate_lt = n_lt / n
    #             rate_gt = n_gt / n
    #             rate_eq = n_eq / n
    #             support = n / domain_rows

    #             both_sides = (n_lt > 0) and (n_gt > 0)
    #             side_ok = (min(rate_lt, rate_gt) >= float(min_side_rate)) and (min(n_lt, n_gt) >= int(min_side_count))
    #             support_ok = (support >= float(min_support))

    #             gaps = np.abs(axm - aym)
    #             median_gap = float(np.nanmedian(gaps)) if n > 0 else 0.0
    #             mean_gap   = float(np.nanmean(gaps))   if n > 0 else 0.0
    #             gap_ok = (median_gap >= float(min_median_gap)) or (mean_gap >= float(min_mean_gap))

    #             selected = bool(both_sides and side_ok and support_ok and gap_ok)

    #             expr_name = None
    #             if selected:
    #                 # Build symbolic Expr: |x - y|
    #                 expr = abs_(self.exprs[x] - self.exprs[y])

    #                 # Register with a stable/unique key in exprs
    #                 proposed = _propose_name(x, y)
    #                 key = proposed if overwrite_existing else _unique_name(proposed)
    #                 self.exprs[key] = expr

    #                 # Track synthetic exprs if you want to use them later
    #                 synthetic = getattr(self, "synthetic_expr_names_", set())
    #                 synthetic.add(key)
    #                 self.synthetic_expr_names_ = synthetic

    #                 expr_name = key

    #             rows.append({
    #                 "inv1": x, "inv2": y,
    #                 "n_rows": n,
    #                 "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
    #                 "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
    #                 "support": support,
    #                 "median_gap": median_gap, "mean_gap": mean_gap,
    #                 "selected": selected, "expr_name": expr_name,
    #             })

    #     out = pd.DataFrame(rows, columns=[
    #         "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
    #         "rate_lt","rate_gt","rate_eq","support",
    #         "median_gap","mean_gap","selected","expr_name"
    #     ])

    #     # ── Rank by mean gap and store the top abs-diff Exprs ──────────────────────
    #     out_sorted = out.sort_values("mean_gap", ascending=False).reset_index(drop=True)

    #     # Keep only Exprs actually registered (selected==True and expr_name not None)
    #     selected_exprs = [
    #         (row["expr_name"], self.exprs[row["expr_name"]])
    #         for _, row in out_sorted.iterrows()
    #         if row["selected"] and isinstance(row["expr_name"], str)
    #     ]

    #     # Save the top-N (e.g., all or top 20) to self.abs_exprs
    #     top_n = min(20, len(selected_exprs))
    #     self.abs_exprs = selected_exprs[:top_n]


    #     # For inspection
    #     self.last_meaningful_incomparability_exprs_ = out_sorted
    #     return out_sorted

    # ------------------------------ Pretty Print ------------------------------

    def print_class_characterization_summary(self) -> None:
        """
        Print a compact summary of equivalences and inclusions obtained from
        `characterize_constant_classes()`.

        Output sections
        ---------------
        - Class Equivalences
        - Inclusions (equalities ⇒ class)
        - Inclusions (class ⇒ equalities)

        Notes
        -----
        Uses `pretty()` of `ClassEquivalence` and `ClassInclusion` items.
        """
        eqv = getattr(self, "class_equivalences_", [])
        ab  = getattr(self, "class_inclusions_AB_", [])
        ba  = getattr(self, "class_inclusions_BA_", [])

        print("=== Class Equivalences ===")
        for E in eqv:
            print(" ", E.pretty())

        print("\n=== Inclusions (equalities ⇒ class) ===")
        for I in ab:
            print(" ", I.pretty())

        print("\n=== Inclusions (class ⇒ equalities) ===")
        for I in ba:
            print(" ", I.pretty())

    # ───────────────────────────── Internal Helpers ───────────────────────────

    def _reset_parameters(self) -> None:
        """
        (Re)initialize caches and derived structures from `self.df`.

        Side effects
        ------------
        - Resets mask cache and cache version.
        - Partitions columns into `boolean_cols` and `expr_cols`.
        - Builds `exprs` and `predicates`.
        - Detects `base_hypothesis` and `base_hypothesis_name`.
        - Auto-registers |x−y| Exprs for meaningfully-incomparable pairs (Expr-only).
        - Clears derived attributes (enumeration/results).
        - Runs baseline pipeline:
            enumerate_conjunctions → find_redundant_conjunctions →
            characterize_constant_classes → generate_ratio_bounds_on_base →
            sort_conjunctions_by_generality
        (These baseline results remain stable.)
        - Optionally post-mines equality-classes from ratio touches and integrates them
        **without** disturbing the baseline; any classes identical to the base mask
        are discarded prior to integration.
        """
        # ---- init cache first (so _always_true_boolean_cols can use it) ----
        self._mask_cache_version = id(self.df)
        self._mask_cache: dict[int, np.ndarray] = {}

        # ---- column partitions ----
        self.boolean_cols = [c for c in self.df.columns if _is_boolean_like(self.df[c])]
        self.expr_cols    = [c for c in self.df.columns if c not in self.boolean_cols]

        # ---- wrappers ----
        # Start with df-backed numeric columns
        self.exprs = {c: ColumnTerm(c) for c in self.expr_cols}
        self.predicates = {c: Predicate.from_column(c, truthy_only=True) for c in self.boolean_cols}

        # ---- base hypothesis: ∧ of all universally true predicates ----
        always_true_cols = self._always_true_boolean_cols()
        if always_true_cols:
            always_true_cols.sort()
            self.base_hypothesis = reduce(operator.and_, (self.predicates[c] for c in always_true_cols))
            self.base_hypothesis_name = " ∧ ".join(always_true_cols)
        else:
            self.base_hypothesis = TRUE
            self.base_hypothesis_name = "TRUE"

        self.base_predicates = {k: v for k, v in self.predicates.items() if k not in always_true_cols}

        # ---- synthetic bookkeeping ----
        # Names of Exprs we synthesize (e.g., |x−y|); ranked top list by mean gap.
        self.synthetic_expr_names_ = set()
        self.abs_exprs: list[tuple[str, Expr]] = []

        # ---- auto-register |x−y| Exprs for meaningfully incomparable pairs ----
        # This writes derived Exprs directly into self.exprs (no df mutation).
        try:
            abs_summary = self.add_absdiff_exprs_for_meaningfully_incomparable(
                condition=None,            # will use base if available
                use_base_if_none=True,
                require_finite=True,
                # incomparability thresholds (tune as desired)
                min_support=0.10,
                min_side_rate=0.10,
                min_side_count=5,
                # gap-size thresholds
                min_median_gap=0.5,
                min_mean_gap=0.5,
                # naming & registry
                name_style="abs_x_minus_y",
                prefix="abs_",
                suffix="",
                overwrite_existing=False,
            )
            # Store top-N |x−y| Exprs by mean gap for quick access
            if not abs_summary.empty:
                abs_sorted = abs_summary.sort_values("mean_gap", ascending=False).reset_index(drop=True)
                selected_exprs = [
                    (row["expr_name"], self.exprs[row["expr_name"]])
                    for _, row in abs_sorted.iterrows()
                    if row.get("selected", False) and isinstance(row.get("expr_name"), str)
                ]
                top_n = min(20, len(selected_exprs))
                self.abs_exprs = selected_exprs[:top_n]
        except Exception:
            # Keep initialization resilient; absence/failure of the helper shouldn’t break init.
            pass

        # ---- clear derived/cached attributes (optional) ----
        for attr in (
            "base_conjunctions", "nonredundant_conjunctions_", "redundant_conjunctions_",
            "equivalent_conjunction_groups_", "sorted_conjunctions_", "constant_exprs_",
            "class_equivalences_", "class_inclusions_AB_", "class_inclusions_BA_",
            "class_non_equivalences_", "class_cond_implies_equalities_"
        ):
            if hasattr(self, attr):
                delattr(self, attr)

        # =========================
        # Baseline pipeline (stable)
        # =========================
        self.enumerate_conjunctions()
        self.find_redundant_conjunctions()
        # Build grouped constant equalities and characterize classes BEFORE any mining
        self.characterize_constant_classes(tol=0.0, group_per_hypothesis=True)
        self.generate_ratio_bounds_on_base()
        self.sort_conjunctions_by_generality()

        # =======================================
        # Optional: post-mine equality-class hyps
        # =======================================
        # # We keep this post-baseline and filtered so the baseline report stays meaningful.
        # try:
        #     mined = self.use_top_equality_classes_as_hypotheses(
        #         # Slightly stricter defaults to avoid base-wide "equalities"
        #         min_support=0.10,
        #         min_touch_count=3,
        #         min_touch_rate=0.20,
        #         which="auto",
        #         touch_atol=1e-9,      # good for integer invariants
        #         touch_rtol=0.0,
        #         rationalize=True,
        #         max_denom=30,
        #         top_k=20,
        #         conjoin_base=True,    # keep the base context in names/logic
        #         analyze_bounds_per_class=False,
        #         per_class_min_support=0.10,
        #     )

        #     kept = list(mined.get("kept", []))

        #     if kept:
        #         # Filter out any equality-class whose mask is IDENTICAL to the base mask.
        #         # This prevents “representative swapping” during later merges/sorts.
        #         base_mask = self._mask_cached(self.base_hypothesis if self.base_hypothesis is not TRUE else Where(lambda d: np.ones(len(d), dtype=bool)))
        #         base_sig = base_mask.tobytes()

        #         kept_filtered: list[tuple[str, Predicate]] = []
        #         for (n, p) in kept:
        #             pm = self._mask_cached(p)
        #             if pm.tobytes() != base_sig:
        #                 kept_filtered.append((n, p))

        #         if kept_filtered:
        #             # Integrate without recomputing baseline constant characterization.
        #             # Sorting is refreshed so kept classes appear in generality order,
        #             # but we DO NOT rerun characterize_constant_classes() here.
        #             self._integrate_derived_hypotheses(kept_filtered)

        #             # If you’d like to update bounds for the *display* (optional), you can:
        #             # self.generate_ratio_bounds_on_base()
        #             # self.sort_conjunctions_by_generality()
        # except Exception:
        #     # Best-effort; don’t let optional mining break initialization
        #     pass



    def analyze_incomparability(
        self,
        condition: Predicate | None = None,
        *,
        use_base_if_none: bool = True,
        require_finite: bool = True,
        include_all_pairs: bool = False,   # if False, keep only incomparable pairs
    ) -> pd.DataFrame:
        """
        Find unordered pairs (x, y) of numeric invariants that are *incomparable*
        on a given domain: there exist rows with x<y and rows with x>y.
        Also report how frequently each side happens.

        Parameters
        ----------
        condition : Predicate or None, default=None
            Restrict the analysis to rows where `condition` holds. If None and
            `use_base_if_none=True`, uses the base hypothesis; otherwise uses all rows.
        use_base_if_none : bool, default=True
            When `condition` is None, use `self.base_hypothesis` (if not TRUE) as the domain.
        require_finite : bool, default=True
            If True, drop rows where either invariant is NaN or ±inf.
        include_all_pairs : bool, default=False
            If False, return only pairs that are incomparable. If True, include all pairs
            with a boolean `incomparable` flag.

        Returns
        -------
        pd.DataFrame
            Columns:
            - inv1, inv2           : column names (unordered; inv1 < inv2 lexicographically)
            - n_rows               : # rows compared (after masks/finite filtering)
            - n_lt, n_gt, n_eq     : counts with x<y, x>y, x==y (within tolerance-free strict compare)
            - rate_lt, rate_gt, rate_eq : counts divided by n_rows
            - incomparable         : bool, True iff (n_lt > 0 and n_gt > 0)
            - balance              : min(rate_lt, rate_gt)   # higher ⇒ more “two-sided”
            - support              : n_rows / domain_rows    # fraction of domain used
            Sorted by (incomparable desc, balance desc, support desc, inv1, inv2).
        """
        # --- choose domain mask ---
        if condition is not None:
            domain_mask = self._mask_cached(condition)
            domain_name = repr(condition)
        elif use_base_if_none and (self.base_hypothesis is not TRUE):
            domain_mask = self._mask_cached(self.base_hypothesis)
            domain_name = self.base_hypothesis_name
        else:
            # all rows
            domain_mask = np.ones(len(self.df), dtype=bool)
            domain_name = "TRUE"

        domain_rows = int(domain_mask.sum())
        if domain_rows == 0:
            return pd.DataFrame(columns=[
                "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
                "rate_lt","rate_gt","rate_eq","incomparable","balance","support"
            ])

        # --- numeric columns & cached eval ---
        num_cols = list(self.expr_cols)
        eval_cache: Dict[str, pd.Series] = {c: self.exprs[c].eval(self.df) for c in num_cols}

        # preconvert to numpy floats once
        arrays: Dict[str, np.ndarray] = {}
        finite_masks: Dict[str, np.ndarray] = {}
        for c in num_cols:
            s = eval_cache[c]
            if require_finite:
                s = s.replace([np.inf, -np.inf], np.nan)
            a = s.to_numpy(dtype=float, copy=False)
            arrays[c] = a
            finite_masks[c] = np.isfinite(a) if require_finite else np.ones_like(a, dtype=bool)

        rows = []
        # iterate unordered pairs
        for i in range(len(num_cols)):
            x = num_cols[i]; ax = arrays[x]; fx = finite_masks[x]
            for j in range(i+1, len(num_cols)):
                y = num_cols[j]; ay = arrays[y]; fy = finite_masks[y]

                m = domain_mask & fx & fy
                n = int(m.sum())
                if n == 0:
                    continue

                axm = ax[m]; aym = ay[m]
                lt = axm < aym
                gt = axm > aym
                # exact equality (no tolerance here; add atol/rtol if you prefer)
                eq = ~(lt | gt)

                n_lt = int(lt.sum())
                n_gt = int(gt.sum())
                n_eq = int(eq.sum())

                incomparable = (n_lt > 0) and (n_gt > 0)
                if include_all_pairs or incomparable:
                    rate_lt = n_lt / n
                    rate_gt = n_gt / n
                    rate_eq = n_eq / n
                    balance = min(rate_lt, rate_gt)  # how “two-sided” the incomparability is
                    support = n / domain_rows

                    rows.append({
                        "inv1": x, "inv2": y,
                        "n_rows": n,
                        "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
                        "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                        "incomparable": bool(incomparable),
                        "balance": balance,
                        "support": support,
                    })

        out = pd.DataFrame(rows, columns=[
            "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
            "rate_lt","rate_gt","rate_eq","incomparable","balance","support"
        ])
        if out.empty:
            return out

        out = out.sort_values(
            ["incomparable","balance","support","inv1","inv2"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)

        # Store last result for quick access/inspection
        self.last_incomparability_ = {
            "domain": domain_name,
            "result": out,
        }
        return out


    def _always_true_boolean_cols(self) -> List[str]:
        """
        Identify boolean-like columns that are universally true on `df`.

        Returns
        -------
        list[str]
            Column names whose predicate masks are all True.
        """
        out: List[str] = []
        for c, pred in self.predicates.items():
            if self._mask_cached(pred).all():
                out.append(c)
        return out

    @staticmethod
    def _relation_to_predicate(rel: Relation, name: str | None = None) -> Predicate:
        """
        Wrap a `Relation` as a row-wise `Predicate`, using `Relation.evaluate(df)`.

        Parameters
        ----------
        rel : Relation
            An atomic relation or `AllOf([...])`.
        name : str or None, default=None
            Optional display name; if None, uses `rel.pretty()` when available,
            otherwise `repr(rel)`.

        Returns
        -------
        Predicate
            A `Where` that evaluates the relation row-wise.
        """
        disp = name or (getattr(rel, "pretty", None) and rel.pretty()) or repr(rel)
        return Where(lambda d, r=rel: r.evaluate(d), name=disp)

    def _conjecture_cond_implies_equalities(
        self, cond: Predicate, rel: Relation, *, tol: float = 0.0
    ) -> Conjecture:
        """
        Build a conjecture stating that `cond` implies all equalities in `rel`.

        Parameters
        ----------
        cond : Predicate
            Hypothesis class predicate.
        rel : Relation
            Either a single equality-like relation or `AllOf([...])`.
        tol : float, default=0.0
            Included for interface symmetry (not used here).

        Returns
        -------
        Conjecture
            A conjecture `AllOf([...]) | cond` with a canonical name.
        """
        relation = rel if isinstance(rel, AllOf) else AllOf([rel])
        return Conjecture(relation=relation, condition=cond, name=f"Const[{repr(cond)}]")

    # ---- mask caching tied to current DataFrame ----

    def _mask_cache_clear(self) -> None:
        """
        Clear the internal predicate-mask cache.

        Notes
        -----
        Cache is also cleared automatically when `id(self.df)` changes.
        """
        self._mask_cache: Dict[int, np.ndarray] = {}

    def _mask_cached(self, pred: Predicate) -> np.ndarray:
        """
        Return a cached boolean mask for `pred` evaluated on `self.df`.

        Parameters
        ----------
        pred : Predicate
            Predicate to evaluate.

        Returns
        -------
        np.ndarray
            Dense boolean mask aligned with `df`.

        Notes
        -----
        - Uses `id(pred)` as the cache key.
        - Rebuilds cache if `id(self.df)` changed since last evaluation.
        """
        # If df object identity changed, clear cache (already reset in _reset_parameters)
        if self._mask_cache_version != id(self.df):
            self._mask_cache_version = id(self.df)
            self._mask_cache_clear()
        key = id(pred)
        m = self._mask_cache.get(key)
        if m is None:
            # Convert to stable boolean ndarray
            m = pred.mask(self.df).to_numpy(dtype=np.bool_, copy=False)
            self._mask_cache[key] = m
        return m

    def _compress_conjectures(self, conjs: List[Conjecture]) -> List[Conjecture]:
        """
        Merge conjectures by identical conditions, dedup relations by signature,
        and emit one conjecture per condition (atomic relation or `AllOf`).

        Parameters
        ----------
        conjs : list[Conjecture]
            Conjectures to merge by condition.

        Returns
        -------
        list[Conjecture]
            One conjecture per unique condition; with relation deduplication.

        Notes
        -----
        - Condition equality is decided by `repr(cond)` for None/TRUE vs. others.
        - Relation identity is deduped using `pretty()` when available, else `repr()`.
        """
        groups: Dict[str, List[Relation]] = defaultdict(list)
        cond_objs: Dict[str, Predicate | None] = {}

        for cj in conjs:
            cond = cj.condition  # may be None (interpreted as TRUE)
            key = repr(cond) if cond is not None else "TRUE"
            groups[key].append(cj.relation)
            cond_objs.setdefault(key, cond)

        out: List[Conjecture] = []
        for key, rels in groups.items():
            cond = cond_objs[key]
            unique: List[Relation] = []
            seen = set()
            for r in rels:
                sig = (getattr(r, "pretty", None) and r.pretty()) or repr(r)
                if sig not in seen:
                    seen.add(sig)
                    unique.append(r)

            if len(unique) == 1:
                out.append(Conjecture(relation=unique[0], condition=cond,
                                      name=f"{key} | {unique[0].__class__.__name__}"))
            else:
                out.append(Conjecture(relation=AllOf(unique), condition=cond, name=f"Const[{key}]"))
        return out

    def _mask_of_relation(self, rel: Relation) -> np.ndarray:
        """
        Compute the boolean mask where `rel` holds, via a temporary `Where` predicate.

        Parameters
        ----------
        rel : Relation
            Relation to evaluate on `df`.

        Returns
        -------
        np.ndarray
            Dense boolean mask aligned with `df`.
        """
        pred = self._relation_to_predicate(rel)
        return self._mask_cached(pred)

    def prune_conjectures(self, conjs: list[Conjecture]):
        """
        Deduplicate and reduce a list of conjectures by condition-equivalence and relation truth masks,
        and keep sharpest one-sided bounds when recognizable.

        Steps
        -----
        1) **Merge by condition**: group conjectures whose condition masks are identical
           (TRUE/None are treated uniformly).
        2) **Dedup by relation mask**: within a condition, drop relations that yield the
           same truth mask on `df`.
        3) **Keep sharpest inequalities**: for recognizable one-sided forms like
           `Ge(lhs, k*rhs)` / `Le(lhs, k*rhs)`, keep the extremal `k` per `(lhs, rhs)`.

        Parameters
        ----------
        conjs : list[Conjecture]
            Input conjectures to prune.

        Returns
        -------
        list[Conjecture]
            Pruned set where each item contains `AllOf(minimal_relations)` under a
            representative condition.

        Notes
        -----
        - The heuristic in (3) depends on relation-class attributes (`lhs`, `rhs`, `k`)
          if available. Non-matching relations are retained as-is in `others`.
        - If you prefer atomics instead of `AllOf`, re-run `_compress_conjectures(pruned)`.
        """
        if not conjs:
            return []

        # 1) group by condition mask signature (merge equivalent conditions)
        cond_groups = {}     # sig -> {"mask": m, "cond": Predicate, "items": []}
        cond_reprs  = {}     # sig -> representative predicate
        for cj in conjs:
            cond = cj.condition or TRUE
            cmask = self._mask_cached(cond)
            sig = cmask.tobytes()
            if sig not in cond_groups:
                cond_groups[sig] = {"mask": cmask, "items": []}
                cond_reprs[sig] = cond
            cond_groups[sig]["items"].append(cj)

        pruned = []
        for sig, bucket in cond_groups.items():
            cond_rep = cond_reprs[sig]

            # 2) within each condition, dedup relations by their truth mask on df
            seen_rel_sigs = set()
            rel_buckets = []  # store individually to later do "keep sharpest" per (lhs,rhs)
            for cj in bucket["items"]:
                rmask = self._mask_of_relation(cj.relation)
                rsig = rmask.tobytes()
                if rsig in seen_rel_sigs:
                    continue
                seen_rel_sigs.add(rsig)
                rel_buckets.append(cj.relation)

            # 3) keep sharpest inequality per (lhs,rhs) pair when possible
            #    We detect simple forms Ge(lhs, k*rhs) and Le(lhs, k*rhs) via introspection.
            #    If your Relation classes expose structured fields, adapt below accordingly.
            lowers = {}
            uppers = {}
            others = []
            for rel in rel_buckets:
                # very gentle detection: look for .lhs/.rhs/.k attributes if you have them;
                # otherwise skip to "others".
                lhs = getattr(rel, "lhs", None)
                rhs = getattr(rel, "rhs", None)
                k   = getattr(rel, "k",   None)
                is_ge = rel.__class__.__name__ in {"Ge","GE","GreaterEqual"}
                is_le = rel.__class__.__name__ in {"Le","LE","LessEqual"}
                key = None
                if lhs is not None and rhs is not None and isinstance(k, (int, float)):
                    key = (repr(lhs), repr(rhs))
                if is_ge and key:
                    prev = lowers.get(key)
                    if (prev is None) or (k > prev[0]):
                        lowers[key] = (k, rel)
                elif is_le and key:
                    prev = uppers.get(key)
                    if (prev is None) or (k < prev[0]):
                        uppers[key] = (k, rel)
                else:
                    others.append(rel)

            # rebuild minimal set: best lowers, best uppers, plus unique others
            keep_relations = [r for (_, r) in lowers.values()] + [r for (_, r) in uppers.values()] + others
            pruned.append(Conjecture(relation=AllOf(keep_relations), condition=cond_rep,
                                    name=f"Const[{repr(cond_rep)}]"))

        # If you prefer splitting AllOf back into atomics, use _compress_conjectures(pruned)
        return pruned

    def print_initialization_report(
        self,
        *,
        max_list: int = 8,
        show_top_conjunctions: bool = True,
        show_top_abs_exprs: bool = True,
        show_top_char_results: bool = True,
        indent: int = 2,
    ) -> dict:
        """
        Print a concise summary of discoveries performed during initialization.

        Sections
        --------
        • Dataset & Partitions
        • Base Hypothesis
        • Conjunction Enumeration
        • Constants & Class Characterization
        • Ratio Bounds (base)
        • Derived Equality Classes (from ratio touches)
        • Meaningfully Incomparable Pairs (|x−y| exprs)

        Parameters
        ----------
        max_list : int, default=8
            Max items to print in each list preview.
        show_top_conjunctions, show_top_abs_exprs, show_top_char_results : bool
            Toggle sub-lists in the report.
        indent : int, default=2
            Spaces used for bullet indentation.

        Returns
        -------
        dict
            Machine-readable summary of the same statistics printed.
        """
        pad = " " * indent
        def _safe_len(x) -> int:
            try:
                return len(x)
            except Exception:
                return 0

        def _pct(n: int, d: int) -> str:
            return f"{(100.0 * n / d):.1f}%" if d > 0 else "—"

        def _support(pred) -> tuple[int, float]:
            if pred is None or pred is TRUE:
                n = len(self.df)
                return n, 1.0
            m = self._mask_cached(pred)
            n = int(m.sum())
            return n, (n / len(self.df) if len(self.df) else 0.0)

        # ──────────────────────────────────────────────────────────────────────
        # Dataset & Partitions
        # ──────────────────────────────────────────────────────────────────────
        n_rows = len(self.df)
        n_cols = self.df.shape[1] if hasattr(self.df, "shape") else 0
        bool_cols = getattr(self, "boolean_cols", [])
        expr_cols = getattr(self, "expr_cols", [])
        n_bool, n_expr = _safe_len(bool_cols), _safe_len(expr_cols)

        # Synthetic expr bookkeeping (created by add_absdiff_exprs_for_meaningfully_incomparable)
        synth_names = sorted(getattr(self, "synthetic_expr_names_", set()))
        n_synth = _safe_len(synth_names)
        abs_exprs = getattr(self, "abs_exprs", [])  # list[(name, Expr)]
        n_abs_top = _safe_len(abs_exprs)

        print("────────────────────────────────────────────────────────────────")
        print("TxGraffiti • Initialization Summary")
        print("────────────────────────────────────────────────────────────────")
        print(f"Rows × Cols: {n_rows} × {n_cols}")
        print(f"Boolean-like columns: {n_bool}")
        print(f"Numeric expr columns: {n_expr}")
        print(f"Synthesized exprs (e.g., |x−y|): {n_synth}  (top saved: {n_abs_top})")

        # ──────────────────────────────────────────────────────────────────────
        # Base Hypothesis
        # ──────────────────────────────────────────────────────────────────────
        base_name = getattr(self, "base_hypothesis_name", "TRUE")
        base_pred = getattr(self, "base_hypothesis", TRUE)
        supp_n, supp_r = _support(base_pred)
        print("\nBase Hypothesis")
        print("---------------")
        print(f"{pad}{base_name}   [support: {supp_n}/{n_rows} = {_pct(supp_n, n_rows)}]")

        # ──────────────────────────────────────────────────────────────────────
        # Conjunction Enumeration
        # ──────────────────────────────────────────────────────────────────────
        base_conjs = getattr(self, "base_conjunctions", [])
        nonred = getattr(self, "nonredundant_conjunctions_", [])
        red = getattr(self, "redundant_conjunctions_", [])
        eq_groups = getattr(self, "equivalent_conjunction_groups_", [])

        print("\nConjunctions")
        print("------------")
        print(f"{pad}Enumerated: {_safe_len(base_conjs)}")
        print(f"{pad}Nonredundant (incl. base): {_safe_len(nonred)}")
        print(f"{pad}Redundant: {_safe_len(red)}")
        print(f"{pad}Equivalence groups: {_safe_len(eq_groups)}")

        if show_top_conjunctions and nonred:
            # Show top by support (already sorted by sort_conjunctions_by_generality)
            sorted_conjs = getattr(self, "sorted_conjunctions_", nonred)
            print(f"{pad}Top by generality (support):")
            shown = 0
            for name, pred in sorted_conjs[:max_list]:
                s_n, _ = _support(pred)
                print(f"{pad*2}• {name}   [{s_n}/{n_rows} = {_pct(s_n, n_rows)}]")
                shown += 1
            if len(sorted_conjs) > shown:
                print(f"{pad*2}… +{len(sorted_conjs) - shown} more")

        # ──────────────────────────────────────────────────────────────────────
        # Constants & Class Characterization
        # ──────────────────────────────────────────────────────────────────────
        const_map = getattr(self, "constant_exprs_", {})
        n_hyps_with_consts = sum(1 for v in const_map.values() if v)
        n_total_consts = sum(len(v) for v in const_map.values())

        eqv = getattr(self, "class_equivalences_", [])
        inc_ab = getattr(self, "class_inclusions_AB_", [])
        inc_ba = getattr(self, "class_inclusions_BA_", [])
        non_eq = getattr(self, "class_non_equivalences_", [])

        print("\nConstants & Class Characterization")
        print("----------------------------------")
        print(f"{pad}Hypotheses with constants: {n_hyps_with_consts}")
        print(f"{pad}Total constant equalities: {n_total_consts}")
        print(f"{pad}Class equivalences (A ≡ B): {_safe_len(eqv)}")
        print(f"{pad}Inclusions A⊆B (eq ⇒ class): {_safe_len(inc_ab)}")
        print(f"{pad}Inclusions B⊆A (class ⇒ eq): {_safe_len(inc_ba)}")
        if show_top_char_results:
            if eqv:
                print(f"{pad}Examples (equivalences):")
                for E in eqv[:max_list]:
                    print(f"{pad*2}• {E.pretty()}")
                if len(eqv) > max_list:
                    print(f"{pad*2}… +{len(eqv) - max_list} more")
            elif inc_ab or inc_ba:
                # Show a couple of inclusions if no equivalences
                if inc_ab:
                    print(f"{pad}Examples (eq ⇒ class):")
                    for I in inc_ab[:max_list]:
                        print(f"{pad*2}• {I.pretty()}")
                    if len(inc_ab) > max_list:
                        print(f"{pad*2}… +{len(inc_ab) - max_list} more")
                if inc_ba:
                    print(f"{pad}Examples (class ⇒ eq):")
                    for I in inc_ba[:max_list]:
                        print(f"{pad*2}• {I.pretty()}")
                    if len(inc_ba) > max_list:
                        print(f"{pad*2}… +{len(inc_ba) - max_list} more")

        # ──────────────────────────────────────────────────────────────────────
        # Ratio Bounds (base)
        # ──────────────────────────────────────────────────────────────────────

        ratio_conjs_raw = getattr(self, "simple_ratio_bounds_raw", [])
        n_ratio_conjs_raw = len(ratio_conjs_raw)
        ratio_conjs = getattr(self, "simple_ratio_bounds", [])
        n_ratio_conjs = len(ratio_conjs)

        est_pairs = (n_ratio_conjs_raw // 2) if n_ratio_conjs_raw else (n_ratio_conjs // 2)

        print("\nRatio Bounds on Base")
        print("--------------------")
        print(f"{pad}Emitted conjectures (atomic): {n_ratio_conjs_raw or '—'}")
        print(f"{pad}Conjecture groups (compressed): {n_ratio_conjs}")
        print(f"{pad}≈ distinct (inv1, inv2) pairs: {est_pairs}")

        # ratio_conjs = getattr(self, "simple_ratio_bounds", [])
        # n_ratio_conjs = _safe_len(ratio_conjs)
        # # Heuristic: 2 conjectures (lower+upper) per (inv1, inv2) pair
        # est_pairs = n_ratio_conjs // 2

        # print("\nRatio Bounds on Base")
        # print("--------------------")
        # print(f"{pad}Emitted conjectures: {n_ratio_conjs}  (≈ {est_pairs} pairs)")
        # # (If you later store the summary DF, you can print top touchers here.)

        # ──────────────────────────────────────────────────────────────────────
        # Derived Equality Classes (from ratio touches)
        # ──────────────────────────────────────────────────────────────────────
        derived = getattr(self, "derived_conjunctions_", [])
        print("\nDerived Equality Classes")
        print("------------------------")
        print(f"{pad}Kept (after merge/dominance): {_safe_len(derived)}")
        if derived:
            print(f"{pad}Examples:")
            shown = 0
            for name, pred in derived[:max_list]:
                s_n, _ = _support(pred)
                print(f"{pad*2}• {name}   [{s_n}/{n_rows} = {_pct(s_n, n_rows)}]")
                shown += 1
            if len(derived) > shown:
                print(f"{pad*2}… +{len(derived) - shown} more")

        # ──────────────────────────────────────────────────────────────────────
        # Meaningfully Incomparable Pairs (|x−y| exprs)
        # ──────────────────────────────────────────────────────────────────────
        last_inc = getattr(self, "last_meaningful_incomparability_exprs_", None)
        print("\nMeaningfully Incomparable Pairs (|x−y|)")
        print("---------------------------------------")
        print(f"{pad}Synthesized |x−y| Exprs registered: {n_synth}")
        if show_top_abs_exprs and n_abs_top:
            print(f"{pad}Top |x−y| by mean gap:")
            # Crosswalk names to mean_gap
            gap_map = {}
            if isinstance(last_inc, pd.DataFrame) and not last_inc.empty and "expr_name" in last_inc and "mean_gap" in last_inc:
                gap_map = {r["expr_name"]: r["mean_gap"] for _, r in last_inc.iterrows()}
            shown = 0
            for name, _expr in abs_exprs[:max_list]:
                gap = gap_map.get(name, None)
                if gap is None:
                    print(f"{pad*2}• {name}")
                else:
                    print(f"{pad*2}• {name}   [mean gap ≈ {gap:.4g}]")
                shown += 1
            if n_abs_top > shown:
                print(f"{pad*2}… +{n_abs_top - shown} more")
        elif n_synth and not n_abs_top:
            print(f"{pad}Note: synthesized exprs exist, but no top list recorded.")

        print("\nDone.\n")

        # ──────────────────────────────────────────────────────────────────────
        # Machine-readable return
        # ──────────────────────────────────────────────────────────────────────
        return {
            "rows": n_rows,
            "cols": n_cols,
            "n_boolean_cols": n_bool,
            "n_expr_cols": n_expr,
            "n_synth_exprs": n_synth,
            "base": {
                "name": base_name,
                "support_n": supp_n,
                "support_rate": supp_r,
            },
            "enumeration": {
                "enumerated": _safe_len(base_conjs),
                "nonredundant": _safe_len(nonred),
                "redundant": _safe_len(red),
                "equiv_groups": _safe_len(eq_groups),
                "top_conjunctions": [
                    {"name": n, "support_n": _support(p)[0]}
                    for (n, p) in (getattr(self, "sorted_conjunctions_", nonred)[:max_list] or [])
                ],
            },
            "constants": {
                "hyps_with_constants": n_hyps_with_consts,
                "total_constants": n_total_consts,
            },
            "characterization": {
                "equivalences": _safe_len(eqv),
                "inclusions_AB": _safe_len(inc_ab),
                "inclusions_BA": _safe_len(inc_ba),
                "non_equivalences": _safe_len(non_eq),
            },
            "ratio_bounds_base": {
                "emitted_conjectures": n_ratio_conjs,
                "approx_distinct_pairs": est_pairs,
            },
            "derived_equality_classes": {
                "kept": _safe_len(derived),
                "examples": [n for (n, _) in derived[:max_list]] if derived else [],
            },
            "abs_diff_exprs": {
                "top_saved": n_abs_top,
                "synth_total": n_synth,
                "examples": [name for (name, _) in abs_exprs[:max_list]],
            },
        }
