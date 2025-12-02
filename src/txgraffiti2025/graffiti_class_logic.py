# src/txgraffiti2025/graffiti_class_logic.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from itertools import combinations
import operator
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd

from .graffiti_base import GraffitiBase
from .graffiti_predicates import Predicate, Where, TRUE
from .graffiti_generic_conjecture import (
    BoolFormula, Conjecture, AllOf, Eq, coerce_formula,
)
from .graffiti_utils import min_, max_, absdiff

# For pairwise synthesis & types
from .graffiti_utils import Expr, BinOp, abs_  # reuse DSL nodes

__all__ = ["GraffitiClassLogic"]


# ─────────────────────────── small helpers ─────────────────────────── #


# --- paste near top of graffiti_class_logic.py ---
import re
import textwrap

_COL_ALIASES = {
    "order": "n",
    "size": "m",
    "avg_deg": "d̄",
    "radius": "radius",
    "diameter": "diameter",
}

def _alias_tokens(s: str) -> str:
    # Replace bare column tokens when they appear as identifiers (not substrings)
    # e.g., order -> n, size -> m, avg_deg -> d̄
    def repl(m):
        tok = m.group(0)
        return _COL_ALIASES.get(tok, tok)
    return re.sub(r"[A-Za-z_][A-Za-z0-9_]*", repl, s)

def _pretty_func_calls(s: str) -> str:
    # absdiff(x,y) -> |x − y|
    s = re.sub(r"absdiff\(\s*([^,()]+)\s*,\s*([^)]+)\)",
               r"|\1 − \2|", s)
    # min(a,b) / max(a,b)
    s = re.sub(r"min\(\s*([^,()]+)\s*,\s*([^)]+)\)", r"min(\1, \2)", s)
    s = re.sub(r"max\(\s*([^,()]+)\s*,\s*([^)]+)\)", r"max(\1, \2)", s)
    return s

def _pretty_conj_name(name: str) -> str:
    """Tighten and humanize a conjunction string."""
    s = name
    s = s.replace("  ", " ").strip()
    s = _pretty_func_calls(s)
    s = _alias_tokens(s)
    # remove redundant parentheses around simple tokens
    s = re.sub(r"\((\w+)\)", r"\1", s)
    # tighten spaces around operators/symbols
    s = re.sub(r"\s*([∧∨=<>±])\s*", r" \1 ", s)
    s = re.sub(r"\s*\|\s*", r"|", s)  # keep |x − y| tight
    # ensure minus is nice
    s = s.replace("-", "−")
    return s

def _wrap_lines(lines, width=88, indent="  • ", hang="    "):
    wrapped = []
    for line in lines:
        wrapped.append(textwrap.fill(
            line,
            width=width,
            initial_indent=indent,
            subsequent_indent=hang,
            break_on_hyphens=False,
        ))
    return "\n".join(wrapped)

def _format_const_eqs(eq_pairs):
    """eq_pairs like [('radius', 2.0), ('absdiff(order,size)', 1.0), ...] -> pretty uniques."""
    seen = set()
    out = []
    for lhs, val in eq_pairs:
        # lhs may already be pretty-printed by your Expr.pretty(); run through prettyifier anyway
        pretty_lhs = _pretty_conj_name(str(lhs))
        key = (pretty_lhs, val)
        if key in seen:
            continue
        seen.add(key)
        # int-looking floats as ints
        v = int(val) if isinstance(val, (int, float)) and abs(val - int(val)) < 1e-12 else val
        out.append(f"{pretty_lhs} = {v}")
    return out


def _holds_all(df: pd.DataFrame, formula_or_pred: Union[BoolFormula, Predicate]) -> bool:
    """True iff the BoolFormula/Predicate holds on all rows (NA→False upstream)."""
    bf = coerce_formula(formula_or_pred)
    m = bf.evaluate(df)
    return bool(pd.Series(m).all())


def _sig_mask(a: np.ndarray) -> bytes:
    """Stable byte signature for a boolean mask."""
    return np.asarray(a, dtype=bool).tobytes()


# ─────────────────────────── main class ─────────────────────────── #

class GraffitiClassLogic(GraffitiBase):
    """
    Class/hypothesis logic on top of GraffitiBase (new DSL).

    • Enumerates base∧(p1∧…∧pk) conjunctions over non-base boolean predicates.
    • Detects redundancy/equivalence among conjunctions (by mask).
    • Mines per-hypothesis constants; builds equality conjectures.
    • Compares “equalities hold” (A) vs class predicate (B): emits DSL-native
      Characterizations A ⇔ B and Inclusions A ⇒ B / B ⇒ A as Conjectures.

    Extras:
    • Before the pipeline: synthesize pairwise invariants (min/max/absdiff)
      from non-comparable numeric pairs; keep the most interesting.
    • Before the pipeline: add comparison booleans x>y / x<y if their class
      is a superset of a nonredundant conjunction and not redundant.
    """

    # storage (attributes set in __init__)
    base_conjunctions: List[Tuple[str, Predicate]]
    nonredundant_conjunctions_: List[Tuple[str, Predicate]]
    redundant_conjunctions_: List[Tuple[str, Predicate]]
    equivalent_conjunction_groups_: List[List[Tuple[str, Predicate]]]
    constant_exprs_: Dict[str, List[Tuple[str, float]]]

    class_equivalences_cj_: List[Conjecture]
    class_inclusions_AB_cj_: List[Conjecture]
    class_inclusions_BA_cj_: List[Conjecture]
    class_cond_implies_equalities_: List[Conjecture]

    sorted_conjunctions_: List[Tuple[str, Predicate]]
    simple_ratio_bounds: List[Conjecture]  # optional external attachment

    def __init__(
        self,
        base_or_df: Union[pd.DataFrame, GraffitiBase],
        *,
        run_pipeline: bool = True,
        max_arity: int = 2,
        tol: float = 0.0,
        # Pairwise synthesis knobs
        synthesize_pairwise: bool = True,
        synth_top_k: int = 24,                 # total new Exprs to keep (across all three kinds)
        synth_min_pair_support: int = 2,       # require each side to “win” at least this many rows
        synth_min_toggle_frac: float = 0.10,   # require ≥10% of rows on each side
        synth_max_corr_keep: float = 0.995,    # drop new expr if ~identical to a parent (|r|≥0.995)
        # Comparison-boolean knobs
        add_comparison_booleans: bool = True,
    ):
        # Adopt base
        if isinstance(base_or_df, GraffitiBase):
            self._adopt_base(base_or_df)
        else:
            super().__init__(base_or_df)

        # Initialize storages BEFORE any pipeline runs
        self.base_conjunctions = []
        self.nonredundant_conjunctions_ = []
        self.redundant_conjunctions_ = []
        self.equivalent_conjunction_groups_ = []
        self.constant_exprs_ = {}

        self.class_equivalences_cj_ = []
        self.class_inclusions_AB_cj_ = []
        self.class_inclusions_BA_cj_ = []
        self.class_cond_implies_equalities_ = []

        self.sorted_conjunctions_ = []
        self.simple_ratio_bounds = []
        self._ran_pipeline_ = False

        # Step 0: (optional) synthesize new Exprs (min/max/absdiff) from non-comparable pairs
        if synthesize_pairwise:
            self._synthesize_pairwise_invariants(
                top_k=synth_top_k,
                min_pair_support=synth_min_pair_support,
                min_toggle_frac=synth_min_toggle_frac,
                max_corr_keep=synth_max_corr_keep,
            )

        # Step 1: (optional) add comparison booleans x>y / x<y that are useful supersets
        if add_comparison_booleans:
            self._discover_noncomparable_comparison_booleans(max_arity_for_probe=max_arity)

        # Step 2: run baseline pipeline
        if run_pipeline:
            self._run_baseline(max_arity=max_arity, tol=tol)

    # ───────────────────── adopt-from-base (no shared caches) ───────────────────── #

    def _adopt_base(self, base: GraffitiBase) -> None:
        # core reference
        self.df = base.df

        # exact attribute names as defined by GraffitiBase
        self.boolean_cols = list(base.boolean_cols)
        self.expr_cols = list(base.expr_cols)

        self.base_hypothesis = base.base_hypothesis
        self.base_hypothesis_name = base.base_hypothesis_name

        self.predicates = dict(base.predicates)
        self.base_predicates = dict(base.base_predicates)
        self.exprs = dict(base.exprs)

        # hooks used by higher layers
        self.synthetic_expr_names_ = set(getattr(base, "synthetic_expr_names_", set()))
        self.abs_exprs = list(getattr(base, "abs_exprs", []))

        # fresh local caches bound to this df
        self._mask_cache_version: int = id(self.df)
        self._mask_cache: Dict[int, np.ndarray] = {}
        self._formula_pred_cache: Dict[str, Predicate] = {}

    # ───────────────────────────── pipeline ───────────────────────────── #

    def _run_baseline(self, *, max_arity: int, tol: float) -> None:
        self.enumerate_conjunctions(max_arity=max_arity)
        nr, rd, eqv = self.find_redundant_conjunctions()
        self.nonredundant_conjunctions_ = nr
        self.redundant_conjunctions_ = rd
        self.equivalent_conjunction_groups_ = eqv

        self.constant_exprs_ = self.find_constant_exprs(tol=tol)
        self.characterize_constant_classes(tol=tol, group_per_hypothesis=True, recompute=False)
        self.sort_conjunctions_by_generality(ascending=False)
        self._ran_pipeline_ = True

    # ───────────────────────── enumeration ───────────────────────── #

    def enumerate_conjunctions(self, *, max_arity: int = 2) -> List[Tuple[str, Predicate]]:
        if max_arity < 1:
            raise ValueError("max_arity must be ≥ 1")

        items = sorted(self.base_predicates.items(), key=lambda t: t[0])
        conjs: List[Tuple[str, Predicate]] = []

        base_parts = () if self.base_hypothesis is TRUE else tuple(self.base_hypothesis_name.split(" ∧ "))

        for k in range(1, max_arity + 1):
            for combo in combinations(items, k):
                names = [n for (n, _) in combo]
                preds = [p for (_, p) in combo]
                name = " ∧ ".join((*base_parts, *names)) if base_parts else " ∧ ".join(names)
                conj = reduce(operator.and_, preds, self.base_hypothesis)
                conjs.append((name, conj))

        # Always include the base hypothesis itself if not present
        if self.base_hypothesis_name not in [n for (n, _) in conjs]:
            conjs.append((self.base_hypothesis_name, self.base_hypothesis))

        self.base_conjunctions = conjs
        return conjs

    def find_redundant_conjunctions(
        self,
        *,
        equivalences_only_minimal: bool = True,
    ) -> tuple[list[tuple[str, Predicate]], list[tuple[str, Predicate]], list[list[tuple[str, Predicate]]]]:
        if not hasattr(self, "base_conjunctions"):
            raise AttributeError("Call enumerate_conjunctions() first.")

        base_parts = tuple(self.base_hypothesis_name.split(" ∧ ")) if getattr(self, "base_hypothesis", None) is not TRUE else tuple()

        records: list[dict] = []
        for idx, (name, pred) in enumerate(self.base_conjunctions):
            mask = np.asarray(self.mask(pred), dtype=bool)
            sig = _sig_mask(mask)
            parts = tuple(name.split(" ∧ "))
            extras = parts[len(base_parts):]
            records.append({
                "idx": idx, "name": name, "pred": pred,
                "sig": sig, "extras": frozenset(extras), "size": len(extras),
            })

        by_sig: dict[bytes, list[dict]] = defaultdict(list)
        for r in records:
            by_sig[r["sig"]].append(r)

        redundant_ids: set[int] = set()
        equivalence_groups: list[list[tuple[str, Predicate]]] = []

        for sig, group in by_sig.items():
            group_sorted = sorted(group, key=lambda r: r["size"])
            minimal_extras: list[frozenset] = []
            minimal_records: list[dict] = []

            for r in group_sorted:
                if any(me < r["extras"] for me in minimal_extras):
                    redundant_ids.add(r["idx"])
                else:
                    minimal_extras.append(r["extras"])
                    minimal_records.append(r)

            if equivalences_only_minimal:
                if len(minimal_records) > 1:
                    equivalence_groups.append([(rr["name"], rr["pred"]) for rr in minimal_records])
            else:
                by_sig_size: dict[tuple[bytes, int], list[dict]] = defaultdict(list)
                for r in group:
                    by_sig_size[(r["sig"], r["size"])].append(r)
                for _, grp in by_sig_size.items():
                    if len(grp) > 1:
                        equivalence_groups.append([(rr["name"], rr["pred"]) for rr in grp])

        nonredundant = [(r["name"], r["pred"]) for r in records if r["idx"] not in redundant_ids]
        redundant    = [(r["name"], r["pred"]) for r in records if r["idx"] in redundant_ids]

        if self.base_hypothesis_name not in [n for (n, _) in nonredundant]:
            nonredundant.append((self.base_hypothesis_name, self.base_hypothesis))

        self.nonredundant_conjunctions_ = nonredundant
        self.redundant_conjunctions_ = redundant
        self.equivalent_conjunction_groups_ = equivalence_groups
        return nonredundant, redundant, equivalence_groups

    def sort_conjunctions_by_generality(self, *, ascending: bool = False) -> list[tuple[str, Predicate]]:
        if not hasattr(self, "nonredundant_conjunctions_"):
            raise AttributeError("Call find_redundant_conjunctions() first.")
        scored: list[tuple[str, Predicate, int]] = []
        for name, pred in self.nonredundant_conjunctions_:
            m = np.asarray(self.mask(pred), dtype=bool)
            scored.append((name, pred, int(m.sum())))
        scored.sort(key=lambda t: (t[2], t[0]), reverse=not ascending)
        self.sorted_conjunctions_ = [(n, p) for (n, p, _) in scored]
        return self.sorted_conjunctions_

    # ───────────────────────── constants & conjectures ───────────────────── #

    def find_constant_exprs(self, *, tol: float = 0.0, require_finite: bool = True) -> Dict[str, List[Tuple[str, float]]]:
        if not hasattr(self, "nonredundant_conjunctions_"):
            self.enumerate_conjunctions()
            self.find_redundant_conjunctions()

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

        out: Dict[str, List[Tuple[str, float]]] = {}
        for name, pred in self.nonredundant_conjunctions_:
            mask = self.mask(pred)
            consts: List[Tuple[str, float]] = []
            for col in self.expr_cols:
                s = self.exprs[col].evaluate(self.df)[mask]
                ok, val = _constant_value(s)
                if ok and val is not None:
                    consts.append((col, float(val)))
            out[name] = consts

        self.constant_exprs_ = out
        return out

    def build_constant_conjectures(self, *, tol: float = 0.0, group_per_hypothesis: bool = True) -> List[Conjecture]:
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
                parts = [Eq(self.exprs[col], val) for (col, val) in pairs]
                conjs.append(Conjecture(relation=AllOf(parts), condition=cond, name=f"Const[{hyp_name}]"))
            else:
                for col, val in pairs:
                    conjs.append(Conjecture(relation=Eq(self.exprs[col], val), condition=cond, name=f"{col} const | {hyp_name}"))
        return self.compress_conjectures(conjs)

    # ───────────────────────── pairwise invariant synthesis ───────────────────────── #

    def _synthesize_pairwise_invariants(
        self,
        *,
        top_k: int = 24,
        min_pair_support: int = 2,
        min_toggle_frac: float = 0.10,
        max_corr_keep: float = 0.995,
    ) -> None:
        """
        Discover pairs of numeric invariants (x,y) that are *non-comparable* under the base
        hypothesis (i.e., both x<y and y<x occur) and synthesize:
            • min(x,y), max(x,y), abs(x−y)
        Keep only the 'most interesting' ones via a novelty/variability score.

        Adds selected Exprs into self.exprs (and self.expr_cols), and records names in
        self.synthetic_expr_names_.
        """
        if not getattr(self, "expr_cols", None):
            return

        # base-mask (under base hypothesis)
        base_mask = self.mask(self.base_hypothesis).astype(bool, copy=False)

        # quick access to evaluated numeric columns under base mask
        eval_cache: dict[str, pd.Series] = {}
        for col in self.expr_cols:
            try:
                s = self.exprs[col].evaluate(self.df)
                s = pd.to_numeric(s, errors="coerce")
                eval_cache[col] = s[base_mask]
            except Exception:
                continue

        cols = sorted(eval_cache.keys())
        if len(cols) < 2:
            return

        import numpy as _np

        def _min_expr(a: Expr, b: Expr) -> Expr:
            return BinOp(_np.minimum, a, b, "min")

        def _max_expr(a: Expr, b: Expr) -> Expr:
            return BinOp(_np.maximum, a, b, "max")

        def _absdiff_expr(a: Expr, b: Expr) -> Expr:
            return abs_(a - b)

        def _finite_pair(sa: pd.Series, sb: pd.Series) -> tuple[pd.Series, pd.Series]:
            m = sa.notna() & sb.notna()
            return sa[m], sb[m]

        def _is_noncomparable(sa: pd.Series, sb: pd.Series) -> tuple[bool, dict]:
            """Return (ok, stats) where ok=True if both x<y and y<x occur robustly."""
            sa, sb = _finite_pair(sa, sb)
            if sa.empty:
                return False, {}
            lt = (sa < sb)
            gt = (sb < sa)
            n_lt = int(lt.sum())
            n_gt = int(gt.sum())
            n = len(sa)
            if n_lt >= min_pair_support and n_gt >= min_pair_support:
                frac_lt = n_lt / n
                frac_gt = n_gt / n
                if min(frac_lt, frac_gt) >= min_toggle_frac:
                    return True, {"n": n, "frac_lt": frac_lt, "frac_gt": frac_gt}
            return False, {}

        def _pearson_abs(sa: pd.Series, sb: pd.Series) -> float:
            sa, sb = _finite_pair(sa, sb)
            if len(sa) < 3:
                return 0.0
            try:
                c = float(sa.corr(sb))
                if _np.isnan(c):
                    return 0.0
                return abs(c)
            except Exception:
                return 0.0

        def _range_std(s: pd.Series) -> tuple[float, float]:
            s = s.dropna()
            if s.empty:
                return 0.0, 0.0
            a = s.to_numpy(_np.float64, copy=False)
            return float(a.max() - a.min()), float(a.std(ddof=0))

        def _score_new(new_s: pd.Series, a_s: pd.Series, b_s: pd.Series) -> float:
            """
            Interestingness = novelty * variability
            - novelty: 1 - max(|corr(new,a)|, |corr(new,b)|)
            - variability: combine std and nonzero proportion
            """
            new_s = new_s.dropna()
            if new_s.empty:
                return 0.0
            r_new_a = _pearson_abs(new_s, a_s)
            r_new_b = _pearson_abs(new_s, b_s)
            novelty = max(0.0, 1.0 - max(r_new_a, r_new_b))
            rng, std = _range_std(new_s)
            p_nz = float((new_s != 0).mean()) if len(new_s) > 0 else 0.0
            variability = 0.6 * (std / (1.0 + rng)) + 0.4 * p_nz
            return novelty * variability

        candidates: list[tuple[float, str, Expr]] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                s1, s2 = eval_cache[c1], eval_cache[c2]
                ok, _ = _is_noncomparable(s1, s2)
                if not ok:
                    continue

                e_min  = min_(self.exprs[c1], self.exprs[c2])
                e_max  = max_(self.exprs[c1], self.exprs[c2])
                e_absd = absdiff(self.exprs[c1], self.exprs[c2])

                v_min  = pd.to_numeric(e_min.evaluate(self.df), errors="coerce")[base_mask]
                v_max  = pd.to_numeric(e_max.evaluate(self.df), errors="coerce")[base_mask]
                v_absd = pd.to_numeric(e_absd.evaluate(self.df), errors="coerce")[base_mask]

                if _pearson_abs(v_min, s1) < max_corr_keep and _pearson_abs(v_min, s2) < max_corr_keep:
                    score = _score_new(v_min, s1, s2)
                    if score > 0:
                        candidates.append((score, f"min({c1},{c2})", e_min))

                if _pearson_abs(v_max, s1) < max_corr_keep and _pearson_abs(v_max, s2) < max_corr_keep:
                    score = _score_new(v_max, s1, s2)
                    if score > 0:
                        candidates.append((score, f"max({c1},{c2})", e_max))

                if not (_pearson_abs(v_absd, s1) >= max_corr_keep and _pearson_abs(v_absd, s2) >= max_corr_keep):
                    score = _score_new(v_absd, s1, s2)
                    if score > 0:
                        candidates.append((score, f"absdiff({c1},{c2})", e_absd))

        if not candidates:
            return

        candidates.sort(key=lambda t: t[0], reverse=True)

        kept_names: list[str] = []
        for score, name, expr in candidates:
            if name in self.exprs:
                continue
            self.exprs[name] = expr
            self.synthetic_expr_names_.add(name)
            kept_names.append(name)
            if len(kept_names) >= max(0, int(top_k)):
                break

        # Append to column list in the order we actually kept them
        for nm in kept_names:
            if nm not in self.expr_cols:
                self.expr_cols.append(nm)

    # ───────────────────── comparison boolean discovery (x>y/x<y) ─────────────────── #

    def _mask_sig(self, m: np.ndarray) -> bytes:
        return np.asarray(m, dtype=bool).tobytes()

    def _make_cmp_where(self, a_name: str, b_name: str, op: str) -> Where:
        A = self.exprs[a_name]
        B = self.exprs[b_name]
        if op == ">":
            fn = lambda df: (A.evaluate(df) > B.evaluate(df)).fillna(False).astype(bool)
            nm = f"{a_name}>{b_name}"
        elif op == "<":
            fn = lambda df: (A.evaluate(df) < B.evaluate(df)).fillna(False).astype(bool)
            nm = f"{a_name}<{b_name}"
        else:
            raise ValueError("op must be '>' or '<'")
        return Where(fn, name=nm)

    def _discover_noncomparable_comparison_booleans(self, *, max_arity_for_probe: int = 2) -> list[str]:
        """
        Add comparison predicates x>y / x<y for non-comparable numeric pairs (both occur),
        but keep a candidate only if:
          • its mask is non-trivial (not all/none),
          • not redundant with any existing predicate,
          • and it is a *superset* of at least one current nonredundant conjunction.
        Returns the list of new predicate names added.
        """
        if not self.expr_cols:
            return []

        # Build a probe of current nonredundant conjunction masks
        self.enumerate_conjunctions(max_arity=max_arity_for_probe)
        nonred_probe, _, _ = self.find_redundant_conjunctions()
        probe_masks: list[tuple[str, np.ndarray]] = []
        for name, pred in nonred_probe:
            try:
                probe_masks.append((name, self.mask(pred).astype(bool, copy=False)))
            except Exception:
                continue

        # Existing signatures to avoid duplicates
        existing_sigs: set[bytes] = set()
        for p in self.predicates.values():
            try:
                existing_sigs.add(self._mask_sig(self.mask(p)))
            except Exception:
                pass

        # Evaluate numeric columns once
        eval_all: dict[str, pd.Series] = {}
        for c in self.expr_cols:
            try:
                s = pd.to_numeric(self.exprs[c].evaluate(self.df), errors="coerce")
                eval_all[c] = s
            except Exception:
                continue

        cols = sorted(eval_all.keys())
        new_names: list[str] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                sa, sb = eval_all[a], eval_all[b]
                m_gt = (sa > sb).fillna(False).to_numpy(dtype=bool, copy=False)
                m_lt = (sa < sb).fillna(False).to_numpy(dtype=bool, copy=False)

                # Non-comparable if both occur
                if not (m_gt.any() and m_lt.any()):
                    continue

                def _try(mask: np.ndarray, op: str) -> Optional[str]:
                    if not mask.any() or mask.all():
                        return None
                    sig = self._mask_sig(mask)
                    if sig in existing_sigs:
                        return None
                    # superset check: mask ⊇ pm  ⇔  pm ⊆ mask
                    is_superset = any((mask | (~pm)).all() for _, pm in probe_masks)
                    if not is_superset:
                        return None

                    pred = self._make_cmp_where(a, b, op)
                    name = pred.pretty() if hasattr(pred, "pretty") else repr(pred)
                    base_name = name
                    k = 2
                    while name in self.predicates:
                        name = f"{base_name}__{k}"
                        k += 1

                    # Register predicate
                    self.predicates[name] = pred
                    self.base_predicates[name] = pred
                    existing_sigs.add(sig)
                    new_names.append(name)
                    return name

                _try(m_gt, ">")
                _try(m_lt, "<")

        return new_names

    # ───────────────────── class characterization (A vs B) ─────────────────── #

    def characterize_constant_classes(
        self,
        *,
        tol: float = 0.0,
        group_per_hypothesis: bool = True,
        limit: Optional[int] = None,
        recompute: bool = False,
    ) -> dict:
        if recompute or not hasattr(self, "nonredundant_conjunctions_"):
            self.enumerate_conjunctions()
            self.find_redundant_conjunctions()

        grouped = self.build_constant_conjectures(tol=tol, group_per_hypothesis=group_per_hypothesis)
        if limit is not None:
            grouped = grouped[:max(0, int(limit))]

        equivalences_cj: List[Conjecture] = []
        inclusions_AB_cj: List[Conjecture] = []
        inclusions_BA_cj: List[Conjecture] = []
        non_equivalences: List[dict] = []

        for cj in grouped:
            B: Predicate = cj.condition or TRUE                    # class predicate
            A: Predicate = self.formula_to_predicate(cj.relation)  # equalities bundle as predicate

            eq_holds = _holds_all(self.df, A.iff(B))
            a_holds  = _holds_all(self.df, A >> B)
            b_holds  = _holds_all(self.df, B >> A)

            if eq_holds:
                equivalences_cj.append(Conjecture(relation=A.iff(B), name=f"Char[{A.pretty()} ⇔ {B.pretty()}]"))
            else:
                if a_holds:
                    inclusions_AB_cj.append(Conjecture(relation=(A >> B), name=f"Inclusion[{A.pretty()} ⇒ {B.pretty()}]"))
                if b_holds:
                    inclusions_BA_cj.append(Conjecture(relation=(B >> A), name=f"Inclusion[{B.pretty()} ⇒ {A.pretty()}]"))
                non_equivalences.append({
                    "A": A, "B": B,
                    "A_subset_B": bool(a_holds),
                    "B_subset_A": bool(b_holds),
                })

        # Keep the grouped “condition ⇒ equalities” as DSL-native Conjectures
        self.class_equivalences_cj_ = equivalences_cj
        self.class_inclusions_AB_cj_ = inclusions_AB_cj
        self.class_inclusions_BA_cj_ = inclusions_BA_cj
        self.class_cond_implies_equalities_ = [
            Conjecture(relation=cj.relation, condition=(cj.condition or TRUE), name=f"Const[{(cj.condition or TRUE).pretty()}]")
            for cj in grouped
        ]

        # Return a summary record
        return {
            "equivalences_cj": equivalences_cj,
            "inclusions_AB_cj": inclusions_AB_cj,
            "inclusions_BA_cj": inclusions_BA_cj,
            "non_equivalences": non_equivalences,
        }

    def _constants_by_condition(self) -> Dict[str, List[str]]:
        """
        Map each nonredundant conjunction (condition name) to a list of
        human-readable equality strings like 'radius = 1' built from
        self.constant_exprs_. Computes constants if absent.
        """
        const_map = getattr(self, "constant_exprs_", None)
        if not const_map:
            const_map = self.find_constant_exprs()

        out: Dict[str, List[str]] = {}

        def _fmt_val(v: float) -> str:
            try:
                vf = float(v)
            except Exception:
                return str(v)
            if abs(vf - round(vf)) < 1e-12:
                return str(int(round(vf)))
            return f"{vf:g}"

        for hyp_name, pairs in const_map.items():
            eqs: List[str] = []
            for col, val in pairs:
                lhs = getattr(self.exprs[col], "pretty", None)
                lhs = lhs() if callable(lhs) else repr(self.exprs[col])
                eqs.append(f"{lhs} = {_fmt_val(val)}")
            out[hyp_name] = eqs

        return out

    # ─────────────────────────────── summary ─────────────────────────────── #

    # def summary_conjectures(
    #     self,
    #     *,
    #     max_per_section: int = 12,
    #     show_characterizations: bool = True,
    #     show_implications: bool = True,
    #     show_const_equalities: bool = True,
    #     show_ratio_bounds: bool = True,
    #     include_counts_bar: bool = True,
    #     verbose: bool = True,
    #     skip_characterized: bool = True,
    #     list_const_equalities: bool = True,
    #     max_equalities_per_cond: int = 8,
    # ) -> str:
    #     """
    #     Concise catalog of *meaningful conjectures* represented in your DSL:

    #     • Characterizations (A ⇔ B) as IffPredicates inside Conjectures.
    #     • Implications A ⇒ B and B ⇒ A as ImpliesPredicates Conjectures.
    #     • Condition ⇒ Equalities (grouped constants).
    #     • Ratio-bound conjectures (if attached externally).
    #     """
    #     lines: list[str] = []
    #     clip = lambda L: (L[:max_per_section], max(0, len(L) - max_per_section))

    #     # ───── helpers ───── #

    #     def _clean_cond_label(s: str) -> str:
    #         if s is None:
    #             return ""
    #         s = s.strip()
    #         if s.startswith("(") and s.endswith(")"):
    #             s = s[1:-1]
    #         if s.startswith("TRUE ∧ "):
    #             s = s[len("TRUE ∧ "):]
    #         s = s.replace("(TRUE ∧ ", "(")
    #         return s

    #     def _pretty_cj(x):
    #         nm = getattr(x, "name", None)
    #         if nm:
    #             nm = nm.replace("(TRUE ∧ ", "(").replace("TRUE ∧ ", "")
    #             return nm
    #         rel = getattr(x, "relation", None)
    #         cond = getattr(x, "condition", None)
    #         rel_s = rel.pretty() if hasattr(rel, "pretty") else repr(rel)
    #         cond_s = ""
    #         if cond not in (None, True):
    #             raw = cond.pretty() if hasattr(cond, "pretty") else repr(cond)
    #             cond_s = _clean_cond_label(raw)
    #         return f"{rel_s}" + (f" | {cond_s}" if cond_s else "")

    #     def _support(pred) -> int:
    #         try:
    #             m = self.mask(pred)
    #             return int(np.asarray(m, dtype=bool).sum())
    #         except Exception:
    #             return 0

    #     def _balancedness(pred) -> float:
    #         N = len(self.df.index)
    #         if N <= 0:
    #             return 0.0
    #         p = _support(pred) / float(N)
    #         return max(0.0, 1.0 - abs(p - 0.5) * 2.0)

    #     def _arity_of_name(hyp_name: str) -> int:
    #         base_parts = tuple(self.base_hypothesis_name.split(" ∧ ")) if getattr(self, "base_hypothesis", None) is not TRUE else tuple()
    #         parts = tuple(hyp_name.split(" ∧ "))
    #         return max(0, len(parts) - len(base_parts))

    #     def _simplicity_penalty(*, arity: int, k_eq: int, text_len: int) -> float:
    #         return 0.15 * arity + 0.05 * max(0, k_eq - 1) + 0.01 * (text_len / 80.0)

    #     def _eq_bundle_stats(rel) -> tuple[int, float]:
    #         from .graffiti_generic_conjecture import AllOf
    #         parts = list(rel.parts) if isinstance(rel, AllOf) else [rel]
    #         k = 0
    #         nice = 0.0
    #         from fractions import Fraction
    #         for p in parts:
    #             v = getattr(p, "rhs", None) or getattr(p, "value", None)
    #             if v is None:
    #                 continue
    #             k += 1
    #             try:
    #                 vf = float(v)
    #                 if vf.is_integer():
    #                     nice += 1.0
    #                 else:
    #                     fr = Fraction(vf).limit_denominator(64)
    #                     nice += 0.5 if fr.denominator <= 7 else 0.2
    #             except Exception:
    #                 nice += 0.2
    #         return k, nice

    #     def _incl_precision_recall(A, B) -> tuple[float, float, int, int]:
    #         mA = np.asarray(self.mask(A), dtype=bool)
    #         mB = np.asarray(self.mask(B), dtype=bool)
    #         sA = int(mA.sum()); sB = int(mB.sum())
    #         if sA == 0 or sB == 0:
    #             return 0.0, 0.0, sA, sB
    #         inter = int((mA & mB).sum())
    #         prec = inter / float(sA)
    #         rec  = inter / float(sB)
    #         return prec, rec, sA, sB

    #     def _sig_of_pred(p) -> bytes:
    #         return np.asarray(self.mask(p), dtype=bool).tobytes()

    #     def _support_of_name(hyp_name: str, name_to_pred: dict[str, Any]) -> int:
    #         pred = name_to_pred.get(hyp_name)
    #         return 0 if pred is None else _support(pred)

    #     # Importance keys
    #     def _key_equivalence(E) -> float:
    #         pred = getattr(E, "A", None) or getattr(E, "B", None)
    #         if pred is None:
    #             return -1e9
    #         sup = _support(pred)
    #         bal = _balancedness(pred)
    #         txt = len(E.pretty()) if hasattr(E, "pretty") else len(repr(E))
    #         return 2.0 * bal + np.log1p(sup) - _simplicity_penalty(arity=0, k_eq=0, text_len=txt)

    #     def _key_inclusion_AB(I) -> float:
    #         A = getattr(I, "A", None); B = getattr(I, "B", None)
    #         if A is None or B is None:
    #             return -1e9
    #         prec, rec, sA, _ = _incl_precision_recall(A, B)
    #         balB = _balancedness(B)
    #         txt = len(I.pretty()) if hasattr(I, "pretty") else len(repr(I))
    #         f = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
    #         return 1.2*f + 0.6*rec + 0.4*balB + 0.3*np.log1p(sA) - _simplicity_penalty(arity=0, k_eq=0, text_len=txt)

    #     def _key_inclusion_BA(I) -> float:
    #         B = getattr(I, "A", None); A = getattr(I, "B", None)  # (ClassInclusion(cond, rel_pred))
    #         if A is None or B is None:
    #             return -1e9
    #         prec, rec, sB, _ = _incl_precision_recall(B, A)
    #         balB = _balancedness(B)
    #         k_eq, nice = (0, 0.0)
    #         try:
    #             rel = getattr(getattr(A, "source_relation", None), "relation", None) or getattr(A, "relation", None)
    #             if rel is not None:
    #                 k_eq, nice = _eq_bundle_stats(rel)
    #         except Exception:
    #             pass
    #         txt = len(I.pretty()) if hasattr(I, "pretty") else len(repr(I))
    #         return 0.8*np.log1p(sB) + 0.8*prec + 0.4*balB + 0.2*nice - _simplicity_penalty(arity=0, k_eq=k_eq, text_len=txt)

    #     def _key_const_group(hyp_name: str, eq_list: list[str], pred) -> float:
    #         s = _support(pred)
    #         bal = _balancedness(pred)
    #         ar = _arity_of_name(hyp_name)
    #         k = len(eq_list)
    #         txtlen = sum(len(x) for x in eq_list[:8]) + len(hyp_name)
    #         return 0.9*np.log1p(s) + 0.9*bal + 0.5*k - _simplicity_penalty(arity=ar, k_eq=k, text_len=txtlen)

    #     # ───────────────── gather DSL-native conjectures ───────────────── #

    #     eqv_cj   = list(getattr(self, "class_equivalences_cj_", []) or [])
    #     incAB_cj = list(getattr(self, "class_inclusions_AB_cj_", []) or [])
    #     incBA_cj = list(getattr(self, "class_inclusions_BA_cj_", []) or [])
    #     cie      = list(getattr(self, "class_cond_implies_equalities_", []) or [])
    #     ratio    = list(getattr(self, "simple_ratio_bounds", []) or [])

    #     # Header
    #     lines.append("──────────────────────────────────────────────")
    #     lines.append("GraffitiClassLogic • Meaningful Conjectures (DSL)")
    #     lines.append("──────────────────────────────────────────────")
    #     lines.append(f"Base hypothesis: {getattr(self, 'base_hypothesis_name', 'TRUE')}")
    #     lines.append(f"DataFrame: {self.df.shape[0]} rows × {self.df.shape[1]} cols")
    #     lines.append("")

    #     if include_counts_bar:
    #         lines.append("Counts:")
    #         lines.append(f"  Characterizations (A ⇔ B): {len(eqv_cj)}")
    #         lines.append(f"  Implications A ⇒ B:        {len(incAB_cj)}")
    #         lines.append(f"  Implications B ⇒ A:        {len(incBA_cj)}")
    #         lines.append(f"  Cond ⇒ Equalities:         {len(cie)}")
    #         lines.append(f"  Ratio bounds (groups):     {len(ratio)}")
    #         lines.append("")

    #     # Characterizations
    #     if show_characterizations:
    #         lines.append("Characterizations (A ⇔ B):")
    #         if not eqv_cj:
    #             lines.append("  (none)")
    #         else:
    #             eqv_sorted = sorted(eqv_cj, key=_key_equivalence, reverse=True)
    #             items, extra = clip(eqv_sorted)
    #             for E in items:
    #                 lines.append("  • " + _pretty_cj(E))
    #             if extra:
    #                 lines.append(f"  … +{extra} more")
    #         lines.append("")

    #     # Implications
    #     if show_implications:
    #         lines.append("Implications:")
    #         if not incAB_cj and not incBA_cj:
    #             lines.append("  (none)")
    #         else:
    #             if incAB_cj:
    #                 lines.append("  • A ⇒ B (equalities ⇒ class):")
    #                 incAB_sorted = sorted(incAB_cj, key=_key_inclusion_AB, reverse=True)
    #                 items, extra = clip(incAB_sorted)
    #                 for I in items:
    #                     lines.append("      " + _pretty_cj(I))
    #                 if extra:
    #                     lines.append(f"      … +{extra} more")
    #             if incBA_cj:
    #                 lines.append("  • B ⇒ A (class ⇒ equalities):")
    #                 incBA_sorted = sorted(incBA_cj, key=_key_inclusion_BA, reverse=True)
    #                 items, extra = clip(incBA_cj)
    #                 for I in items:
    #                     lines.append("      " + _pretty_cj(I))
    #                 if extra:
    #                     lines.append(f"      … +{extra} more")
    #         lines.append("")

    #     # Condition ⇒ Equalities
    #     if show_const_equalities:
    #         lines.append("Condition ⇒ Equalities (grouped constants):")

    #         const_map = self._constants_by_condition()  # {hyp_name -> [eq strings]}
    #         name_to_pred = {name: pred for (name, pred) in getattr(self, "nonredundant_conjunctions_", [])}

    #         characterized_sigs: set[bytes] = set()
    #         if skip_characterized:
    #             native_eqv = list(getattr(self, "class_equivalences_", []) or [])
    #             for E in native_eqv:
    #                 for side in (getattr(E, "A", None), getattr(E, "B", None)):
    #                     if side is None:
    #                         continue
    #                     try:
    #                         characterized_sigs.add(np.asarray(self.mask(side), dtype=bool).tobytes())
    #                     except Exception:
    #                         lbl = getattr(side, "pretty", None)
    #                         lbl = lbl() if callable(lbl) else repr(side)
    #                         if lbl in name_to_pred:
    #                             characterized_sigs.add(np.asarray(self.mask(name_to_pred[lbl]), dtype=bool).tobytes())

    #         items_all = []
    #         for hyp_name, eq_list in const_map.items():
    #             if len(eq_list) == 0:
    #                 continue
    #             pred = name_to_pred.get(hyp_name)
    #             if pred is None:
    #                 continue
    #             items_all.append((hyp_name, eq_list, pred))

    #         items_all.sort(key=lambda t: _key_const_group(*t), reverse=True)

    #         ranked = []
    #         for hyp_name, eq_list, pred in items_all:
    #             if skip_characterized and np.asarray(self.mask(pred), dtype=bool).tobytes() in characterized_sigs:
    #                 continue
    #             ranked.append((hyp_name, eq_list, pred))

    #         if not ranked:
    #             lines.append("  (none)")
    #         else:
    #             items, extra_after_clip = clip(ranked)
    #             for hyp_name, eq_list, pred in items:
    #                 clean = _clean_cond_label(hyp_name)
    #                 k = len(eq_list)
    #                 sup = _support_of_name(hyp_name, name_to_pred)
    #                 lines.append(f"  • Const[{clean}] — {k} equalit{'y' if k==1 else 'ies'} (support={sup})")
    #                 if list_const_equalities and k > 0:
    #                     for eq_str in eq_list[:max_equalities_per_cond]:
    #                         lines.append(f"      - {eq_str}")
    #                     if k > max_equalities_per_cond:
    #                         lines.append(f"        … +{k - max_equalities_per_cond} more")
    #             if extra_after_clip:
    #                 lines.append(f"  … +{extra_after_clip} more")
    #         lines.append("")

    #     lines.append("──────────────────────────────────────────────")
    #     s = "\n".join(lines)
    #     if verbose:
    #         print(s)
    #     return s


    def summary_conjectures(
        self,
        *,
        max_per_section: int = 12,
        show_characterizations: bool = True,
        show_implications: bool = True,
        show_const_equalities: bool = True,
        show_ratio_bounds: bool = True,
        include_counts_bar: bool = True,
        verbose: bool = True,
        skip_characterized: bool = True,
        list_const_equalities: bool = True,
        max_equalities_per_cond: int = 8,
    ) -> str:
        """
        Concise catalog of *meaningful conjectures* represented in the DSL.

        Uses pretty-printers:
        - absdiff(a,b) -> |a − b|
        - min/max rendered as min(a,b), max(a,b)
        - column aliases: order→n, size→m, avg_deg→d̄
        Adds small witness lists (e.g., 'K4, C4, Q3') so you can sanity-check quickly.

        Sections:
        • Characterizations (A ⇔ B)
        • Implications (A ⇒ B and B ⇒ A)
        • Condition ⇒ Equalities (grouped constants)
        • Ratio bounds (if available)
        """
        import numpy as _np

        # --- config for visual presentation ---
        _WIDTH = 92
        _EXAMPLE_K = 3
        _SHOW_EXAMPLES = True  # always show a few example rows

        lines: list[str] = []
        clip = lambda L: (L[:max_per_section], max(0, len(L) - max_per_section))

        # ---------- utilities (local wrappers) ----------
        def _support(pred) -> int:
            try:
                m = self.mask(pred)
                return int(_np.asarray(m, dtype=bool).sum())
            except Exception:
                return 0

        def _balancedness(pred) -> float:
            N = len(self.df.index)
            if N <= 0:
                return 0.0
            p = _support(pred) / float(N)
            return max(0.0, 1.0 - abs(p - 0.5) * 2.0)

        def _arity_of_name(hyp_name: str) -> int:
            base_parts = tuple(getattr(self, "base_hypothesis_name", "TRUE").split(" ∧ "))
            parts = tuple(hyp_name.split(" ∧ "))
            return max(0, len(parts) - len(base_parts))

        def _simplicity_penalty(*, arity: int, k_eq: int, text_len: int) -> float:
            return 0.15 * arity + 0.05 * max(0, k_eq - 1) + 0.01 * (text_len / 80.0)

        def _incl_precision_recall(A, B) -> tuple[float, float, int, int]:
            mA = _np.asarray(self.mask(A), dtype=bool)
            mB = _np.asarray(self.mask(B), dtype=bool)
            sA = int(mA.sum()); sB = int(mB.sum())
            if sA == 0 or sB == 0:
                return 0.0, 0.0, sA, sB
            inter = int((mA & mB).sum())
            prec = inter / float(sA)
            rec  = inter / float(sB)
            return prec, rec, sA, sB

        # rank keys (same spirit as your prior version)
        def _key_equivalence(E) -> float:
            txt = len(E.pretty()) if hasattr(E, "pretty") else len(repr(E))
            # prefer more balanced, larger-support classes
            A = getattr(E, "A", None)
            if A is None:
                return -1e9
            sup = _support(A)
            bal = _balancedness(A)
            return 2.0 * bal + _np.log1p(sup) - _simplicity_penalty(arity=0, k_eq=0, text_len=txt)

        def _key_inclusion_AB(I) -> float:
            A = getattr(I, "A", None); B = getattr(I, "B", None)
            if A is None or B is None: return -1e9
            prec, rec, sA, _ = _incl_precision_recall(A, B)
            balB = _balancedness(B)
            txt = len(I.pretty()) if hasattr(I, "pretty") else len(repr(I))
            f = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
            return 1.2*f + 0.6*rec + 0.4*balB + 0.3*_np.log1p(sA) - _simplicity_penalty(arity=0, k_eq=0, text_len=txt)

        def _key_inclusion_BA(I) -> float:
            B = getattr(I, "A", None); A = getattr(I, "B", None)  # (class ⇒ equalities)
            if A is None or B is None: return -1e9
            prec, rec, sB, _ = _incl_precision_recall(B, A)
            balB = _balancedness(B)
            txt = len(I.pretty()) if hasattr(I, "pretty") else len(repr(I))
            return 0.8*_np.log1p(sB) + 0.8*prec + 0.4*balB - _simplicity_penalty(arity=0, k_eq=0, text_len=txt)

        # name -> predicate map (for supports/examples)
        name_to_pred = {name: pred for (name, pred) in getattr(self, "nonredundant_conjunctions_", [])}

        # ---------- header ----------
        lines.append("──────────────────────────────────────────────")
        lines.append("GraffitiClassLogic • Meaningful Conjectures (DSL)")
        lines.append("──────────────────────────────────────────────")
        base_name = getattr(self, "base_hypothesis_name", "TRUE")
        try:
            base_name = _pretty_conj_name(base_name)
        except Exception:
            pass
        lines.append(f"Base hypothesis: {base_name}")
        lines.append(f"DataFrame: {self.df.shape[0]} rows × {self.df.shape[1]} cols")
        lines.append("")

        # ---------- gather DSL-native conjectures ----------
        eqv_cj   = list(getattr(self, "class_equivalences_cj_", []) or [])
        incAB_cj = list(getattr(self, "class_inclusions_AB_cj_", []) or [])
        incBA_cj = list(getattr(self, "class_inclusions_BA_cj_", []) or [])
        cie      = list(getattr(self, "class_cond_implies_equalities_", []) or [])
        ratio    = list(getattr(self, "simple_ratio_bounds", []) or [])

        if include_counts_bar:
            lines.append("Counts:")
            lines.append(f"  Characterizations (A ⇔ B): {len(eqv_cj)}")
            lines.append(f"  Implications A ⇒ B:        {len(incAB_cj)}")
            lines.append(f"  Implications B ⇒ A:        {len(incBA_cj)}")
            lines.append(f"  Cond ⇒ Equalities:         {len(cie)}")
            lines.append(f"  Ratio bounds (groups):     {len(ratio)}")
            lines.append("")

        # ---------- Characterizations ----------
        if show_characterizations:
            lines.append("Characterizations (A ⇔ B):")
            if not eqv_cj:
                lines.append("  (none)\n")
            else:
                eqv_sorted = sorted(eqv_cj, key=_key_equivalence, reverse=True)
                items, extra = clip(eqv_sorted)
                bullets = []
                for E in items:
                    # E.pretty() should already include both sides (A ⇔ B), but make it cleaner:
                    text = E.pretty() if hasattr(E, "pretty") else repr(E)
                    try:
                        text = _pretty_conj_name(text)
                    except Exception:
                        pass

                    # attach example witnesses if we can access mask
                    mask = getattr(E, "mask", None)
                    if callable(mask):
                        try:
                            m = _np.asarray(mask(), dtype=bool)
                            names = self.df.index[m].tolist()
                        except Exception:
                            names = []
                    else:
                        names = []
                    if _SHOW_EXAMPLES and names:
                        text += f"   (e.g., {', '.join(names[:_EXAMPLE_K])})"

                    bullets.append(text)

                lines.append(_wrap_lines([f"Char[{b}]" if not b.startswith("Char[") else b
                                        for b in bullets], width=_WIDTH))
                if extra:
                    lines.append(f"\n  … +{extra} more\n")
                else:
                    lines.append("")

        # ---------- Implications ----------
        if show_implications:
            lines.append("Implications:")
            if not incAB_cj and not incBA_cj:
                lines.append("  (none)\n")
            else:
                if incAB_cj:
                    lines.append("  • A ⇒ B (equalities ⇒ class):")
                    incAB_sorted = sorted(incAB_cj, key=_key_inclusion_AB, reverse=True)
                    items, extra = clip(incAB_sorted)
                    bullets = []
                    for I in items:
                        t = I.pretty() if hasattr(I, "pretty") else repr(I)
                        try:
                            t = _pretty_conj_name(t)
                        except Exception:
                            pass
                        bullets.append(f"    {t}")
                    lines.append(_wrap_lines(bullets, width=_WIDTH, indent="    ", hang="      "))
                    if extra:
                        lines.append(f"      … +{extra} more")
                if incBA_cj:
                    lines.append("  • B ⇒ A (class ⇒ equalities):")
                    incBA_sorted = sorted(incBA_cj, key=_key_inclusion_BA, reverse=True)
                    items, extra = clip(incBA_sorted)
                    bullets = []
                    for I in items:
                        t = I.pretty() if hasattr(I, "pretty") else repr(I)
                        try:
                            t = _pretty_conj_name(t)
                        except Exception:
                            pass
                        bullets.append(f"    {t}")
                    lines.append(_wrap_lines(bullets, width=_WIDTH, indent="    ", hang="      "))
                    if extra:
                        lines.append(f"      … +{extra} more")
                lines.append("")

        # ---------- Condition ⇒ Equalities ----------
        if show_const_equalities:
            lines.append("Condition ⇒ Equalities (grouped constants):")

            # --- helper: normalize cie items into (name, eq_lines, support, mask_bool_array) ---
            def _normalize_cie(cie_items):
                out = []
                # late import to avoid cycles
                try:
                    from .graffiti_generic_conjecture import AllOf
                except Exception:
                    AllOf = None

                for item in (cie_items or []):
                    # Case 1: already a tuple (name, eqs, support, mask)
                    if isinstance(item, tuple) and len(item) == 4:
                        name, eqs, support, mask = item
                        # make eqs a list of pretty strings
                        eq_lines = []
                        for e in (eqs or []):
                            if isinstance(e, str):
                                eq_lines.append(_pretty_conj_name(e))
                            elif hasattr(e, "pretty"):
                                eq_lines.append(_pretty_conj_name(e.pretty()))
                            else:
                                eq_lines.append(_pretty_conj_name(repr(e)))
                        out.append((name, eq_lines, support, mask))
                        continue

                    # Case 2: Conjecture-like object
                    cond = getattr(item, "condition", None)
                    rel  = getattr(item, "relation",  None)
                    # pretty condition label
                    conj_name = getattr(item, "name", None)
                    if not conj_name:
                        if cond is not None:
                            conj_name = cond.pretty() if hasattr(cond, "pretty") else repr(cond)
                        else:
                            conj_name = "TRUE"
                    conj_name = _pretty_conj_name(conj_name)

                    # collect equality lines from relation
                    eq_parts = []
                    if rel is not None:
                        if AllOf and isinstance(rel, AllOf):
                            eq_parts = list(rel.parts)
                        else:
                            eq_parts = [rel]

                    eq_lines = []
                    for p in eq_parts:
                        # prefer pretty() if available; otherwise repr()
                        txt = p.pretty() if hasattr(p, "pretty") else repr(p)
                        eq_lines.append(_pretty_conj_name(txt))

                    # mask / support from condition
                    mask = None
                    support = None
                    try:
                        mask = np.asarray(self.mask(cond), dtype=bool)
                        support = int(mask.sum())
                    except Exception:
                        pass

                    out.append((conj_name, eq_lines, support, mask))
                return out

            cie_norm = _normalize_cie(getattr(self, "class_cond_implies_equalities_", []))

            if not cie_norm:
                lines.append("  (none)\n")
            else:
                # Skip groups characterized by an equivalence
                characterized_sigs: set[bytes] = set()
                if skip_characterized:
                    native_eqv = list(getattr(self, "class_equivalences_", []) or [])
                    for E in native_eqv:
                        for side in (getattr(E, "A", None), getattr(E, "B", None)):
                            if side is None:
                                continue
                            try:
                                characterized_sigs.add(np.asarray(self.mask(side), dtype=bool).tobytes())
                            except Exception:
                                pass

                # ranking key
                def _key_group(item) -> float:
                    conj_name, eq_lines, support, mask = item
                    pred = {n: p for (n, p) in getattr(self, "nonredundant_conjunctions_", [])}.get(conj_name)
                    s = int(support) if support is not None else (_support(pred) if pred is not None else 0)
                    bal = _balancedness(pred) if pred is not None else 0.0
                    ar  = _arity_of_name(conj_name)
                    k   = len(eq_lines or [])
                    txtlen = sum(len(x) for x in (eq_lines or [])[:8]) + len(conj_name)
                    return 0.9*np.log1p(s) + 0.9*bal + 0.5*k - _simplicity_penalty(arity=ar, k_eq=k, text_len=txtlen)

                ranked = []
                for item in cie_norm:
                    conj_name, eq_lines, support, mask = item
                    try:
                        if skip_characterized and mask is not None and np.asarray(mask, dtype=bool).tobytes() in characterized_sigs:
                            continue
                    except Exception:
                        pass
                    ranked.append(item)

                ranked.sort(key=_key_group, reverse=True)
                items, extra = (ranked[:max_per_section], max(0, len(ranked) - max_per_section))

                for conj_name, eq_lines, support, mask in items:
                    head = f"  • Const[{conj_name}] — {len(eq_lines)} equalit{'y' if len(eq_lines)==1 else 'ies'}"
                    if support is not None:
                        head += f" (support={int(support)})"
                    # small witness list
                    if mask is not None:
                        try:
                            names = self.df.index[np.asarray(mask, dtype=bool)].tolist()
                            if names:
                                head += f"   (e.g., {', '.join(names[:3])})"
                        except Exception:
                            pass
                    lines.append(head)

                    if list_const_equalities and eq_lines:
                        for eq_line in eq_lines[:max_equalities_per_cond]:
                            lines.append(f"      - {eq_line}")
                        if len(eq_lines) > max_equalities_per_cond:
                            lines.append(f"        … +{len(eq_lines) - max_equalities_per_cond} more")

                if extra:
                    lines.append(f"  … +{extra} more")
                lines.append("")


        # ---------- Ratio bounds (optional) ----------
        if show_ratio_bounds:
            lines.append("Ratio bounds (groups):")
            if not ratio:
                lines.append("  (none)\n")
            else:
                items, extra = clip(ratio)
                for r in items:
                    t = r.pretty() if hasattr(r, "pretty") else repr(r)
                    try:
                        t = _pretty_conj_name(t)
                    except Exception:
                        pass
                    lines.append(f"  • {t}")
                if extra:
                    lines.append(f"  … +{extra} more")
                lines.append("")

        lines.append("──────────────────────────────────────────────")
        s = "\n".join(lines)
        if verbose:
            print(s)
        return s
