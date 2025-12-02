# src/txgraffiti2025/relations/class_logic.py

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import combinations
from typing import Dict, List, Tuple, Iterable, Optional, Any

import numpy as np

from .core import DataModel, MaskCache
from ..forms.predicates import Predicate, Where
from ..forms.generic_conjecture import TRUE  # canonical TRUE instance

__all__ = ["ClassLogic"]

# Single source of truth for pretty naming
_NAME_DELIM = " ∧ "

def _and_all(preds: Iterable[Predicate]) -> Predicate:
    preds = list(preds)
    if not preds:
        return TRUE
    return reduce(lambda a, b: a & b, preds)


@dataclass(slots=True)
class _ConjRecord:
    """
    Internal, structural representation of a hypothesis class:
      base_parts: tuple of base boolean column names (sorted)
      extras:     tuple of extra boolean column names (sorted)
      pred:       Predicate = AND of (base predicate and all extras)
      name:       pretty name using _NAME_DELIM
    We never parse names to recover structure; we carry it explicitly.
    """
    base_parts: Tuple[str, ...]
    extras: Tuple[str, ...]
    pred: Predicate
    name: str

    @property
    def arity(self) -> int:
        return len(self.extras)

    def parts(self) -> Tuple[str, ...]:
        return (*self.base_parts, *self.extras)


class ClassLogic:
    """
    Construct and manage hypothesis classes from boolean-like columns of a DataModel.

    Pipeline
    --------
    1) Detect base predicate B as the AND of all boolean columns that are true on every row.
    2) Enumerate conjunctions  B ∧ p1 ∧ ... ∧ pk  for 1 ≤ k ≤ max_arity over non-base columns.
    3) Normalize:
        • Equivalence: same mask & same arity → grouped together
        • Redundancy: within a fixed mask, any strict superset of extras is redundant
    4) Sort by generality (support).
    5) Canonicalize arbitrary (name, Predicate) lists by equivalence-merge + dominance pruning.
    """

    # immutable inputs
    model: DataModel
    cache: MaskCache

    # base
    _base_pred: Predicate
    _base_parts: Tuple[str, ...]
    _base_name: str

    # storage (computed lazily)
    _pool: Dict[str, Predicate]                      # non-base boolean preds
    _enumerated: List[_ConjRecord] | None
    _nonredundant: List[_ConjRecord] | None
    _redundant: List[_ConjRecord] | None
    _equiv_groups: List[List[_ConjRecord]] | None
    _sorted: List[_ConjRecord] | None

    def __init__(self, model: DataModel, cache: MaskCache) -> None:
        self.model = model
        self.cache = cache

        # Identify universally-true boolean columns
        base_cols: List[str] = []
        for name, pred in model.preds.items():
            if bool(self.cache.mask(pred).all()):
                base_cols.append(name)
        base_cols = sorted(base_cols)
        self._base_parts = tuple(base_cols)
        self._base_pred = _and_all([model.preds[c] for c in self._base_parts])
        self._base_name = "TRUE" if not self._base_parts else _NAME_DELIM.join(self._base_parts)

        # Pool = remaining boolean predicates
        self._pool = {n: p for n, p in model.preds.items() if n not in self._base_parts}

        # lazy fields
        self._enumerated = None
        self._nonredundant = None
        self._redundant = None
        self._equiv_groups = None
        self._sorted = None

    # ──────────────────────────────────────────────────────────────────
    # Base API
    # ──────────────────────────────────────────────────────────────────
    def base_predicate(self) -> Predicate:
        return self._base_pred

    def base_name(self) -> str:
        return self._base_name

    def base_parts(self) -> Tuple[str, ...]:
        return self._base_parts

    def base_predicates(self) -> Dict[str, Predicate]:
        """Non-base boolean predicates (available for enumeration)."""
        return dict(self._pool)

    # ──────────────────────────────────────────────────────────────────
    # Enumeration
    # ──────────────────────────────────────────────────────────────────
    def enumerate(
        self,
        *,
        max_arity: int = 2,
        min_support: int = 0,
        include_base: bool = False,
    ) -> List[Tuple[str, Predicate]]:
        """
        Enumerate (B ∧ p1 ∧ … ∧ pk) with 1 ≤ k ≤ max_arity over non-base predicates.

        Parameters
        ----------
        max_arity : int
            Maximum number of extra predicates to conjoin.
        min_support : int
            If > 0, drop conjunctions with support < min_support.
        include_base : bool
            Include the pure base class B (even if B == TRUE).

        Returns
        -------
        List[(name, Predicate)]
        """
        if max_arity < 1:
            raise ValueError("max_arity must be >= 1")

        items = sorted(self._pool.items(), key=lambda t: t[0])
        base_parts = self._base_parts  # tuple
        base_pred = self._base_pred

        def mk_record(extra_names: Tuple[str, ...]) -> _ConjRecord:
            extras_sorted = tuple(sorted(extra_names))
            preds = [self.model.preds[n] for n in extras_sorted]
            conj = _and_all([base_pred, *_safe(preds)])
            name = "TRUE" if not base_parts and not extras_sorted else _NAME_DELIM.join((*base_parts, *extras_sorted))
            return _ConjRecord(base_parts=base_parts, extras=extras_sorted, pred=conj, name=name)

        def _safe(ps: Iterable[Predicate]) -> List[Predicate]:
            return [p for p in ps if p is not None]

        recs: List[_ConjRecord] = []

        # Optional pure base
        if include_base:
            r_base = mk_record(tuple())
            if min_support <= 0 or int(self.cache.mask(r_base.pred).sum()) >= min_support:
                recs.append(r_base)

        # Arity 1..max_arity
        names_only = [n for (n, _) in items]
        for k in range(1, max_arity + 1):
            for combo in combinations(names_only, k):
                r = mk_record(tuple(combo))
                if min_support > 0:
                    if int(self.cache.mask(r.pred).sum()) < min_support:
                        continue
                recs.append(r)

        self._enumerated = recs
        # invalidate downstream caches
        self._nonredundant = self._redundant = self._equiv_groups = self._sorted = None
        return [(r.name, r.pred) for r in recs]

    # ──────────────────────────────────────────────────────────────────
    # Normalization (equivalence + redundancy)
    # ──────────────────────────────────────────────────────────────────
    def normalize(self) -> Tuple[
        List[Tuple[str, Predicate]],
        List[Tuple[str, Predicate]],
        List[List[Tuple[str, Predicate]]],
    ]:
        """
        Partition enumerated conjunctions into:
          • nonredundant
          • redundant
          • equivalence groups (same mask AND same arity)
        Always ensures the pure base (B) appears in nonredundant.
        """
        if self._enumerated is None:
            self.enumerate()

        records = self._enumerated or []

        # Build signatures and supports once
        rec_info: List[dict] = []
        for idx, r in enumerate(records):
            m = self.cache.mask(r.pred)
            arr = m.to_numpy(dtype=np.bool_, copy=False)
            sig = np.packbits(arr, bitorder="little").tobytes()
            rec_info.append({
                "idx": idx,
                "rec": r,
                "mask": arr,
                "sig": sig,
                "arity": r.arity,
                "support": int(arr.sum()),
                "extras_set": frozenset(r.extras),
            })

        # Equivalence groups: by (sig, arity)
        by_sig_arity: Dict[tuple, List[dict]] = {}
        for info in rec_info:
            by_sig_arity.setdefault((info["sig"], info["arity"]), []).append(info)

        equiv_groups_records: List[List[_ConjRecord]] = []
        for (_, _), group in by_sig_arity.items():
            if len(group) > 1:
                equiv_groups_records.append([g["rec"] for g in group])

        # Redundancy within fixed signature:
        # keep minimal extras (by strict subset) per signature
        redundant_ids = set()
        by_sig: Dict[bytes, List[dict]] = {}
        for info in rec_info:
            by_sig.setdefault(info["sig"], []).append(info)

        for sig, group in by_sig.items():
            group_sorted = sorted(group, key=lambda z: z["arity"])
            minimal: List[frozenset] = []
            for g in group_sorted:
                ex = g["extras_set"]
                if any(me < ex for me in minimal):  # strict subset already present
                    redundant_ids.add(g["idx"])
                else:
                    minimal.append(ex)

        nonred_records = [rec_info[i]["rec"] for i in range(len(rec_info)) if rec_info[i]["idx"] not in redundant_ids]
        red_records = [ri["rec"] for ri in rec_info if ri["idx"] in redundant_ids]

        # Ensure base appears in nonredundant (even if not enumerated)
        base_rec = _ConjRecord(self._base_parts, tuple(), self._base_pred, self._base_name)
        if all(br.name != base_rec.name for br in nonred_records):
            nonred_records.append(base_rec)

        self._nonredundant = nonred_records
        self._redundant = red_records
        self._equiv_groups = equiv_groups_records
        self._sorted = None

        return ([(r.name, r.pred) for r in nonred_records],
                [(r.name, r.pred) for r in red_records],
                [[(r.name, r.pred) for r in grp] for grp in equiv_groups_records])

    # ──────────────────────────────────────────────────────────────────
    # Sorting by generality (support)
    # ──────────────────────────────────────────────────────────────────
    def sort_by_generality(self, *, ascending: bool = False) -> List[Tuple[str, Predicate]]:
        if self._nonredundant is None:
            self.normalize()
        scored = []
        for r in self._nonredundant or []:
            support = int(self.cache.mask(r.pred).sum())
            scored.append((support, r.name, r))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=not ascending)
        self._sorted = [r for (_, __, r) in scored]
        return [(r.name, r.pred) for r in self._sorted]

    # ──────────────────────────────────────────────────────────────────
    # Canonicalize external (name, Predicate) sets
    # ──────────────────────────────────────────────────────────────────
    def canonicalize(
        self,
        hyps: List[Tuple[str, Predicate]],
    ) -> Tuple[
        List[Tuple[str, Predicate]],
        List[List[Tuple[str, Predicate]]],
        List[Tuple[str, Predicate, str]],
    ]:
        """
        Merge equivalents and drop dominated classes among an arbitrary hypothesis set.

        Returns
        -------
        kept : List[(name, Predicate)]
            Representatives after equivalence-merge and dominance pruning.
        merged_groups : List[List[(name, Predicate)]]
            Groups of hypotheses merged by mask-equivalence.
        dominated : List[(dominated_name, dominated_pred, by_name)]
            Pairs indicating which names were removed as strict subsets of another.
        """
        if not hyps:
            return [], [], []

        # compute masks & signatures
        recs = []
        for (name, pred) in hyps:
            m = self.cache.mask(pred)
            arr = m.to_numpy(dtype=np.bool_, copy=False)
            sig = np.packbits(arr, bitorder="little").tobytes()
            recs.append({"name": name, "pred": pred, "mask": arr, "sig": sig, "support": int(arr.sum())})

        # group equivalents by signature
        by_sig: Dict[bytes, List[dict]] = {}
        for r in recs:
            by_sig.setdefault(r["sig"], []).append(r)

        reps: List[Tuple[str, Predicate]] = []
        merged_groups: List[List[Tuple[str, Predicate]]] = []
        for sig, group in by_sig.items():
            # Representative: max support, then shortest name, then lexicographic
            rep = sorted(group, key=lambda z: (-z["support"], len(z["name"]), z["name"]))[0]
            reps.append((rep["name"], rep["pred"]))
            if len(group) > 1:
                merged_groups.append([(g["name"], g["pred"]) for g in group])

        # dominance pruning among representatives
        dom: List[Tuple[str, Predicate, str]] = []
        mats = []
        for (n, p) in reps:
            arr = self.cache.mask(p).to_numpy(dtype=np.bool_, copy=False)
            mats.append((n, p, arr))

        for i, (ni, pi, mi) in enumerate(mats):
            for j, (nj, pj, mj) in enumerate(mats):
                if i == j:
                    continue
                # strict subset: mi ⊂ mj
                if (mi & ~mj).sum() == 0 and (mj & ~mi).sum() > 0:
                    dom.append((ni, pi, nj))

        dom_names = {n for (n, _, __) in dom}
        kept = [(n, p) for (n, p) in reps if n not in dom_names]
        return kept, merged_groups, dom

    def to_registry(
        self,
        *,
        kind: str = "classes",
        sorted_by_generality: bool = True,
        include_mask_sig: bool = True,
        include_pred: bool = False,
    ) -> None:
        """
        Materialize the current nonredundant hypothesis catalogue into model.registry[kind].
        Each record includes: name, arity, support, extras, base_parts, (optional) mask_sig, (optional) pred.
        """
        # Ensure we have the finalized nonredundant set
        if self._nonredundant is None:
            self.normalize()

        # Choose order
        recs = self._nonredundant or []
        if sorted_by_generality:
            if self._sorted is None:
                self.sort_by_generality()
            recs = self._sorted or recs

        # Write
        for r in recs:
            m = self.cache.mask(r.pred).to_numpy(dtype=np.bool_, copy=False)
            payload = {
                "name": r.name,
                "arity": r.arity,
                "support": int(m.sum()),
                "extras": list(r.extras),
                "base_parts": list(r.base_parts),
            }
            if include_mask_sig:
                payload["mask_sig"] = np.packbits(m, bitorder="little").tobytes()
            if include_pred:
                payload["pred"] = r.pred
            self.model.record_class(**payload)

    # ──────────────────────────────────────────────────────────────────
    # Introspection (optional)
    # ──────────────────────────────────────────────────────────────────
    def enumerated(self) -> List[Tuple[str, Predicate]] | None:
        return None if self._enumerated is None else [(r.name, r.pred) for r in self._enumerated]

    def nonredundant(self) -> List[Tuple[str, Predicate]] | None:
        return None if self._nonredundant is None else [(r.name, r.pred) for r in self._nonredundant]

    def redundant(self) -> List[Tuple[str, Predicate]] | None:
        return None if self._redundant is None else [(r.name, r.pred) for r in self._redundant]

    def equivalence_groups(self) -> List[List[Tuple[str, Predicate]]] | None:
        if self._equiv_groups is None:
            return None
        return [[(r.name, r.pred) for r in grp] for grp in self._equiv_groups]

    def sorted_by_generality(self) -> List[Tuple[str, Predicate]] | None:
        return None if self._sorted is None else [(r.name, r.pred) for r in self._sorted]
