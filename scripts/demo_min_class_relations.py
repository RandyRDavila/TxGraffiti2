# # #!/usr/bin/env python3
# # from __future__ import annotations

# # from dataclasses import dataclass
# # from itertools import combinations
# # from typing import Iterable, List, Tuple, Dict, Set

# # import numpy as np
# # import pandas as pd

# # # Your modules
# # from txgraffiti.example_data import graph_data as df
# # df['nontrivial'] = df['connected']
# # from txgraffiti2025.graffiti.core import DataModel, MaskCache  # DataModel has .boolean_names(), .boolean(...)

# # from txgraffiti2025.forms.predicates import Predicate, Where
# # from txgraffiti2025.forms.class_relations import ClassInclusion, ClassEquivalence

# # # ----------------------------
# # # Switches / knobs
# # # ----------------------------
# # MAX_ARITY      = 2
# # MIN_SUPPORT    = 0
# # INCLUDE_BASE   = True
# # USE_NEGATIONS  = False   # keep False for now; turn on later if you want ~preds, doubles results quickly
# # SOPHIE_SCOPE_CUTOFF = 1  # min new coverage rows to accept by Sophie
# # TOP_K_BY_SUPPORT = 10    # for printing “top classes”

# # NAME_DELIM = " ∧ "

# # # ----------------------------
# # # Tiny helpers
# # # ----------------------------
# # def bool_mask(cache: MaskCache, p: Predicate) -> pd.Series:
# #     """Aligned boolean mask (uses cache)."""
# #     return cache.mask(p)

# # def _and_all(preds: Iterable[Predicate]) -> Predicate:
# #     preds = list(preds)
# #     if not preds:
# #         # empty AND == TRUE is represented by a Where that returns True everywhere;
# #         # but we won't actually call this path in this script
# #         base_true = Where(lambda d: pd.Series(True, index=d.index), name="TRUE")
# #         return base_true
# #     out = preds[0]
# #     for q in preds[1:]:
# #         out = out & q
# #     return out

# # @dataclass(frozen=True)
# # class ClassRec:
# #     name: str
# #     pred: Predicate
# #     parts: Tuple[str, ...]     # tuple of boolean names in the conjunction
# #     extras: Tuple[str, ...]    # the parts beyond base
# #     arity: int

# # # ----------------------------
# # # Base predicate: all columns that are identically True
# # # ----------------------------
# # def compute_base(model: DataModel, cache: MaskCache) -> Tuple[Predicate, Tuple[str, ...], str]:
# #     always_true: List[str] = []
# #     for nm in model.boolean_names:
# #         if bool(bool_mask(cache, model.boolean(nm)).all()):
# #             always_true.append(nm)
# #     always_true = sorted(always_true)
# #     if not always_true:
# #         # TRUE base
# #         base_pred = Where(lambda d: pd.Series(True, index=d.index), name="TRUE")
# #         base_name = "TRUE"
# #         base_parts: Tuple[str, ...] = tuple()
# #     else:
# #         preds = [model.boolean(n) for n in always_true]
# #         base_pred = _and_all(preds)
# #         base_parts = tuple(always_true)
# #         base_name = NAME_DELIM.join(base_parts)
# #     return base_pred, base_parts, base_name

# # # ----------------------------
# # # Enumerate conjunctions of non-base predicates (optionally with negations)
# # # ----------------------------
# # def enumerate_conjunctions(
# #     model: DataModel,
# #     cache: MaskCache,
# #     base_pred: Predicate,
# #     base_parts: Tuple[str, ...],
# #     pool: List[str],
# #     *,
# #     max_arity: int,
# #     min_support: int,
# #     include_base: bool,
# #     use_negations: bool,
# # ) -> List[ClassRec]:
# #     def mk_rec(extra_names: Tuple[str, ...]) -> ClassRec:
# #         extras_sorted = tuple(sorted(extra_names))
# #         preds = [model.boolean(n) for n in extras_sorted]
# #         extras_conj = _and_all(preds)
# #         conj = _and_all([base_pred, extras_conj])  # ← fixed: no splat of a Predicate

# #         parts = tuple((*base_parts, *extras_sorted))
# #         name = "TRUE" if not parts else NAME_DELIM.join(parts)
# #         return ClassRec(name=name, pred=conj, parts=parts, extras=extras_sorted, arity=len(extras_sorted))

# #     out: List[ClassRec] = []

# #     # Optional: the pure base class as a record
# #     if include_base:
# #         if not base_parts:
# #             base_rec = ClassRec(name="TRUE", pred=base_pred, parts=tuple(), extras=tuple(), arity=0)
# #         else:
# #             base_rec = ClassRec(name=NAME_DELIM.join(base_parts), pred=base_pred, parts=base_parts, extras=tuple(), arity=0)
# #         if min_support <= 0 or int(bool_mask(cache, base_rec.pred).sum()) >= min_support:
# #             out.append(base_rec)

# #     atoms: List[str] = sorted(pool)

# #     # Enumerate arity 1..max_arity
# #     for k in range(1, max_arity + 1):
# #         for combo in combinations(atoms, k):
# #             r = mk_rec(combo)
# #             if min_support > 0:
# #                 if int(bool_mask(cache, r.pred).sum()) < min_support:
# #                     continue
# #             out.append(r)

# #     # Optionally add negated atoms as well (not common for “class” relations; off by default)
# #     if use_negations:
# #         neg_atoms = [f"¬{a}" for a in atoms]
# #         # add predicates for negated columns
# #         for a in atoms:
# #             pa = model.boolean(a)
# #             neg_pred = Where(lambda d, p=pa: ~p.mask(d), name=f"¬{a}")
# #             model.add_boolean(neg_pred)

# #         # re-enumerate mixed combos of negations if needed (omitted for simplicity)
# #         # (you can add another loop here mirroring the one above)

# #     return out

# # # ----------------------------
# # # Normalize: equivalence by mask+arity, redundancy by extras strict-superset within same mask
# # # ----------------------------
# # @dataclass
# # class _Info:
# #     idx: int
# #     rec: ClassRec
# #     mask: np.ndarray
# #     sig: bytes     # packed bits signature
# #     arity: int
# #     support: int
# #     extras_set: frozenset

# # def normalize_equiv_and_redundancy(
# #     cache: MaskCache,
# #     recs: List[ClassRec],
# # ) -> Tuple[List[Tuple[ClassRec, np.ndarray]], List[Tuple[ClassRec, np.ndarray]], List[List[ClassRec]]]:
# #     infos: List[_Info] = []
# #     for i, r in enumerate(recs):
# #         m = bool_mask(cache, r.pred).to_numpy(dtype=np.bool_, copy=False)
# #         sig = np.packbits(m, bitorder="little").tobytes()
# #         infos.append(_Info(idx=i, rec=r, mask=m, sig=sig, arity=r.arity, support=int(m.sum()), extras_set=frozenset(r.extras)))

# #     # equivalence groups: same sig + same arity
# #     by_sig_arity: Dict[Tuple[bytes, int], List[_Info]] = {}
# #     for info in infos:
# #         by_sig_arity.setdefault((info.sig, info.arity), []).append(info)

# #     equiv_groups: List[List[ClassRec]] = []
# #     for (_, _), group in by_sig_arity.items():
# #         if len(group) > 1:
# #             equiv_groups.append([g.rec for g in group])

# #     # redundancy: within same sig, keep minimal extras (no strict superset)
# #     redundant_ids: Set[int] = set()
# #     by_sig: Dict[bytes, List[_Info]] = {}
# #     for info in infos:
# #         by_sig.setdefault(info.sig, []).append(info)

# #     for sig, group in by_sig.items():
# #         group_sorted = sorted(group, key=lambda z: z.arity)
# #         minimal: List[frozenset] = []
# #         for g in group_sorted:
# #             ex = g.extras_set
# #             if any(me < ex for me in minimal):   # strict subset already present
# #                 redundant_ids.add(g.idx)
# #             else:
# #                 minimal.append(ex)

# #     nonred = [(info.rec, info.mask) for info in infos if info.idx not in redundant_ids]
# #     red    = [(info.rec, info.mask) for info in infos if info.idx in redundant_ids]
# #     return nonred, red, equiv_groups

# # # ----------------------------
# # # R₁ candidates (A ⊆ B) / (A ≡ B)
# # # ----------------------------
# # def r1_candidates(
# #     df: pd.DataFrame,
# #     cache: MaskCache,
# #     nonred: List[Tuple[ClassRec, np.ndarray]],
# # ) -> Tuple[List[ClassInclusion], List[ClassEquivalence]]:
# #     incs: List[ClassInclusion] = []
# #     eqs:  List[ClassEquivalence] = []

# #     # prepack (rec, mask) for speed
# #     for i in range(len(nonred)):
# #         rec_i, mask_i = nonred[i]
# #         for j in range(i + 1, len(nonred)):
# #             rec_j, mask_j = nonred[j]

# #             # equality test
# #             if mask_i.shape == mask_j.shape and np.array_equal(mask_i, mask_j):
# #                 eqs.append(ClassEquivalence(rec_i.pred, rec_j.pred))
# #                 continue

# #             # inclusion tests
# #             # i ⊆ j  iff (~i) | j is all True  ⇔  (i & ~j).any() == False
# #             if (~mask_i & mask_j).sum() == 0 and (mask_i & ~mask_j).sum() == 0:
# #                 # equal; already handled
# #                 pass
# #             if (mask_i & ~mask_j).sum() == 0:
# #                 incs.append(ClassInclusion(rec_i.pred, rec_j.pred))
# #             if (mask_j & ~mask_i).sum() == 0:
# #                 incs.append(ClassInclusion(rec_j.pred, rec_i.pred))

# #     return incs, eqs

# # # ----------------------------
# # # Sophie heuristic (post-filter on hypothesis coverage)
# # # ----------------------------
# # def sophie_accept_conj(
# #     new_hyp_mask: pd.Series,
# #     accepted_union: pd.Series,
# #     *,
# #     min_new_rows: int = 1,
# # ) -> Tuple[bool, pd.Series]:
# #     """Return (accept?, new_union)."""
# #     delta = new_hyp_mask & ~accepted_union
# #     if int(delta.sum()) >= min_new_rows:
# #         return True, (accepted_union | new_hyp_mask)
# #     return False, accepted_union

# # # ----------------------------
# # # Extra inclusion filters to remove noise
# # # ----------------------------
# # def is_tautology_by_parts(lhs_parts: Tuple[str, ...], rhs_parts: Tuple[str, ...]) -> bool:
# #     """Drop (B∧X) ⊆ B and similar: rhs ⊆ lhs."""
# #     set_l, set_r = set(lhs_parts), set(rhs_parts)
# #     return set_r.issubset(set_l)

# # def is_reverse_subset_artifact(lhs_parts: Tuple[str, ...], rhs_parts: Tuple[str, ...]) -> bool:
# #     """Drop A ⊆ (A∧X) (dataset artifact; not a structural fact)."""
# #     set_l, set_r = set(lhs_parts), set(rhs_parts)
# #     return set_l.issubset(set_r)

# # # resolve class name from Predicate by searching nonred
# # def _name_parts_for_pred(p: Predicate, catalog: List[Tuple[ClassRec, np.ndarray]]) -> Tuple[str, Tuple[str, ...]]:
# #     for rec, _ in catalog:
# #         if rec.pred is p:
# #             return rec.name, rec.parts
# #     # fallback (should not happen)
# #     return repr(p), tuple()

# # def rel_scope_size(df: pd.DataFrame, cache: MaskCache, rel) -> int:
# #     if isinstance(rel, ClassEquivalence):
# #         # scope = rows where either side holds (same mask when equivalent)
# #         a = cache.mask(rel.A)
# #         return int(a.sum())
# #     if isinstance(rel, ClassInclusion):
# #         # scope = rows where hypothesis holds
# #         a = cache.mask(rel.A)
# #         return int(a.sum())
# #     return 0

# # # ----------------------------
# # # Main
# # # ----------------------------
# # def main():
# #     model = DataModel(df.copy())
# #     cache = MaskCache(model)

# #     print("\n\n=== Base scan ===")
# #     print("booleans :", ", ".join(model.boolean_names))
# #     base_pred, base_parts, base_name = compute_base(model, cache)
# #     print(f"\nBase: {base_name}   (support={int(bool_mask(cache, base_pred).sum())})")
# #     print(f"  base parts: {base_parts!r}")

# #     # pool = all boolean names excluding base parts
# #     pool = [n for n in model.boolean_names if n not in set(base_parts)]

# #     # enumerate
# #     recs = enumerate_conjunctions(
# #         model, cache, base_pred, base_parts, pool,
# #         max_arity=MAX_ARITY, min_support=MIN_SUPPORT, include_base=INCLUDE_BASE, use_negations=USE_NEGATIONS,
# #     )
# #     print(f"\nEnumerated {len(recs)} classes (including base).")

# #     # normalize
# #     nonred, red, equiv_groups = normalize_equiv_and_redundancy(cache, recs)
# #     print(f"Nonredundant: {len(nonred)} | Redundant: {len(red)} | Equivalence groups: {len(equiv_groups)}")

# #     # top by support
# #     scored = sorted(nonred, key=lambda t: int(t[1].sum()), reverse=True)
# #     print("\nTop classes by support:")
# #     for rec, m in scored[:TOP_K_BY_SUPPORT]:
# #         print(f"  • {rec.name:<60} support={int(m.sum())}")

# #     # pretty-print one example equivalence group if present
# #     if equiv_groups:
# #         print("\nExample equivalence group (same mask & arity):")
# #         for r in equiv_groups[0]:
# #             print(" ", r.name)

# #     # pretty-print a few redundant examples if present
# #     if red:
# #         print("\nExample redundant classes (dominated within same signature):")
# #         for (r, _m) in red[:5]:
# #             print(f"  {r.name:<40} extras={r.extras!r}")

# #     # R1 candidates
# #     incs, eqs = r1_candidates(model.df, cache, nonred)
# #     print(f"\nPre-Sophie: inclusions={len(incs)} | equivalences={len(eqs)}")

# #     # Apply extra inclusion filters first (by parts)
# #     # Build quick lookup map: Predicate -> parts
# #     parts_map: Dict[Predicate, Tuple[str, ...]] = {}
# #     for rec, _ in nonred:
# #         parts_map[rec.pred] = rec.parts

# #     def _parts(p: Predicate) -> Tuple[str, ...]:
# #         return parts_map.get(p, tuple())

# #     filtered_incs: List[ClassInclusion] = []
# #     for r in incs:
# #         lhs_parts = _parts(r.A)
# #         rhs_parts = _parts(r.B)
# #         if is_tautology_by_parts(lhs_parts, rhs_parts):
# #             continue
# #         if is_reverse_subset_artifact(lhs_parts, rhs_parts):
# #             continue
# #         filtered_incs.append(r)

# #     # Sophie post-filter
# #     accepted_incs: List[ClassInclusion] = []
# #     accepted_eqs: List[ClassEquivalence] = []

# #     union_incs = pd.Series(False, index=model.df.index)
# #     union_eqs  = pd.Series(False, index=model.df.index)

# #     # sort stable: larger hypothesis support first (greedy coverage)
# #     filtered_incs_sorted = sorted(
# #         filtered_incs,
# #         key=lambda rel: int(bool_mask(cache, rel.A).sum()),
# #         reverse=True,
# #     )
# #     eqs_sorted = sorted(
# #         eqs,
# #         key=lambda rel: int(bool_mask(cache, rel.A).sum()),
# #         reverse=True,
# #     )

# #     for rel in filtered_incs_sorted:
# #         accept, union_incs = sophie_accept_conj(bool_mask(cache, rel.A), union_incs, min_new_rows=SOPHIE_SCOPE_CUTOFF)
# #         if accept:
# #             accepted_incs.append(rel)

# #     for rel in eqs_sorted:
# #         # for equivalence, “hypothesis” = either side; use A
# #         accept, union_eqs = sophie_accept_conj(bool_mask(cache, rel.A), union_eqs, min_new_rows=SOPHIE_SCOPE_CUTOFF)
# #         if accept:
# #             accepted_eqs.append(rel)

# #     # Print results
# #     print("\nR₁ Conjectures (Class Inclusions, Sophie-selected):")
# #     if not accepted_incs:
# #         print("  (none)")
# #     else:
# #         for r in accepted_incs:
# #             nmA, _pA = _name_parts_for_pred(r.A, nonred)
# #             nmB, _pB = _name_parts_for_pred(r.B, nonred)
# #             scope = int(bool_mask(cache, r.A).sum())
# #             print(f"  • ({nmA}) ⊆ ({nmB})   [scope={scope}]")

# #     print("\nR₁ Conjectures (Class Equivalences, Sophie-selected):")
# #     if not accepted_eqs:
# #         print("  (none)")
# #     else:
# #         for r in accepted_eqs:
# #             nmA, _pA = _name_parts_for_pred(r.A, nonred)
# #             nmB, _pB = _name_parts_for_pred(r.B, nonred)
# #             scope = int(bool_mask(cache, r.A).sum())
# #             print(f"  • ({nmA}) ≡ ({nmB})   [scope={scope}]")

# #     print("\nCoverage summary:")
# #     print(f"  accepted inclusions cover rows:   {int(union_incs.sum())} / {len(model.df)}")
# #     print(f"  accepted equivalences cover rows: {int(union_eqs.sum())} / {len(model.df)}")
# #     print("\nDone.")

# # if __name__ == "__main__":
# #     main()



# #!/usr/bin/env python3
# from __future__ import annotations

# from dataclasses import dataclass
# from itertools import combinations
# from typing import Iterable, List, Tuple, Dict, Set, Optional

# import numpy as np
# import pandas as pd

# # ───────────────────────── Your modules ───────────────────────── #
# from txgraffiti.example_data import graph_data as df
# df['nontrivial'] = df['connected']

# # DataModel exposes .boolean_names (attr or method) and .boolean(name)
# from txgraffiti2025.graffiti.core import DataModel, MaskCache

# from txgraffiti2025.forms.predicates import Predicate, Where
# from txgraffiti2025.forms.class_relations import ClassInclusion, ClassEquivalence

# # ───────────────────────── Switches / knobs ───────────────────────── #
# MAX_ARITY            = 2
# MIN_SUPPORT          = 0
# INCLUDE_BASE         = True
# USE_NEGATIONS        = False   # safe default: off
# NEG_MIN_POS_SUPPORT  = 20      # if negations enabled: require enough positives
# NEG_MIN_NEG_SUPPORT  = 20      # and enough negatives to avoid “nearly-all” predicates

# SOPHIE_SCOPE_CUTOFF  = 1       # min new rows to accept by Sophie
# TOP_K_BY_SUPPORT     = 10      # for printing “top classes”
# NAME_DELIM           = " ∧ "

# # ───────────────────────── Tiny helpers ───────────────────────── #
# def _boolean_names(model: DataModel) -> List[str]:
#     """Support both attribute and method styles."""
#     names = getattr(model, "boolean_names", None)
#     return list(names() if callable(names) else names)

# def bool_mask(cache: MaskCache, p: Predicate) -> pd.Series:
#     """Aligned boolean mask (uses cache)."""
#     return cache.mask(p)

# def _and_all(preds: Iterable[Predicate]) -> Predicate:
#     preds = list(preds)
#     if not preds:
#         # An explicit TRUE predicate; rarely used but safe.
#         return Where(lambda d: pd.Series(True, index=d.index), name="TRUE")
#     out = preds[0]
#     for q in preds[1:]:
#         out = out & q
#     return out

# def _short(name: str, n: int = 60) -> str:
#     return name if len(name) <= n else name[: n - 1] + "…"

# @dataclass(frozen=True)
# class ClassRec:
#     name: str
#     pred: Predicate
#     parts: Tuple[str, ...]     # boolean names in conjunction (including base parts)
#     extras: Tuple[str, ...]    # parts beyond base
#     arity: int                 # len(extras)

# # ───────────────────────── Base predicate ───────────────────────── #
# def compute_base(model: DataModel, cache: MaskCache) -> Tuple[Predicate, Tuple[str, ...], str, bytes]:
#     """Return (base_pred, base_parts, base_name, base_sig)."""
#     always_true: List[str] = []
#     for nm in _boolean_names(model):
#         if bool(bool_mask(cache, model.boolean(nm)).all()):
#             always_true.append(nm)

#     always_true = sorted(always_true)

#     if not always_true:
#         base_pred = Where(lambda d: pd.Series(True, index=d.index), name="TRUE")
#         base_parts: Tuple[str, ...] = tuple()
#         base_name = "TRUE"
#     else:
#         preds = [model.boolean(n) for n in always_true]
#         base_pred = _and_all(preds)
#         base_parts = tuple(always_true)
#         base_name = NAME_DELIM.join(base_parts)

#     # base signature (bit-packed) for quick comparisons
#     base_mask = bool_mask(cache, base_pred).to_numpy(dtype=np.bool_, copy=False)
#     base_sig  = np.packbits(base_mask, bitorder="little").tobytes()

#     return base_pred, base_parts, base_name, base_sig

# # ───────────────────────── Enumeration ───────────────────────── #
# def add_negations_safely(
#     model: DataModel,
#     cache: MaskCache,
#     atoms: List[str],
#     *,
#     min_pos_support: int,
#     min_neg_support: int,
# ) -> List[str]:
#     """Add ¬X only when both X and ¬X have reasonable support."""
#     added = []
#     for a in atoms:
#         pa = model.boolean(a)
#         m = bool_mask(cache, pa)
#         if int(m.sum()) >= min_pos_support and int((~m).sum()) >= min_neg_support:
#             neg_pred = Where(lambda d, p=pa: ~p.mask(d), name=f"¬{a}")
#             # rely on DataModel.add_boolean if available; otherwise assume registration
#             if hasattr(model, "add_boolean"):
#                 model.add_boolean(neg_pred)
#             added.append(f"¬{a}")
#     return added

# def enumerate_conjunctions(
#     model: DataModel,
#     cache: MaskCache,
#     base_pred: Predicate,
#     base_parts: Tuple[str, ...],
#     pool: List[str],
#     *,
#     max_arity: int,
#     min_support: int,
#     include_base: bool,
#     use_negations: bool,
# ) -> List[ClassRec]:

#     def mk_rec(extra_names: Tuple[str, ...]) -> ClassRec:
#         extras_sorted = tuple(sorted(extra_names))
#         preds = [model.boolean(n) for n in extras_sorted]
#         extras_conj = _and_all(preds)
#         conj = _and_all([base_pred, extras_conj])

#         parts = tuple((*base_parts, *extras_sorted))
#         name = "TRUE" if not parts else NAME_DELIM.join(parts)
#         return ClassRec(name=name, pred=conj, parts=parts, extras=extras_sorted, arity=len(extras_sorted))

#     out: List[ClassRec] = []

#     # Pure base as a class (optional)
#     if include_base:
#         base_rec = ClassRec(
#             name=("TRUE" if not base_parts else NAME_DELIM.join(base_parts)),
#             pred=base_pred,
#             parts=base_parts,
#             extras=tuple(),
#             arity=0,
#         )
#         if min_support <= 0 or int(bool_mask(cache, base_rec.pred).sum()) >= min_support:
#             out.append(base_rec)

#     atoms: List[str] = sorted(pool)

#     # Optional negations (controlled)
#     if use_negations:
#         added_negs = add_negations_safely(
#             model, cache, atoms,
#             min_pos_support=NEG_MIN_POS_SUPPORT, min_neg_support=NEG_MIN_NEG_SUPPORT
#         )
#         atoms = sorted(atoms + added_negs)

#     # Enumerate 1..max_arity
#     for k in range(1, max_arity + 1):
#         for combo in combinations(atoms, k):
#             r = mk_rec(combo)
#             if min_support > 0 and int(bool_mask(cache, r.pred).sum()) < min_support:
#                 continue
#             out.append(r)

#     return out

# # ───────────────────────── Normalization ───────────────────────── #
# @dataclass
# class _Info:
#     idx: int
#     rec: ClassRec
#     mask: np.ndarray
#     sig: bytes     # bit-packed signature
#     arity: int
#     support: int
#     extras_set: frozenset

# def normalize_equiv_and_redundancy(
#     cache: MaskCache,
#     recs: List[ClassRec],
# ) -> Tuple[List[Tuple[ClassRec, np.ndarray]], List[Tuple[ClassRec, np.ndarray]], List[List[ClassRec]]]:
#     infos: List[_Info] = []
#     for i, r in enumerate(recs):
#         m = bool_mask(cache, r.pred).to_numpy(dtype=np.bool_, copy=False)
#         sig = np.packbits(m, bitorder="little").tobytes()
#         infos.append(
#             _Info(
#                 idx=i, rec=r, mask=m, sig=sig, arity=r.arity,
#                 support=int(m.sum()), extras_set=frozenset(r.extras)
#             )
#         )

#     # equivalence groups: same sig + same arity
#     by_sig_arity: Dict[Tuple[bytes, int], List[_Info]] = {}
#     for info in infos:
#         by_sig_arity.setdefault((info.sig, info.arity), []).append(info)

#     equiv_groups: List[List[ClassRec]] = []
#     for (_, _), group in by_sig_arity.items():
#         if len(group) > 1:
#             equiv_groups.append([g.rec for g in group])

#     # redundancy: within same sig, keep minimal extras (no strict superset)
#     redundant_ids: Set[int] = set()
#     by_sig: Dict[bytes, List[_Info]] = {}
#     for info in infos:
#         by_sig.setdefault(info.sig, []).append(info)

#     for sig, group in by_sig.items():
#         group_sorted = sorted(group, key=lambda z: z.arity)
#         minimal: List[frozenset] = []
#         for g in group_sorted:
#             ex = g.extras_set
#             if any(me < ex for me in minimal):   # strict subset already present
#                 redundant_ids.add(g.idx)
#             else:
#                 minimal.append(ex)

#     nonred = [(info.rec, info.mask) for info in infos if info.idx not in redundant_ids]
#     red    = [(info.rec, info.mask) for info in infos if info.idx in redundant_ids]
#     return nonred, red, equiv_groups

# # ───────────────────────── R₁ candidates ───────────────────────── #
# def r1_candidates(
#     cache: MaskCache,
#     nonred: List[Tuple[ClassRec, np.ndarray]],
# ) -> Tuple[List[ClassInclusion], List[ClassEquivalence]]:
#     incs: List[ClassInclusion] = []
#     eqs:  List[ClassEquivalence] = []

#     seen_inc: Set[Tuple[int, int]] = set()   # (id(A), id(B))
#     seen_eq_sig: Set[bytes] = set()          # signature of mask

#     for i in range(len(nonred)):
#         rec_i, mask_i = nonred[i]
#         for j in range(i + 1, len(nonred)):
#             rec_j, mask_j = nonred[j]

#             same_shape = mask_i.shape == mask_j.shape
#             if same_shape and np.array_equal(mask_i, mask_j):
#                 sig = np.packbits(mask_i, bitorder="little").tobytes()
#                 if sig not in seen_eq_sig:
#                     eqs.append(ClassEquivalence(rec_i.pred, rec_j.pred))
#                     seen_eq_sig.add(sig)
#                 continue

#             # i ⊆ j if (i & ~j).any() == False
#             if (~mask_i & mask_j).sum() == 0 and (mask_i & ~mask_j).sum() == 0:
#                 # actually equal; already handled
#                 pass
#             if (mask_i & ~mask_j).sum() == 0:
#                 key = (id(rec_i.pred), id(rec_j.pred))
#                 if key not in seen_inc:
#                     incs.append(ClassInclusion(rec_i.pred, rec_j.pred))
#                     seen_inc.add(key)
#             if (mask_j & ~mask_i).sum() == 0:
#                 key = (id(rec_j.pred), id(rec_i.pred))
#                 if key not in seen_inc:
#                     incs.append(ClassInclusion(rec_j.pred, rec_i.pred))
#                     seen_inc.add(key)

#     return incs, eqs

# # ───────────────────────── Filters & selection ───────────────────────── #
# def is_tautology_by_parts(lhs_parts: Tuple[str, ...], rhs_parts: Tuple[str, ...]) -> bool:
#     """Drop (B∧X) ⊆ B and similar: rhs ⊆ lhs (trivial)."""
#     set_l, set_r = set(lhs_parts), set(rhs_parts)
#     return set_r.issubset(set_l)

# def is_reverse_subset_artifact(lhs_parts: Tuple[str, ...], rhs_parts: Tuple[str, ...]) -> bool:
#     """Drop A ⊆ (A∧X) (dataset artifact; not a structural fact)."""
#     set_l, set_r = set(lhs_parts), set(rhs_parts)
#     return set_l.issubset(set_r)

# def sophie_select(
#     relations: List[ClassInclusion | ClassEquivalence],
#     cache: MaskCache,
#     *,
#     min_new_rows: int,
# ) -> Tuple[List[Tuple[ClassInclusion | ClassEquivalence, int]], pd.Series]:
#     """Greedy coverage selection. Returns ([(rel, delta_added)], union_mask)."""
#     accepted: List[Tuple[ClassInclusion | ClassEquivalence, int]] = []
#     union = pd.Series(False, index=cache.model.df.index)

#     # Sort by hypothesis support (stable, largest first)
#     def _hyp_mask(rel):
#         if isinstance(rel, ClassInclusion):
#             return bool_mask(cache, rel.A)
#         return bool_mask(cache, rel.A)  # equivalence: either side the same

#     relations_sorted = sorted(relations, key=lambda r: int(_hyp_mask(r).sum()), reverse=True)

#     for rel in relations_sorted:
#         hyp = _hyp_mask(rel)
#         delta = hyp & ~union
#         add = int(delta.sum())
#         if add >= min_new_rows:
#             accepted.append((rel, add))
#             union |= hyp

#     return accepted, union

# # ───────────────────────── Utilities ───────────────────────── #
# # resolve class name from Predicate by searching nonred
# def _name_parts_for_pred(p: Predicate, catalog: List[Tuple[ClassRec, np.ndarray]]) -> Tuple[str, Tuple[str, ...]]:
#     for rec, _ in catalog:
#         if rec.pred is p:
#             return rec.name, rec.parts
#     return repr(p), tuple()

# def witness_rows(df: pd.DataFrame, cache: MaskCache, rel, *, k: int = 3) -> pd.DataFrame:
#     """Return a few rows witnessing the relation (for sanity/auditing)."""
#     if isinstance(rel, ClassInclusion):
#         A, B = cache.mask(rel.A), cache.mask(rel.B)
#         ok = A & B
#         return df.loc[ok].head(k)
#     if isinstance(rel, ClassEquivalence):
#         A = cache.mask(rel.A)  # same as B
#         return df.loc[A].head(k)
#     return df.iloc[0:0]

# def rel_scope_size(cache: MaskCache, rel) -> int:
#     if isinstance(rel, ClassEquivalence):
#         a = cache.mask(rel.A)
#         return int(a.sum())
#     if isinstance(rel, ClassInclusion):
#         a = cache.mask(rel.A)
#         return int(a.sum())
#     return 0

# # ───────────────────────── Main ───────────────────────── #
# def main():
#     model = DataModel(df.copy())
#     cache = MaskCache(model)

#     print("\n\n=== Base scan ===")
#     names = _boolean_names(model)
#     print("booleans :", ", ".join(names))

#     base_pred, base_parts, base_name, base_sig = compute_base(model, cache)
#     print(f"\nBase: {base_name}   (support={int(bool_mask(cache, base_pred).sum())})")
#     print(f"  base parts: {base_parts!r}")

#     # pool = all boolean names excluding base parts
#     pool = [n for n in _boolean_names(model) if n not in set(base_parts)]

#     # enumerate
#     recs = enumerate_conjunctions(
#         model, cache, base_pred, base_parts, pool,
#         max_arity=MAX_ARITY, min_support=MIN_SUPPORT,
#         include_base=INCLUDE_BASE, use_negations=USE_NEGATIONS,
#     )
#     print(f"\nEnumerated {len(recs)} classes (including base).")

#     # normalize
#     nonred, red, equiv_groups = normalize_equiv_and_redundancy(cache, recs)
#     print(f"Nonredundant: {len(nonred)} | Redundant: {len(red)} | Equivalence groups: {len(equiv_groups)}")

#     # top by support (deterministic: tie-break by name)
#     scored = sorted(
#         nonred,
#         key=lambda t: (-int(t[1].sum()), t[0].name)
#     )
#     print("\nTop classes by support:")
#     for rec, m in scored[:TOP_K_BY_SUPPORT]:
#         print(f"  • {_short(rec.name, 60):<60} support={int(m.sum())}")

#     # example equivalence group (stable order)
#     if equiv_groups:
#         equiv_groups.sort(key=lambda g: tuple(sorted(r.name for r in g)))
#         print("\nExample equivalence group (same mask & arity):")
#         for r in equiv_groups[0]:
#             print(" ", r.name)

#     # pretty-print a few redundant examples if present,
#     # but SKIP redundancies whose signature equals the base signature
#     # (this was the “base treated wrong” annoyance)
#     if red:
#         print("\nExample redundant classes (dominated within same signature):")
#         shown = 0
#         for (r, m) in red:
#             sig = np.packbits(m, bitorder="little").tobytes()
#             if sig == base_sig:
#                 continue  # don't showcase base-caused redundancies
#             print(f"  {_short(r.name, 40):<40} extras={r.extras!r}")
#             shown += 1
#             if shown == 5:
#                 break

#     # R1 candidates
#     incs, eqs = r1_candidates(cache, nonred)
#     print(f"\nPre-Sophie: inclusions={len(incs)} | equivalences={len(eqs)}")

#     # Build quick lookup map: Predicate -> parts
#     parts_map: Dict[Predicate, Tuple[str, ...]] = {rec.pred: rec.parts for rec, _ in nonred}
#     def _parts(p: Predicate) -> Tuple[str, ...]:
#         return parts_map.get(p, tuple())

#     # Apply extra inclusion filters (by parts)
#     filtered_incs: List[ClassInclusion] = []
#     for r in incs:
#         lhs_parts = _parts(r.A)
#         rhs_parts = _parts(r.B)
#         if is_tautology_by_parts(lhs_parts, rhs_parts):
#             continue
#         if is_reverse_subset_artifact(lhs_parts, rhs_parts):
#             continue
#         filtered_incs.append(r)

#     # Sophie selections (greedy coverage)
#     accepted_incs, union_incs = sophie_select(filtered_incs, cache, min_new_rows=SOPHIE_SCOPE_CUTOFF)
#     accepted_eqs,  union_eqs  = sophie_select(eqs,           cache, min_new_rows=SOPHIE_SCOPE_CUTOFF)

#     # Print results
#     print("\nR₁ Conjectures (Class Inclusions, Sophie-selected):")
#     if not accepted_incs:
#         print("  (none)")
#     else:
#         for r, add in accepted_incs:
#             nmA, _ = _name_parts_for_pred(r.A, nonred)
#             nmB, _ = _name_parts_for_pred(r.B, nonred)
#             scope = rel_scope_size(cache, r)
#             print(f"  • ({nmA}) ⊆ ({nmB})   [scope={scope}, Δ={add}]")

#     print("\nR₁ Conjectures (Class Equivalences, Sophie-selected):")
#     if not accepted_eqs:
#         print("  (none)")
#     else:
#         for r, add in accepted_eqs:
#             nmA, _ = _name_parts_for_pred(r.A, nonred)
#             nmB, _ = _name_parts_for_pred(r.B, nonred)
#             scope = rel_scope_size(cache, r)
#             print(f"  • ({nmA}) ≡ ({nmB})   [scope={scope}, Δ={add}]")

#     print("\nCoverage summary:")
#     print(f"  accepted inclusions cover rows:   {int(union_incs.sum())} / {len(model.df)}")
#     print(f"  accepted equivalences cover rows: {int(union_eqs.sum())} / {len(model.df)}")

#     # Optional: show a tiny witness table for the first accepted relation (handy in practice)
#     if accepted_incs:
#         w = witness_rows(model.df, cache, accepted_incs[0][0], k=3)
#         if not w.empty:
#             print("\nWitness rows (first accepted inclusion):")
#             print(w.iloc[:, : min(8, w.shape[1])])

#     print("\nDone.")

# # ───────────────────────── Entrypoint ───────────────────────── #
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Tuple, Dict, Set, Optional, Any

import numpy as np
import pandas as pd

# ───────────────────────── Your modules ───────────────────────── #
# from txgraffiti.example_data import graph_data as df
# df['nontrivial'] = df['connected']

df = pd.read_csv('scripts/fake_surface.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

# New minimal base
from txgraffiti2025.graffiti.core import DataModel, MaskCache

from txgraffiti2025.forms.predicates import Predicate, Where
from txgraffiti2025.forms.class_relations import ClassInclusion, ClassEquivalence
from txgraffiti2025.forms.utils import abs_, min_, max_, to_expr  # for derived Exprs

# ───────────────────────── Switches / knobs ───────────────────────── #
MAX_ARITY            = 2
MIN_SUPPORT          = 0
INCLUDE_BASE         = True
USE_NEGATIONS        = False
NEG_MIN_POS_SUPPORT  = 20
NEG_MIN_NEG_SUPPORT  = 20

SOPHIE_SCOPE_CUTOFF  = 1
TOP_K_BY_SUPPORT     = 20
NAME_DELIM           = " ∧ "

# ── Incomparability block toggles (inline funcs; adapted to new DataModel) ── #
RUN_INCOMPARABILITY          = True
INC_PRINT_TOP                = 12

# Register abs(x−y) for meaningfully incomparable pairs
INC_REGISTER_ABS             = True
INC_ABS_MIN_SUPPORT          = 0.10
INC_ABS_MIN_SIDE_RATE        = 0.10
INC_ABS_MIN_SIDE_COUNT       = 5
INC_ABS_MIN_MEDIAN_GAP       = 0.50
INC_ABS_MIN_MEAN_GAP         = 0.50

# Option A: min/max for often-unequal pairs
INC_REGISTER_MINMAX_OFTEN    = False
INC_MINMAX_MIN_SUPPORT       = 0.10
INC_MINMAX_MIN_NEQ_RATE      = 0.50
INC_MINMAX_MIN_NEQ_COUNT     = 8
INC_MINMAX_MUST_BE_INCOMP    = True

# Option B: min/max for meaningfully incomparable pairs
INC_REGISTER_MINMAX_MEANING  = True
INC_MMEAN_MIN_SUPPORT        = 0.10
INC_MMEAN_MIN_SIDE_RATE      = 0.10
INC_MMEAN_MIN_SIDE_COUNT     = 5
INC_MMEAN_MAX_EQ_RATE        = 0.70
INC_KEY_STYLE                = "pretty"   # "slug" or "pretty"

# ───────────────────────── Tiny helpers ───────────────────────── #
def _boolean_names(model: DataModel) -> List[str]:
    # DataModel.boolean_names is a @property returning a List[str]
    return list(model.boolean_names)

def bool_mask(cache: MaskCache, p: Predicate) -> pd.Series:
    return cache.mask(p)

def _and_all(preds: Iterable[Predicate]) -> Predicate:
    preds = list(preds)
    if not preds:
        return Where(lambda d: pd.Series(True, index=d.index), name="TRUE")
    out = preds[0]
    for q in preds[1:]:
        out = out & q
    return out

def _short(name: str, n: int = 60) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"

@dataclass(frozen=True)
class ClassRec:
    name: str
    pred: Predicate
    parts: Tuple[str, ...]
    extras: Tuple[str, ...]
    arity: int

# ───────────────────────── Base predicate ───────────────────────── #
def compute_base(model: DataModel, cache: MaskCache) -> Tuple[Predicate, Tuple[str, ...], str, bytes]:
    always_true: List[str] = []
    for nm in _boolean_names(model):
        if bool(bool_mask(cache, model.boolean(nm)).all()):
            always_true.append(nm)
    always_true = sorted(always_true)
    if not always_true:
        base_pred = Where(lambda d: pd.Series(True, index=d.index), name="TRUE")
        base_parts: Tuple[str, ...] = tuple()
        base_name = "TRUE"
    else:
        preds = [model.boolean(n) for n in always_true]
        base_pred = _and_all(preds)
        base_parts = tuple(always_true)
        base_name = NAME_DELIM.join(base_parts)

    base_mask = bool_mask(cache, base_pred).to_numpy(dtype=np.bool_, copy=False)
    base_sig  = np.packbits(base_mask, bitorder="little").tobytes()
    return base_pred, base_parts, base_name, base_sig

# ───────────────────────── Enumeration ───────────────────────── #
def add_negations_safely(
    model: DataModel,
    cache: MaskCache,
    atoms: List[str],
    *,
    min_pos_support: int,
    min_neg_support: int,
) -> List[str]:
    added = []
    for a in atoms:
        pa = model.boolean(a)
        m = bool_mask(cache, pa)
        if int(m.sum()) >= min_pos_support and int((~m).sum()) >= min_neg_support:
            neg_pred = Where(lambda d, p=pa: ~p.mask(d), name=f"¬{a}")
            model.add_boolean(neg_pred)
            added.append(f"¬{a}")
    return added

def enumerate_conjunctions(
    model: DataModel,
    cache: MaskCache,
    base_pred: Predicate,
    base_parts: Tuple[str, ...],
    pool: List[str],
    *,
    max_arity: int,
    min_support: int,
    include_base: bool,
    use_negations: bool,
) -> List[ClassRec]:

    def mk_rec(extra_names: Tuple[str, ...]) -> ClassRec:
        extras_sorted = tuple(sorted(extra_names))
        preds = [model.boolean(n) for n in extras_sorted]
        extras_conj = _and_all(preds)
        conj = _and_all([base_pred, extras_conj])
        parts = tuple((*base_parts, *extras_sorted))
        name = "TRUE" if not parts else NAME_DELIM.join(parts)
        return ClassRec(name=name, pred=conj, parts=parts, extras=extras_sorted, arity=len(extras_sorted))

    out: List[ClassRec] = []

    if include_base:
        base_rec = ClassRec(
            name=("TRUE" if not base_parts else NAME_DELIM.join(base_parts)),
            pred=base_pred,
            parts=base_parts,
            extras=tuple(),
            arity=0,
        )
        if min_support <= 0 or int(bool_mask(cache, base_rec.pred).sum()) >= min_support:
            out.append(base_rec)

    atoms: List[str] = sorted(pool)

    if use_negations:
        added_negs = add_negations_safely(
            model, cache, atoms,
            min_pos_support=NEG_MIN_POS_SUPPORT, min_neg_support=NEG_MIN_NEG_SUPPORT
        )
        atoms = sorted(atoms + added_negs)

    for k in range(1, max_arity + 1):
        for combo in combinations(atoms, k):
            r = mk_rec(combo)
            if min_support > 0 and int(bool_mask(cache, r.pred).sum()) < min_support:
                continue
            out.append(r)
    return out

# ───────────────────────── Normalization ───────────────────────── #
@dataclass
class _Info:
    idx: int
    rec: ClassRec
    mask: np.ndarray
    sig: bytes
    arity: int
    support: int
    extras_set: frozenset

def normalize_equiv_and_redundancy(
    cache: MaskCache,
    recs: List[ClassRec],
) -> Tuple[List[Tuple[ClassRec, np.ndarray]], List[Tuple[ClassRec, np.ndarray]], List[List[ClassRec]]]:
    infos: List[_Info] = []
    for i, r in enumerate(recs):
        m = bool_mask(cache, r.pred).to_numpy(dtype=np.bool_, copy=False)
        sig = np.packbits(m, bitorder="little").tobytes()
        infos.append(_Info(i, r, m, sig, r.arity, int(m.sum()), frozenset(r.extras)))

    by_sig_arity: Dict[Tuple[bytes, int], List[_Info]] = {}
    for info in infos:
        by_sig_arity.setdefault((info.sig, info.arity), []).append(info)

    equiv_groups: List[List[ClassRec]] = []
    for (_, _), group in by_sig_arity.items():
        if len(group) > 1:
            equiv_groups.append([g.rec for g in group])

    redundant_ids: Set[int] = set()
    by_sig: Dict[bytes, List[_Info]] = {}
    for info in infos:
        by_sig.setdefault(info.sig, []).append(info)

    for sig, group in by_sig.items():
        group_sorted = sorted(group, key=lambda z: z.arity)
        minimal: List[frozenset] = []
        for g in group_sorted:
            ex = g.extras_set
            if any(me < ex for me in minimal):   # strict subset present
                redundant_ids.add(g.idx)
            else:
                minimal.append(ex)

    nonred = [(info.rec, info.mask) for info in infos if info.idx not in redundant_ids]
    red    = [(info.rec, info.mask) for info in infos if info.idx in redundant_ids]
    return nonred, red, equiv_groups

# ───────────────────────── R₁ candidates ───────────────────────── #
def r1_candidates(
    cache: MaskCache,
    nonred: List[Tuple[ClassRec, np.ndarray]],
) -> Tuple[List[ClassInclusion], List[ClassEquivalence]]:
    incs: List[ClassInclusion] = []
    eqs:  List[ClassEquivalence] = []
    seen_inc: Set[Tuple[int, int]] = set()
    seen_eq_sig: Set[bytes] = set()

    for i in range(len(nonred)):
        rec_i, mask_i = nonred[i]
        for j in range(i + 1, len(nonred)):
            rec_j, mask_j = nonred[j]

            same_shape = mask_i.shape == mask_j.shape
            if same_shape and np.array_equal(mask_i, mask_j):
                sig = np.packbits(mask_i, bitorder="little").tobytes()
                if sig not in seen_eq_sig:
                    eqs.append(ClassEquivalence(rec_i.pred, rec_j.pred))
                    seen_eq_sig.add(sig)
                continue

            if (mask_i & ~mask_j).sum() == 0:
                key = (id(rec_i.pred), id(rec_j.pred))
                if key not in seen_inc:
                    incs.append(ClassInclusion(rec_i.pred, rec_j.pred))
                    seen_inc.add(key)
            if (mask_j & ~mask_i).sum() == 0:
                key = (id(rec_j.pred), id(rec_i.pred))
                if key not in seen_inc:
                    incs.append(ClassInclusion(rec_j.pred, rec_i.pred))
                    seen_inc.add(key)
    return incs, eqs

# ───────────────────────── Filters & selection ───────────────────────── #
def is_tautology_by_parts(lhs_parts: Tuple[str, ...], rhs_parts: Tuple[str, ...]) -> bool:
    set_l, set_r = set(lhs_parts), set(rhs_parts)
    return set_r.issubset(set_l)

def is_reverse_subset_artifact(lhs_parts: Tuple[str, ...], rhs_parts: Tuple[str, ...]) -> bool:
    set_l, set_r = set(lhs_parts), set(rhs_parts)
    return set_l.issubset(set_r)

def sophie_select(
    relations: List[ClassInclusion | ClassEquivalence],
    cache: MaskCache,
    *,
    min_new_rows: int,
) -> Tuple[List[Tuple[ClassInclusion | ClassEquivalence, int]], pd.Series]:
    accepted: List[Tuple[ClassInclusion | ClassEquivalence, int]] = []
    union = pd.Series(False, index=cache.model.df.index)
    def _hyp_mask(rel):
        if isinstance(rel, ClassInclusion):
            return bool_mask(cache, rel.A)
        return bool_mask(cache, rel.A)
    relations_sorted = sorted(relations, key=lambda r: int(_hyp_mask(r).sum()), reverse=True)
    for rel in relations_sorted:
        hyp = _hyp_mask(rel)
        delta = hyp & ~union
        add = int(delta.sum())
        if add >= min_new_rows:
            accepted.append((rel, add))
            union |= hyp
    return accepted, union

# ───────────────────────── Utilities ───────────────────────── #
def _name_parts_for_pred(p: Predicate, catalog: List[Tuple[ClassRec, np.ndarray]]) -> Tuple[str, Tuple[str, ...]]:
    for rec, _ in catalog:
        if rec.pred is p:
            return rec.name, rec.parts
    return repr(p), tuple()

def witness_rows(df: pd.DataFrame, cache: MaskCache, rel, *, k: int = 3) -> pd.DataFrame:
    if isinstance(rel, ClassInclusion):
        A, B = cache.mask(rel.A), cache.mask(rel.B)
        ok = A & B
        return df.loc[ok].head(k)
    if isinstance(rel, ClassEquivalence):
        A = cache.mask(rel.A)
        return df.loc[A].head(k)
    return df.iloc[0:0]

def rel_scope_size(cache: MaskCache, rel) -> int:
    if isinstance(rel, ClassEquivalence):
        return int(bool_mask(cache, rel.A).sum())
    if isinstance(rel, ClassInclusion):
        return int(bool_mask(cache, rel.A).sum())
    return 0

# ───────────────────────── Incomparability (inline funcs) ───────────────────────── #
def _domain_array_mask(cache: MaskCache, model: DataModel, condition: Optional[Predicate], base_fallback: Optional[Predicate]) -> np.ndarray:
    if condition is not None:
        return bool_mask(cache, condition).to_numpy(dtype=np.bool_, copy=False)
    if base_fallback is not None:
        return bool_mask(cache, base_fallback).to_numpy(dtype=np.bool_, copy=False)
    return np.ones(len(model.df), dtype=bool)

def _invariant_arrays(model: DataModel, require_finite: bool) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Evaluate all numeric invariants (Expr) to arrays and provide finite masks.
    Names use repr(Expr); ColumnTerm repr is the column name.
    """
    names: List[str] = []
    arrays: Dict[str, np.ndarray] = {}
    finite_masks: Dict[str, np.ndarray] = {}
    for e in model.invariants:
        key = repr(e)
        s = e.eval(model.df)  # assumes Expr has .eval(df)
        if require_finite:
            s = s.replace([np.inf, -np.inf], np.nan)
        a = s.to_numpy(dtype=float, copy=False)
        names.append(key)
        arrays[key] = a
        finite_masks[key] = np.isfinite(a) if require_finite else np.ones_like(a, dtype=bool)
    return names, arrays, finite_masks

def analyze_incomparability(
    model: DataModel,
    cache: MaskCache,
    condition: Optional[Predicate] = None,
    *,
    base_fallback: Optional[Predicate] = None,
    require_finite: bool = True,
    include_all_pairs: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame with pairwise directionality counts/rates on a domain.
    Columns: inv1, inv2, n_rows, n_lt, n_gt, n_eq, rate_lt, rate_gt, rate_eq,
             incomparable, balance=min(rate_lt,rate_gt), support
    """
    domain = _domain_array_mask(cache, model, condition, base_fallback)
    domain_rows = int(domain.sum())
    if domain_rows == 0:
        return pd.DataFrame(columns=[
            "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
            "rate_lt","rate_gt","rate_eq","incomparable","balance","support"
        ])

    inv_names, arrays, finite_masks = _invariant_arrays(model, require_finite)
    rows: List[Dict[str, Any]] = []

    for i in range(len(inv_names)):
        x = inv_names[i]; ax = arrays[x]; fx = finite_masks[x]
        for j in range(i + 1, len(inv_names)):
            y = inv_names[j]; ay = arrays[y]; fy = finite_masks[y]

            m = domain & fx & fy
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

            incomparable = (n_lt > 0) and (n_gt > 0)
            if include_all_pairs or incomparable:
                rate_lt = n_lt / n
                rate_gt = n_gt / n
                rate_eq = n_eq / n
                rows.append({
                    "inv1": x, "inv2": y,
                    "n_rows": n,
                    "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
                    "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                    "incomparable": bool(incomparable),
                    "balance": min(rate_lt, rate_gt),
                    "support": n / domain_rows,
                })

    out = pd.DataFrame(rows, columns=[
        "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
        "rate_lt","rate_gt","rate_eq","incomparable","balance","support"
    ])
    if out.empty:
        return out
    return out.sort_values(
        ["incomparable","balance","support","inv1","inv2"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

def _add_invariant_expr(model: DataModel, expr) -> str:
    """Add a derived Expr to the model and return its key (repr)."""
    key = model.add_invariant(to_expr(expr))
    return key

def register_absdiff_for_meaningful_pairs(
    model: DataModel,
    cache: MaskCache,
    condition: Optional[Predicate] = None,
    *,
    base_fallback: Optional[Predicate] = None,
    require_finite: bool = True,
    min_support: float = 0.10,
    min_side_rate: float = 0.10,
    min_side_count: int = 5,
    min_median_gap: float = 0.5,
    min_mean_gap: float = 0.5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    For meaningfully incomparable pairs, register abs(x - y) as a new invariant.
    Returns (report_df, added_keys).
    """
    domain = _domain_array_mask(cache, model, condition, base_fallback)
    domain_rows = int(domain.sum())
    if domain_rows == 0:
        return pd.DataFrame(), []

    inv_names, arrays, finite_masks = _invariant_arrays(model, require_finite)
    rows: List[Dict[str, Any]] = []
    added: List[str] = []

    for i in range(len(inv_names)):
        x = inv_names[i]; ax = arrays[x]; fx = finite_masks[x]
        for j in range(i + 1, len(inv_names)):
            y = inv_names[j]; ay = arrays[y]; fy = finite_masks[y]

            m = domain & fx & fy
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
            side_ok = (min(rate_lt, rate_gt) >= float(min_side_rate)) and (min(n_lt, n_gt) >= int(min_side_count))
            support_ok = (support >= float(min_support))

            gaps = np.abs(axm - aym)
            median_gap = float(np.nanmedian(gaps)) if n > 0 else 0.0
            mean_gap   = float(np.nanmean(gaps))   if n > 0 else 0.0
            gap_ok = (median_gap >= float(min_median_gap)) or (mean_gap >= float(min_mean_gap))

            selected = bool(both_sides and side_ok and support_ok and gap_ok)

            expr_key = None
            if selected:
                ex = model.invariant(x)
                ey = model.invariant(y)
                expr_key = _add_invariant_expr(model, abs_(ex - ey))
                added.append(expr_key)
                expr_key = _add_invariant_expr(model, min_(ex, ey))

            rows.append({
                "inv1": x, "inv2": y,
                "n_rows": n,
                "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
                "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                "support": support,
                "median_gap": median_gap, "mean_gap": mean_gap,
                "selected": selected, "expr_key": expr_key,
            })

    out = pd.DataFrame(rows, columns=[
        "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
        "rate_lt","rate_gt","rate_eq","support",
        "median_gap","mean_gap","selected","expr_key"
    ])
    if out.empty:
        return out, added
    return out.sort_values("mean_gap", ascending=False, kind="mergesort").reset_index(drop=True), added

def register_minmax_for_often_unequal_pairs(
    model: DataModel,
    cache: MaskCache,
    condition: Optional[Predicate] = None,
    *,
    base_fallback: Optional[Predicate] = None,
    require_finite: bool = True,
    min_support: float = 0.10,
    min_neq_rate: float = 0.50,
    min_neq_count: int = 8,
    must_be_incomparable: bool = True,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Register min(x,y) and max(x,y) where the pair is often unequal (and optionally two-sided incomparable).
    Returns (report_df, added_key_pairs).
    """
    domain = _domain_array_mask(cache, model, condition, base_fallback)
    domain_rows = int(domain.sum())
    if domain_rows == 0:
        return pd.DataFrame(), []

    inv_names, arrays, finite_masks = _invariant_arrays(model, require_finite)
    rows: List[Dict[str, Any]] = []
    added_pairs: List[Tuple[str, str]] = []

    for i in range(len(inv_names)):
        x = inv_names[i]; ax = arrays[x]; fx = finite_masks[x]
        for j in range(i + 1, len(inv_names)):
            y = inv_names[j]; ay = arrays[y]; fy = finite_masks[y]

            m = domain & fx & fy
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

            incomparable = (n_lt > 0) and (n_gt > 0)
            if must_be_incomparable and not incomparable:
                selected = False
                often_unequal = False
            else:
                neq = n - n_eq
                often_unequal = (n > 0) and ((neq / n) >= float(min_neq_rate)) and (neq >= int(min_neq_count))
                selected = bool((support >= float(min_support)) and often_unequal)

            key_min = None
            key_max = None
            if selected:
                ex = model.invariant(x)
                ey = model.invariant(y)
                key_min = _add_invariant_expr(model, min_(ex, ey))
                key_max = _add_invariant_expr(model, max_(ex, ey))
                added_pairs.append((key_min, key_max))

            rows.append({
                "inv1": x, "inv2": y,
                "n_rows": n,
                "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
                "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                "support": support,
                "often_unequal": bool((n - n_eq) / n >= float(min_neq_rate)) if n > 0 else False,
                "selected": selected,
                "key_min": key_min, "key_max": key_max,
            })

    out = pd.DataFrame(rows, columns=[
        "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
        "rate_lt","rate_gt","rate_eq","support",
        "often_unequal","selected","key_min","key_max"
    ])
    return out.sort_values(
        ["selected","support","inv1","inv2"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True), added_pairs

def register_minmax_for_meaningful_pairs(
    model: DataModel,
    cache: MaskCache,
    condition: Optional[Predicate] = None,
    *,
    base_fallback: Optional[Predicate] = None,
    require_finite: bool = True,
    min_support: float = 0.10,
    min_side_rate: float = 0.10,
    min_side_count: int = 5,
    max_eq_rate: float = 0.70,
    style: str = "slug",        # "slug" -> min_a_b / max_a_b, "pretty" -> min(a,b)/max(a,b)
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Register min(x,y), max(x,y) for pairs that are meaningfully two-sided incomparable (and not mostly equal).
    Returns (report_df, added_key_pairs).
    """
    domain = _domain_array_mask(cache, model, condition, base_fallback)
    domain_rows = int(domain.sum())
    if domain_rows == 0:
        return pd.DataFrame(), []

    inv_names, arrays, finite_masks = _invariant_arrays(model, require_finite)

    def _keypair(x: str, y: str) -> Tuple[str, str]:
        a, b = sorted([x, y])
        if style == "pretty":
            return f"min({a}, {b})", f"max({a}, {b})"
        return f"min_{a}_{b}", f"max_{a}_{b}"

    rows: List[Dict[str, Any]] = []
    added_pairs: List[Tuple[str, str]] = []

    for i in range(len(inv_names)):
        x = inv_names[i]; ax = arrays[x]; fx = finite_masks[x]
        for j in range(i + 1, len(inv_names)):
            y = inv_names[j]; ay = arrays[y]; fy = finite_masks[y]

            m = domain & fx & fy
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
            side_ok = (min(rate_lt, rate_gt) >= float(min_side_rate)) and (min(n_lt, n_gt) >= int(min_side_count))
            support_ok = (support >= float(min_support))
            not_mostly_equal = (rate_eq <= float(max_eq_rate))

            selected = bool(both_sides and side_ok and support_ok and not_mostly_equal)

            key_min = None
            key_max = None
            if selected:
                ex = model.invariant(x)
                ey = model.invariant(y)
                # Use Expr directly; model.add_invariant returns repr key
                key_min = _add_invariant_expr(model, min_(ex, ey))
                key_max = _add_invariant_expr(model, max_(ex, ey))
                added_pairs.append((key_min, key_max))

            rows.append({
                "inv1": x, "inv2": y,
                "n_rows": n,
                "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
                "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                "support": support,
                "selected": selected,
                "key_min": key_min, "key_max": key_max,
            })

    out = pd.DataFrame(rows, columns=[
        "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
        "rate_lt","rate_gt","rate_eq","support",
        "selected","key_min","key_max",
    ])
    return out.sort_values(
        ["selected","support","inv1","inv2"],
        ascending=[False, False, True, True]
    ).reset_index(drop=True), added_pairs

# ───────────────────────── Main ───────────────────────── #
def main():
    model = DataModel(df.copy())
    cache = MaskCache(model)

    print("\n\n=== Base scan ===")
    print("booleans :", ", ".join(_boolean_names(model)))

    base_pred, base_parts, base_name, base_sig = compute_base(model, cache)
    print(f"\nBase: {base_name}   (support={int(bool_mask(cache, base_pred).sum())})")
    print(f"  base parts: {base_parts!r}")

    # pool = all boolean names excluding base parts
    pool = [n for n in _boolean_names(model) if n not in set(base_parts)]

    # enumerate
    recs = enumerate_conjunctions(
        model, cache, base_pred, base_parts, pool,
        max_arity=MAX_ARITY, min_support=MIN_SUPPORT,
        include_base=INCLUDE_BASE, use_negations=USE_NEGATIONS,
    )
    print(f"\nEnumerated {len(recs)} classes (including base).")

    # normalize
    nonred, red, equiv_groups = normalize_equiv_and_redundancy(cache, recs)
    print(f"Nonredundant: {len(nonred)} | Redundant: {len(red)} | Equivalence groups: {len(equiv_groups)}")

    # top by support (deterministic)
    scored = sorted(nonred, key=lambda t: (-int(t[1].sum()), t[0].name))
    print("\nTop classes by support:")
    for rec, m in scored[:TOP_K_BY_SUPPORT]:
        print(f"  • {_short(rec.name, 60):<60} support={int(m.sum())}")

    # example equivalence group (stable order)
    if equiv_groups:
        equiv_groups.sort(key=lambda g: tuple(sorted(r.name for r in g)))
        print("\nExample equivalence group (same mask & arity):")
        for r in equiv_groups[0]:
            print(" ", r.name)

    # pretty-print redundant examples, skipping base-signature redundancies
    if red:
        print("\nExample redundant classes (dominated within same signature):")
        shown = 0
        for (r, m) in red:
            sig = np.packbits(m, bitorder="little").tobytes()
            if sig == base_sig:
                continue
            print(f"  {_short(r.name, 40):<40} extras={r.extras!r}")
            shown += 1
            if shown == 5:
                break

    # R1 candidates
    incs, eqs = r1_candidates(cache, nonred)
    print(f"\nPre-Sophie: inclusions={len(incs)} | equivalences={len(eqs)}")

    # Lookup map for parts
    parts_map: Dict[Predicate, Tuple[str, ...]] = {rec.pred: rec.parts for rec, _ in nonred}
    def _parts(p: Predicate) -> Tuple[str, ...]:
        return parts_map.get(p, tuple())

    # Inclusion filters
    filtered_incs: List[ClassInclusion] = []
    for r in incs:
        lhs_parts = _parts(r.A)
        rhs_parts = _parts(r.B)
        if is_tautology_by_parts(lhs_parts, rhs_parts):
            continue
        if is_reverse_subset_artifact(lhs_parts, rhs_parts):
            continue
        filtered_incs.append(r)

    # Sophie selections
    accepted_incs, union_incs = sophie_select(filtered_incs, cache, min_new_rows=SOPHIE_SCOPE_CUTOFF)
    accepted_eqs,  union_eqs  = sophie_select(eqs,           cache, min_new_rows=SOPHIE_SCOPE_CUTOFF)

    # Print results
    print("\nR₁ Conjectures (Class Inclusions, Sophie-selected):")
    if not accepted_incs:
        print("  (none)")
    else:
        for r, add in accepted_incs:
            nmA, _ = _name_parts_for_pred(r.A, nonred)
            nmB, _ = _name_parts_for_pred(r.B, nonred)
            scope = rel_scope_size(cache, r)
            print(f"  • ({nmA}) ⊆ ({nmB})   [scope={scope}, Δ={add}]")

    print("\nR₁ Conjectures (Class Equivalences, Sophie-selected):")
    if not accepted_eqs:
        print("  (none)")
    else:
        for r, add in accepted_eqs:
            nmA, _ = _name_parts_for_pred(r.A, nonred)
            nmB, _ = _name_parts_for_pred(r.B, nonred)
            scope = rel_scope_size(cache, r)
            print(f"  • ({nmA}) ≡ ({nmB})   [scope={scope}, Δ={add}]")

    print("\nCoverage summary:")
    print(f"  accepted inclusions cover rows:   {int(union_incs.sum())} / {len(model.df)}")
    print(f"  accepted equivalences cover rows: {int(union_eqs.sum())} / {len(model.df)}")

    if accepted_incs:
        w = witness_rows(model.df, cache, accepted_incs[0][0], k=3)
        if not w.empty:
            print("\nWitness rows (first accepted inclusion):")
            print(w.iloc[:, : min(8, w.shape[1])])

    # ─────────────────────────────────────────────────────────────
    # Incomparability block (inline; adapted to new DataModel)
    # ─────────────────────────────────────────────────────────────
    if RUN_INCOMPARABILITY:
        print("\n=== Incomparability scan (numeric invariants) ===")
        # If there are no invariants, skip gracefully
        if not model.invariants:
            print("  (skipped: no numeric invariants in model)")
        else:
            diag = analyze_incomparability(
                model, cache,
                condition=None,
                base_fallback=base_pred,
                require_finite=True,
                include_all_pairs=False,
            )
            if diag.empty:
                print("  No incomparable pairs detected on the base domain.")
            else:
                print(f"  Incomparable pairs found: {int(diag['incomparable'].sum())} / {len(diag)}")
                print("  Top pairs by balance/support:")
                head = diag.head(INC_PRINT_TOP)
                for _, row in head.iterrows():
                    inv1 = row["inv1"]; inv2 = row["inv2"]
                    bal  = row["balance"]; sup = row["support"]
                    neq  = 1.0 - row["rate_eq"]
                    print(f"   • ({inv1}, {inv2})  balance≈{bal:.2f}  support≈{sup:.2f}  neq≈{neq:.2f}")

            # abs(x−y)
            if INC_REGISTER_ABS:
                absdf, abs_added = register_absdiff_for_meaningful_pairs(
                    model, cache,
                    condition=None, base_fallback=base_pred, require_finite=True,
                    min_support=INC_ABS_MIN_SUPPORT,
                    min_side_rate=INC_ABS_MIN_SIDE_RATE,
                    min_side_count=INC_ABS_MIN_SIDE_COUNT,
                    min_median_gap=INC_ABS_MIN_MEDIAN_GAP,
                    min_mean_gap=INC_ABS_MIN_MEAN_GAP,
                )
                n_new = int(absdf["selected"].sum()) if not absdf.empty else 0
                print(f"\n  Registered absdiff expressions: {n_new} (new invariants added: {len(abs_added)})")

            # min/max option A
            if INC_REGISTER_MINMAX_OFTEN:
                mm, pairs = register_minmax_for_often_unequal_pairs(
                    model, cache,
                    condition=None, base_fallback=base_pred, require_finite=True,
                    min_support=INC_MINMAX_MIN_SUPPORT,
                    min_neq_rate=INC_MINMAX_MIN_NEQ_RATE,
                    min_neq_count=INC_MINMAX_MIN_NEQ_COUNT,
                    must_be_incomparable=INC_MINMAX_MUST_BE_INCOMP,
                )
                n_new = int(mm["selected"].sum()) if not mm.empty else 0
                print(f"  Registered min/max (often-unequal): {n_new} (pairs added: {len(pairs)})")

            # # min/max option B
            # if INC_REGISTER_MINMAX_MEANING:
            #     mm2, pairs2 = register_minmax_for_meaningful_pairs(
            #         model, cache,
            #         condition=None, base_fallback=base_pred, require_finite=True,
            #         min_support=INC_MMEAN_MIN_SUPPORT,
            #         min_side_rate=INC_MMEAN_MIN_SIDE_RATE,
            #         min_side_count=INC_MMEAN_MIN_SIDE_COUNT,
            #         max_eq_rate=INC_MMEAN_MAX_EQ_RATE,
            #         style=INC_KEY_STYLE,
            #     )
            #     n_new = int(mm2["selected"].sum()) if not mm2.empty else 0
            #     print(f"  Registered min/max (meaningful incomparability): {n_new} (pairs added: {len(pairs2)})")


            for inv in model.invariants:
                print(inv)

    print("\nDone.")

# ───────────────────────── Entrypoint ───────────────────────── #
if __name__ == "__main__":
    main()
