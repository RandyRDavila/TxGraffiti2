# # run_integer_txgraffiti_loop.py
# # Iterative TxGraffiti-style pipeline on a minimal integer dataset.
# # Discovers primes from τ sharpness, then mines/tightens implications and
# # promotes useful predicates (twins/sexy primes via τ(n±2)=2, τ(n±6)=2), while
# # keeping strongest & most general hypotheses.

# from __future__ import annotations
# import math
# import numpy as np
# import pandas as pd
# from itertools import combinations

# # --- TxGraffiti core plumbing ---
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Le, Ge
# from txgraffiti2025.forms.predicates import Predicate, Where, AndPred, LT, GT
# from txgraffiti2025.forms.pretty import format_pred
# from txgraffiti2025.forms.utils import to_expr

# # ================================
# # Minimal integer dataset
# # ================================

# def tau_of(n: int) -> int:
#     if n <= 0: return 0
#     t, r = 0, int(math.isqrt(n))
#     for k in range(1, r + 1):
#         if n % k == 0:
#             t += 2 if k * k != n else 1
#     return t

# def build_integer_df(N: int = 20000) -> pd.DataFrame:
#     n = np.arange(1, N + 1, dtype=int)
#     def safe_tau(x): return tau_of(x) if x >= 1 else 0
#     tau    = np.fromiter((tau_of(i)     for i in n), count=N, dtype=int)
#     tau_m2 = np.fromiter((safe_tau(i-2) for i in n), count=N, dtype=int)
#     tau_p2 = np.fromiter((tau_of(i+2)   for i in n), count=N, dtype=int)
#     tau_m6 = np.fromiter((safe_tau(i-6) for i in n), count=N, dtype=int)
#     tau_p6 = np.fromiter((tau_of(i+6)   for i in n), count=N, dtype=int)
#     return pd.DataFrame({
#         "n": n, "tau": tau,
#         "tau_m2": tau_m2, "tau_p2": tau_p2,
#         "tau_m6": tau_m6, "tau_p6": tau_p6
#     })

# # ================================
# # Mask & pretty helpers
# # ================================

# def _mask(df: pd.DataFrame, pred: Predicate | None) -> pd.Series:
#     if pred is None:
#         return pd.Series(True, index=df.index)
#     m = pred.mask(df)
#     return m.reindex(df.index, fill_value=False).astype(bool)

# def pretty_pred(pred: Predicate | None) -> str:
#     return format_pred(pred, unicode_ops=True)

# def _flatten_and_pred(p):
#     if p is None:
#         return []
#     if isinstance(p, AndPred):
#         return _flatten_and_pred(p.a) + _flatten_and_pred(p.b)
#     return [p]

# def _is_true_pred(p) -> bool:
#     return getattr(p, "name", "") == "TRUE"

# def _and_many(parts):
#     parts = [q for q in parts if q is not None and not _is_true_pred(q)]
#     if not parts:
#         return None
#     out = parts[0]
#     for q in parts[1:]:
#         out = AndPred(out, q)
#     return out

# def normalize_and(pred):
#     if pred is None or _is_true_pred(pred):
#         return None
#     parts = _flatten_and_pred(pred)
#     seen, uniq = set(), []
#     for q in parts:
#         k = repr(q)
#         if k not in seen:
#             seen.add(k)
#             uniq.append(q)
#     return _and_many(uniq)

# # stable key to dedupe predicates/implications
# def _pred_key(df: pd.DataFrame, pred) -> tuple:
#     pn = normalize_and(pred)
#     key_repr = repr(pn)
#     m = _mask(df, pn)
#     supp = int(m.sum())
#     nz = tuple(map(int, np.flatnonzero(m.values[:min(len(m), 64)])))
#     return (key_repr, supp, nz)

# # ================================
# # Weakest-true sharp bound to “discover primes”
# # ================================

# def weakest_tau_lower_true_and_sharp(df: pd.DataFrame, H: Predicate | None) -> tuple[int, pd.Series]:
#     Hm = _mask(df, H)
#     if not Hm.any():
#         raise ValueError("Empty hypothesis mask.")
#     tauH = df.loc[Hm, "tau"].to_numpy()
#     c = int(tauH.min())
#     eq_mask = pd.Series(False, index=df.index)
#     eq_mask.loc[Hm] = (df.loc[Hm, "tau"] == c)
#     if not eq_mask.any():
#         raise RuntimeError("No equality points for τ lower bound.")
#     return c, eq_mask

# # ================================
# # Tiny predicate-induction layer (minimal atoms)
# # ================================

# def atom_mod(m: int, r: int) -> Predicate:
#     return Where(lambda d, m=m, r=r: (d["n"] % m) == r, name=f"(n≡{r} mod {m})")

# def atom_tau_eq(k: int) -> Predicate:
#     return Where(lambda d, k=k: (d["tau"] == k), name=f"(τ={k})")

# def atom_tau_ge(k: int) -> Predicate:
#     return Where(lambda d, k=k: (d["tau"] >= k), name=f"(τ≥{k})")

# def atom_tau_shift_eq2(col: str, label: str) -> Predicate:
#     return Where(lambda d, col=col: (d[col] == 2), name=f"(τ{label}=2)")

# def atom_tau_p2_eq2() -> Predicate:
#     return atom_tau_shift_eq2("tau_p2", "(n+2)")

# def atom_tau_m2_eq2() -> Predicate:
#     return atom_tau_shift_eq2("tau_m2", "(n-2)")

# def atom_tau_p6_eq2() -> Predicate:
#     return atom_tau_shift_eq2("tau_p6", "(n+6)")

# def atom_tau_m6_eq2() -> Predicate:
#     return atom_tau_shift_eq2("tau_m6", "(n-6)")

# def induce_min_description_for_mask(
#     df: pd.DataFrame,
#     target_mask: pd.Series,
#     *,
#     max_and_terms: int = 3
# ) -> Predicate:
#     atoms: list[Predicate] = []
#     atoms += [atom_tau_eq(2), atom_tau_eq(3), atom_tau_ge(2)]
#     for m in (2, 3, 6):  # keep focused but mod-6 included
#         for r in range(m):
#             atoms.append(atom_mod(m, r))
#     # shifted τ==2 atoms in the library (lets inducer pick them if exact)
#     atoms += [atom_tau_p2_eq2(), atom_tau_m2_eq2(), atom_tau_p6_eq2(), atom_tau_m6_eq2()]

#     T = target_mask.astype(bool)

#     # exact single-atom match preferred
#     for A in atoms:
#         if _mask(df, A).equals(T):
#             return A

#     # greedy AND to exact cover if possible
#     current = pd.Series(True, index=df.index)
#     picked: list[Predicate] = []

#     def symdiff(a: pd.Series, b: pd.Series) -> int:
#         return int((a ^ b).sum())

#     best = symdiff(current, T)
#     improved = True
#     while improved and len(picked) < max_and_terms:
#         improved = False
#         bestA, bestNew, bestScore = None, None, best
#         for A in atoms:
#             if A in picked:
#                 continue
#             cand = current & _mask(df, A)
#             score = symdiff(cand, T)
#             if score < bestScore:
#                 bestA, bestNew, bestScore = A, cand, score
#         if bestA is not None:
#             picked.append(bestA)
#             current = bestNew
#             best = bestScore
#             improved = True

#     if current.equals(T) and picked:
#         acc = picked[0]
#         for q in picked[1:]:
#             acc = AndPred(acc, q)
#         acc.name = "(" + ") ∧ (".join(format_pred(p, unicode_ops=True) for p in picked) + ")"
#         return acc

#     return picked[0] if picked else atom_tau_ge(2)

# # ================================
# # Tightness predicates from a conjecture
# # ================================

# def _tightness_preds(df: pd.DataFrame, conj: Conjecture, *, atol=1e-9):
#     rel = conj.relation
#     cond = conj.condition

#     def _eq_mask(d, L, R, tol):
#         Ls = pd.to_numeric(L.eval(d), errors="coerce")
#         Rs = pd.to_numeric(R.eval(d), errors="coerce")
#         return pd.Series(np.isclose(Ls - Rs, 0.0, atol=float(tol)), index=d.index)

#     if isinstance(rel, Le):
#         P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, atol),
#                            name=f"(({rel.left}) = ({rel.right}))")
#         P_eq  = AndPred(cond, P_eq_inner) if cond else P_eq_inner
#         P_str = AndPred(cond, LT(rel.left, rel.right)) if cond else LT(rel.left, rel.right)
#         return [normalize_and(P_eq), normalize_and(P_str)]

#     if isinstance(rel, Ge):
#         P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, atol),
#                            name=f"(({rel.left}) = ({rel.right}))")
#         P_eq  = AndPred(cond, P_eq_inner) if cond else P_eq_inner
#         P_str = AndPred(cond, GT(rel.left, rel.right)) if cond else GT(rel.left, rel.right)
#         return [normalize_and(P_eq), normalize_and(P_str)]

#     if isinstance(rel, Eq):
#         P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, rel.tol),
#                            name=f"(({rel.left}) = ({rel.right}))")
#         P_eq = AndPred(cond, P_eq_inner) if cond else P_eq_inner
#         return [normalize_and(P_eq)]

#     return []

# # ================================
# # Candidate boolean hypotheses (AND/OR) & dedup
# # ================================

# def _and_all(preds):
#     it = iter(preds)
#     acc = next(it)
#     for p in it:
#         acc = AndPred(acc, p)
#     parts = [format_pred(p, unicode_ops=True) for p in preds]
#     acc.name = "(" + ") ∧ (".join(parts) + ")"
#     return acc

# def _or_pred(a: Predicate, b: Predicate) -> Predicate:
#     name = f"({format_pred(a, unicode_ops=True)}) ∨ ({format_pred(b, unicode_ops=True)})"
#     return Where(lambda d, A=a, B=b: _mask(d, A) | _mask(d, B), name=name)

# def build_candidate_hypotheses_with_disjunctions(
#     df: pd.DataFrame,
#     base_hyps,
#     *,
#     include_or: bool = True,
#     min_support: int = 1,
# ):
#     cands = list(base_hyps)
#     base = list(base_hyps)

#     # Pairwise ANDs
#     for A, B in combinations(base, 2):
#         try:
#             cand = _and_all([A, B])
#             m = _mask(df, cand)
#             if int(m.sum()) >= min_support:
#                 cands.append(cand)
#         except Exception:
#             pass

#     # Pairwise ORs (avoid A⊆B / B⊆A and skip ORs that cover all rows)
#     if include_or:
#         seen_keys = set()
#         universe = pd.Series(True, index=df.index)
#         for A, B in combinations(base, 2):
#             try:
#                 mA = _mask(df, A); mB = _mask(df, B)
#                 if not (mA & ~mB).any() or not (mB & ~mA).any():
#                     continue
#                 OR = _or_pred(A, B)
#                 mOR = _mask(df, OR)
#                 if int(mOR.sum()) < min_support or mOR.equals(universe):
#                     continue
#                 key = tuple(sorted([getattr(A, "name", repr(A)), getattr(B, "name", repr(B))]))
#                 if key in seen_keys:
#                     continue
#                 seen_keys.add(key)
#                 cands.append(OR)
#             except Exception:
#                 pass

#     uniq, seen = [], set()
#     for p in cands:
#         k = getattr(p, "name", repr(p))
#         if k not in seen:
#             seen.add(k)
#             uniq.append(p)
#     return uniq

# # ================================
# # Implication miner & filters
# # ================================

# def mine_implications(df, tight_preds, candidate_preds, *, min_support=1):
#     out = []
#     for H in tight_preds:
#         Hn = normalize_and(H)
#         Hm = _mask(df, Hn)
#         supH = int(Hm.sum())
#         if supH < min_support:
#             continue
#         for C in candidate_preds:
#             Cn = normalize_and(C)
#             Cm = _mask(df, Cn)
#             if not Cm.any():
#                 continue
#             implies = not (Hm & ~Cm).any()
#             if implies:
#                 equiv = not (Cm & ~Hm).any()
#                 out.append({
#                     "H": Hn, "C": Cn,
#                     "support_H": supH,
#                     "support_C": int(Cm.sum()),
#                     "equiv": bool(equiv),
#                 })
#     return out

# def sophie_filter_implications(
#     df: pd.DataFrame,
#     mined_rows: list,
#     *,
#     group_by: str = "consequent",
#     min_support: int = 1,
# ):
#     uniq = []
#     seen = set()
#     for r in mined_rows:
#         k = (_pred_key(df, r["H"]), _pred_key(df, r["C"]))
#         if k not in seen:
#             seen.add(k)
#             uniq.append(r)

#     pools = {}
#     kept = []
#     for r in uniq:
#         H = normalize_and(r["H"]); C = normalize_and(r["C"])
#         Hm = _mask(df, H)
#         if int(Hm.sum()) < min_support:
#             continue
#         gkey = None if group_by == "global" else repr(C)
#         if gkey not in pools:
#             pools[gkey] = pd.Series(False, index=df.index)
#         delta = Hm & ~pools[gkey]
#         if delta.any():
#             kept.append(r)
#             pools[gkey] = pools[gkey] | Hm
#     return kept

# def hazel_like_rank_implications(
#     mined: list[dict],
#     *,
#     drop_frac: float = 0.25,
#     min_support: int | None = None,
# ):
#     if not mined:
#         return []
#     items = [r for r in mined if (min_support is None or r["support_H"] >= min_support)]
#     if not items:
#         return []
#     items_sorted = sorted(items, key=lambda r: r["support_H"])
#     k = max(0, int(len(items_sorted) * drop_frac))
#     kept = items_sorted[k:]
#     kept.sort(
#         key=lambda r: (
#             not r["equiv"],
#             -r["support_H"],
#             r["support_C"],
#             format_pred(r["H"], unicode_ops=True),
#             format_pred(r["C"], unicode_ops=True),
#         )
#     )
#     return kept

# # ================================
# # Conjecture templates for the integer dataset
# # ================================

# def mk_ge(hyp: Predicate | None, left: str, right, name="") -> Conjecture:
#     return Conjecture(Ge(to_expr(left), to_expr(right)), hyp, name=name or f"{left}_ge_{right}")

# def mk_eq(hyp: Predicate | None, left: str, right, name="") -> Conjecture:
#     return Conjecture(Eq(to_expr(left), to_expr(right), tol=1e-9), hyp, name=name or f"{left}_eq_{right}")

# def conjecture_bank_for_round(df: pd.DataFrame, hyp_bank: list[Predicate]) -> list[Conjecture]:
#     """
#     Build a small, targeted set of conjectures using current hypotheses.
#     - Always include τ ≥ 2 under (n≥2).
#     - If 'prime' is present (i.e., τ=2), also include τ(n±2)=2 and τ(n±6)=2 under prime.
#     """
#     out: list[Conjecture] = []

#     # Always test the weakest sharp bound again (lets tightness feed new H's)
#     H_ge2 = Where(lambda d: d["n"] >= 2, name="(n≥2)")
#     out.append(mk_ge(H_ge2, "tau", 2, name="tau_ge_2_on_n_ge_2"))

#     # If prime-like present, test shifted prime patterns under prime
#     has_prime_like = any("(τ=2)" in getattr(H, "name", "") for H in hyp_bank)
#     if has_prime_like:
#         P = next(H for H in hyp_bank if "(τ=2)" in getattr(H, "name", ""))
#         out.append(mk_eq(P, "tau_p2", 2, name="tau_p2_eq_2_on_prime"))  # twin (forward)
#         out.append(mk_eq(P, "tau_m2", 2, name="tau_m2_eq_2_on_prime"))  # twin (backward)
#         out.append(mk_eq(P, "tau_p6", 2, name="tau_p6_eq_2_on_prime"))  # sexy forward
#         out.append(mk_eq(P, "tau_m6", 2, name="tau_m6_eq_2_on_prime"))  # sexy backward

#     return out

# # ================================
# # Promotion rules
# # ================================

# def promote_from_tightness(
#     df: pd.DataFrame,
#     tight_preds: list[Predicate],
#     hyp_bank: list[Predicate],
#     *,
#     support_floor: int = 50
# ) -> list[Predicate]:
#     """
#     If we see tightness predicates like (Prime ∧ (τ(n+2)=2)), promote:
#       1) the atomic consequent (τ(n+2)=2) if not present
#       2) the conjunction (Prime ∧ τ(n+2)=2) if not present
#     Same for ±2 and ±6.
#     """
#     def have_name(s: str) -> bool:
#         return any(s == getattr(h, "name", "") for h in hyp_bank)

#     new_hyps: list[Predicate] = []

#     # atomic consequent atoms
#     atoms = {
#         "(τ(n+2)=2)": atom_tau_p2_eq2(),
#         "(τ(n-2)=2)": atom_tau_m2_eq2(),
#         "(τ(n+6)=2)": atom_tau_p6_eq2(),
#         "(τ(n-6)=2)": atom_tau_m6_eq2(),
#     }

#     # scan tightness predicates
#     for P in tight_preds:
#         s = pretty_pred(P)
#         supp = int(_mask(df, P).sum())
#         if supp < support_floor:
#             continue
#         # If this tightness contains one of our atomic atoms, consider promotion
#         for label, atom in atoms.items():
#             if label in s:
#                 # add atomic if missing
#                 if not have_name(atom.name):
#                     new_hyps.append(atom)
#                 # also add the conjunction explicitly if missing
#                 # try to recover a "prime-like" conjunct by name
#                 prime_like = next((H for H in hyp_bank if "(τ=2)" in getattr(H, "name", "")), None)
#                 if prime_like is not None:
#                     conj = AndPred(prime_like, atom)
#                     conj.name = f"({pretty_pred(prime_like)}) ∧ {atom.name}"
#                     if not have_name(conj.name):
#                         new_hyps.append(conj)

#     # dedup by name
#     out, seen = [], set()
#     for h in new_hyps:
#         nm = getattr(h, "name", repr(h))
#         if nm not in seen:
#             seen.add(nm)
#             out.append(h)
#     # filter ones already in bank
#     filtered = [h for h in out if all(getattr(h, "name", "") != getattr(old, "name", "") for old in hyp_bank)]
#     return filtered

# # ================================
# # Main loop
# # ================================

# def main(N: int = 20000, rounds: int = 2):
#     df = build_integer_df(N)

#     # Round 0: start with H=(n≥2) and discover equality mask for τ lower bound
#     H_ge2 = Where(lambda d: d["n"] >= 2, name="(n≥2)")
#     c, eq_mask = weakest_tau_lower_true_and_sharp(df, H_ge2)
#     print("=== Round 0: Weakest-true sharp bound on H=(n≥2) ===")
#     print(f"τ minimal c: {c}; equality hits τ=={c}: {int(eq_mask.sum())}")

#     # Induce minimal predicate for equality set (should be τ=2)
#     P_eq = induce_min_description_for_mask(df, eq_mask)
#     print("\nInduced base predicate (expected prime):", pretty_pred(P_eq))

#     # Initialize hypothesis bank with H_ge2 + induced predicate; add mod/parity atoms
#     hyp_bank: list[Predicate] = [
#         H_ge2, P_eq,
#         atom_mod(2,0), atom_mod(2,1),
#         atom_mod(3,0), atom_mod(3,1), atom_mod(3,2),
#         atom_mod(6,1), atom_mod(6,5),
#     ]
#     print("\nInitial hypothesis bank:")
#     for h in hyp_bank:
#         print(" -", pretty_pred(h))

#     # Iterative rounds
#     for r in range(1, rounds + 1):
#         print(f"\n================ Round {r} ================")
#         # 1) build conjectures with current bank
#         conj = conjecture_bank_for_round(df, hyp_bank)
#         print(f"Conjectures this round ({len(conj)}):")
#         for cjt in conj:
#             print(" -", cjt.pretty(arrow="⇒"))

#         # 2) promote inequalities to equalities if tight everywhere on H
#         promoted = []
#         for cjt in conj:
#             rel = cjt.relation
#             if isinstance(rel, (Le, Ge)):
#                 applicable = _mask(df, cjt.condition)
#                 if applicable.any():
#                     holds = rel.evaluate(df).reindex(df.index, fill_value=False).astype(bool)
#                     slack = rel.slack(df)
#                     tight = pd.Series(np.isclose(slack, 0.0, atol=1e-9), index=df.index)
#                     if bool((applicable & holds).all()) and bool((applicable & tight).all()):
#                         new_rel = Eq(rel.left, rel.right, tol=1e-9)
#                         promoted.append(Conjecture(new_rel, cjt.condition, name=cjt.name + " [PROMOTED]"))
#                         continue
#             promoted.append(cjt)

#         # 3) derive tightness predicates
#         tight_preds = []
#         print("\nDerived tightness predicates:")
#         for cjt in promoted:
#             T = _tightness_preds(df, cjt, atol=1e-9)
#             for P in T:
#                 supp = int(_mask(df, P).sum())
#                 print(f" • {pretty_pred(P)}   support={supp}")
#             tight_preds.extend(T)

#         # 4) build candidate boolean predicates from hypothesis bank (AND/OR)
#         # include shifted τ==2 atoms as stand-alone candidates to enable implications
#         shifted_atoms = [atom_tau_p2_eq2(), atom_tau_m2_eq2(), atom_tau_p6_eq2(), atom_tau_m6_eq2()]
#         candidates = build_candidate_hypotheses_with_disjunctions(
#             df, hyp_bank + shifted_atoms, include_or=True, min_support=1
#         )

#         # 5) mine implications H_tight ⇒ C
#         mined = mine_implications(df, tight_preds, candidates, min_support=1)
#         mined = sophie_filter_implications(df, mined, group_by="consequent", min_support=1)
#         ranked = hazel_like_rank_implications(mined, drop_frac=0.25, min_support=None)

#         print("\n=== Implications mined from tightness classes (Hazel-like filtered) ===")
#         if not ranked:
#             print("(none)")
#         else:
#             for row in ranked:
#                 arrow = "≡" if row["equiv"] else "⇒"
#                 print(f"{format_pred(row['H'], unicode_ops=True)} {arrow} "
#                       f"{format_pred(row['C'], unicode_ops=True)}   "
#                       f"[support(H)={row['support_H']}, support(C)={row['support_C']}]")

#         # 6) Promotion: introduce useful atoms & conjunctions discovered via tightness
#         new_hyps = promote_from_tightness(df, tight_preds, hyp_bank, support_floor=50)
#         if new_hyps:
#             print("\n→ Promoting new hypotheses:")
#             for h in new_hyps:
#                 hyp_bank.append(h)
#                 print("   +", pretty_pred(h))

#         # 7) Show bank at end of round
#         print("\nHypothesis bank (end of round):")
#         for h in hyp_bank:
#             print(" -", pretty_pred(h))

#     # Final small sanity: count primes and show a few twin/sexy examples if present
#     PrimePred = [h for h in hyp_bank if "(τ=2)" in getattr(h, "name", "")]
#     if PrimePred:
#         prime_mask = _mask(df, PrimePred[0])
#         print(f"\n#primes discovered (τ=2): {int(prime_mask.sum())}")

#     def show_examples(label: str, atom: Predicate, k: int = 10):
#         m = _mask(df, atom)
#         idxs = list(df.index[m])[:k]
#         if idxs:
#             cols = ["n", "tau", "tau_m2", "tau_p2", "tau_m6", "tau_p6"]
#             print(f"\nSample rows for {label}:")
#             print(df.loc[idxs, cols].to_string(index=False))

#     show_examples("twin-forward (τ(n+2)=2)", atom_tau_p2_eq2())
#     show_examples("twin-backward (τ(n-2)=2)", atom_tau_m2_eq2())
#     show_examples("sexy-forward (τ(n+6)=2)", atom_tau_p6_eq2())
#     show_examples("sexy-backward (τ(n-6)=2)", atom_tau_m6_eq2())

# if __name__ == "__main__":
#     # Increase rounds for deeper bootstrapping (2–4 is good on this setup).
#     main(N=20000, rounds=2)


# run_integer_txgraffiti_loop.py
# Integer discovery loop with OR throttling, “new-only RHS” filters, and IG/lift ranking.

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from itertools import combinations

# --- TxGraffiti core plumbing ---
from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Le, Ge
from txgraffiti2025.forms.predicates import Predicate, Where, AndPred, LT, GT
from txgraffiti2025.forms.pretty import format_pred
from txgraffiti2025.forms.utils import to_expr

# ================================
# Tunables
# ================================

OR_COVER_CAP       = 0.90   # skip candidate ORs that cover > 90% of the universe
MIN_PHC            = 0.15   # require p(H|C) >= 0.15 to avoid “RHS true almost everywhere”
MIN_DISJ_PHC       = 0.20   # for OR A∨B consequents: each disjunct needs p(disjunct|H) >= 0.20
PROMOTE_ATOM_FLOOR = 50     # support threshold to promote atoms like τ(n±k)=2 to the hyp bank

# ================================
# Minimal integer dataset
# ================================

def tau_of(n: int) -> int:
    if n <= 0: return 0
    t, r = 0, int(math.isqrt(n))
    for k in range(1, r + 1):
        if n % k == 0:
            t += 2 if k * k != n else 1
    return t

def build_integer_df(N: int = 20000) -> pd.DataFrame:
    n = np.arange(1, N + 1, dtype=int)
    tau = np.fromiter((tau_of(i) for i in n), count=N, dtype=int)
    # shifted τ for small k
    tau_p2 = np.fromiter((tau_of(i + 2) for i in n), count=N, dtype=int)
    tau_m2 = np.fromiter((tau_of(i - 2) for i in n), count=N, dtype=int)
    tau_p6 = np.fromiter((tau_of(i + 6) for i in n), count=N, dtype=int)
    tau_m6 = np.fromiter((tau_of(i - 6) for i in n), count=N, dtype=int)
    return pd.DataFrame({
        "n": n, "tau": tau,
        "tau_p2": tau_p2, "tau_m2": tau_m2,
        "tau_p6": tau_p6, "tau_m6": tau_m6,
    })

# ================================
# Mask & pretty helpers
# ================================

def _mask(df: pd.DataFrame, pred: Predicate | None) -> pd.Series:
    if pred is None:
        return pd.Series(True, index=df.index)
    m = pred.mask(df)
    return m.reindex(df.index, fill_value=False).astype(bool)

def pretty_pred(pred: Predicate | None) -> str:
    return format_pred(pred, unicode_ops=True)

def _flatten_and_pred(p):
    if p is None:
        return []
    if isinstance(p, AndPred):
        return _flatten_and_pred(p.a) + _flatten_and_pred(p.b)
    return [p]

def _is_true_pred(p) -> bool:
    return getattr(p, "name", "") == "TRUE"

def _and_many(parts):
    parts = [q for q in parts if q is not None and not _is_true_pred(q)]
    if not parts:
        return None
    out = parts[0]
    for q in parts[1:]:
        out = AndPred(out, q)
    return out

def normalize_and(pred):
    if pred is None or _is_true_pred(pred):
        return None
    parts = _flatten_and_pred(pred)
    seen, uniq = set(), []
    for q in parts:
        k = repr(q)
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    return _and_many(uniq)

def _pred_key(df: pd.DataFrame, pred) -> tuple:
    pn = normalize_and(pred)
    key_repr = repr(pn)
    m = _mask(df, pn)
    supp = int(m.sum())
    nz = tuple(map(int, np.flatnonzero(m.values[:min(len(m), 64)])))
    return (key_repr, supp, nz)

# ================================
# Weakest-true sharp bound to “discover primes”
# ================================

def weakest_tau_lower_true_and_sharp(df: pd.DataFrame, H: Predicate | None) -> tuple[int, pd.Series]:
    Hm = _mask(df, H)
    if not Hm.any():
        raise ValueError("Empty hypothesis mask.")
    tauH = df.loc[Hm, "tau"].to_numpy()
    c = int(tauH.min())
    eq_mask = pd.Series(False, index=df.index)
    eq_mask.loc[Hm] = (df.loc[Hm, "tau"] == c)
    if not eq_mask.any():
        raise RuntimeError("No equality points for τ lower bound.")
    return c, eq_mask

# ================================
# Minimal atoms
# ================================

def atom_mod(m: int, r: int) -> Predicate:
    return Where(lambda d, m=m, r=r: (d["n"] % m) == r, name=f"(n≡{r} mod {m})")

def atom_tau_eq(k: int) -> Predicate:
    return Where(lambda d, k=k: (d["tau"] == k), name=f"(τ={k})")

def atom_tau_ge(k: int) -> Predicate:
    return Where(lambda d, k=k: (d["tau"] >= k), name=f"(τ≥{k})")

def atom_tau_shift_eq2(col: str, label: str) -> Predicate:
    return Where(lambda d, col=col: (d[col] == 2), name=f"(τ({label})=2)")

def induce_min_description_for_mask(
    df: pd.DataFrame,
    target_mask: pd.Series,
    *,
    max_and_terms: int = 3
) -> Predicate:
    atoms: list[Predicate] = []
    atoms += [atom_tau_eq(2), atom_tau_eq(3), atom_tau_ge(2)]
    for m in (2, 3, 6):
        for r in range(m):
            atoms.append(atom_mod(m, r))
    atoms += [
        atom_tau_shift_eq2("tau_p2", "n+2"),
        atom_tau_shift_eq2("tau_m2", "n-2"),
        atom_tau_shift_eq2("tau_p6", "n+6"),
        atom_tau_shift_eq2("tau_m6", "n-6"),
    ]

    T = target_mask.astype(bool)

    # exact single-atom match preferred
    for A in atoms:
        if _mask(df, A).equals(T):
            return A

    # greedy AND to exact cover if possible
    current = pd.Series(True, index=df.index)
    picked: list[Predicate] = []

    def symdiff(a: pd.Series, b: pd.Series) -> int:
        return int((a ^ b).sum())

    best = symdiff(current, T)
    improved = True
    while improved and len(picked) < max_and_terms:
        improved = False
        bestA, bestNew, bestScore = None, None, best
        for A in atoms:
            if A in picked:
                continue
            cand = current & _mask(df, A)
            score = symdiff(cand, T)
            if score < bestScore:
                bestA, bestNew, bestScore = A, cand, score
        if bestA is not None:
            picked.append(bestA)
            current = bestNew
            best = bestScore
            improved = True

    if current.equals(T) and picked:
        acc = picked[0]
        for q in picked[1:]:
            acc = AndPred(acc, q)
        acc.name = "(" + ") ∧ (".join(format_pred(p, unicode_ops=True) for p in picked) + ")"
        return acc

    return picked[0] if picked else atom_tau_ge(2)

# ================================
# Tightness predicates from a conjecture
# ================================

def _tightness_preds(df: pd.DataFrame, conj: Conjecture, *, atol=1e-9):
    rel = conj.relation
    cond = conj.condition

    def _eq_mask(d, L, R, tol):
        Ls = pd.to_numeric(L.eval(d), errors="coerce")
        Rs = pd.to_numeric(R.eval(d), errors="coerce")
        return pd.Series(np.isclose(Ls - Rs, 0.0, atol=float(tol)), index=d.index)

    if isinstance(rel, Le):
        P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, atol),
                           name=f"(({rel.left}) = ({rel.right}))")
        P_eq  = AndPred(cond, P_eq_inner) if cond else P_eq_inner
        P_str = AndPred(cond, LT(rel.left, rel.right)) if cond else LT(rel.left, rel.right)
        return [normalize_and(P_eq), normalize_and(P_str)]

    if isinstance(rel, Ge):
        P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, atol),
                           name=f"(({rel.left}) = ({rel.right}))")
        P_eq  = AndPred(cond, P_eq_inner) if cond else P_eq_inner
        P_str = AndPred(cond, GT(rel.left, rel.right)) if cond else GT(rel.left, rel.right)
        return [normalize_and(P_eq), normalize_and(P_str)]

    if isinstance(rel, Eq):
        P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, rel.tol),
                           name=f"(({rel.left}) = ({rel.right}))")
        P_eq = AndPred(cond, P_eq_inner) if cond else P_eq_inner
        return [normalize_and(P_eq)]

    return []

# ================================
# Candidate boolean hypotheses (AND/OR) & dedup
# ================================

def _and_all(preds):
    it = iter(preds)
    acc = next(it)
    for p in it:
        acc = AndPred(acc, p)
    parts = [format_pred(p, unicode_ops=True) for p in preds]
    acc.name = "(" + ") ∧ (".join(parts) + ")"
    return acc

def _or_pred(a: Predicate, b: Predicate) -> Predicate:
    name = f"({format_pred(a, unicode_ops=True)}) ∨ ({format_pred(b, unicode_ops=True)})"
    OR = Where(lambda d, A=a, B=b: _mask(d, A) | _mask(d, B), name=name)
    # store parts for later contribution checks
    setattr(OR, "_left", a)
    setattr(OR, "_right", b)
    return OR

def build_candidate_hypotheses_with_disjunctions(
    df: pd.DataFrame,
    base_hyps,
    *,
    include_or: bool = True,
    min_support: int = 1,
):
    cands = list(base_hyps)
    base = list(base_hyps)

    # Pairwise ANDs
    for A, B in combinations(base, 2):
        try:
            cand = _and_all([A, B])
            m = _mask(df, cand)
            if int(m.sum()) >= min_support:
                cands.append(cand)
        except Exception:
            pass

    # Pairwise ORs (avoid trivial subset and avoid broad ORs)
    if include_or:
        seen_keys = set()
        N = len(df)
        for A, B in combinations(base, 2):
            try:
                mA = _mask(df, A); mB = _mask(df, B)
                if not (mA & ~mB).any() or not (mB & ~mA).any():
                    continue  # trivial
                OR = _or_pred(A, B)
                mOR = _mask(df, OR)
                cover = mOR.mean()
                if cover > OR_COVER_CAP:   # throttle very broad ORs
                    continue
                if int(mOR.sum()) < min_support:
                    continue
                key = tuple(sorted([getattr(A, "name", repr(A)), getattr(B, "name", repr(B))]))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                cands.append(OR)
            except Exception:
                pass

    uniq, seen = [], set()
    for p in cands:
        k = getattr(p, "name", repr(p))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq

# ================================
# Mining + interest filters + ranking
# ================================

def _entropy(p: float) -> float:
    if p <= 0 or p >= 1: return 0.0
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def _kl_bernoulli(p: float, q: float) -> float:
    # KL(p || q) for Bernoulli; safe for edges.
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def _is_or(pred: Predicate) -> bool:
    return "∨" in getattr(pred, "name", "")

def _or_parts(pred: Predicate):
    return getattr(pred, "_left", None), getattr(pred, "_right", None)

def mine_implications(df, tight_preds, candidate_preds, *, min_support=1):
    out = []
    N = len(df)
    for H in tight_preds:
        Hn = normalize_and(H)
        Hm = _mask(df, Hn)
        supH = int(Hm.sum())
        if supH < min_support:
            continue
        pH = supH / N

        for C in candidate_preds:
            Cn = normalize_and(C)
            Cm = _mask(df, Cn)
            supC = int(Cm.sum())
            if supC == 0:
                continue
            pC = supC / N

            implies = not (Hm & ~Cm).any()          # H ⊆ C (precision 1)
            if not implies:
                continue

            equiv = not (Cm & ~Hm).any()            # C ⊆ H
            # “new-only RHS” meaningful: require p(H|C) not tiny
            pH_given_C = (int((Cm & Hm).sum()) / supC) if supC > 0 else 0.0
            if pH_given_C < MIN_PHC:
                continue

            # For OR consequents, each disjunct must carry weight on H
            if _is_or(Cn):
                A, B = _or_parts(C)
                if (A is None) or (B is None):
                    pass
                else:
                    pA_given_H = int((_mask(df, A) & Hm).sum()) / supH
                    pB_given_H = int((_mask(df, B) & Hm).sum()) / supH
                    if (pA_given_H < MIN_DISJ_PHC) or (pB_given_H < MIN_DISJ_PHC):
                        continue

            # Scores
            pC_given_H = 1.0  # by implication
            lift = pC_given_H / pC
            ig = _kl_bernoulli(pC_given_H, pC)      # IG of predicting C under H vs baseline

            out.append({
                "H": Hn, "C": Cn,
                "support_H": supH,
                "support_C": supC,
                "equiv": bool(equiv),
                "pC": pC,
                "pH": pH,
                "pH_given_C": pH_given_C,
                "lift": lift,
                "ig": ig,
            })
    return out

def hazel_like_rank_implications(mined: list[dict]):
    if not mined:
        return []
    # Sort by: equivalence first, IG desc, lift desc, support_H desc, support_C asc, names
    mined.sort(
        key=lambda r: (
            not r["equiv"],
            -r["ig"],
            -r["lift"],
            -r["support_H"],
            r["support_C"],
            format_pred(r["H"], unicode_ops=True),
            format_pred(r["C"], unicode_ops=True),
        )
    )
    return mined

def sophie_filter_implications(
    df: pd.DataFrame,
    mined_rows: list,
    *,
    group_by: str = "consequent",
    min_support: int = 1,
):
    uniq = []
    seen = set()
    for r in mined_rows:
        k = (_pred_key(df, r["H"]), _pred_key(df, r["C"]))
        if k not in seen:
            seen.add(k)
            uniq.append(r)

    pools = {}
    kept = []
    for r in uniq:
        H = normalize_and(r["H"]); C = normalize_and(r["C"])
        Hm = _mask(df, H)
        if int(Hm.sum()) < min_support:
            continue
        gkey = None if group_by == "global" else repr(C)
        if gkey not in pools:
            pools[gkey] = pd.Series(False, index=df.index)
        delta = Hm & ~pools[gkey]
        if delta.any():
            kept.append(r)
            pools[gkey] = pools[gkey] | Hm
    return kept

# ================================
# Conjecture templates
# ================================

def mk_ge(hyp: Predicate | None, left: str, right, name="") -> Conjecture:
    return Conjecture(Ge(to_expr(left), to_expr(right)), hyp, name=name or f"{left}_ge_{right}")

def mk_eq(hyp: Predicate | None, left: str, right, name="") -> Conjecture:
    return Conjecture(Eq(to_expr(left), to_expr(right), tol=1e-9), hyp, name=name or f"{left}_eq_{right}")

def conjecture_bank_for_round(df: pd.DataFrame, hyp_bank: list[Predicate]) -> list[Conjecture]:
    out = []
    H_ge2 = Where(lambda d: d["n"] >= 2, name="(n≥2)")
    out.append(mk_ge(H_ge2, "tau", 2, name="tau_ge_2_on_n_ge_2"))

    has_prime_like = any("τ=2" in getattr(H, "name", "") for H in hyp_bank)
    if has_prime_like:
        # twin/sexy templates under prime
        P = [H for H in hyp_bank if "τ=2" in getattr(H, "name", "")][0]
        out.append(mk_eq(P, "tau_p2", 2, name="tau_p2_eq_2_on_prime"))
        out.append(mk_eq(P, "tau_m2", 2, name="tau_m2_eq_2_on_prime"))
        out.append(mk_eq(P, "tau_p6", 2, name="tau_p6_eq_2_on_prime"))
        out.append(mk_eq(P, "tau_m6", 2, name="tau_m6_eq_2_on_prime"))
    return out

# ================================
# Main loop
# ================================

def main(N: int = 20000, rounds: int = 2):
    df = build_integer_df(N)

    # Round 0: start with H=(n≥2) and discover equality mask for τ lower bound
    H_ge2 = Where(lambda d: d["n"] >= 2, name="(n≥2)")
    c, eq_mask = weakest_tau_lower_true_and_sharp(df, H_ge2)
    print("=== Round 0: Weakest-true sharp bound on H=(n≥2) ===")
    print(f"τ minimal c: {c}; equality hits τ=={c}: {int(eq_mask.sum())}")

    # Induce minimal predicate for equality set (should be τ=2)
    P_eq = induce_min_description_for_mask(df, eq_mask)
    print("\nInduced base predicate (expected prime):", pretty_pred(P_eq))

    # Initialize hypothesis bank with H_ge2 + induced predicate; add small congruence atoms
    hyp_bank: list[Predicate] = [H_ge2, P_eq]
    for m, residues in [(2, [0,1]), (3, [0,1,2]), (6, [1,5])]:
        for r in residues:
            hyp_bank.append(atom_mod(m, r))

    print("\nInitial hypothesis bank:")
    for h in hyp_bank:
        print(" -", pretty_pred(h))

    # Also prepare a sieve OR candidate (n≡1 mod 6) ∨ (n≡5 mod 6)
    sieve_or = _or_pred(atom_mod(6,1), atom_mod(6,5))

    for r in range(1, rounds + 1):
        print(f"\n================ Round {r} ================")
        # 1) build conjectures
        conj = conjecture_bank_for_round(df, hyp_bank)
        print(f"Conjectures this round ({len(conj)}):")
        for cjt in conj:
            print(" -", cjt.pretty(arrow="⇒"))

        # 2) maybe promote to equality if tight everywhere on H
        promoted = []
        for cjt in conj:
            rel = cjt.relation
            if isinstance(rel, (Le, Ge)):
                applicable = _mask(df, cjt.condition)
                if applicable.any():
                    holds = rel.evaluate(df).reindex(df.index, fill_value=False).astype(bool)
                    slack = rel.slack(df)
                    tight = pd.Series(np.isclose(slack, 0.0, atol=1e-9), index=df.index)
                    if bool((applicable & holds).all()) and bool((applicable & tight).all()):
                        new_rel = Eq(rel.left, rel.right, tol=1e-9)
                        promoted.append(Conjecture(new_rel, cjt.condition, name=cjt.name + " [PROMOTED]"))
                        continue
            promoted.append(cjt)

        # 3) derive tightness predicates
        tight_preds = []
        print("\nDerived tightness predicates:")
        for cjt in promoted:
            T = _tightness_preds(df, cjt, atol=1e-9)
            for P in T:
                supp = int(_mask(df, P).sum())
                print(f" • {pretty_pred(P)}   support={supp}")
            tight_preds.extend(T)

        # 4) candidate boolean predicates from hypothesis bank (AND/OR), plus sieve OR and shift atoms
        base_cands = list(hyp_bank) + [
            atom_tau_shift_eq2("tau_p2", "n+2"),
            atom_tau_shift_eq2("tau_m2", "n-2"),
            atom_tau_shift_eq2("tau_p6", "n+6"),
            atom_tau_shift_eq2("tau_m6", "n-6"),
            sieve_or,
        ]
        candidates = build_candidate_hypotheses_with_disjunctions(
            df, base_cands, include_or=True, min_support=1
        )

        # 5) mine implications + filters + ranking
        mined = mine_implications(df, tight_preds, candidates, min_support=1)
        mined = sophie_filter_implications(df, mined, group_by="consequent", min_support=1)
        ranked = hazel_like_rank_implications(mined)

        print("\n=== Implications mined from tightness classes (filtered & IG-ranked) ===")
        if not ranked:
            print("(none)")
        else:
            for row in ranked[:50]:  # keep print concise
                arrow = "≡" if row["equiv"] else "⇒"
                Hs = format_pred(row['H'], unicode_ops=True)
                Cs = format_pred(row['C'], unicode_ops=True)
                print(f"{Hs} {arrow} {Cs}   "
                      f"[support(H)={row['support_H']}, support(C)={row['support_C']}, "
                      f"lift={row['lift']:.2f}, IG={row['ig']:.3f}, p(H|C)={row['pH_given_C']:.2f}]")

        # 6) Promote useful atoms if they exist
        for col, label in [("tau_p2","n+2"), ("tau_m2","n-2"), ("tau_p6","n+6"), ("tau_m6","n-6")]:
            atom = atom_tau_shift_eq2(col, label)
            supp = int(_mask(df, atom).sum())
            already = any(getattr(h, "name", "") == atom.name for h in hyp_bank)
            if (supp >= PROMOTE_ATOM_FLOOR) and not already:
                hyp_bank.append(atom)
                print(f"→ Promoted atom into hypothesis bank: {pretty_pred(atom)} [support={supp}]")

        # 7) Show bank at end of round
        print("\nHypothesis bank (end of round):")
        for h in hyp_bank:
            print(" -", pretty_pred(h))

    # Final small sanity: prime & twin samples
    PrimePred = [h for h in hyp_bank if "(τ=2)" in getattr(h, "name", "")]
    if PrimePred:
        prime_mask = _mask(df, PrimePred[0])
        print(f"\n#primes discovered (τ=2): {int(prime_mask.sum())}")

    # quick samples for the shift classes
    for col, label in [("tau_p2","n+2"), ("tau_m2","n-2"), ("tau_p6","n+6"), ("tau_m6","n-6")]:
        mask = df[col] == 2
        idxs = list(df.index[mask])[:10]
        if idxs:
            print(f"\nSample rows for τ({label})=2:")
            cols = ["n","tau","tau_m2","tau_p2","tau_m6","tau_p6"]
            print(df.loc[idxs, cols].to_string(index=False))

if __name__ == "__main__":
    # Increase rounds for deeper bootstrapping (2–3 is plenty on this minimal setup).
    main(N=20000, rounds=2)
