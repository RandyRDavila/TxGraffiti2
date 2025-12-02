# src/txgraffiti2025/systems/conjecture_types_runner.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Literal, Any

import numpy as np
import pandas as pd

# --- core types / utilities ---
from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Le, Ge
from txgraffiti2025.forms.predicates import Predicate, Where, AndPred
from txgraffiti2025.forms.pretty import format_pred
from txgraffiti2025.forms.utils import to_expr

# --- pre ---
from txgraffiti2025.processing.pre.hypotheses import enumerate_boolean_hypotheses
from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses
from txgraffiti2025.processing.pre.constants_cache import precompute_constant_ratios_pairs

# --- generators / post-generalizers / refiner ---
from txgraffiti2025.generators import ratios
from txgraffiti2025.processing.post.generalize_from_constants import propose_generalizations_from_constants
from txgraffiti2025.processing.post.generalize import propose_joint_generalizations
from txgraffiti2025.processing.post.refine_numeric import refine_numeric_bounds, RefinementConfig

# --- filters ---
from txgraffiti2025.processing.post import hazel_rank, morgan_filter
from txgraffiti2025.processing.post import dalmatian_filter


# =====================================================================================
# Unified class combining: (A–E) conjecture generation path + tightness/implication miner
# =====================================================================================

@dataclass
class ConjectureTypesResult:
    # A) hypotheses
    kept_hypotheses: List[Predicate] = field(default_factory=list)
    saved_equivalence_conjs: List[Any] = field(default_factory=list)

    # B–D) type-one & generalizations
    type_one: List[Conjecture] = field(default_factory=list)
    generalized_from_constants: List[Conjecture] = field(default_factory=list)
    joint_generals: List[Conjecture] = field(default_factory=list)

    # D3) numeric refinement
    refined: List[Conjecture] = field(default_factory=list)

    # E) final filtered
    hazel_kept_sorted: List[Conjecture] = field(default_factory=list)
    morgan_kept: List[Conjecture] = field(default_factory=list)
    dalmatian_kept: List[Conjecture] = field(default_factory=list)

    # Tightness + implication mining
    promoted_seeds: List[Conjecture] = field(default_factory=list)
    tightness_preds: List[Predicate] = field(default_factory=list)
    mined_implications_ranked: List[dict] = field(default_factory=list)


class ConjectureTypesRunner:
    """
    A single entry-point class that:
      (1) builds/simplifies hypotheses (base-aware),
      (2) generates type-one ratio conjectures for a target vs. other numeric columns,
      (3) generalizes from constant slopes & joint slope/reciprocal/intercept,
      (4) optionally refines numerics,
      (5) Hazel → Morgan → Dalmatian filters,
      (6) mines OR/AND-style implications from tightness classes of chosen seed conjectures.

    Usage
    -----
    runner = ConjectureTypesRunner(df, target="independence_number")
    res = runner.run_all(seed_mode="system", tightness_source="final")
    runner.print_summary(res)
    """

    # -------------------------
    # Construction / config
    # -------------------------
    def __init__(
        self,
        df: pd.DataFrame,
        target: str = "independence_number",
        *,
        min_support_frac: float = 0.05,
        atol: float = 1e-9,
        use_refiner: bool = True,
        max_denominator: int = 50,
        const_cache_shifts: Tuple[int, ...] = (-1, 0, 1),
        hazel_drop_frac: float = 0.25,
    ):
        self.df = df
        self.target = target
        self.min_support_frac = float(min_support_frac)
        self.atol = float(atol)
        self.use_refiner = bool(use_refiner)
        self.max_denominator = int(max_denominator)
        self.const_cache_shifts = tuple(const_cache_shifts)
        self.hazel_drop_frac = float(hazel_drop_frac)

        # numeric columns used as features (excluding target)
        self.numeric_cols = self._numeric_columns_excluding_target()

    # -------------------------
    # Public API
    # -------------------------
    def run_all(
        self,
        seed_conjectures: Optional[List[Conjecture]] = None,
        *,
        seed_mode: Literal["system", "given", "default"] = "system",
        tightness_source: Literal["final", "type_one", "all"] = "final",
    ) -> ConjectureTypesResult:
        res = ConjectureTypesResult()

        # ===== (A)–(E) generation pipeline =====
        hyps = enumerate_boolean_hypotheses(
            self.df, include_base=True, include_pairs=True, skip_always_false=True
        )
        min_support = max(1, int(self.min_support_frac * len(self.df)))
        kept, eqs = simplify_and_dedup_hypotheses(self.df, hyps, min_support=min_support)
        res.kept_hypotheses = kept
        res.saved_equivalence_conjs = eqs

        type_one: List[Conjecture] = []
        for other in self.numeric_cols:
            for H in kept:
                type_one.extend(ratios(self.df, features=[other], target=self.target, hypothesis=H))
        res.type_one = type_one

        base = Conjecture(relation=None, condition=None)._auto_base(self.df)
        const_cache = precompute_constant_ratios_pairs(
            self.df,
            base=base,
            numeric_cols=self.numeric_cols,
            shifts=self.const_cache_shifts,
            atol=self.atol, rtol=self.atol,
            min_support=max(8, int(self.min_support_frac * len(self.df))),
            max_denominator=self.max_denominator,
            skip_identity=True,
        )

        res.generalized_from_constants = self._generalize_from_constants(type_one, kept, const_cache)
        res.joint_generals = self._joint_generalize(type_one, kept, const_cache)

        merged_before_refine = res.type_one + res.generalized_from_constants + res.joint_generals
        res.refined = self._maybe_refine_numeric(merged_before_refine)

        merged_all = merged_before_refine + res.refined
        hazel_res = hazel_rank(self.df, merged_all, drop_frac=self.hazel_drop_frac, atol=self.atol)
        res.hazel_kept_sorted = list(hazel_res.kept_sorted)
        morgan_res = morgan_filter(self.df, res.hazel_kept_sorted)
        res.morgan_kept = list(morgan_res.kept)
        res.dalmatian_kept = dalmatian_filter(res.morgan_kept, self.df)

        # ===== choose seeds for tightness miner (single, clean place) =====
        if seed_mode == "given" and seed_conjectures is not None:
            seeds = seed_conjectures
        elif seed_mode == "default":
            seeds = self._default_seed_conjectures()
        else:
            # seed_mode == "system"
            if tightness_source == "final":
                seeds = res.dalmatian_kept or res.type_one
            elif tightness_source == "type_one":
                seeds = res.type_one
            else:  # "all"
                seeds = res.type_one + res.generalized_from_constants + res.joint_generals + res.refined

        # keep only inequalities/equalities (as the demo expects)
        seeds = [c for c in seeds if isinstance(c.relation, (Le, Ge, Eq))]

        # ===== promote and derive tightness predicates (demo logic) =====
        promoted = [self._maybe_promote_to_eq(c, atol=self.atol) for c in seeds]
        res.promoted_seeds = promoted

        tight_preds: List[Predicate] = []
        for c in promoted:
            tight_preds.extend(self._tightness_preds(c, atol=self.atol))
        # keep only those with nonempty support
        res.tightness_preds = [p for p in tight_preds if self._mask(p).any()]

        # ===== rebuild candidate pool like the demo =====
        kept_signed, atomic_pool = self._signed_pool(res.kept_hypotheses)
        candidates = self._build_candidate_hypotheses_with_disjunctions(
            kept_signed, include_or=True, min_support=1
        )

        implies_map = self._learn_atomic_implications(atomic_pool)

        # ===== mine implications H_tight ⇒ C (and equivalences) =====
        mined = self._mine_implications(
            tight_preds=res.tightness_preds, candidates=candidates, min_support=1
        )
        mined = self._dedupe_implications(mined)
        mined = self._drop_trivial_consequents(mined)
        mined = self._drop_h_implies_h_or_x(mined)
        mined = self._keep_most_general_per_consequent(mined)
        ranked = self._hazel_like_rank_implications(mined, drop_frac=0.25, min_support=None)
        res.mined_implications_ranked = ranked

        return res

    def print_summary(self, res: ConjectureTypesResult) -> None:
        # A) hypotheses
        print("=== kept (base-aware, simplified) ===")
        for h in res.kept_hypotheses:
            print(" ", self._pretty_pred(h))

        print("\n=== saved equivalence conjectures ===")
        for c in res.saved_equivalence_conjs:
            print(" ", self._format_equivalence_any(c))

        # B–D
        print(f"\nGenerated {len(res.type_one)} raw type-one conjectures.")
        print(f"Proposed {len(res.generalized_from_constants)} generalized conjectures from constants.")
        print(f"Proposed {len(res.joint_generals)} joint generalizations (slope/intercept/reciprocals).")
        if self.use_refiner:
            print(f"Refined {len(res.refined)} conjectures numerically.")
        print(f"Hazel → Morgan kept {len(res.morgan_kept)} / {len(res.type_one) + len(res.generalized_from_constants) + len(res.joint_generals) + len(res.refined)}")

        print("\n=== Final (Hazel → Morgan → Dalmatian) Conjectures ===")
        for c in res.dalmatian_kept:
            print(" -", c.pretty(arrow='⇒'))

        # Tightness mining
        print("\n=== Seed conjectures after promotion (used for tightness miner) ===")
        for c in res.promoted_seeds:
            extra = " (PROMOTED)" if isinstance(c.relation, Eq) else ""
            print(" -", c.pretty(arrow="⇒"), extra)

        print("\n=== Derived tightness predicates and supports ===")
        for p in res.tightness_preds:
            supp = int(self._mask(p).sum())
            print(f" • {self._pretty_pred(p)} support={supp}")

        print("\n=== Implications mined from tightness classes (Hazel-like filtered) ===")
        if not res.mined_implications_ranked:
            print("(none)")
        else:
            implies_map = self._learn_atomic_implications(self._atomic_from(res.kept_hypotheses))
            for r in res.mined_implications_ranked:
                arrow = "≡" if r.get("equiv") else "⇒"
                Hs = self._pretty_factored(r['H'])
                Cs_clean = self._pretty_rhs_conditioned_on_H(r['H'], r['C'], implies_map)
                print(f"{Hs} {arrow} {Cs_clean}   [support(H)={r['support_H']}, support(C)={r['support_C']}]")

    # -------------------------
    # Internals — (A–E) pipeline
    # -------------------------
    def _numeric_columns_excluding_target(self) -> List[str]:
        cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        return [c for c in cols if c != self.target]

    def _is_subset(self, A: Optional[Predicate], B: Optional[Predicate]) -> bool:
        a = (self._mask(A) if A is not None else pd.Series(True, index=self.df.index)).astype(bool)
        b = (self._mask(B) if B is not None else pd.Series(True, index=self.df.index)).astype(bool)
        return not (a & ~b).any()

    def _generalize_from_constants(self, type_one: List[Conjecture], kept: List[Predicate], const_cache) -> List[Conjecture]:
        out = []
        for conj in type_one:
            supersets = [H for H in kept if self._is_subset(conj.condition, H)]
            gens = propose_generalizations_from_constants(
                self.df, conj, const_cache, candidate_hypotheses=supersets
            )
            out.extend([g.new_conjecture for g in gens])
        return out

    def _joint_generalize(self, type_one: List[Conjecture], kept: List[Predicate], const_cache) -> List[Conjecture]:
        out = []
        for conj in type_one:
            supersets = [H for H in kept if self._is_subset(conj.condition, H)]
            out.extend(
                propose_joint_generalizations(
                    self.df,
                    conj,
                    cache=const_cache,
                    candidate_hypotheses=supersets,
                    candidate_intercepts=None,
                    relaxers_Z=None,
                    require_superset=True,
                    atol=self.atol,
                )
            )
        return out

    def _maybe_refine_numeric(self, conjs: List[Conjecture]) -> List[Conjecture]:
        if not self.use_refiner:
            return []
        cfg = RefinementConfig(
            try_whole_rhs_floor=True,
            try_whole_rhs_ceil=False,
            try_prod_floor=False,
            try_coeff_round_const=True,
            try_intercept_round_const=True,
            try_sqrt_intercept=True,
            try_log_intercept=False,
            require_tighter=True,
        )
        out = []
        for conj in conjs:
            out.extend(refine_numeric_bounds(self.df, conj, config=cfg))
        return out

    # -------------------------
    # Internals — Tightness & implication miner (self-contained helpers)
    # -------------------------
    @staticmethod
    def _flatten_and_pred(p):
        if p is None: return []
        if isinstance(p, AndPred):
            return ConjectureTypesRunner._flatten_and_pred(p.a) + ConjectureTypesRunner._flatten_and_pred(p.b)
        return [p]

    @staticmethod
    def _is_true_pred(p) -> bool:
        return getattr(p, "name", "") == "TRUE"

    @staticmethod
    def _normalize_and(pred):
        if pred is None or ConjectureTypesRunner._is_true_pred(pred): return None
        parts = ConjectureTypesRunner._flatten_and_pred(pred)
        seen, uniq = set(), []
        for q in parts:
            k = repr(q)
            if k not in seen:
                seen.add(k); uniq.append(q)
        if not uniq: return None
        out = uniq[0]
        for q in uniq[1:]:
            out = AndPred(out, q)
        return out

    def _mask(self, pred: Predicate | None) -> pd.Series:
        if pred is None: return pd.Series(True, index=self.df.index)
        m = pred.mask(self.df)
        return m.reindex(self.df.index, fill_value=False).astype(bool)

    def _pretty_pred(self, pred: Predicate | None) -> str:
        return format_pred(self._normalize_and(pred), unicode_ops=True)

    @staticmethod
    def _pretty_factored(pred: Predicate | None) -> str:
        return format_pred(ConjectureTypesRunner._normalize_and(pred), unicode_ops=True)

    # ---- Promotion & tightness predicates ----
    def _maybe_promote_to_eq(self, conj: Conjecture, *, atol=1e-9) -> Conjecture:
        rel = conj.relation
        if isinstance(rel, (Le, Ge)):
            applicable = self._mask(conj.condition)
            if not applicable.any(): return conj
            holds = rel.evaluate(self.df).reindex(self.df.index, fill_value=False).astype(bool)
            slack = rel.slack(self.df)
            tight = pd.Series(np.isclose(slack, 0.0, atol=atol), index=self.df.index)
            if bool(holds[applicable].all()) and bool(tight[applicable].all()):
                new_rel = Eq(rel.left, rel.right, tol=atol)
                return Conjecture(new_rel, conj.condition, name=getattr(conj, "name", "conj"))
        return conj

    def _tightness_preds(self, conj: Conjecture, *, atol=1e-9) -> List[Predicate]:
        rel = conj.relation
        cond = conj.condition

        def _eq_mask(d, L, R, tol):
            Ls = pd.to_numeric(L.eval(d), errors="coerce")
            Rs = pd.to_numeric(R.eval(d), errors="coerce")
            return pd.Series(np.isclose(Ls - Rs, 0.0, atol=float(tol)), index=d.index)

        from txgraffiti2025.forms.predicates import LT, GT  # local import to avoid circulars
        if isinstance(rel, Le):
            P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, atol),
                               name=f"(({rel.left}) = ({rel.right}))")
            P_eq  = AndPred(cond, P_eq_inner) if cond else P_eq_inner
            P_str = AndPred(cond, LT(rel.left, rel.right)) if cond else LT(rel.left, rel.right)
            return [self._normalize_and(P_eq), self._normalize_and(P_str)]
        if isinstance(rel, Ge):
            P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, atol),
                               name=f"(({rel.left}) = ({rel.right}))")
            P_eq  = AndPred(cond, P_eq_inner) if cond else P_eq_inner
            P_str = AndPred(cond, GT(rel.left, rel.right)) if cond else GT(rel.left, rel.right)
            return [self._normalize_and(P_eq), self._normalize_and(P_str)]
        if isinstance(rel, Eq):
            P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, rel.tol),
                               name=f"(({rel.left}) = ({rel.right}))")
            P_eq = AndPred(cond, P_eq_inner) if cond else P_eq_inner
            return [self._normalize_and(P_eq)]
        return []

    # ---- Candidate hypothesis builder with ORs (uses AND of predicates + Where OR) ----
    @staticmethod
    def _and_all(preds: Sequence[Predicate]) -> Predicate:
        it = iter(preds)
        acc = next(it)
        for p in it:
            acc = AndPred(acc, p)
        parts = [format_pred(p, unicode_ops=True) for p in preds]
        acc.name = "(" + ") ∧ (".join(parts) + ")"
        return acc

    @staticmethod
    def _or_pred(a: Predicate, b: Predicate) -> Predicate:
        name = f"({format_pred(a, unicode_ops=True)}) ∨ ({format_pred(b, unicode_ops=True)})"
        return Where(lambda d, A=a, B=b: A.mask(d) | B.mask(d), name=name)

    def _build_candidate_hypotheses_with_disjunctions(
        self,
        base_hyps: Sequence[Predicate],
        *,
        include_or: bool = True,
        min_support: int = 1,
    ) -> List[Predicate]:
        cands = list(base_hyps)
        base = list(base_hyps)

        from itertools import combinations
        # Pairwise ANDs
        for A, B in combinations(base, 2):
            try:
                cand = self._and_all([A, B])
                m = self._mask(cand)
                if int(m.sum()) >= min_support:
                    cands.append(cand)
            except Exception:
                pass

        # Pairwise ORs (non-trivial only)
        if include_or:
            seen_keys = set()
            for A, B in combinations(base, 2):
                try:
                    mA = self._mask(A); mB = self._mask(B)
                    if not (mA & ~mB).any() or not (mB & ~mA).any():
                        continue
                    OR = self._or_pred(A, B)
                    mOR = self._mask(OR)
                    if int(mOR.sum()) < min_support:
                        continue
                    key = tuple(sorted([getattr(A, "name", repr(A)), getattr(B, "name", repr(B))]))
                    if key in seen_keys: continue
                    seen_keys.add(key)
                    cands.append(OR)
                except Exception:
                    pass

        # Deduplicate by name
        uniq, seen = [], set()
        for p in cands:
            k = getattr(p, "name", repr(p))
            if k not in seen:
                seen.add(k); uniq.append(p)
        return uniq

    # ---- Signed pool (positive + atomic negations that are nonempty) ----
    def _signed_pool(self, kept: List[Predicate]) -> Tuple[List[Predicate], List[Predicate]]:
        signed_kept: List[Predicate] = []
        atomic_pool: List[Predicate] = []
        for p in kept:
            m = self._mask(p)
            if m.any():
                signed_kept.append(p)
                if self._is_atomic_pred(p):
                    atomic_pool.append(p)
            if self._is_atomic_pred(p):
                pn = self._negate(p)
                mn = self._mask(pn)
                if mn.any():
                    signed_kept.append(pn)
        return signed_kept, atomic_pool

    # ---- Atomic check / Not-predicate ----
    class _NotPred(Predicate):
        def __init__(self, p: Predicate, name: str | None = None):
            self.p = p
            self.name = name or f"¬({format_pred(p, unicode_ops=True)})"
        def mask(self, df: pd.DataFrame) -> pd.Series:
            return (~self.p.mask(df)).astype(bool)
        def __repr__(self):
            return f"NotPred({repr(self.p)})"

    @staticmethod
    def _is_atomic_pred(p: Predicate) -> bool:
        return not isinstance(p, AndPred)

    def _negate(self, p: Predicate) -> Predicate:
        if isinstance(p, ConjectureTypesRunner._NotPred):
            return p.p
        return ConjectureTypesRunner._NotPred(p)

    # ---- Learn implications among atomic positives ----
    def _learn_atomic_implications(self, atoms: List[Predicate]) -> dict[str, set[str]]:
        uniq: List[Predicate] = []
        names: List[str] = []
        masks: List[pd.Series] = []
        seen = set()
        for p in atoms:
            if not self._is_atomic_pred(p): continue
            key = repr(self._normalize_and(p))
            if key in seen: continue
            seen.add(key)
            m = self._mask(p)
            if not m.any():  # skip empty
                continue
            uniq.append(p)
            names.append(format_pred(p, unicode_ops=True))
            masks.append(m.astype(bool))

        n = len(uniq)
        implies: dict[str, set[str]] = {names[i]: set() for i in range(n)}
        for i in range(n):
            mi = masks[i]
            for j in range(n):
                if i == j: continue
                mj = masks[j]
                if not (mi & ~mj).any():
                    implies[names[i]].add(names[j])
        return implies

    # ---- Mining implications H_tight ⇒ C from candidate pool ----
    def _mine_implications(self, tight_preds: List[Predicate], candidates: List[Predicate], *, min_support: int = 1) -> List[dict]:
        out = []
        for H in tight_preds:
            Hn = self._normalize_and(H); Hm = self._mask(Hn)
            supH = int(Hm.sum())
            if supH < min_support: continue
            for C in candidates:
                Cn = self._normalize_and(C); Cm = self._mask(Cn)
                if not Cm.any(): continue
                implies = not (Hm & ~Cm).any()  # H ⊆ C
                if implies:
                    equiv = not (Cm & ~Hm).any()  # C ⊆ H
                    out.append({
                        "H": Hn, "C": Cn,
                        "support_H": supH,
                        "support_C": int(Cm.sum()),
                        "equiv": bool(equiv),
                    })
        return out

    # ---- Cleanup + rank mined implications ----
    def _cleanup_and_rank_mined(self, mined: List[dict], implies_map: dict[str, set[str]]) -> List[dict]:
        mined = self._dedupe_implications(mined)
        mined = self._drop_trivial_consequents(mined)
        mined = self._drop_h_implies_h_or_x(mined)
        mined = self._keep_most_general_per_consequent(mined)
        return self._hazel_like_rank_implications(mined, drop_frac=0.25, min_support=None)

    def _pred_key(self, pred) -> tuple:
        pn = self._normalize_and(pred)
        key_repr = repr(pn)
        m = self._mask(pn)
        supp = int(m.sum())
        nz = tuple(map(int, np.flatnonzero(m.values[:min(len(m), 64)])))
        return (key_repr, supp, nz)

    def _dedupe_implications(self, mined: List[dict]) -> List[dict]:
        seen, out = set(), []
        for r in mined:
            Hk = self._pred_key(r["H"]); Ck = self._pred_key(r["C"])
            key = (Hk, Ck, bool(r.get("equiv", False)))
            if key in seen: continue
            seen.add(key); out.append(r)
        return out

    def _drop_trivial_consequents(self, mined: List[dict]) -> List[dict]:
        out = []
        for r in mined:
            Hn = self._normalize_and(r["H"]); Cn = self._normalize_and(r["C"])
            if Hn is None or Cn is None: continue
            Cm = self._mask(Cn)
            trivial = False
            for part in self._flatten_and_pred(Hn):
                pm = self._mask(part)
                if not (pm & ~Cm).any():
                    trivial = True; break
            if not trivial: out.append(r)
        return out

    def _drop_h_implies_h_or_x(self, mined: List[dict]) -> List[dict]:
        out = []
        for r in mined:
            Hm = self._mask(self._normalize_and(r["H"]))
            Cm = self._mask(self._normalize_and(r["C"]))
            if (Cm & ~Hm).any():  # keep only if C strictly extends H
                out.append(r)
        return out

    def _keep_most_general_per_consequent(self, mined: List[dict]) -> List[dict]:
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in mined:
            Ck = self._pred_key(r["C"])
            buckets[Ck].append(r)
        kept_all = []
        for _, items in buckets.items():
            masks = []
            for r in items:
                Hn = self._normalize_and(r["H"])
                masks.append((self._mask(Hn), r))
            n = len(masks); dominated = [False]*n
            for i in range(n):
                if dominated[i]: continue
                mi, ri = masks[i]
                for j in range(n):
                    if i == j or dominated[i]: continue
                    mj, rj = masks[j]
                    if not (mi & ~mj).any():
                        si, sj = int(mi.sum()), int(mj.sum())
                        if si < sj: dominated[i] = True
                        elif si == sj:
                            hi = self._pretty_factored(ri["H"]); hj = self._pretty_factored(rj["H"])
                            if hi > hj: dominated[i] = True
            for k, d in enumerate(dominated):
                if not d: kept_all.append(masks[k][1])
        return kept_all

    @staticmethod
    def _hazel_like_rank_implications(mined: List[dict], *, drop_frac: float = 0.25, min_support: int | None = None) -> List[dict]:
        if not mined: return []
        items = [r for r in mined if (min_support is None or r["support_H"] >= min_support)]
        if not items: return []
        items_sorted = sorted(items, key=lambda r: r["support_H"])
        k = max(0, int(len(items_sorted) * drop_frac))
        kept = items_sorted[k:]
        kept.sort(
            key=lambda r: (
                not r["equiv"],
                -r["support_H"],
                r["support_C"],
                format_pred(ConjectureTypesRunner._normalize_and(r["H"]), unicode_ops=True),
                format_pred(ConjectureTypesRunner._normalize_and(r["C"]), unicode_ops=True),
            )
        )
        return kept

    # ---- Pretty RHS conditioned on H (subtract H and positives implied by H) ----
    def _atomic_from(self, preds: List[Predicate]) -> List[Predicate]:
        return [p for p in preds if self._is_atomic_pred(p)]

    @staticmethod
    def _is_neg_token(s: str) -> bool:
        s = s.strip()
        return s.startswith("¬(") and s.endswith(")")

    @staticmethod
    def _strip_outer_parens(s: str) -> str:
        s = s.strip()
        while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
            depth = 0; ok = True
            for i, ch in enumerate(s):
                if ch == "(": depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and i != len(s) - 1:
                        ok = False; break
            if ok: s = s[1:-1].strip()
            else: break
        return s

    @staticmethod
    def _split_top_level(s: str, sep: str) -> list[str]:
        parts, depth, last, i = [], 0, 0, 0
        while i < len(s):
            ch = s[i]
            if ch == "(": depth += 1
            elif ch == ")": depth -= 1
            elif ch == sep and depth == 0:
                parts.append(s[last:i].strip()); last = i + 1
            i += 1
        parts.append(s[last:].strip())
        return parts

    @classmethod
    def _and_atoms(cls, s: str) -> list[str]:
        s = cls._strip_outer_parens(s)
        if not s: return []
        return [cls._strip_outer_parens(p.strip())
                for p in (cls._split_top_level(s, "∧") if "∧" in s else [s])
                if p.strip()]

    @classmethod
    def _or_disjuncts(cls, s: str) -> list[str]:
        s = cls._strip_outer_parens(s)
        if "∨" not in s:
            return [s] if s else []
        return [cls._strip_outer_parens(t.strip())
                for t in cls._split_top_level(s, "∨") if t.strip()]

    @staticmethod
    def _prune_and_atoms_with_implications(and_atoms: list[str], implies_map: dict[str, set[str]]) -> list[str]:
        A = list(dict.fromkeys(and_atoms))  # stable unique
        keep = [True] * len(A)
        for i, ai in enumerate(A):
            if not keep[i]: continue
            if ConjectureTypesRunner._is_neg_token(ai): continue
            ai0 = ConjectureTypesRunner._strip_outer_parens(ai)
            implied_set = implies_map.get(ai0, set())
            if not implied_set: continue
            for j, aj in enumerate(A):
                if i == j or not keep[j]: continue
                if ConjectureTypesRunner._is_neg_token(aj): continue
                aj0 = ConjectureTypesRunner._strip_outer_parens(aj)
                if aj0 in implied_set:
                    keep[j] = False
        return [A[k] for k in range(len(A)) if keep[k]]

    def _implied_by_set(self, H_pos: set[str], implies_map: dict[str, set[str]]) -> set[str]:
        out = set()
        for a in H_pos:
            out |= implies_map.get(a, set())
        return out

    def _pretty_rhs_conditioned_on_H(self, H: Predicate | None, C: Predicate | None,
                                     implies_map: dict[str, set[str]]) -> str:
        Hs = self._pretty_factored(H); Cs = self._pretty_factored(C)
        H_atoms_all = set(self._and_atoms(Hs))
        H_pos = {a for a in H_atoms_all if not self._is_neg_token(a)}
        implied_by_H = self._implied_by_set(H_pos, implies_map)

        disjuncts = self._or_disjuncts(Cs)
        residual = []
        for Dj in disjuncts:
            Dj_atoms = self._and_atoms(Dj)
            kept = []
            for a in Dj_atoms:
                a0 = self._strip_outer_parens(a)
                if a0 in H_atoms_all:
                    continue
                if (not self._is_neg_token(a0)) and (a0 in implied_by_H):
                    continue
                kept.append(a0)
            kept = self._prune_and_atoms_with_implications(kept, implies_map)
            if kept:
                residual.append("(" + ") ∧ (".join(sorted(set(kept))) + ")")
        if not residual: return "(⊤)"
        if len(residual) == 1: return residual[0]
        return " ∨ ".join(residual)

    # -------------------------
    # Equivalences pretty-printer (robust)
    # -------------------------
    def _format_equivalence_any(self, obj: Any) -> str:
        """
        Pretty-print any 'ClassEquivalence'-like record returned by
        simplify_and_dedup_hypotheses(...).
        """
        # 1) direct pretty
        if hasattr(obj, "pretty") and callable(getattr(obj, "pretty")):
            try:
                return obj.pretty(arrow="≡")
            except Exception:
                pass
        # 2) convert to conjecture if possible
        if hasattr(obj, "to_conjecture") and callable(getattr(obj, "to_conjecture")):
            try:
                cj = obj.to_conjecture()
                if hasattr(cj, "pretty"):
                    return cj.pretty(arrow="≡")
            except Exception:
                pass
        # 3) lhs/rhs style
        for left_name, right_name in (("lhs", "rhs"), ("A", "B"), ("left", "right")):
            L = getattr(obj, left_name, None)
            R = getattr(obj, right_name, None)
            if isinstance(L, Predicate) and isinstance(R, Predicate):
                try:
                    Ls = format_pred(L, unicode_ops=True)
                    Rs = format_pred(R, unicode_ops=True)
                    return f"{Ls} ≡ {Rs}"
                except Exception:
                    pass
        # 4) Conjecture instance
        try:
            from txgraffiti2025.forms.generic_conjecture import Conjecture as _C
            if isinstance(obj, _C):
                return obj.pretty(arrow="≡")
        except Exception:
            pass
        # 5) last resort
        return str(obj)

    # -------------------------
    # Default seeds (optional demo)
    # -------------------------
    def _default_seed_conjectures(self) -> List[Conjecture]:
        def bool_col_pred(col: str) -> Predicate:
            return Where(lambda d, c=col: d[c].astype(bool), name=f"({col})")
        def mk_le(hyp: Predicate | None, left: str, right: str, name="") -> Conjecture:
            return Conjecture(Le(to_expr(left), to_expr(right)), hyp, name=name or f"{left}_le_{right}")
        def mk_ge(hyp: Predicate | None, left: str, right: str, name="") -> Conjecture:
            return Conjecture(Ge(to_expr(left), to_expr(right)), hyp, name=name or f"{left}_ge_{right}")

        H_conn         = bool_col_pred("connected")
        H_trianglefree = bool_col_pred("triangle_free")
        H_clawfree     = bool_col_pred("claw_free")

        return [
            mk_le(H_conn, "independence_number", "annihilation_number",
                  name="alpha_le_a(G)"),
            mk_ge(H_conn, "independence_number", "independent_domination_number",
                  name="alpha_ge_i(G)"),
            mk_ge(AndPred(H_conn, H_trianglefree), "independence_number", "maximum_degree",
                  name="alpha_ge_D(G) on triangle_free"),
            mk_ge(AndPred(H_conn, H_clawfree), "independent_domination_number", "domination_number",
                  name="i(G)_ge_gamma(G) on claw_free"),
        ]

    # -------------------------
    # Exporter
    # -------------------------
    def write_conjectures_to_file(
        self,
        res: ConjectureTypesResult,
        filename: str = "conjectures_output.txt",
        *,
        include_equivalences: bool = True,
        include_generalizations: bool = True,
        include_refinements: bool = True,
        include_dalmatian: bool = True,
        include_tightness: bool = True,       # NEW
        include_implications: bool = True,    # NEW
        include_candidates: bool = False,     # NEW (can be large)
    ) -> None:
        """
        Write conjectures and optional metadata to a .txt file in a clean, human-readable format.
        Now includes the 'further hypotheses search' results: tightness miner + mined implications.
        """
        lines: list[str] = []
        add = lines.append

        add("========================================")
        add("TxGraffiti2025 Conjecture Export")
        add("========================================\n")

        # (A) Equivalence conjectures
        if include_equivalences and res.saved_equivalence_conjs:
            add("=== Saved Equivalence Conjectures ===")
            for c in res.saved_equivalence_conjs:
                add(f"- {self._format_equivalence_any(c)}")
            add("")

        # (B) Type-one
        add("=== Type-One (Ratio) Conjectures ===")
        for c in res.type_one:
            add(f"- {c.pretty(arrow='⇒')}")
        add("")

        # (C) Generalized
        if include_generalizations:
            add("=== Generalized from Constants ===")
            for c in res.generalized_from_constants:
                add(f"- {c.pretty(arrow='⇒')}")
            add("")

            add("=== Joint Generalizations (Slope/Intercept/Reciprocals) ===")
            for c in res.joint_generals:
                add(f"- {c.pretty(arrow='⇒')}")
            add("")

        # (D) Numeric refinement
        if include_refinements and res.refined:
            add("=== Refined Conjectures ===")
            for c in res.refined:
                add(f"- {c.pretty(arrow='⇒')}")
            add("")

        # (E) Hazel → Morgan → Dalmatian (final)
        if include_dalmatian and res.dalmatian_kept:
            add("=== Final (Hazel → Morgan → Dalmatian) Conjectures ===")
            for c in res.dalmatian_kept:
                add(f"- {c.pretty(arrow='⇒')}")
            add("")

        # (F) Further hypotheses search: tightness miner (PROMOTED seeds + tightness preds)
        if include_tightness:
            if res.promoted_seeds:
                add("=== Tightness Miner Seeds (after promotion when tight everywhere) ===")
                for c in res.promoted_seeds:
                    tag = " (PROMOTED)" if isinstance(c.relation, Eq) else ""
                    add(f"- {c.pretty(arrow='⇒')}{tag}")
                add("")
            if res.tightness_preds:
                add("=== Derived Tightness Predicates (with supports) ===")
                for p in res.tightness_preds:
                    supp = int(self._mask(p).sum())
                    add(f"- {self._pretty_pred(p)}   [support={supp}]")
                add("")

        # (G) Further hypotheses search: mined implications from tightness classes
        if include_implications:
            if res.mined_implications_ranked:
                # Rebuild implies_map for nice RHS presentation C \ H
                implies_map = self._learn_atomic_implications(self._atomic_from(res.kept_hypotheses))
                add("=== Mined Implications from Tightness Classes (Hazel-like filtered) ===")
                for r in res.mined_implications_ranked:
                    arrow = "≡" if r.get("equiv") else "⇒"
                    Hs = self._pretty_factored(r["H"])
                    Cs_clean = self._pretty_rhs_conditioned_on_H(r["H"], r["C"], implies_map)
                    add(f"- {Hs} {arrow} {Cs_clean}   [support(H)={r['support_H']}, support(C)={r['support_C']}]")
                add("")
            else:
                add("=== Mined Implications from Tightness Classes (Hazel-like filtered) ===")
                add("(none)\n")

        # (H) Optional: list the AND/OR candidate hypotheses considered in mining
        if include_candidates:
            kept_signed, _atomic = self._signed_pool(res.kept_hypotheses)
            candidates = self._build_candidate_hypotheses_with_disjunctions(
                kept_signed, include_or=True, min_support=1
            )
            if candidates:
                add("=== Candidate Hypotheses Used in Mining (signed AND/OR) ===")
                for p in candidates:
                    add(f"- {self._pretty_pred(p)}")
                add("")

        # Write to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✅ Conjectures (incl. tightness miner & implications) written to '{filename}' "
            f"({len(lines)} lines total).")


        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✅ Conjectures written to '{filename}' ({len(lines)} lines total).")




from txgraffiti.example_data import graph_data as df
# from txgraffiti2025.systems.conjecture_types_runner import ConjectureTypesRunner

runner = ConjectureTypesRunner(df, target="harmonic_index", use_refiner=True)
# Use system-produced conjectures as seeds (default behavior)
res = runner.run_all(seed_mode="system")
# runner.print_summary(res)

runner.write_conjectures_to_file(
    res,
    filename="harmonic_index_conjectures.txt",
    include_equivalences=True,
    include_generalizations=True,
    include_refinements=True,
    include_dalmatian=True,
)
