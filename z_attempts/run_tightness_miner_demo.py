# run_tightness_miner_demo.py

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from itertools import combinations

# --- data ---
from txgraffiti.example_data import graph_data as df

# --- core types / utilities ---
from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Le, Ge
from txgraffiti2025.forms.predicates import Predicate, Where, AndPred, LT, GT
from txgraffiti2025.forms.pretty import format_pred
from txgraffiti2025.forms.utils import to_expr

# --- boolean hyp enumeration (for candidates) ---
from txgraffiti2025.processing.pre.hypotheses import enumerate_boolean_hypotheses
from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses

# =========================================================
# Negation support (atomic only)
# =========================================================

class NotPred(Predicate):
    def __init__(self, p: Predicate, name: str | None = None):
        self.p = p
        self.name = name or f"¬({format_pred(p, unicode_ops=True)})"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return (~_mask(df, self.p)).astype(bool)
    def __repr__(self):
        return f"NotPred({repr(self.p)})"

def negate(p: Predicate) -> Predicate:
    if isinstance(p, NotPred):
        return p.p
    return NotPred(p)

def _is_atomic_pred(p: Predicate) -> bool:
    # Atomic = not an And chain (you can extend if you add OrPred later)
    return not isinstance(p, AndPred)

# =========================================================
# Pretty factoring utilities (presentation only) — robust
# =========================================================

def _strip_all_outer_parens(s: str) -> str:
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

def _tokenize_or_of_ands_robust(s: str):
    s = _strip_all_outer_parens(s)
    if "∨" not in s: return None
    disjunct_strs = _split_top_level(s, "∨")
    if len(disjunct_strs) < 2: return None
    disjuncts = []
    for ds in disjunct_strs:
        ds = _strip_all_outer_parens(ds)
        atoms = _split_top_level(ds, "∧") if "∧" in ds else [ds]
        cleaned = []
        for a in atoms:
            a = _strip_all_outer_parens(a.strip())
            if a: cleaned.append(a)
        if not cleaned: return None
        disjuncts.append(set(cleaned))
    return disjuncts

def _rebuild_and(atoms):
    atoms = sorted(set(atoms))
    if not atoms: return ""
    # Redundancy pruning is injected later once we learn implications from data
    return "(" + ") ∧ (".join(atoms) + ")"

def _rebuild_or(conjs):
    return " ∨ ".join("(" + _rebuild_and(c) + ")" for c in conjs)

def factor_or_string(pretty: str) -> str:
    disj = _tokenize_or_of_ands_robust(pretty)
    if not disj: return pretty
    common = set.intersection(*disj) if disj else set()
    if not common: return pretty
    residuals, has_empty = [], False
    for s in disj:
        r = sorted(list(s - common))
        if len(r) == 0: has_empty = True
        residuals.append(r)
    common_str = _rebuild_and(common)
    if has_empty: return common_str
    inner_conjs = [set(r) for r in residuals]
    inner = _rebuild_or(inner_conjs)
    return f"{common_str} ∧ ({inner})"

def _pretty_factored(pred: Predicate | None) -> str:
    s = format_pred(pred, unicode_ops=True)
    return factor_or_string(s)

# =========================================================
# Helpers: normalization, masks, pretty
# =========================================================

def _flatten_and_pred(p):
    if p is None: return []
    if isinstance(p, AndPred):
        return _flatten_and_pred(p.a) + _flatten_and_pred(p.b)
    return [p]

def _is_true_pred(p) -> bool:
    return getattr(p, "name", "") == "TRUE"

def normalize_and(pred):
    if pred is None or _is_true_pred(pred): return None
    parts = _flatten_and_pred(pred)
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

def pretty_pred(pred):
    return format_pred(normalize_and(pred), unicode_ops=True)

def _mask(df: pd.DataFrame, pred: Predicate | None) -> pd.Series:
    if pred is None: return pd.Series(True, index=df.index)
    m = pred.mask(df)
    return m.reindex(df.index, fill_value=False).astype(bool)

def _name(pred: Predicate | None) -> str:
    return pretty_pred(pred)

def _pred_key(df: pd.DataFrame, pred) -> tuple:
    pn = normalize_and(pred)
    key_repr = repr(pn)
    m = _mask(df, pn)
    supp = int(m.sum())
    nz = tuple(map(int, np.flatnonzero(m.values[:min(len(m), 64)])))
    return (key_repr, supp, nz)

# =========================================================
# Tightness predicates
# =========================================================

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

# =========================================================
# Promote inequalities to equalities when tight everywhere
# =========================================================

def _maybe_promote_to_eq(df: pd.DataFrame, conj: Conjecture, *, atol=1e-9) -> Conjecture:
    rel = conj.relation
    if isinstance(rel, (Le, Ge)):
        applicable = _mask(df, conj.condition)
        if not applicable.any(): return conj
        holds = rel.evaluate(df).reindex(df.index, fill_value=False).astype(bool)
        slack = rel.slack(df)
        tight = pd.Series(np.isclose(slack, 0.0, atol=atol), index=df.index)
        if bool(holds[applicable].all()) and bool(tight[applicable].all()):
            new_rel = Eq(rel.left, rel.right, tol=atol)
            return Conjecture(new_rel, conj.condition, name=getattr(conj, "name", "conj"))
    return conj

# =========================================================
# Candidate hypothesis builder (ANDs + optional ORs)
# =========================================================

def _contradictory(parts: list[Predicate]) -> bool:
    reps = {repr(normalize_and(p)) for p in parts}
    for r in list(reps):
        if r and r.startswith("NotPred("):
            core = r[len("NotPred("):-1]
            if core in reps: return True
    return False

def _and_all(preds):
    if _contradictory(list(preds)):
        raise ValueError("contradictory conjunction P ∧ ¬P")
    it = iter(preds)
    acc = next(it)
    for p in it:
        acc = AndPred(acc, p)
    parts = [format_pred(p, unicode_ops=True) for p in preds]
    acc.name = "(" + ") ∧ (".join(parts) + ")"
    return acc

def _or_pred(a: Predicate, b: Predicate) -> Predicate:
    name = f"({format_pred(a, unicode_ops=True)}) ∨ ({format_pred(b, unicode_ops=True)})"
    return Where(lambda d, A=a, B=b: _mask(d, A) | _mask(d, B), name=name)

def build_candidate_hypotheses_with_disjunctions(
    df: pd.DataFrame,
    base_hyps,           # simplified boolean hyps (possibly signed)
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
    # Pairwise ORs (non-trivial only)
    if include_or:
        seen_keys = set()
        for A, B in combinations(base, 2):
            try:
                mA = _mask(df, A); mB = _mask(df, B)
                if not (mA & ~mB).any() or not (mB & ~mA).any():
                    continue
                OR = _or_pred(A, B)
                mOR = _mask(df, OR)
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

# =========================================================
# Mine implications
# =========================================================

def mine_implications(df, tight_preds, candidate_preds, *, min_support=1):
    out = []
    for H in tight_preds:
        Hn = normalize_and(H); Hm = _mask(df, Hn)
        supH = int(Hm.sum())
        if supH < min_support: continue
        for C in candidate_preds:
            Cn = normalize_and(C); Cm = _mask(df, Cn)
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

# =========================================================
# Post-mining cleanup & ranking
# =========================================================

def dedupe_implications(df: pd.DataFrame, mined: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in mined:
        Hk = _pred_key(df, r["H"]); Ck = _pred_key(df, r["C"])
        key = (Hk, Ck, bool(r.get("equiv", False)))
        if key in seen: continue
        seen.add(key); out.append(r)
    return out

def drop_trivial_consequents(df: pd.DataFrame, mined: list[dict]) -> list[dict]:
    out = []
    for r in mined:
        Hn = normalize_and(r["H"]); Cn = normalize_and(r["C"])
        if Hn is None or Cn is None: continue
        Cm = _mask(df, Cn)
        trivial = False
        for part in _flatten_and_pred(Hn):
            pm = _mask(df, part)
            if not (pm & ~Cm).any():
                trivial = True; break
        if not trivial: out.append(r)
    return out

def drop_h_implies_h_or_x(df: pd.DataFrame, mined: list[dict]) -> list[dict]:
    out = []
    for r in mined:
        Hm = _mask(df, normalize_and(r["H"]))
        Cm = _mask(df, normalize_and(r["C"]))
        if (Cm & ~Hm).any():  # keep only if C strictly extends H
            out.append(r)
    return out

def keep_most_general_per_consequent(df: pd.DataFrame, mined: list[dict]) -> list[dict]:
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in mined:
        Ck = _pred_key(df, r["C"])
        buckets[Ck].append(r)
    kept_all = []
    for _, items in buckets.items():
        masks = []
        for r in items:
            Hn = normalize_and(r["H"])
            masks.append((_mask(df, Hn), r))
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
                        hi = _pretty_factored(ri["H"]); hj = _pretty_factored(rj["H"])
                        if hi > hj: dominated[i] = True
        for k, d in enumerate(dominated):
            if not d: kept_all.append(masks[k][1])
    return kept_all

def hazel_like_rank_implications(
    mined: list[dict], *, drop_frac: float = 0.25, min_support: int | None = None,
):
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
            _pretty_factored(r["H"]),
            _pretty_factored(r["C"]),
        )
    )
    return kept

# =========================================================
# Data-driven implication lattice (learned from DF)
# =========================================================

def learn_atomic_implications(df: pd.DataFrame, atoms: list[Predicate]) -> dict[str, set[str]]:
    """
    Learn A ⇒ B if mask(A) ⊆ mask(B) on the dataset (nonempty A, B).
    Returns mapping from pretty(atom A) -> set of pretty(atom B).
    Only considers *atomic* predicates passed in `atoms`.
    """
    # Build masks and names for unique atoms
    uniq: list[Predicate] = []
    names: list[str] = []
    masks: list[pd.Series] = []
    seen = set()
    for p in atoms:
        if not _is_atomic_pred(p): continue
        key = repr(normalize_and(p))
        if key in seen: continue
        seen.add(key)
        m = _mask(df, p)
        if not m.any():  # skip empty
            continue
        uniq.append(p)
        names.append(_strip_all_outer_parens(_pretty_factored(p)))
        masks.append(m.astype(bool))

    n = len(uniq)
    implies: dict[str, set[str]] = {names[i]: set() for i in range(n)}
    for i in range(n):
        mi = masks[i]
        for j in range(n):
            if i == j: continue
            mj = masks[j]
            # if support(i) > 0 and mi ⊆ mj, record i ⇒ j
            if not (mi & ~mj).any():
                implies[names[i]].add(names[j])
    return implies

# =========================================================
# Presentation helpers using the learned implications
# =========================================================

def _is_neg_token(s: str) -> bool:
    s = s.strip()
    return s.startswith("¬(") and s.endswith(")")

def _and_atoms(s: str) -> list[str]:
    s = _strip_all_outer_parens(s)
    if not s: return []
    return [ _strip_all_outer_parens(p.strip())
             for p in (_split_top_level(s, "∧") if "∧" in s else [s])
             if p.strip() ]

def _or_disjuncts(s: str) -> list[str]:
    s = _strip_all_outer_parens(s)
    if "∨" not in s:
        return [s] if s else []
    return [ _strip_all_outer_parens(t.strip())
             for t in _split_top_level(s, "∨") if t.strip() ]

def _prune_and_atoms_with_implications(and_atoms: list[str], implies_map: dict[str, set[str]]) -> list[str]:
    """
    Drop positive literals that are implied by other positive literals present in the same AND.
    Negations are NOT dropped via positive implications.
    """
    A = list(dict.fromkeys(and_atoms))  # stable unique
    keep = [True] * len(A)
    # Build quick index
    for i, ai in enumerate(A):
        if not keep[i]: continue
        if _is_neg_token(ai): continue
        ai0 = _strip_all_outer_parens(ai)
        implied_set = implies_map.get(ai0, set())
        if not implied_set: continue
        for j, aj in enumerate(A):
            if i == j or not keep[j]: continue
            if _is_neg_token(aj): continue
            aj0 = _strip_all_outer_parens(aj)
            if aj0 in implied_set:
                # ai ⇒ aj, so drop aj
                keep[j] = False
    return [A[k] for k in range(len(A)) if keep[k]]

def _implied_by_set(H_pos: set[str], implies_map: dict[str, set[str]]) -> set[str]:
    out = set()
    for a in H_pos:
        out |= implies_map.get(a, set())
    return out

def pretty_consequent_minus_h(H: Predicate | None, C: Predicate | None,
                              implies_map: dict[str, set[str]]) -> str:
    """Return C \\ H, OR-aware, using learned implications to prune positives implied by H."""
    Hs = _pretty_factored(H); Cs = _pretty_factored(C)
    H_atoms_all = set(_and_atoms(Hs))
    H_pos = {a for a in H_atoms_all if not _is_neg_token(a)}
    implied_by_H = _implied_by_set(H_pos, implies_map)

    disjuncts = _or_disjuncts(Cs)
    residual = []
    for Dj in disjuncts:
        Dj_atoms = _and_atoms(Dj)
        # subtract H literals and positives implied by H
        kept = []
        for a in Dj_atoms:
            a0 = _strip_all_outer_parens(a)
            if a0 in H_atoms_all:  # already in H
                continue
            if (not _is_neg_token(a0)) and (a0 in implied_by_H):
                continue
            kept.append(a0)
        # prune redundancies inside this AND via learned implications
        kept = _prune_and_atoms_with_implications(kept, implies_map)
        if kept:
            residual.append("(" + ") ∧ (".join(sorted(set(kept))) + ")")
    if not residual: return "(⊤)"
    if len(residual) == 1: return residual[0]
    return " ∨ ".join(residual)

def pretty_rhs_conditioned_on_H(H: Predicate | None, C: Predicate | None,
                                implies_map: dict[str, set[str]]) -> str:
    # Show only what C adds beyond H; you can reattach H if you prefer.
    return pretty_consequent_minus_h(H, C, implies_map)

# =========================================================
# Seed conjectures (the small example)
# =========================================================

def bool_col_pred(col: str) -> Predicate:
    return Where(lambda d, c=col: d[c].astype(bool), name=f"({col})")

def mk_le(hyp: Predicate | None, left: str, right: str, name="") -> Conjecture:
    return Conjecture(Le(to_expr(left), to_expr(right)), hyp, name=name or f"{left}_le_{right}")

def mk_ge(hyp: Predicate | None, left: str, right: str, name="") -> Conjecture:
    return Conjecture(Ge(to_expr(left), to_expr(right)), hyp, name=name or f"{left}_ge_{right}")

H_conn         = bool_col_pred("connected")
H_trianglefree = bool_col_pred("triangle_free")
H_clawfree     = bool_col_pred("claw_free")

seed = [
    mk_le(H_conn, "independence_number", "annihilation_number",
          name="alpha_le_a(G)"),
    mk_ge(H_conn, "independence_number", "independent_domination_number",
          name="alpha_ge_i(G)"),
    mk_ge(AndPred(H_conn, H_trianglefree), "independence_number", "maximum_degree",
          name="alpha_ge_D(G) on triangle_free"),
    mk_ge(AndPred(H_conn, H_clawfree), "independent_domination_number", "domination_number",
          name="i(G)_ge_gamma(G) on claw_free"),
]

# =========================================================
# Run the demo
# =========================================================

if __name__ == "__main__":
    # 0) Show seed
    print("=== Seed conjectures ===")
    for c in seed:
        print(" -", c.pretty(arrow="⇒"))

    # 1) Promote tight-inequalities to equalities
    promoted = [_maybe_promote_to_eq(df, c, atol=1e-9) for c in seed]

    print("\n=== After promotion ===")
    for c in promoted:
        extra = " (PROMOTED)" if isinstance(c.relation, Eq) else ""
        print(" -", c.pretty(arrow="⇒"), extra)

    # 2) Derive tightness predicates
    print("\n=== Derived tightness predicates and supports ===")
    tight_preds = []
    for c in promoted:
        T = _tightness_preds(df, c, atol=1e-9)
        tight_preds.extend(T)
        print(f"\nFrom: {c.pretty(arrow='⇒')}")
        for P in T:
            supp = int(_mask(df, P).sum())
            print(f" • {pretty_pred(P)} support={supp}")

    # 3) Candidate hypotheses = simplified boolean hyps (+ atomic negations) + AND/ORs
    hyps = enumerate_boolean_hypotheses(
        df, include_base=True, include_pairs=True, skip_always_false=True
    )
    kept, _eqs = simplify_and_dedup_hypotheses(
        df, hyps, min_support=max(1, int(0.05 * len(df)))
    )

    # Build signed pool: positive and *atomic* negated variants with nonempty support
    signed_kept = []
    atomic_pool = []  # collect atomic positives for learning implications
    for p in kept:
        m = _mask(df, p)
        if m.any():
            signed_kept.append(p)
            if _is_atomic_pred(p):
                atomic_pool.append(p)
        if _is_atomic_pred(p):
            pn = negate(p)
            mn = _mask(df, pn)
            if mn.any():
                signed_kept.append(pn)

    candidates = build_candidate_hypotheses_with_disjunctions(
        df, signed_kept, include_or=True, min_support=1
    )

    # 3.5) Learn data-driven implications among atomic predicates
    implies_map = learn_atomic_implications(df, atomic_pool)

    # 4) Mine implications: H_tight ⇒ C (and equivalences)
    mined = mine_implications(df, tight_preds, candidates, min_support=1)

    # 5) Cleanup + ranking pipeline
    mined = dedupe_implications(df, mined)
    mined = drop_trivial_consequents(df, mined)
    mined = drop_h_implies_h_or_x(df, mined)
    mined = keep_most_general_per_consequent(df, mined)
    ranked = hazel_like_rank_implications(mined, drop_frac=0.25, min_support=None)

    print("\n=== Implications mined from tightness classes (Hazel-like filtered) ===")
    if not ranked:
        print("(none)")
    else:
        for r in ranked:
            arrow = "≡" if r["equiv"] else "⇒"
            Hs = _pretty_factored(r['H'])
            # Show RHS conditioned on H (subtract H, prune via learned implications)
            Cs_clean = pretty_rhs_conditioned_on_H(r['H'], r['C'], implies_map)
            print(f"{Hs} {arrow} {Cs_clean}   "
                  f"[support(H)={r['support_H']}, support(C)={r['support_C']}]")
