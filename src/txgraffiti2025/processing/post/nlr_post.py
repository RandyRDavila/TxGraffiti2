# src/txgraffiti2025/processing/post/nlr_post.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Literal
import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Relation, Ge, Le, Eq
from txgraffiti2025.forms.utils import Expr

__all__ = [
    "ConjStats",
    "score_conjecture",
    "dedup_conjectures",
    "dominance_prune",
    "morgan_prune",
    "summarize_and_prune",
]

# ───────── stats & scoring (unchanged) ───────── #

@dataclass(frozen=True)
class ConjStats:
    support: int
    tight: int
    tight_ratio: float
    violation: int
    mean_margin: float

def score_conjecture(df: pd.DataFrame, c: Conjecture, *, atol: float = 1e-9) -> ConjStats:
    applicable, _, _ = c.check(df, auto_base=True)
    if not applicable.any():
        return ConjStats(0, 0, 0.0, 0, 0.0)

    s = c.relation.slack(df).reindex(df.index)
    app = applicable.values.astype(bool)
    sup = int(app.sum())

    s_app = s.values[app]
    violation = int(np.sum(np.isfinite(s_app) & (s_app < -atol)))

    tight_mask = c.relation.is_tight(df, atol=atol).reindex(df.index).values
    tight_cnt = int(np.sum(app & tight_mask))

    mean_margin = float(np.nanmean(s_app)) if sup > 0 else 0.0
    return ConjStats(
        support=sup,
        tight=tight_cnt,
        tight_ratio=(tight_cnt / sup) if sup else 0.0,
        violation=violation,
        mean_margin=mean_margin,
    )

# ───────── utils & pruning (same as before) ───────── #

# ── NEW: promote identically-tight inequalities to Eq ─────────────────────────
from txgraffiti2025.forms.generic_conjecture import Eq, Ge, Le, Conjecture

def _promote_equalities(
    df: pd.DataFrame,
    conjs: List[Conjecture],
    *,
    atol: float = 1e-9
) -> Tuple[List[Conjecture], List[Conjecture]]:
    """
    Scan Ge/Le conjectures; if slack == 0 (within atol) on *all applicable rows*,
    convert to Eq(left, right, tol=atol). Return (eqs, remaining_ineqs).
    """
    eqs: List[Conjecture] = []
    kept: List[Conjecture] = []

    for c in conjs:
        rel = c.relation
        if isinstance(rel, (Ge, Le)):
            applicable, _, _ = c.check(df, auto_base=True)
            app = applicable.values.astype(bool)
            if not app.any():
                kept.append(c); continue

            s = rel.slack(df).reindex(df.index).values
            s_app = s[app]
            # All applicable must be finite & |slack| <= atol
            finite = np.isfinite(s_app)
            if finite.all() and np.all(np.abs(s_app) <= atol):
                eqs.append(Conjecture(Eq(rel.left, rel.right, tol=atol), c.condition))
                continue

        kept.append(c)

    return eqs, kept


def _conj_key(c: Conjecture) -> str:
    return c.signature()

def dedup_conjectures(conjs: Iterable[Conjecture]) -> List[Conjecture]:
    seen: Dict[str, Conjecture] = {}
    for c in conjs:
        k = _conj_key(c)
        if k not in seen:
            seen[k] = c
    return list(seen.values())

def _eval_expr(df: pd.DataFrame, e: Expr) -> np.ndarray:
    v = e.eval(df) if hasattr(e, "eval") else e(df)  # type: ignore
    return np.asarray(pd.to_numeric(v, errors="coerce"), dtype=float)

def _dominates_ge(rhs_a: np.ndarray, rhs_b: np.ndarray, atol: float) -> bool:
    ok = np.isfinite(rhs_a) & np.isfinite(rhs_b)
    return bool(ok.any() and np.all(rhs_a[ok] >= rhs_b[ok] - atol))

def _dominates_le(rhs_a: np.ndarray, rhs_b: np.ndarray, atol: float) -> bool:
    ok = np.isfinite(rhs_a) & np.isfinite(rhs_b)
    return bool(ok.any() and np.all(rhs_a[ok] <= rhs_b[ok] + atol))

def dominance_prune(df: pd.DataFrame, conjs: List[Conjecture], *, atol: float = 1e-9) -> List[Conjecture]:
    from collections import defaultdict
    def _group_key(c: Conjecture) -> Tuple[str, str, str]:
        rel = c.relation
        op = "ge" if isinstance(rel, Ge) else "le" if isinstance(rel, Le) else rel.__class__.__name__
        cond_repr = repr(c.condition) if c.condition is not None else "AUTO-BASE"
        left_repr = getattr(rel, "left", None)
        left_repr = repr(left_repr) if left_repr is not None else repr(rel)
        return (cond_repr, op, left_repr)

    groups: Dict[Tuple[str, str, str], List[Conjecture]] = defaultdict(list)
    for c in conjs:
        if isinstance(c.relation, (Ge, Le)):
            groups[_group_key(c)].append(c)

    kept: List[Conjecture] = []
    for key, lst in groups.items():
        if len(lst) == 1:
            kept.append(lst[0]); continue

        survivors = [True] * len(lst)
        op = key[1]
        for i in range(len(lst)):
            if not survivors[i]: continue
            for j in range(len(lst)):
                if i == j or not survivors[j]: continue
                app_i, _, _ = lst[i].check(df, auto_base=True)
                app_j, _, _ = lst[j].check(df, auto_base=True)
                inter = (app_i.values.astype(bool) & app_j.values.astype(bool))
                if not inter.any(): continue

                rhs_i = _eval_expr(df.loc[inter], lst[i].relation.right)  # type: ignore[attr-defined]
                rhs_j = _eval_expr(df.loc[inter], lst[j].relation.right)  # type: ignore[attr-defined]

                if op == "ge" and _dominates_ge(rhs_j, rhs_i, atol):
                    survivors[i] = False; break
                if op == "le" and _dominates_le(rhs_j, rhs_i, atol):
                    survivors[i] = False; break

        kept.extend([c for c, alive in zip(lst, survivors) if alive])

    for c in conjs:
        if not isinstance(c.relation, (Ge, Le)):
            kept.append(c)
    return kept

def _conclusion_key(c: Conjecture) -> str:
    # conclusion only (ignore condition)
    return repr(c.relation)

def _strict_subset(a: np.ndarray, b: np.ndarray) -> bool:
    if a.size == 0 or b.size == 0:
        return False
    a_in_b = (~a) | b
    proper = np.any(b & ~a)
    return bool(np.all(a_in_b) and proper)

def morgan_prune(df: pd.DataFrame, conjs: List[Conjecture]) -> List[Conjecture]:
    from collections import defaultdict
    groups: Dict[str, List[Conjecture]] = defaultdict(list)
    for c in conjs:
        groups[_conclusion_key(c)].append(c)

    kept: List[Conjecture] = []
    for _, lst in groups.items():
        if len(lst) == 1:
            kept.append(lst[0]); continue

        masks = []
        for c in lst:
            applicable, _, _ = c.check(df, auto_base=True)
            masks.append(applicable.values.astype(bool))

        survivors = [True] * len(lst)
        for i in range(len(lst)):
            if not survivors[i]: continue
            for j in range(len(lst)):
                if i == j or not survivors[j]: continue
                if _strict_subset(masks[j], masks[i]):
                    survivors[j] = False

        kept.extend([c for c, alive in zip(lst, survivors) if alive])
    return kept

# ───────── split & summarize ───────── #

def _split_eq_ineq(conjs: Iterable[Conjecture]):
    eqs, ineqs = [], []
    for c in conjs:
        (eqs if isinstance(c.relation, Eq) else ineqs).append(c)
    return eqs, ineqs

def _sort_ineq_key(cs: Tuple[Conjecture, ConjStats]):
    c, s = cs
    # tight ratio desc, then support desc, then mean_margin desc, then fewer violations, then shorter signature
    return (s.tight_ratio, s.support, s.mean_margin, -s.violation, -len(c.signature()))

def _sort_eq_key(cs: Tuple[Conjecture, ConjStats]):
    c, s = cs
    # equalities: rank by coverage/support first, then by mean margin (bigger means closer to exact), then fewer violations
    return (s.support, s.mean_margin, -s.violation, -len(c.signature()))

def summarize_and_prune(
    df: pd.DataFrame,
    lowers: List[Conjecture],
    uppers: List[Conjecture],
    *,
    atol: float = 1e-9,
    per_condition: bool = True,
    keep_top_per_condition: int = 5,
    use_dominance: bool = False,
):
    # 0) Promote equalities out of both pools
    eq_L, lowers = _promote_equalities(df, lowers, atol=atol)
    eq_U, uppers = _promote_equalities(df, uppers, atol=atol)
    eqs = dedup_conjectures(eq_L + eq_U)

    # ===== INEQUALITIES (as before, but now without promoted Eq’s) =====
    l_scored_all = [(c, score_conjecture(df, c, atol=atol)) for c in lowers]
    u_scored_all = [(c, score_conjecture(df, c, atol=atol)) for c in uppers]
    l_scored = [(c, s) for c, s in l_scored_all if s.tight > 0]
    u_scored = [(c, s) for c, s in u_scored_all if s.tight > 0]

    lowers = dedup_conjectures([c for c, _ in l_scored])
    uppers = dedup_conjectures([c for c, _ in u_scored])

    if use_dominance:
        lowers = dominance_prune(df, lowers, atol=atol)
        uppers = dominance_prune(df, uppers, atol=atol)

    lowers = morgan_prune(df, lowers)
    uppers = morgan_prune(df, uppers)

    l_scored = [(c, score_conjecture(df, c, atol=atol)) for c in lowers]
    u_scored = [(c, score_conjecture(df, c, atol=atol)) for c in uppers]

    # rank inequalities by tight ratio (desc), then support, margin, …
    def _ineq_key(cs):
        c, s = cs
        return (s.tight_ratio, s.support, s.mean_margin, -s.violation, -len(c.signature()))

    from collections import defaultdict
    def _rank_bucketed(scored):
        if not per_condition:
            scored.sort(key=_ineq_key, reverse=True)
            return scored[:keep_top_per_condition]
        buckets: Dict[str, List[Tuple[Conjecture, ConjStats]]] = defaultdict(list)
        for cs in scored:
            cond_key = repr(cs[0].condition) if cs[0].condition is not None else "AUTO-BASE"
            buckets[cond_key].append(cs)
        out = []
        for _, arr in buckets.items():
            arr.sort(key=_ineq_key, reverse=True)
            out.extend(arr[:keep_top_per_condition])
        out.sort(key=_ineq_key, reverse=True)
        return out

    L_ranked = _rank_bucketed(l_scored)
    U_ranked = _rank_bucketed(u_scored)

    # ===== EQUALITIES =====
    # Morgan on equalities too (same-conclusion, subset-of-condition rule)
    eqs = morgan_prune(df, eqs)
    eq_scored = [(c, score_conjecture(df, c, atol=atol)) for c in eqs]

    # rank equalities by coverage, then margin
    def _eq_key(cs):
        c, s = cs
        return (s.support, s.mean_margin, -s.violation, -len(c.signature()))

    eq_scored.sort(key=_eq_key, reverse=True)

    return L_ranked, U_ranked, eq_scored
