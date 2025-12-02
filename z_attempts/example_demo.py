# examples/pipeline_ratios_demo.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# --- data ---
from txgraffiti.example_data import graph_data as df  # demo DataFrame

# --- pretty printers (side-effect import to patch .pretty) ---
from txgraffiti2025.forms.pretty import format_conjecture  # noqa: F401

# --- core types ---
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.utils import Expr, BinOp, ColumnTerm, to_expr

# --- generators ---
from txgraffiti2025.generators.ratios import ratios
from txgraffiti2025.generators.lp import lp_bounds, LPConfig

# --- hypotheses ---
from txgraffiti2025.processing.pre.hypotheses import (
    detect_base_hypothesis,
    enumerate_boolean_hypotheses,
)

# --- constants cache (fast warmup + on-demand) ---
from txgraffiti2025.processing.pre.constants_cache import (
    precompute_constant_ratios,
    precompute_constant_ratios_pairs,
    ConstantsCache,
)

# --- post-processing (Morgan, Hazel, generalizers) ---
from txgraffiti2025.processing.post.morgan import morgan_filter
from txgraffiti2025.processing.post.hazel import hazel_rank
from txgraffiti2025.processing.post.generalize_from_constants import (
    propose_generalizations_from_constants,
    Generalization,
)
from txgraffiti2025.processing.post.reciprocal_generalizer import (
    propose_generalizations_from_reciprocals,
)

# --- diversity metrics ---
from txgraffiti2025.processing.metrics.diversity import compute_diversity, DiversityConfig


# -----------------------------------------------------------------------------
# Algebraic simplification for RHS of ratio-style relations
# -----------------------------------------------------------------------------

def _structurally_equal(a: Expr, b: Expr) -> bool:
    return repr(a) == repr(b)

def _try_cancel_product(rhs: Expr) -> Expr:
    """
    Try to simplify a product of the form:
        (A / Col(f)) * Col(f)  or  Col(f) * (A / Col(f))  ->  A
    Otherwise return `rhs` unchanged.
    """
    if not isinstance(rhs, BinOp) or rhs.fn is not np.multiply:
        return rhs

    def _cancel(one: Expr, other: Expr) -> Expr | None:
        if isinstance(other, BinOp) and other.fn is np.divide and isinstance(other.right, ColumnTerm) and isinstance(one, ColumnTerm):
            if other.right.col == one.col:
                return other.left
        return None

    left, right = rhs.left, rhs.right
    if isinstance(left, ColumnTerm):
        cand = _cancel(left, right)
        if cand is not None:
            return cand
    if isinstance(right, ColumnTerm):
        cand = _cancel(right, left)
        if cand is not None:
            return cand
    return rhs

def simplify_ratio_conjecture(c: Conjecture) -> Conjecture:
    """
    If relation is of the form target (≤/≥) coeff * feature, simplify coeff*feature
    by cancelling simple (A/feature)*feature patterns. Avoid creating tautologies.
    """
    rel = c.relation
    if not isinstance(rel, (Ge, Le)):
        return c

    left = rel.left
    rhs_simpl = _try_cancel_product(rel.right)

    if _structurally_equal(left, rhs_simpl):
        return c

    new_rel = Ge(left, rhs_simpl) if isinstance(rel, Ge) else Le(left, rhs_simpl)
    return Conjecture(relation=new_rel, condition=c.condition, name=c.name)

def is_trivial_constant_bound(df: pd.DataFrame, c: Conjecture, atol: float = 1e-9) -> bool:
    """
    If after simplification the RHS is numerically constant on the hypothesis mask,
    and equals the trivial extreme of the LHS (min for ≥, max for ≤), drop it.
    """
    s = simplify_ratio_conjecture(c)
    rel = s.relation
    if not isinstance(rel, (Ge, Le)):
        return False

    mask = (s.condition.mask(df) if s.condition is not None else pd.Series(True, index=df.index)).astype(bool)
    if not mask.any():
        return True

    lhs = pd.to_numeric(rel.left.eval(df)[mask], errors="coerce").dropna()
    rhs = pd.to_numeric(rel.right.eval(df)[mask], errors="coerce").dropna()
    if lhs.empty or rhs.empty:
        return True

    if np.nanmax(rhs) - np.nanmin(rhs) > atol:
        return False
    const_val = float(rhs.iloc[0])

    if isinstance(rel, Ge):
        return np.isclose(const_val, float(lhs.min()), atol=atol)
    else:
        return np.isclose(const_val, float(lhs.max()), atol=atol)


# -----------------------------------------------------------------------------
# Helpers: touch, diversity, printing, dedupe
# -----------------------------------------------------------------------------

def _touch_number(df: pd.DataFrame, conj: Conjecture, atol: float = 1e-9) -> int:
    applicable, holds, _ = conj.check(df)
    if not applicable.any():
        return 0
    s = conj.relation.slack(df).reindex(df.index)
    mask = applicable & holds
    s = s.where(mask)
    return int(np.sum(np.isfinite(s) & (np.abs(s) <= atol)))

def print_conjectures_with_metrics(
    df: pd.DataFrame,
    conjs: list[Conjecture],
    *,
    diversity_config: DiversityConfig | None = None,
    atol: float = 1e-9,
) -> None:
    cfg = diversity_config or DiversityConfig(
        w_coverage=0.5, w_boolean=1.0, w_categorical=1.0, w_numeric=1.0
    )
    for c in conjs:
        touch = _touch_number(df, c, atol=atol)
        div = compute_diversity(df, c.condition, config=cfg)
        score = div.get("DiversityScore", float("nan"))
        print(c.pretty(arrow="⇒"))
        print(f"touch={touch}, diversity={score:.3f}\n")

def header(title: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")

def dedupe_conjectures(conjs: Iterable[Conjecture]) -> list[Conjecture]:
    """
    Remove duplicates by (repr(condition), repr(relation)).
    """
    seen = set()
    out: list[Conjecture] = []
    for c in conjs:
        key = (repr(c.condition), repr(c.relation))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


# -----------------------------------------------------------------------------
# Ratio & LP generation + constants cache helpers
# -----------------------------------------------------------------------------

@dataclass
class RatioGenConfig:
    features: Sequence[str]
    target: str
    max_denominator: int = 50
    direction: str = "both"        # "both" | "upper" | "lower"
    q_clip: Optional[float] = None
    min_support: int = 2

def generate_ratio_conjectures(
    df: pd.DataFrame,
    hypotheses: Iterable[Optional[Predicate]],
    cfg: RatioGenConfig,
):
    out: list[Conjecture] = []
    for hyp in hypotheses:
        out.extend(
            ratios(
                df,
                features=list(cfg.features),
                target=cfg.target,
                hypothesis=hyp,
                max_denominator=cfg.max_denominator,
                direction=cfg.direction,
                q_clip=cfg.q_clip,
                min_support=cfg.min_support,
            )
        )
    return out

def generate_lp_conjectures(
    df: pd.DataFrame,
    hypotheses: Iterable[Optional[Predicate]],
    cfg: LPConfig,
):
    out: list[Conjecture] = []
    for hyp in hypotheses:
        try:
            out.extend(lp_bounds(df, hypothesis=hyp, config=cfg))
        except RuntimeError as e:
            # No solver or infeasible: skip gracefully
            print(f"[lp] Skipping hypothesis {getattr(hyp, 'pretty', lambda **_: repr(hyp))()}: {e}")
    return out

def ensure_constants_for(
    df: pd.DataFrame,
    cache: ConstantsCache,
    hypothesis: Optional[Predicate],
    *,
    shifts = (-2, -1, 0, 1, 2),
    min_support: int = 8,
    max_denominator: int = 50,
) -> ConstantsCache:
    if cache.hyp_to_key.get(repr(hypothesis)) in cache.key_to_constants:
        return cache
    mini = precompute_constant_ratios(
        df,
        hypotheses=[hypothesis],
        shifts=shifts,
        min_support=min_support,
        max_denominator=max_denominator,
    )
    cache.hyp_to_key.update(mini.hyp_to_key)
    cache.key_to_constants.update(mini.key_to_constants)
    return cache


# -----------------------------------------------------------------------------
# Superset acceptance for generalizations
# -----------------------------------------------------------------------------

def _mask(df: pd.DataFrame, pred: Predicate | None) -> pd.Series:
    if pred is None:
        return pd.Series(True, index=df.index)
    return pred.mask(df).reindex(df.index, fill_value=False).astype(bool)

def _is_superset(df: pd.DataFrame, old: Predicate | None, new: Predicate | None, *, strict: bool) -> bool:
    m_old = _mask(df, old)
    m_new = _mask(df, new)
    subset = not (m_old & ~m_new).any()
    if not subset:
        return False
    return (m_new & ~m_old).any() if strict else True

def _accept_generalization(df: pd.DataFrame, old: Conjecture, new: Conjecture) -> bool:
    if not _is_superset(df, old.condition, new.condition, strict=True):
        return False
    if not new.is_true(df):
        return False
    m_old = _mask(df, old.condition)
    return bool(new.relation.evaluate(df).reindex(df.index, fill_value=False)[m_old].all())

def _normalize_generalizations(
    old: Conjecture, objs: List[Generalization | Conjecture]
) -> List[Generalization]:
    out: List[Generalization] = []
    for g in objs:
        if isinstance(g, Conjecture):
            out.append(Generalization(from_conjecture=old, new_conjecture=g, reason=""))
        else:
            out.append(g)
    return out

def collect_generalizations(
    df: pd.DataFrame,
    kept_conjectures: List[Conjecture],
    cache: ConstantsCache,
    candidate_hypotheses: Sequence[Optional[Predicate]],
) -> List[Generalization]:
    gens_all: List[Generalization] = []
    seen_conditions = set()

    for c in kept_conjectures:
        key = repr(c.condition)
        if key not in seen_conditions:
            ensure_constants_for(df, cache, c.condition)
            seen_conditions.add(key)

        g1 = _normalize_generalizations(
            c, propose_generalizations_from_constants(df, c, cache, candidate_hypotheses=candidate_hypotheses)
        )
        g2 = _normalize_generalizations(
            c, propose_generalizations_from_reciprocals(
                df, c, candidate_hypotheses=candidate_hypotheses,
                candidate_cols=None, shifts=(0, 1), min_support=8
            )
        )
        gens_all.extend(g1)
        gens_all.extend(g2)
    return gens_all


# -----------------------------------------------------------------------------
# Demo / test pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    header("Sample DataFrame")
    print(df.head())

    TARGET = 'independence_number'
    header("Detect base & warm up constants (pairs only)")
    base = detect_base_hypothesis(df)
    print(f"Base: {getattr(base, 'pretty', lambda **_: repr(base))()}")

    const_cache = precompute_constant_ratios_pairs(
        df, base,
        shifts=(-2, -1, 0, 1, 2),
        min_support=8,
        max_denominator=50,
    )
    print(f"Warm cache for {len(const_cache.key_to_constants)} hypotheses.")

    header("Enumerate boolean hypotheses")
    H = enumerate_boolean_hypotheses(df, include_base=True)
    print(f"Hypotheses: {len(H)}")

    NUM_COLUMNS = df.select_dtypes(include=['number']).columns.tolist()
    NUM_COLUMNS = [invar for invar in NUM_COLUMNS if invar != TARGET]
    # --- Generate: ratios FIRST, then LP bounds ---
    header("Generate ratio conjectures")
    r_cfg = RatioGenConfig(
        features=NUM_COLUMNS,
        target=TARGET,
        max_denominator=50,
        direction="both",
        q_clip=None,
        min_support=2,
    )
    ratio_conjs = generate_ratio_conjectures(df, H, r_cfg)
    print(f"Ratios generated: {len(ratio_conjs)}")

    header("Generate LP bounds")
    lp_cfg = LPConfig(
        features=['residue', 'independence_number'],
        target=TARGET,
        direction="both",
        max_denominator=50,
        tol=1e-9,
        min_support=8,
    )
    lp_conjs = generate_lp_conjectures(df, H, lp_cfg)
    print(f"LP bounds generated: {len(lp_conjs)}")

    # Combine + dedupe BEFORE Morgan/Hazel
    all_gen = dedupe_conjectures(ratio_conjs + lp_conjs)
    print(f"Combined (deduped): {len(all_gen)}")

    # --- Post 1: Morgan → Hazel ---
    header("Morgan → Hazel post-processing (pass 1)")
    m1 = morgan_filter(df, all_gen)
    h1 = hazel_rank(df, m1.kept, drop_frac=0.25)
    kept1 = h1.kept_sorted
    print(f"Kept after pass 1: {len(kept1)}")

    # --- Generalizations on pass-1 kept ---
    gens = collect_generalizations(df, kept1, const_cache, candidate_hypotheses=H)

    accepted, superseded_ids = [], set()
    for g in gens:
        old, new = g.from_conjecture, g.new_conjecture
        if _accept_generalization(df, old, new):
            superseded_ids.add(id(old))
            accepted.append(new)
    print(f"Generalizations accepted: {len(accepted)}; superseding {len(superseded_ids)} originals.")

    # Merge originals (not superseded) + accepted generalizations
    merged = [c for c in kept1 if id(c) not in superseded_ids] + accepted

    # Simplify and drop trivial constant bounds (works well for ratio-like forms)
    normalized = []
    for c in merged:
        s = simplify_ratio_conjecture(c)
        if not is_trivial_constant_bound(df, s, atol=1e-9):
            normalized.append(s)

    # --- Post 2: Morgan → Hazel on normalized set ---
    header("Morgan → Hazel post-processing (pass 2 on merged set)")
    m2 = morgan_filter(df, normalized)
    h2 = hazel_rank(df, m2.kept, drop_frac=0.25)
    final_conjs = h2.kept_sorted
    print(f"Final kept: {len(final_conjs)}\n")

    # Output (pretty + metrics)
    header("Final conjectures")
    print_conjectures_with_metrics(df, final_conjs[:50])
    header("Done")


if __name__ == "__main__":
    main()
