# demos/demo_generalize_from_constants.py
from __future__ import annotations
import numpy as np
import pandas as pd

from txgraffiti.example_data import graph_data as df

# Pre: hypotheses
from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
    detect_base_hypothesis,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_and_dedup_hypotheses,
)

# Forms
from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le

# Post: constant finder + generalizer
from txgraffiti2025.processing.post.constant_ratios import (
    find_constant_ratios_over_hypotheses,
)
from txgraffiti2025.processing.post.generalize_from_constants import (
    constants_from_ratios_hits,
    propose_generalizations_from_constants,
)


def _header(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78 + "\n")


def _pick_hypotheses(df: pd.DataFrame, *, min_support: int) -> tuple[Conjecture | None, list]:
    base = detect_base_hypothesis(df)
    hyps_all = enumerate_boolean_hypotheses(
        df,
        treat_binary_ints=True,
        include_base=True,
        include_pairs=True,
        skip_always_false=True,
    )
    kept, _ = simplify_and_dedup_hypotheses(
        df, hyps_all, min_support=min_support, treat_binary_ints=True
    )
    pairs = [H for H in kept if "AndPred" in type(H).__name__]
    singles = [H for H in kept if "AndPred" not in type(H).__name__]
    pairs_sorted = sorted(pairs, key=lambda H: int(H.mask(df).sum()), reverse=True)
    singles_sorted = sorted(singles, key=lambda H: int(H.mask(df).sum()), reverse=True)
    H0 = pairs_sorted[0] if pairs_sorted else (singles_sorted[0] if singles_sorted else None)
    return H0, kept


def _touch_count(conj: Conjecture, df: pd.DataFrame) -> int:
    """Count how many rows achieve equality lhs == rhs (within tolerance) on conj.condition."""
    try:
        lhs = conj.relation.left.eval(df)
        rhs = conj.relation.right.eval(df)
        mask = conj.condition.mask(df) if conj.condition else np.ones(len(df), bool)
        touch = np.isclose(lhs[mask], rhs[mask], rtol=1e-8, atol=1e-8)
        return int(np.sum(touch))
    except Exception:
        return 0


def main():
    _header("Demo: Generalize from constants (coefficient lifts)")
    print(f"Dataset: {df.shape}")

    # 1) Pick hypotheses
    H0, kept_hyps = _pick_hypotheses(df, min_support=max(30, int(0.10 * len(df))))
    print("Base (narrow) hypothesis H0:", repr(H0) if H0 else "None")
    print(f"Candidate hypotheses to test as supersets: {len(kept_hyps)}")

    if H0 is None:
        print("No suitable H0 found; exiting.")
        return

    # 2) Base conjecture (seed)
    target, feature = "domination_number", "order"
    base_conj = Conjecture(
        relation=Le(to_expr(target), Const(1.0) * to_expr(feature)),
        condition=H0,
        name=f"{target}_vs_{feature}_on_H0",
    )
    print("Base conjecture:", base_conj.pretty(unicode_ops=True))

    # 3) Constant-ratio mining → candidate K(N,D)
    _header("Mining structural coefficients K = (N+a)/(D+b) across subclasses")
    hits = find_constant_ratios_over_hypotheses(
        df,
        hypotheses=kept_hyps,
        shifts=tuple(range(-1, 1)),            # -1, 0, 1
        constancy="cv",
        cv_tol=0.10,
        min_support=max(12, int(0.08 * len(df))),
        max_denominator=30,
        top_k_per_hypothesis=25,
    )
    print(f"Candidate constant-ratio hits: {len(hits)}")
    for h in hits[:8]:
        hyp_txt = repr(h.hypothesis) if h.hypothesis is not None else "TRUE"
        print(f"  [{hyp_txt}]  ({h.numerator}+{h.shift_num})/({h.denominator}+{h.shift_den})"
              f" ≈ {h.value_display} | support={h.support} | cv={h.cv:.3f}")

    cands = constants_from_ratios_hits(hits)
    print(f"\nConverted to constant candidates: {len(cands)}")

    # 4) Propose generalizations c → K(N,D) on strict supersets H ⊃ H0
    _header("Proposing generalizations (strict supersets, filters enabled)")
    gens = propose_generalizations_from_constants(
        df,
        base_conj,
        candidate_hypotheses=kept_hyps,
        constant_candidates=cands,
        constants_cache=None,
        require_strict_superset=True,
        drop_near_one_coeffs=True,
        near_one_med_tol=1e-2,
        near_one_cv_tol=2e-2,
        atol=1e-6,
        rtol=5e-2,
    )

    print(f"Accepted generalizations: {len(gens)}")
    # --- NEW: always include the original conjecture as a fallback/keeper ---
    # We’ll merge it with the generalized ones and sort the combined list.
    rows = []

    # Original (label + conj)
    rows.append({
        "kind": "original",
        "conj": base_conj,
        "witness_superset": base_conj.condition,
        "meta": {},
        "align": None,
    })

    # Generalized
    for g in gens:
        rows.append({
            "kind": "generalized",
            "conj": g.new_conjecture,
            "witness_superset": g.witness_superset,
            "meta": g.meta.get("candidate_info", {}),
            "align": g.meta.get("align", None),
        })

    # Sort by touch count (descending)
    print("\nSorting by touch count (including original)...\n")
    rows_with_touch = [(row, _touch_count(row["conj"], df)) for row in rows]
    rows_sorted = sorted(rows_with_touch, key=lambda t: t[1], reverse=True)

    # 5) Preview (sorted)
    _header("Preview (top 12 by touch count)")
    for j, (row, tcount) in enumerate(rows_sorted[:12], 1):
        conj = row["conj"]
        app, holds, _ = conj.check(df)
        Hsup = repr(row["witness_superset"]) if row["witness_superset"] else "None"
        meta = row["meta"]
        align = row["align"]
        tag = "(original)" if row["kind"] == "original" else "(generalized)"

        print(f"[G{j:02d}] {tag}  {conj.pretty(unicode_ops=True)}")
        print(f"     hypothesis: {Hsup}")
        print(f"     touch count: {tcount}")
        print(f"     coverage: applicable={int(app.sum())}, satisfied={int((app & holds).sum())}, violations={int((app & ~holds).sum())}")
        if meta:
            cv = meta.get("cv")
            qsp = meta.get("qspan")
            cv_txt = f"{cv:.3f}" if isinstance(cv, (int, float)) else str(cv)
            qsp_txt = f"{qsp:.3f}" if isinstance(qsp, (int, float)) else str(qsp)
            print(f"     candidate: support={meta.get('support')}  cv={cv_txt}  qspan={qsp_txt}  value≈{meta.get('value_display')}")
        if align is not None:
            print(f"     align: K@base_median={align['K@base_median']:.4g}, |median-c|={align['abs_err_to_c']:.4g}")
        print()

    _header("Done")


if __name__ == "__main__":
    main()
