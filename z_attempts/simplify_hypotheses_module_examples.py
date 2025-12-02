# simplify_hypotheses_module_examples.py
# -----------------------------------------------------------------------------
# Educational walkthrough for:
#   - simplify_predicate_via_df
#   - simplify_and_dedup_hypotheses
#
# We craft several hypotheses (some redundant), simplify them against the DF,
# dedup identical masks, and collect ClassEquivalence witnesses when renaming.
# -----------------------------------------------------------------------------

from __future__ import annotations
import pandas as pd

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
    detect_base_hypothesis,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_predicate_via_df,
    simplify_and_dedup_hypotheses,
)

def show_df(df: pd.DataFrame):
    print("\n" + "="*78)
    print("0) DataFrame")
    print("="*78)
    print(df, "\n")

def build_custom_hypotheses(df: pd.DataFrame):
    """
    Create a few ad-hoc hypotheses:
      - some equal to simple columns,
      - some equal to conjunctions,
      - some that differ only by naming / structure but have identical masks.
    """
    P = Predicate.from_column("planar",        truthy_only=True)
    R = Predicate.from_column("regular",       truthy_only=True)
    B = Predicate.from_column("bipartite",     truthy_only=True)
    T = Predicate.from_column("triangle_free", truthy_only=True)
    C = Predicate.from_column("connected",     truthy_only=True)
    S = Predicate.from_column("simple",        truthy_only=True)
    # nullable bool
    Ch = Predicate.from_column("chordal",      truthy_only=True)

    # Singles
    H1 = P;                           H1.name = "(planar)"
    H2 = R;                           H2.name = "(regular)"
    H3 = B;                           H3.name = "(bipartite)"

    # Conjunctions (vary structure to create equivalent-but-different forms)
    H4 = (P & R);                     H4.name = "((planar) ∧ (regular))"
    H5 = (R & P);                     H5.name = "((regular) ∧ (planar))"     # same mask as H4

    # Base-like conjunctions that include always-true columns (C and S)
    H6 = (C & S & P);                 H6.name = "((connected) ∧ (simple) ∧ (planar))"
    H7 = (P & C & S);                 H7.name = "((planar) ∧ (connected) ∧ (simple))"  # same mask as H6

    # Mix with nullable boolean, and another column
    H8 = (Ch & T);                    H8.name = "((chordal) ∧ (triangle_free))"

    return [H1, H2, H3, H4, H5, H6, H7, H8]

def show_simplify_each(df: pd.DataFrame, hyps):
    print("="*78)
    print("1) Simplify individual hypotheses (and optional equivalence witness)")
    print("="*78)
    for i, h in enumerate(hyps, 1):
        Hs, eq = simplify_predicate_via_df(h, df, treat_binary_ints=True)
        print(f"[h{i:02d}] original:  {getattr(h, 'name', repr(h))}")
        print(f"      mask:     ", Predicate.from_column("connected", True).mask(df).mul(0) + h.mask(df))  # quick visual
        print(f"      simplified: {getattr(Hs, 'name', repr(Hs))}")
        if eq is not None:
            print("      witness:   ", repr(eq))
        else:
            print("      witness:    (none needed; identical atom list)")
        print()

def show_dedup_pipeline(df: pd.DataFrame, hyps):
    print("="*78)
    print("2) End-to-end: simplify, enforce min_support, deduplicate by mask")
    print("="*78)

    # Mix in hypotheses auto-enumerated from the DF
    auto = enumerate_boolean_hypotheses(
        df, treat_binary_ints=True, include_base=True, include_pairs=True
    )
    pool = hyps + auto

    kept, witnesses = simplify_and_dedup_hypotheses(
        df,
        pool,
        min_support=2,          # require at least 2 rows of support
        treat_binary_ints=True,
    )

    print(f"Kept hypotheses: {len(kept)}\n")
    for i, H in enumerate(kept, 1):
        name = getattr(H, "name", repr(H))
        m = H.mask(df).astype(int).tolist()
        print(f"[K{i:02d}] {name}")
        print("repr:", repr(H))
        print("mask:", m, "\n")

    if witnesses:
        print("ClassEquivalence witnesses (renamings where atom lists changed):\n")
        for W in witnesses:
            print(" -", repr(W))
    else:
        print("No renaming witnesses emitted.")

def main():
    df = pd.DataFrame(
        {
            "connected":     [True,  True,  True,  True,  True,  True,  True],   # always True
            "simple":        [True,  True,  True,  True,  True,  True,  True],   # always True
            "planar":        [True,  True,  True,  False, False, False, True],
            "regular":       [True,  True,  False, False, True,  False, True],
            "chordal":       [True,  None,  True,  False, None, False, True],     # nullable bool
            "bipartite":     pd.Series([1, 1, 0, 1, 0, 0, 1], dtype="Int64"),     # 0/1 Int64
            "triangle_free": pd.Series([0, 0, 1, 1, 1, 0, 1], dtype="Int64"),
            "order":         [3, 4, 5, 6, 8, 10, 7],
        },
        index=[f"G{i}" for i in range(1, 8)],
    )

    show_df(df)

    base = detect_base_hypothesis(df)
    print("Detected base:", getattr(base, "name", repr(base)), "\n")

    hyps = build_custom_hypotheses(df)
    show_simplify_each(df, hyps)
    show_dedup_pipeline(df, hyps)

if __name__ == "__main__":
    main()
