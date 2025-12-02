# hypotheses_module_examples.py
# -----------------------------------------------------------------------------
# Educational walkthrough for:
#   - list_boolean_columns
#   - detect_base_hypothesis
#   - enumerate_boolean_hypotheses
#
# Data includes: bool, nullable-bool, 0/1 Int64, and a few always-True columns.
# -----------------------------------------------------------------------------

from __future__ import annotations
import pandas as pd

from txgraffiti2025.processing.pre.hypotheses import (
    list_boolean_columns,
    detect_base_hypothesis,
    enumerate_boolean_hypotheses,
)
from txgraffiti2025.forms.predicates import Predicate

def show_df(df: pd.DataFrame):
    print("\n" + "="*78)
    print("0) DataFrame")
    print("="*78)
    print(df, "\n")

def show_columns(df: pd.DataFrame):
    print("="*78)
    print("1) Boolean-like column discovery")
    print("="*78)
    cols_default = list_boolean_columns(df)  # treat_binary_ints=True (default)
    cols_strict  = list_boolean_columns(df, treat_binary_ints=False)
    print("[treat_binary_ints=True]  ->", cols_default)
    print("[treat_binary_ints=False] ->", cols_strict, "\n")

def show_base(df: pd.DataFrame):
    print("="*78)
    print("2) Base hypothesis detection (always-True columns)")
    print("="*78)
    base = detect_base_hypothesis(df)
    print("base repr: ", repr(base))
    print("base name: ", getattr(base, "name", repr(base)))
    print("base mask: ", base.mask(df).astype(int).tolist(), "\n")

def show_enum(df: pd.DataFrame):
    print("="*78)
    print("3) Enumerate hypotheses (base, singles, pairs)")
    print("="*78)
    hyps = enumerate_boolean_hypotheses(
        df,
        treat_binary_ints=True,
        include_base=True,
        include_pairs=True,
        skip_always_false=True,
    )

    print(f"total hypotheses: {len(hyps)}\n")

    # Display each hypothesis: name, mask, and support
    for i, H in enumerate(hyps, 1):
        name = getattr(H, "name", repr(H))
        m = H.mask(df).astype(bool)
        support = int(m.sum())
        print(f"[H{i:02d}] {name}")
        print("repr:", repr(H))
        print("mask:", m.astype(int).tolist(), " support:", support, "\n")

def main():
    # Build a dataframe with a mix of boolean shapes
    df = pd.DataFrame(
        {
            # A few graph-like flags
            "connected":     [True,  True,  True,  True,  True,  True,  True],     # always True
            "simple":        [True,  True,  True,  True,  True,  True,  True],     # always True
            "planar":        [True,  True,  True,  False, False, False, True],
            "regular":       [True,  True,  False, False, True,  False, True],
            # Nullable boolean
            "chordal":       [True,  None,  True,  False, None, False, True],
            # 0/1 Int64 (nullable)
            "bipartite":     pd.Series([1, 1, 0, 1, 0, 0, 1], dtype="Int64"),
            "triangle_free": pd.Series([0, 0, 1, 1, 1, 0, 1], dtype="Int64"),
            # Non-boolean numeric (won't be listed)
            "order":         [3, 4, 5, 6, 8, 10, 7],
        },
        index=[f"G{i}" for i in range(1, 8)],
    )

    show_df(df)
    show_columns(df)
    show_base(df)
    show_enum(df)

if __name__ == "__main__":
    main()
