# qualitative_module_examples.py
"""
Educational walkthrough for txgraffiti2025.forms.qualitative.MonotoneRelation

What this script covers
-----------------------
1) Basic increasing/decreasing trends
2) Spearman (rank) vs Pearson (linear) correlation checks
3) Using a Predicate as a mask (e.g., subset of rows)
4) Thresholds on |rho| and minimum sample size
5) Edge cases: constant series, NaNs, small n
6) Unicode vs ASCII pretty() output

Run with:
    PYTHONPATH=src python qualitative_module_examples.py
"""

from __future__ import annotations
import math
import pandas as pd
import numpy as np

from txgraffiti2025.forms.qualitative import MonotoneRelation
from txgraffiti2025.forms.predicates import Predicate


# ------------------------------------------------------------------------------
# Pretty printing helpers
# ------------------------------------------------------------------------------

def banner(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def show(title: str, rel: MonotoneRelation, df: pd.DataFrame, *, mask=None):
    print(f"\n[{title}] {rel.pretty(unicode_ops=True)}")
    res = rel.evaluate_global(df, mask=mask)
    print("-" * 78)
    print("result dict:")
    # Keep a stable order for readability
    keys = ["ok", "rho", "direction", "method", "n", "x", "y"]
    for k in keys:
        print(f"  {k:<10}: {res[k]}")
    if mask is not None:
        print("  mask      : provided")
    print("ASCII pretty:", rel.pretty(unicode_ops=False))


# ------------------------------------------------------------------------------
# 0) Synthetic data setup
# ------------------------------------------------------------------------------

def build_dataframes():
    # Strictly increasing x; monotone nonlinear y=f(x)=x^2 (good for Spearman)
    x = np.arange(0, 11, dtype=float)               # 0..10
    y_inc_nonlinear = x**2                          # monotone increasing

    # Decreasing linear relation
    y_dec_linear = 10.0 - x                         # strictly decreasing linear

    # Mixed data with NaNs sprinkled in
    y_inc_with_nans = y_inc_nonlinear.astype(float).copy()
    y_inc_with_nans[[2, 7]] = np.nan

    # Constant series (edge case)
    y_constant = np.ones_like(x) * 3.14

    # Small sample (n=1 or n=2) to test min_n behavior
    x_small = np.array([0.0, 1.0])
    y_small_inc = np.array([0.0, 2.0])

    # Boolean mask column defining a subset; here we keep even indices
    subset_mask = (x % 2 == 0)

    df = pd.DataFrame({
        "x": x,
        "y_inc_nonlinear": y_inc_nonlinear,
        "y_dec_linear": y_dec_linear,
        "y_inc_with_nans": y_inc_with_nans,
        "y_constant": y_constant,
        "use_subset": subset_mask,
    }, index=[f"G{i}" for i in range(len(x))])

    df_small = pd.DataFrame({
        "x": x_small,
        "y_small_inc": y_small_inc,
    }, index=["H1", "H2"])

    return df, df_small


# ------------------------------------------------------------------------------
# 1) Basic increasing / decreasing
# ------------------------------------------------------------------------------

def demo_basic(df: pd.DataFrame):
    banner("1) Basic increasing / decreasing")

    # Spearman is robust to monotone nonlinear trends (e.g., x^2)
    rel_A = MonotoneRelation("x", "y_inc_nonlinear", direction="increasing",
                             method="spearman", min_abs_rho=0.8)
    show("A: y_inc_nonlinear increases with x (Spearman)", rel_A, df)

    # Pearson for linear decreasing
    rel_B = MonotoneRelation("x", "y_dec_linear", direction="decreasing",
                             method="pearson", min_abs_rho=0.8)
    show("B: y_dec_linear decreases with x (Pearson)", rel_B, df)


# ------------------------------------------------------------------------------
# 2) Spearman vs Pearson contrast
# ------------------------------------------------------------------------------

def demo_spearman_vs_pearson(df: pd.DataFrame):
    banner("2) Spearman vs Pearson (nonlinear monotone example)")

    # Nonlinear monotone increasing: Pearson still high but Spearman is the safer choice
    rel_S = MonotoneRelation("x", "y_inc_nonlinear", direction="increasing",
                             method="spearman", min_abs_rho=0.9)
    rel_P = MonotoneRelation("x", "y_inc_nonlinear", direction="increasing",
                             method="pearson", min_abs_rho=0.9)

    show("S: Spearman on x vs x^2", rel_S, df)
    show("P: Pearson on x vs x^2", rel_P, df)


# ------------------------------------------------------------------------------
# 3) Using a Predicate as a mask
# ------------------------------------------------------------------------------

def demo_with_predicate_mask(df: pd.DataFrame):
    banner("3) Restricting to a class via Predicate mask")

    # Keep only rows where use_subset is True (even indices)
    P_use = Predicate.from_column("use_subset")

    rel_C = MonotoneRelation("x", "y_inc_nonlinear", direction="increasing",
                              method="spearman", min_abs_rho=0.9)
    show("C: increasing on subset (Predicate mask)", rel_C, df, mask=P_use)

    rel_D = MonotoneRelation("x", "y_dec_linear", direction="decreasing",
                              method="pearson", min_abs_rho=0.9)
    show("D: decreasing on subset (Predicate mask)", rel_D, df, mask=P_use)


# ------------------------------------------------------------------------------
# 4) Threshold and min_n effects
# ------------------------------------------------------------------------------

def demo_thresholds_and_min_n(df: pd.DataFrame, df_small: pd.DataFrame):
    banner("4) Threshold (|rho| ≥ τ) and minimum sample size (min_n)")

    # Same relation with different thresholds
    rel_low_tau  = MonotoneRelation("x", "y_inc_nonlinear", "increasing", "spearman", min_abs_rho=0.5)
    rel_high_tau = MonotoneRelation("x", "y_inc_nonlinear", "increasing", "spearman", min_abs_rho=0.99)
    show("E: low threshold τ=0.5", rel_low_tau, df)
    show("F: high threshold τ=0.99", rel_high_tau, df)

    # Minimum n: with only 2 points, min_n=2 passes; min_n=3 fails early
    rel_min2 = MonotoneRelation("x", "y_small_inc", "increasing", "pearson", min_abs_rho=0.8, min_n=2)
    rel_min3 = MonotoneRelation("x", "y_small_inc", "increasing", "pearson", min_abs_rho=0.8, min_n=3)
    show("G: small sample (n=2), min_n=2", rel_min2, df_small)
    show("H: small sample (n=2), min_n=3 (should fail due to n<3)", rel_min3, df_small)


# ------------------------------------------------------------------------------
# 5) Edge cases: constants and NaNs
# ------------------------------------------------------------------------------

def demo_edge_cases(df: pd.DataFrame):
    banner("5) Edge cases: constants and NaNs")

    # Constant y: correlation defined as 0.0 by design -> likely not ok unless min_abs_rho==0 and direction allows
    rel_const = MonotoneRelation("x", "y_constant", "increasing", "pearson", min_abs_rho=0.0)
    show("I: constant y (Pearson)", rel_const, df)

    # NaNs are dropped internally; result depends on remaining pairs
    rel_nans = MonotoneRelation("x", "y_inc_with_nans", "increasing", "spearman", min_abs_rho=0.9)
    show("J: NaNs in y (Spearman)", rel_nans, df)


# ------------------------------------------------------------------------------
# 6) Unicode vs ASCII pretty strings
# ------------------------------------------------------------------------------

def demo_pretty_variants(df: pd.DataFrame):
    banner("6) Unicode vs ASCII pretty() examples")

    rel_u = MonotoneRelation("x", "y_inc_nonlinear", "increasing", "spearman", min_abs_rho=0.7)
    print("Unicode:", rel_u.pretty(unicode_ops=True,  show_threshold=True))
    print("ASCII  :", rel_u.pretty(unicode_ops=False, show_threshold=True))
    print("Signature (stable):", rel_u.signature())


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    df, df_small = build_dataframes()

    banner("0) DataFrames")
    print("Full df:")
    print(df.head(12))   # all rows here, but keep consistent style
    print("\nSmall df (for min_n):")
    print(df_small)

    demo_basic(df)
    demo_spearman_vs_pearson(df)
    demo_with_predicate_mask(df)
    demo_thresholds_and_min_n(df, df_small)
    demo_edge_cases(df)
    demo_pretty_variants(df)


if __name__ == "__main__":
    main()
