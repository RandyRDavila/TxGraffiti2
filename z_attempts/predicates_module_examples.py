# predicates_module_examples.py
"""
Educational walkthrough for txgraffiti2025.forms.predicates

What this demonstrates:
- Column-based predicates and composition with &, |, ~ (unicode ∧, ∨, ¬)
- Vectorized comparisons: LT, LE, GT, GE, EQ, NE (+ handy shorthands)
- Set/range predicates with Unicode reprs:
    [mu ∈ {2, 3, 4}], [4 ≤ order ≤ 8], etc.
- Numeric property predicates:
    [order ∈ ℤ], [isnan(alpha)], [|size / order| < ∞]
- Functional predicates: Where (vectorized), RowWhere (row-wise)
- Quantifier-style semantic sugar:
    (∀ : |expr| < ∞), (∃ : expr → ∞) with optional symbols
- Using Predicates as CONDITIONS in Conjectures

Run:
    PYTHONPATH=src python predicates_module_examples.py
"""

from __future__ import annotations
import pandas as pd

from txgraffiti2025.forms.utils import to_expr, floor
from txgraffiti2025.forms.predicates import (
    Predicate, AndPred, OrPred, NotPred,
    Compare, LT, LE, GT, GE, EQ, NE,
    InSet, Between,
    IsInteger, IsNaN, IsFinite,
    Where, RowWhere,
    ForallFinite, ExistsDivergent,
    GEQ, LEQ, GT0, LT0, EQ0, BETWEEN, IN, IS_INT, IS_NAN, IS_FINITE
)
from txgraffiti2025.forms.generic_conjecture import (
    Conjecture, Eq as RelEq, Le as RelLe, Ge as RelGe, AllOf, AnyOf, TRUE
)

# ---------------------------
# Pretty printing helpers
# ---------------------------
def H1(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def H2(title: str):
    print("\n" + title)
    print("-" * len(title))

def show_mask(title: str, pred: Predicate, df: pd.DataFrame):
    print(f"\n[{title}] {pred!r}")
    print(pred.mask(df).to_string())

def show_conj(title: str, conj: Conjecture, df: pd.DataFrame):
    print(f"\n[{title}]")
    print("pretty():", conj.pretty())  # relies on your unicode-aware pretty in generic_conjecture.py
    applicable, holds, failures = conj.check(df)
    print("applicable.sum():", int(applicable.sum()))
    print("holds among applicable:", bool(holds[applicable].all()))
    if not failures.empty:
        print("-- failures --")
        print(failures.to_string())


def main():
    # ----------------------------------------------------------------
    # 0) DataFrame (includes a divergent row for ExistsDivergent demo)
    # ----------------------------------------------------------------
    H1("0) DataFrame")
    df = pd.DataFrame({
        "order":   [3,  4,  5,  6,  8, 10,  7],
        "size":    [2,  3,  4,  5,  8, 12,  0],  # size=0 on last row → division by zero
        "alpha":   [1,  2,  2,  3,  3,  4,  1],
        "mu":      [1,  2,  3,  3,  4,  5,  2],
        "regular": [True, True, False, False, True, False, True],
        "planar":  [True, True, True,  False, False, False, True],
    }, index=[f"G{i}" for i in range(1, 8)])
    print(df)

    # ----------------------------------------------------------------
    # 1) Column-based predicates via from_column()
    # ----------------------------------------------------------------
    H1("1) Column predicates via Predicate.from_column(...)")
    P_regular = Predicate.from_column("regular")
    P_planar  = Predicate.from_column("planar")
    show_mask("regular", P_regular, df)
    show_mask("planar", P_planar, df)
    show_mask("~regular (negation)", ~P_regular, df)
    show_mask("(planar ∧ regular)", P_planar & P_regular, df)
    show_mask("(planar ∨ regular)", P_planar | P_regular, df)

    # ----------------------------------------------------------------
    # 2) Vectorized comparisons (and shorthands)
    # ----------------------------------------------------------------
    H1("2) Vectorized comparisons (LT/LE/GT/GE/EQ/NE) + shorthands")
    show_mask("alpha ≤ mu", LE("alpha", "mu"), df)
    show_mask("alpha ≥ floor(order/3)", GE("alpha", floor(to_expr("order") / 3)), df)
    show_mask("alpha < mu", LT("alpha", "mu"), df)
    show_mask("alpha > mu", GT("alpha", "mu"), df)
    show_mask("alpha == mu", EQ("alpha", "mu"), df)
    show_mask("alpha != mu", NE("alpha", "mu"), df)

    H2("Shorthands")
    show_mask("GEQ(order, 5)", GEQ("order", 5), df)
    show_mask("LEQ(size, 5)", LEQ("size", 5), df)
    show_mask("GT0(mu)", GT0("mu"), df)
    show_mask("LT0(alpha - mu)", LT0(to_expr("alpha") - to_expr("mu")), df)
    show_mask("EQ0((alpha - mu), tol=0)", EQ0(to_expr("alpha") - to_expr("mu"), tol=0), df)

    # ----------------------------------------------------------------
    # 3) Set membership / range (Unicode reprs)
    # ----------------------------------------------------------------
    H1("3) Set membership and Between (Unicode reprs)")
    show_mask("mu ∈ {2,3,4}", IN("mu", {2,3,4}), df)
    show_mask("4 ≤ order ≤ 8", BETWEEN("order", 4, 8, inc_lo=True, inc_hi=True), df)
    show_mask("4 < order ≤ 8 (left-open interval)", BETWEEN("order", 4, 8, inc_lo=False, inc_hi=True), df)
    show_mask("4 ≤ order < 8 (right-open interval)", BETWEEN("order", 4, 8, inc_lo=True, inc_hi=False), df)

    # ----------------------------------------------------------------
    # 4) Numeric properties (ℤ, isnan, finite)
    # ----------------------------------------------------------------
    H1("4) Numeric properties: IsInteger, IsNaN, IsFinite")
    show_mask("order ∈ ℤ", IS_INT("order"), df)

    df_nan = df.copy()
    df_nan.loc["G2", "alpha"] = float("nan")
    show_mask("isnan(alpha) [with G2 alpha=NaN]", IS_NAN("alpha"), df_nan)

    ratio = to_expr("size") / to_expr("order")
    show_mask("|size / order| < ∞", IS_FINITE(ratio), df)  # all rows finite here

    blowup = to_expr("order") / to_expr("size")            # div by zero at G7
    show_mask("|order / size| < ∞", IS_FINITE(blowup), df) # shows False at G7

    # ----------------------------------------------------------------
    # 5) Functional predicates: Where (vectorized), RowWhere (row-wise)
    # ----------------------------------------------------------------
    H1("5) Functional predicates: Where / RowWhere")
    P_even_order = Where(lambda d: (d["order"] % 2 == 0), name="(order even)")
    P_big_size   = Where(lambda d: (d["size"] >= 5),       name="(size ≥ 5)")
    show_mask("Where: (order even)", P_even_order, df)
    show_mask("Where: (size ≥ 5)", P_big_size, df)

    P_row_custom = RowWhere(lambda row: (row["alpha"] + row["mu"]) <= row["order"], name="(alpha+mu ≤ order)")
    show_mask("RowWhere: (alpha+mu ≤ order)", P_row_custom, df)

    show_mask("(order even) ∧ (size ≥ 5)", P_even_order & P_big_size, df)
    show_mask("¬(alpha+mu ≤ order)", ~P_row_custom, df)

    # ----------------------------------------------------------------
    # 6) Quantifier-style predicates (semantic sugar)
    #     (∀ : |expr| < ∞) and (∃ : expr → ∞) with optional symbol
    # ----------------------------------------------------------------
    H1("6) Quantifier-style predicates (∀, ∃, ∞)")
    show_mask("ForallFinite(size/order)", ForallFinite(ratio), df)
    show_mask("ExistsDivergent(order/size)", ExistsDivergent(blowup), df)
    show_mask("ExistsDivergent(order/size, symbol='G')", ExistsDivergent(blowup, symbol="G"), df)

    # ----------------------------------------------------------------
    # 7) Using predicates as CONDITIONS inside Conjectures
    # ----------------------------------------------------------------
    H1("7) Predicates as conditions in Conjectures")
    # (planar ∧ regular) ⇒ (alpha ≤ mu) ∧ (alpha ≥ ⌊order/3⌋)
    rel_le = RelLe("alpha", "mu")
    rel_ge = RelGe("alpha", floor(to_expr("order") / 3))
    conj_bounds = Conjecture(AllOf([rel_le, rel_ge]), condition=P_planar & P_regular)
    show_conj("D: (planar ∧ regular) ⇒ (alpha ≤ mu ∧ alpha ≥ ⌊order/3⌋)", conj_bounds, df)

    # TRUE ⇒ (alpha ≤ mu) ∨ (size = order − 1)
    conj_either = Conjecture(AnyOf([RelLe("alpha","mu"), RelEq("size", to_expr("order") - 1)]), condition=TRUE)
    show_conj("E: TRUE ⇒ (alpha ≤ mu) ∨ (size = order − 1)", conj_either, df)

    # Condition can be any Predicate; e.g., only where the ratio is finite:
    conj_under_finite_ratio = Conjecture(RelLe("alpha", "mu"), condition=IS_FINITE(ratio))
    show_conj("F: (|size/order| < ∞) ⇒ (alpha ≤ mu)", conj_under_finite_ratio, df)

    print("\nDone. Tweak the DataFrame (e.g., set size=0, alpha=NaN) and re-run to explore.")

if __name__ == "__main__":
    main()
