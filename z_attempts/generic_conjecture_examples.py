# generic_conjecture_examples.py
"""
Educational walkthrough for txgraffiti2025.forms.generic_conjecture

This script demonstrates how to build mathematical conjectures of the form
(C) ⇒ R over a pandas DataFrame, using Relation primitives (Le/Ge/Eq, AllOf/AnyOf)
and Predicate-based class conditions.

Run:
    PYTHONPATH=src python generic_conjecture_examples.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# --- forms layer ---
from txgraffiti2025.forms.utils import to_expr, LinearForm, floor, log, sqrt
from txgraffiti2025.forms.generic_conjecture import (
    Eq, Le, Ge, AllOf, AnyOf, Conjecture, TRUE
)
from txgraffiti2025.forms.predicates import Predicate, Where, AndPred

# ----------------------------
# small print helpers
# ----------------------------
def h1(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def h2(title: str):
    print("\n" + title)
    print("-" * len(title))

def show_conjecture_eval(title: str, conj: Conjecture, df: pd.DataFrame, *, auto_base=True):
    print(f"\n[{title}]")
    print("pretty():", conj.pretty())
    applicable, holds, failures = conj.check(df, auto_base=auto_base)
    print("applicable.sum():", int(applicable.sum()))
    print("holds among applicable:", bool(holds[applicable].all()))
    print("touch_count():", conj.touch_count(df, auto_base=auto_base))
    print("violation_count():", conj.violation_count(df, auto_base=auto_base))
    if not failures.empty:
        print("\n-- failures (head) --")
        print(failures.head().to_string())
    print("signature():", conj.signature())

def main():
    # ------------------------------------------------------------------
    # 0) Example DataFrame
    # ------------------------------------------------------------------
    h1("0) DataFrame")
    # toy invariant table; imagine rows are graphs with named invariants
    df = pd.DataFrame({
        "order":   [3,  4,  5,  6,  8, 10],
        "size":    [2,  3,  4,  5,  8, 12],
        "alpha":   [1,  2,  2,  3,  3,  4],  # independence number
        "mu":      [1,  2,  3,  3,  4,  5],  # matching number
        "regular": [1,  1,  0,  0,  1,  0],  # bool-ish class flags
        "planar":  [1,  1,  1,  0,  0,  0],
    }, index=[f"G{i}" for i in range(1, 7)]).astype({
        "regular": "boolean",
        "planar": "boolean",
    })
    print(df)

    # ------------------------------------------------------------------
    # 1) Relations R: Le / Ge / Eq (row-wise predicates with slack)
    # ------------------------------------------------------------------
    h1("1) Relation primitives (Le, Ge, Eq)")
    # Example: alpha <= mu  (classic style bound)
    r1 = Le("alpha", "mu")

    # Example: alpha >= order/(max_deg+1) surrogate -> use a simpler RHS here:
    # We'll just show sqrt/log to demonstrate non-linear exprs on the RHS:
    r2 = Ge("alpha", floor(to_expr("order") / 3))

    # Equality with tolerance
    r3 = Eq(LinearForm(0, [(1.0, "size")]), to_expr("order") - 1, tol=1e-9)

    # Check evaluate/slack directly (row-wise)
    print("\nr1.evaluate (alpha <= mu):")
    print(r1.evaluate(df).to_string())
    print("r1.slack (mu - alpha):")
    print(r1.slack(df).to_string())

    # ------------------------------------------------------------------
    # 2) Conjecture: (R | C), i.e., 'For all rows in class C, R holds'
    # ------------------------------------------------------------------
    h1("2) Conjectures: (R | C)")

    # Build a class predicate C using Where(lambda df: mask)
    C_regular: Predicate = Where(lambda d: d["regular"], name="(regular)")
    C_planar:  Predicate = Where(lambda d: d["planar"],  name="(planar)")

    # (A) On regular graphs: alpha <= mu
    conj_A = Conjecture(r1, condition=C_regular, name="alpha_le_mu_on_regular")
    show_conjecture_eval("A: (regular) ⇒ alpha ≤ mu", conj_A, df)

    # (B) On planar graphs: alpha ≥ floor(order/3)
    conj_B = Conjecture(r2, condition=C_planar, name="alpha_ge_floor_order_over_3_on_planar")
    show_conjecture_eval("B: (planar) ⇒ alpha ≥ floor(order/3)", conj_B, df)

    # (C) Equality: size = order - 1 with small tol, no explicit condition
    #     Here, auto_base will use the conjunction of always-True boolean columns (if any)
    conj_C = Conjecture(r3, condition=None, name="size_eq_order_minus_1")
    show_conjecture_eval("C: (auto-base) ⇒ size = order - 1 (±tol)", conj_C, df, auto_base=True)

    # ------------------------------------------------------------------
    # 3) Composition: AllOf / AnyOf (and &, | sugar)
    # ------------------------------------------------------------------
    h1("3) Composing relations (AllOf / AnyOf)")

    # AllOf: (alpha ≤ mu) ∧ (alpha ≥ floor(order/3))
    r_both = r1 & r2   # AllOf([r1, r2])
    conj_D = Conjecture(r_both, condition=C_planar & C_regular, name="planar_and_regular_alpha_bounds")
    show_conjecture_eval("D: (planar ∧ regular) ⇒ (alpha ≤ mu ∧ alpha ≥ floor(order/3))", conj_D, df)

    # AnyOf: r1 ∨ r3 (either alpha ≤ mu OR size ≈ order-1)
    r_either = r1 | r3
    conj_E = Conjecture(r_either, condition=TRUE, name="either_alpha_le_mu_or_tree_like_size")
    show_conjecture_eval("E: TRUE ⇒ (alpha ≤ mu) ∨ (size ≈ order−1)", conj_E, df)

    # ------------------------------------------------------------------
    # 4) Tightness, violations, and failures report
    # ------------------------------------------------------------------
    h1("4) Tightness & failures")

    # Make a bound with obvious tight rows: alpha ≥ floor(order/2) will usually fail
    bound = Ge("alpha", floor(to_expr("order") / 2))
    conj_F = Conjecture(bound, condition=TRUE, name="alpha_ge_floor(order/2)")
    app, holds, fails = conj_F.check(df)
    tight = bound.is_tight(df)

    print("\nF: TRUE ⇒ alpha ≥ floor(order/2)")
    print("pretty():", conj_F.pretty())
    print("applicable rows:", int(app.sum()))
    print("holds on applicable:", bool(holds[app].all()))
    print("tight rows (head):")
    print(tight.head().to_string())
    if not fails.empty:
        print("\n-- failures with __slack__ (head) --")
        print(fails.head().to_string())

    # ------------------------------------------------------------------
    # 5) Pretty and signature
    # ------------------------------------------------------------------
    h1("5) pretty() and signature() examples")
    print("A.pretty():", conj_A.pretty())
    print("B.pretty():", conj_B.pretty())
    print("C.pretty():", conj_C.pretty())
    print("D.pretty():", conj_D.pretty())
    print("E.pretty():", conj_E.pretty())

    print("\nSignatures (canonical-ish strings for hashing/dedup):")
    for cj in [conj_A, conj_B, conj_C, conj_D, conj_E, conj_F]:
        print(f" - {cj.name}: {cj.signature()}")

    # ------------------------------------------------------------------
    # 6) Notes on auto_base
    # ------------------------------------------------------------------
    h1("6) Auto-base note")
    print(
        "If a conjecture has condition=None and auto_base=True, the system "
        "conjoins all boolean columns that are identically True to create a base "
        "predicate. You can disable this by calling check(..., auto_base=False)."
    )

    print("\nAll done. Tweak the DataFrame or relations and re-run to explore!")

if __name__ == "__main__":
    main()
