# implication_module_examples.py
"""
Educational walkthrough for txgraffiti2025.forms.implication

What this demonstrates:
- Implication:    R1 ⇒ R2   (with/without conditions)
- Equivalence:    R1 ⇔ R2   (with/without conditions)
- Vacuous truth outside the condition
- Failures table (with "__slack__" for Implication, "__lhs__/__rhs__" for Equivalence)
- Helper stats: violation_count(), touch_count(), signature()

Run:
    PYTHONPATH=src python implication_module_examples.py
"""

from __future__ import annotations
import pandas as pd

from txgraffiti2025.forms.utils import to_expr, floor
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.generic_conjecture import (
    Le as RelLe, Ge as RelGe, Eq as RelEq, AllOf, AnyOf
)
from txgraffiti2025.forms.implication import Implication, Equivalence


# ---------------------------
# Pretty console helpers
# ---------------------------
def H1(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def H2(title: str):
    print("\n" + title)
    print("-" * len(title))

def show_imp(title: str, impl: Implication, df: pd.DataFrame, show_failures: bool = True):
    print(f"\n[{title}]")
    print("pretty():", impl.pretty())        # unicode arrows by default
    applicable, holds, failures = impl.check(df)
    print("applicable.sum():", int(applicable.sum()))
    print("holds among applicable:", bool(holds[applicable].all()))
    print("violation_count():", impl.violation_count(df))
    print("touch_count():", impl.touch_count(df))
    print("signature():", impl.signature())
    if show_failures and not failures.empty:
        print("-- failures --")
        print(failures.to_string())

def show_eqv(title: str, eqv: Equivalence, df: pd.DataFrame, show_failures: bool = True):
    print(f"\n[{title}]")
    print("pretty():", eqv.pretty())         # unicode ⇔, and condition prefix ⇒
    applicable, holds, failures = eqv.check(df)
    print("applicable.sum():", int(applicable.sum()))
    print("holds among applicable:", bool(holds[applicable].all()))
    print("violation_count():", eqv.violation_count(df))
    print("touch_count():", eqv.touch_count(df))
    print("signature():", eqv.signature())
    if show_failures and not failures.empty:
        print("-- failures --")
        print(failures.to_string())


def main():
    # ----------------------------------------------------------------
    # 0) DataFrame
    # ----------------------------------------------------------------
    H1("0) DataFrame")
    df = pd.DataFrame({
        "order":   [3,  4,  5,  6,  8, 10,  7],
        "size":    [2,  3,  4,  5,  8, 12,  6],
        "alpha":   [1,  2,  3,  3,  3,  5,  4],
        "mu":      [1,  2,  2,  3,  4,  4,  3],
        "regular": [True, True, False, False, True, False, True],
        "planar":  [True, True, True,  False, False, False, True],
        "rare":    [False]*7,   # for a vacuous-truth demo
    }, index=[f"G{i}" for i in range(1, 8)])
    print(df)

    # Common predicates (conditions)
    P_regular = Predicate.from_column("regular")
    P_planar  = Predicate.from_column("planar")
    P_rare    = Predicate.from_column("rare")   # all False ⇒ vacuous

    # Common relations (premises/conclusions)
    R_le_alpha_mu  = RelLe("alpha", "mu")                                       # alpha ≤ mu
    R_ge_mu_alpha  = RelGe("mu", "alpha")                                       # mu ≥ alpha
    R_ge_alpha_flo = RelGe("alpha", floor(to_expr("order") / 3))                # alpha ≥ ⌊order/3⌋
    R_ge_size_ord  = RelGe("size", "order")                                     # size ≥ order
    R_eq_size_ordm = RelEq("size", to_expr("order") - 1)                        # size = order − 1
    R_le_size_ord  = RelLe("size", "order")                                     # size ≤ order

    # ----------------------------------------------------------------
    # 1) Global implications
    # ----------------------------------------------------------------
    H1("1) Implications (global / no condition)")
    # Tautological implication: (alpha ≤ mu) ⇒ (mu ≥ alpha)
    impl_A = Implication(R_le_alpha_mu, R_ge_mu_alpha)
    show_imp("A: (alpha ≤ mu) ⇒ (mu ≥ alpha)", impl_A, df)

    # False-ish implication to show failures: (alpha ≤ mu) ⇒ (size ≥ order)
    impl_B = Implication(R_le_alpha_mu, R_ge_size_ord)
    show_imp("B: (alpha ≤ mu) ⇒ (size ≥ order)", impl_B, df)

    # ----------------------------------------------------------------
    # 2) Implications under a condition (planar ∧ regular)
    # ----------------------------------------------------------------
    H1("2) Implications under a condition (planar ∧ regular)")
    cond = P_planar & P_regular
    # (planar ∧ regular) ⇒ (alpha ≤ mu ∧ alpha ≥ ⌊order/3⌋)
    impl_C = Implication(
        premise=R_le_alpha_mu,
        conclusion=AllOf([R_le_alpha_mu, R_ge_alpha_flo]),
        condition=cond
    )
    show_imp("C: (planar ∧ regular) ⇒ (alpha ≤ mu ∧ alpha ≥ ⌊order/3⌋)", impl_C, df)

    # Vacuous truth: condition 'rare' is False everywhere
    impl_D = Implication(R_le_alpha_mu, R_ge_size_ord, condition=P_rare)
    show_imp("D: (rare) ⇒ (size ≥ order)  [vacuous]", impl_D, df)

    # ----------------------------------------------------------------
    # 3) Equivalences (global and conditional)
    # ----------------------------------------------------------------
    H1("3) Equivalences")
    # Tautological equivalence: (alpha ≤ mu) ⇔ (mu ≥ alpha)
    eqv_E = Equivalence(R_le_alpha_mu, R_ge_mu_alpha)
    show_eqv("E: (alpha ≤ mu) ⇔ (mu ≥ alpha)", eqv_E, df)

    # Non-equivalence demo (should have failures):
    # size = order−1  ⇔  size ≤ order    (only one direction true)
    eqv_F = Equivalence(R_eq_size_ordm, R_le_size_ord)
    show_eqv("F: (size = order−1) ⇔ (size ≤ order)", eqv_F, df)

    # Conditional equivalence: same as F but only on planar rows
    eqv_G = Equivalence(R_eq_size_ordm, R_le_size_ord, condition=P_planar)
    show_eqv("G: (planar) ⇒ ((size = order−1) ⇔ (size ≤ order))", eqv_G, df)

    # ----------------------------------------------------------------
    # 4) Implication chained body (R1 ⇒ (R2 ∨ R3)) under condition
    # ----------------------------------------------------------------
    H1("4) Implication with composite RHS under a condition")
    impl_H = Implication(
        premise=R_le_alpha_mu,
        conclusion=AnyOf([R_ge_alpha_flo, R_eq_size_ordm]),
        condition=P_planar
    )
    show_imp("H: (planar) ⇒ [ (alpha ≤ mu) ⇒ (alpha ≥ ⌊order/3⌋ ∨ size = order−1) ]", impl_H, df)

    print("\nDone. Tweak the DataFrame (e.g., swap alpha/mu, change size/order) and re-run to explore.")

if __name__ == "__main__":
    main()
