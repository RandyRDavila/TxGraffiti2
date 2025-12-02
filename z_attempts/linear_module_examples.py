# linear_module_examples.py
"""
Demonstration of txgraffiti2025.forms.linear

Covers:
  1) Building LinearForm expressions (sequence & dict inputs)
  2) Constructing Le / Ge / Eq relations from linear forms
  3) Slack intuition and evaluation on a small DataFrame
  4) Aliases (lin, leq, geq) and from_dict helper
  5) Composition with Conjecture + pretty printing (optional)

Run:
  PYTHONPATH=src python linear_module_examples.py
"""

from __future__ import annotations
import pandas as pd

from txgraffiti2025.forms.linear import (
    linear_expr, linear_le, linear_ge, linear_eq,
    lin, leq, geq, from_dict,
)
from txgraffiti2025.forms.generic_conjecture import Conjecture, TRUE
from txgraffiti2025.forms.utils import to_expr

# ---------------------------
# Display helpers
# ---------------------------
def H(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def show_expr(label: str, e, df: pd.DataFrame, head: int = None):
    print(f"\n[{label}]")
    print("repr:", repr(e))
    vals = e.eval(df)
    print("eval head:")
    print(vals if head is None else vals.head(head))

def show_rel(label: str, R, df: pd.DataFrame):
    print(f"\n[{label}]")
    print("repr:", repr(R))
    mask = R.evaluate(df)
    print("evaluate:", mask.astype(int).to_string())
    print("slack head:")
    print(R.slack(df).head())


def show_conj(label: str, R, df: pd.DataFrame):
    # Optional: wrap the relation into a Conjecture (TRUE ⇒ R)
    conj = Conjecture(R, condition=TRUE)
    applicable, holds, failures = conj.check(df, auto_base=False)
    print(f"\n[{label}] (as Conjecture)")
    print("pretty():", conj.pretty())  # thanks to generic_conjecture.pretty
    print("applicable.sum():", int(applicable.sum()))
    print("holds among applicable:", bool(holds[applicable].all()))
    print("failures:\n", failures if not failures.empty else "(none)")


def main():
    H("0) DataFrame")
    df = pd.DataFrame({
        "order": [3, 4, 5, 6, 8, 10],
        "size":  [2, 3, 4, 5, 8, 12],
        "alpha": [1, 2, 3, 3, 3, 5],
        "mu":    [1, 2, 2, 3, 4, 4],
        "flag":  [True, True, False, False, True, False],
    }, index=[f"G{i}" for i in range(1, 7)])
    print(df)

    # ------------------------------------------------------------------
    H("1) LinearForm expressions")
    e1 = linear_expr(1.0, [(2.0, "alpha"), (-1.0, "mu")])              # 1 + 2·alpha - mu
    e2 = linear_expr(0.0, {"order": 1.0, "size": -1.0})                 # order - size
    e3 = from_dict(0.5, order=0.5, mu=2.0)                              # 0.5 + 0.5·order + 2·mu

    show_expr("e1 = 1 + 2·alpha - mu", e1, df)
    show_expr("e2 = order - size", e2, df)
    show_expr("e3 = 0.5 + 0.5·order + 2·mu", e3, df)

    # Demonstrate deterministic normalization/merging:
    e_dup = linear_expr(0.0, [(1.0, "alpha"), (2.0, "alpha"), (-3.0, "mu"), (3.0, "mu")])
    show_expr("e_dup (merged & zero-dropped) = 3·alpha + 0·mu", e_dup, df)

    # ------------------------------------------------------------------
    H("2) Relations from linear forms (≤, ≥, ≈)")
    R_le = linear_le(0.0, {"alpha": 2.0, "mu": -1.0}, 4)    # 2·alpha - mu ≤ 4
    R_ge = linear_ge(1.0, [(-1.0, "size")], -10)            # 1 - size ≥ -10  ⇔ size ≤ 11
    R_eq = linear_eq(0.0, {"order": 1.0, "size": -1.0}, 1)  # order - size ≈ 1

    show_rel("R_le: 2·alpha - mu ≤ 4", R_le, df)
    show_rel("R_ge: 1 - size ≥ -10", R_ge, df)
    show_rel("R_eq: order - size ≈ 1 (tol=1e-9)", R_eq, df)

    # ------------------------------------------------------------------
    H("3) Aliases and to_expr interop")
    # Same as above, via aliases
    R_le2 = leq(0.0, dict(alpha=2.0, mu=-1.0), 4)
    R_ge2 = geq(1.0, [(-1.0, "size")], -10)
    R_eq2 = linear_eq(0.0, [ (1.0, "order"), (-1.0, "size") ], to_expr(1))  # right can be Expr/col/scalar

    show_rel("R_le2 (alias): 2·alpha - mu ≤ 4", R_le2, df)
    show_rel("R_ge2 (alias): 1 - size ≥ -10", R_ge2, df)
    show_rel("R_eq2: order - size ≈ 1", R_eq2, df)

    # ------------------------------------------------------------------
    H("4) Wrap as Conjectures (for reporting / pretty())")
    show_conj("C1: TRUE ⇒ (2·alpha - mu ≤ 4)", R_le, df)
    show_conj("C2: TRUE ⇒ (1 - size ≥ -10)", R_ge, df)
    show_conj("C3: TRUE ⇒ (order - size ≈ 1)", R_eq, df)


if __name__ == "__main__":
    main()
