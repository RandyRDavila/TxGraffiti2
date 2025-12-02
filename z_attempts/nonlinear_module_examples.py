# nonlinear_module_examples.py
"""
Demonstration of txgraffiti2025.forms.nonlinear

Covers:
  1) Products (binary & n-ary)
  2) Powers, nth_root, square, cube
  3) Ratios (plain & safe), reciprocals (plain & guarded)
  4) Geometric mean
  5) Using nonlinear expressions inside relations & conjectures

Run:
  PYTHONPATH=src python nonlinear_module_examples.py
"""

from __future__ import annotations
import pandas as pd

from txgraffiti2025.forms.nonlinear import (
    product, product_n,
    power, square, cube, nth_root,
    ratio, safe_ratio, reciprocal, reciprocal_eps,
    geometric_mean,
)
from txgraffiti2025.forms.utils import to_expr
from txgraffiti2025.forms.generic_conjecture import Le, Ge, Eq, Conjecture, TRUE


# -------------------------------------------------------------
# Small print helpers
# -------------------------------------------------------------
def H(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def show_expr(label: str, e, df: pd.DataFrame, head: int | None = 6):
    print(f"\n[{label}]")
    print("repr(expr):", repr(e))
    vals = e.eval(df)
    print("eval(expr) head:")
    print(vals if head is None else vals.head(head))

def show_rel(label: str, R, df: pd.DataFrame):
    print(f"\n[{label}]")
    print("repr(relation):", repr(R))
    m = R.evaluate(df)
    print("evaluate mask:")
    print(m.astype(int))
    print("slack head:")
    print(R.slack(df).head())

def show_conj(label: str, conj: Conjecture, df: pd.DataFrame):
    print(f"\n[{label}]")
    print("pretty():", conj.pretty())
    applicable, holds, failures = conj.check(df, auto_base=False)
    print("applicable.sum():", int(applicable.sum()))
    print("holds among applicable:", bool(holds[applicable].all()))
    if failures.empty:
        print("failures: (none)")
    else:
        print("failures:\n", failures)


def main():
    # ---------------------------------------------------------
    H("0) DataFrame")
    df = pd.DataFrame({
        "order": [5, 10, 12, 20, 30, 0],
        "alpha": [2, 3, 4, 5, 6, 0],
        "omega": [2, 1, 3, 2, 1, 0],
        "mu":    [1, 2, 2, 3, 5, 0],
    }, index=[f"G{i}" for i in range(1, 7)])
    print(df)

    # ---------------------------------------------------------
    H("1) Products")
    show_expr("α·ω", product("alpha", "omega"), df)
    show_expr("product_n([alpha, omega, mu])", product_n(["alpha", "omega", "mu"]), df)

    # ---------------------------------------------------------
    H("2) Powers & roots")
    show_expr("power(order, 0.5) = √order", power("order", 0.5), df)
    show_expr("square(alpha) = α²", square("alpha"), df)
    show_expr("cube(omega) = ω³", cube("omega"), df)
    show_expr("nth_root(order, 3) = ³√order", nth_root("order", 3), df)

    # ---------------------------------------------------------
    H("3) Ratios & reciprocals")
    show_expr("ratio(alpha, omega) = α/ω", ratio("alpha", "omega"), df)
    show_expr("safe_ratio(alpha, omega, 1e-9)", safe_ratio("alpha", "omega", eps=1e-9), df)
    show_expr("reciprocal(mu) = 1/μ", reciprocal("mu"), df)
    show_expr("reciprocal_eps(mu, 1e-9)", reciprocal_eps("mu", eps=1e-9), df)

    # ---------------------------------------------------------
    H("4) Geometric mean")
    show_expr("geometric_mean([alpha, omega])", geometric_mean(["alpha", "omega"]), df)

    # ---------------------------------------------------------
    H("5) Nonlinear expressions inside relations & conjectures")
    # Nordhaus–Gaddum-style: α·ω ≤ order
    R1 = Le(product("alpha", "omega"), "order")
    show_rel("R1: α·ω ≤ n", R1, df)

    # α / (ω+1) ≥ 1 (just an illustrative inequality)
    R2 = Ge(safe_ratio("alpha", "omega", eps=1.0), 1.0)
    show_rel("R2: α/(ω+1) ≥ 1", R2, df)

    # (α·ω) ≈ order within tolerance (toy)
    R3 = Eq(product("alpha", "omega"), "order", tol=1e-6)
    show_rel("R3: α·ω ≈ n  (tol=1e-6)", R3, df)

    # Conjectures
    C1 = Conjecture(R1, condition=TRUE)
    show_conj("C1: TRUE ⇒ α·ω ≤ n", C1, df)

    C2 = Conjecture(R2)  # no condition (TRUE omitted in pretty)
    show_conj("C2: α/(ω+1) ≥ 1", C2, df)


if __name__ == "__main__":
    main()
