# logexp_module_examples.py
"""
Demonstration of txgraffiti2025.forms.logexp

Covers:
  1) Natural/base-2/base-10 logs (with/without epsilon clamp)
  2) Exponential and square-root
  3) Composition with arithmetic and LinearForm
  4) Using logs inside relations & conjectures

Run:
  PYTHONPATH=src python logexp_module_examples.py
"""

from __future__ import annotations
import pandas as pd

from txgraffiti2025.forms.logexp import ln, log2, log10, log_base, exp_e, sqrt
from txgraffiti2025.forms.utils import to_expr
from txgraffiti2025.forms.linear import linear_expr
from txgraffiti2025.forms.generic_conjecture import Le, Ge, Eq, Conjecture, TRUE

# -------------------------------------------------------------------
# Small print helpers
# -------------------------------------------------------------------
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
    # ----------------------------------------------------------------
    H("0) DataFrame")
    df = pd.DataFrame({
        "order": [1, 2, 4, 8, 16, 0],
        "size":  [1, 3, 4, 9, 16, 5],
        "alpha": [0, 1, 2, 3, 4, 0],
        "mu":    [1, 1, 2, 3, 5, 2],
        "flag":  [True, True, False, True, True, False],
    }, index=[f"G{i}" for i in range(1, 7)])
    print(df)

    # ----------------------------------------------------------------
    H("1) Logs: ln, log2, log10 (epsilon clamp optional)")
    show_expr("ln(order)", ln("order"), df)
    show_expr("log2(order)", log2("order"), df)
    show_expr("log10(order)", log10("order"), df)

    # At order=0, ln(0) = -inf. Use epsilon to clamp:
    show_expr("ln(order) with epsilon=1e-9 (avoids -inf at 0)", ln("order", epsilon=1e-9), df)

    # Custom base via log_base:
    show_expr("log_base(size, base=3)", log_base("size", base=3), df)

    # ----------------------------------------------------------------
    H("2) Exponential and sqrt")
    show_expr("exp_e(alpha)", exp_e("alpha"), df)
    show_expr("sqrt(size)", sqrt("size"), df)

    # Composition example:
    composed = sqrt("size") + ln("order", epsilon=1e-9) * 2
    show_expr("composed: √(size) + 2·ln(order)", composed, df)

    # ----------------------------------------------------------------
    H("3) Logs inside relations")
    # ln(mu) <= ln(order + 1)
    R1 = Le(ln("mu", epsilon=1e-9), ln(to_expr("order") + 1, epsilon=0.0))
    show_rel("R1: ln(mu) ≤ ln(order+1)", R1, df)

    # log2(size) >= 2  (i.e., size >= 4)
    R2 = Ge(log2("size", epsilon=1e-9), 2)
    show_rel("R2: log₂(size) ≥ 2", R2, df)

    # ln(size) ≈ ln(order) within tol
    R3 = Eq(ln("size", epsilon=1e-9), ln("order", epsilon=1e-9), tol=1e-6)
    show_rel("R3: ln(size) ≈ ln(order)  (tol=1e-6)", R3, df)

    # ----------------------------------------------------------------
    H("4) Wrap as Conjectures for reporting")
    C1 = Conjecture(R1, condition=TRUE)
    show_conj("C1: TRUE ⇒ ln(mu) ≤ ln(order+1)", C1, df)

    # Let’s constrain to a subclass: flag==True rows only
    from txgraffiti2025.forms.predicates import Predicate
    Cflag = Predicate.from_column("flag")
    C2 = Conjecture(R2, condition=Cflag)
    show_conj("C2: (flag) ⇒ log₂(size) ≥ 2", C2, df)

    # Linear+log mixed expression on left vs column on right
    left = ln(linear_expr(0.0, {"alpha": 1.0, "mu": 1.0}), epsilon=1e-9)  # ln(alpha + mu)
    R4 = Le(left, log10("size", epsilon=1e-9))
    C3 = Conjecture(R4, condition=TRUE)
    show_conj("C3: TRUE ⇒ ln(alpha+mu) ≤ log₁₀(size)", C3, df)


if __name__ == "__main__":
    main()
