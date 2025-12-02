# forms_module_examples.py
"""
Educational walkthrough for txgraffiti2025.forms

Run:
    PYTHONPATH=src python forms_module_examples.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import (
    Expr, Const, ColumnTerm, LinearForm,
    to_expr, floor, ceil, abs_, log, exp, sqrt
)

# Pretty print helpers
def h1(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def h2(title: str):
    print("\n" + title)
    print("-" * len(title))

def show_eval(label: str, expr: Expr, df: pd.DataFrame, n=5):
    print(f"\n[{label}]")
    print("repr(expr):", repr(expr))
    out = expr.eval(df)
    print("eval(expr) head:")
    print(out.head(n).to_string())

def main():
    # ------------------------------------------------------------------
    # 0) Data
    # ------------------------------------------------------------------
    h1("0) DataFrame setup")
    df = pd.DataFrame({
        "order": [3, 4, 5, 6, 8],
        "size":  [2, 3, 4, 5, 8],
        "alpha": [1, 2, 2, 3, 3],
        "mu":    [1, 2, 3, 3, 4],
        "regular": [True, True, False, False, True],
    }, index=[f"G{i}" for i in range(1, 6)])
    print(df)

    # ------------------------------------------------------------------
    # 1) Atoms & to_expr
    # ------------------------------------------------------------------
    h1("1) Atoms: Const, ColumnTerm, to_expr")
    x = ColumnTerm("alpha")
    y = ColumnTerm("mu")
    c = Const(3)
    show_eval("ColumnTerm('alpha')", x, df)
    show_eval("Const(3)", c, df)

    h2("to_expr convenience")
    show_eval("to_expr('order')", to_expr("order"), df)
    show_eval("to_expr(5)", to_expr(5), df)
    show_eval("to_expr(ColumnTerm('alpha'))", to_expr(x), df)

    # ------------------------------------------------------------------
    # 2) Arithmetic
    # ------------------------------------------------------------------
    h1("2) Arithmetic: +, -, *, /, %, **")
    expr = x + 2 * y - 1
    show_eval("alpha + 2*mu - 1", expr, df)

    expr_a = (x + y) * to_expr("size")
    expr_b = x + (y * to_expr("size"))
    show_eval("(alpha + mu) * size", expr_a, df)
    show_eval("alpha + (mu * size)", expr_b, df)

    expr_pow = to_expr("size") ** (to_expr("alpha") ** 2)
    show_eval("size ** (alpha ** 2)", expr_pow, df)

    expr_mod = to_expr("size") % 3
    show_eval("size % 3", expr_mod, df)

    # ------------------------------------------------------------------
    # 3) LinearForm (atomic numerator) and an explicit-parentheses version
    # ------------------------------------------------------------------
    h1("3) LinearForm examples (atomic printing; no trailing '+ 0')")

    lin1 = LinearForm(0.0, [(1.0, "alpha"), (2.0, "mu")])   # alpha + 2*mu
    lin2 = LinearForm(1.0, [(1.0, "alpha"), (-1.0, "order")])  # 1 + alpha - order
    lin3 = LinearForm(0.0, [(-2.0, "mu")])                  # -2*mu
    show_eval("LinearForm(0; alpha + 2*mu)", lin1, df)
    show_eval("LinearForm(1; alpha - order)", lin2, df)
    show_eval("LinearForm(0; -2*mu)", lin3, df)

    # NOTE: LinearForm is an *atomic* expression for printing purposes.
    # The following is (alpha + 2*mu) / (order + 1) even if parentheses
    # aren’t shown around the numerator in repr:
    expr_lin_atomic = lin1 / (to_expr("order") + 1)
    show_eval("(alpha + 2*mu) / (order + 1)  [LinearForm as atomic numerator]",
              expr_lin_atomic, df)

    # If you want unambiguous printing, build the numerator from primitives:
    expr_lin_explicit = (to_expr("alpha") + 2*to_expr("mu")) / (to_expr("order") + 1)
    show_eval("(alpha + 2*mu) / (order + 1)  [explicit parentheses version]",
              expr_lin_explicit, df)

    # ------------------------------------------------------------------
    # 4) Unary ops
    # ------------------------------------------------------------------
    h1("4) Unary ops (floor, ceil, abs, sqrt, exp, log)")
    ratio = to_expr("size") / to_expr("order")
    show_eval("size / order", ratio, df)
    show_eval("floor(size / order)", floor(ratio), df)
    show_eval("ceil(size / order)", ceil(ratio), df)

    show_eval("abs(alpha - mu)", abs_(x - y), df)
    show_eval("sqrt(order)", sqrt("order"), df)
    show_eval("exp(alpha - 2)", exp(x - 2), df)

    show_eval("log(order)", log("order"), df)
    show_eval("log2(order)", log("order", base=2), df)
    show_eval("log10(order)", log("order", base=10), df)

    df_with_zero = df.copy()
    df_with_zero.loc["G1", "alpha"] = 0
    show_eval("log(alpha) without epsilon (note -inf at alpha=0)", log("alpha"), df_with_zero)
    show_eval("log(alpha) with epsilon=1e-9", log("alpha", epsilon=1e-9), df_with_zero)

    # ------------------------------------------------------------------
    # 5) Booleans → floats
    # ------------------------------------------------------------------
    h1("5) Boolean columns are cast to floats automatically")
    show_eval("regular (bool) as 0/1", to_expr("regular"), df)
    show_eval("alpha + regular", to_expr("alpha") + to_expr("regular"), df)

    # ------------------------------------------------------------------
    # 6) Common pitfalls (revised)
    # ------------------------------------------------------------------
    h1("6) Common pitfalls (revised)")

    h2("Missing column")
    try:
        show_eval("ColumnTerm('does_not_exist')", ColumnTerm("does_not_exist"), df)
    except KeyError as e:
        print("Caught KeyError (as expected):", e)

    h2("Raw NumPy arrays are NOT valid Expr operands")
    try:
        arr = np.arange(len(df))  # OK length but still not an Expr
        _ = to_expr("alpha") + arr  # this will raise TypeError in to_expr
    except TypeError as e:
        print("Caught TypeError (as expected):", e)
        print("Tip: add arrays as columns aligned to df.index and reference by name.")

    # Correct pattern: align as a column, then use ColumnTerm
    df2 = df.copy()
    df2["arr"] = np.arange(len(df2))  # aligned array
    show_eval("alpha + arr   [arr as aligned column]", to_expr("alpha") + to_expr("arr"), df2)

    print("\nAll done. Explore and modify this script as you wish!")

if __name__ == "__main__":
    main()
