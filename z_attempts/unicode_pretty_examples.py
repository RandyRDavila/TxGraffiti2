# unicode_pretty_examples.py
"""
Demonstrates the upgraded unicode pretty-printing and epsilon-safe logs.

What you'll see:
- Atoms, LinearForm pretty output (no dataclass repr)
- Binary ops with ·, proper precedence, and superscripts ²/³
- floor ⌊x⌋, ceil ⌈x⌉, |x|, √(x), ln(x), log₂(x), log₁₀(x), exp(x)
- log(x, epsilon=...) avoids -inf
- Predicates and Conjectures rendered with unicode math (≤, ≥, ⇒, ⌊ ⌋)

Run:
    PYTHONPATH=src python unicode_pretty_examples.py
"""

from __future__ import annotations
import pandas as pd

# Expr system
from txgraffiti2025.forms.utils import (
    to_expr, Const, ColumnTerm, LinearForm,
    floor, ceil, abs_, sqrt, exp, log
)

# Predicates + Conjectures
from txgraffiti2025.forms.predicates import (
    Predicate, IN, BETWEEN, IS_INT, IS_NAN, IS_FINITE, GE, LE
)
from txgraffiti2025.forms.generic_conjecture import (
    Le, Ge, Eq, AllOf, AnyOf, Conjecture, TRUE
)

# ------------------------------------------------------------
# Pretty console helpers
# ------------------------------------------------------------
def H1(title: str):
    print("\n" + "="*78)
    print(title)
    print("="*78)

def H2(title: str):
    print("\n" + title)
    print("-"*len(title))

def show_expr(title: str, expr, df):
    print(f"\n[{title}]")
    print("repr/pretty:", repr(expr))
    print("eval head:")
    print(expr.eval(df).head().to_string())

def show_mask(title: str, pred, df):
    print(f"\n[{title}] {pred!r}")
    m = pred.mask(df)
    print(m.to_string())

def show_conj(title: str, conj: Conjecture, df):
    print(f"\n[{title}]")
    print("pretty():", conj.pretty())
    applicable, holds, failures = conj.check(df)
    print("applicable.sum():", int(applicable.sum()))
    print("holds among applicable:", bool(holds[applicable].all()))
    if not failures.empty:
        print("-- failures --")
        print(failures.head().to_string())


def main():
    # ----------------------------------------------------------------
    # 0) Data
    # ----------------------------------------------------------------
    H1("0) DataFrame")
    df = pd.DataFrame({
        "order":   [3,  4,  5,  6,  8, 10],
        "size":    [2,  3,  4,  5,  8, 12],
        "alpha":   [1,  2,  2,  3,  3,  4],
        "mu":      [1,  2,  3,  3,  4,  5],
        "regular": [True, True, False, False, True, False],
        "planar":  [True, True, True,  False, False, False],
    }, index=[f"G{i}" for i in range(1, 7)])
    print(df)

    # ----------------------------------------------------------------
    # 1) Atoms & LinearForm (math-style pretty)
    # ----------------------------------------------------------------
    H1("1) Atoms & LinearForm (unicode pretty)")
    show_expr("ColumnTerm('alpha')", ColumnTerm("alpha"), df)
    show_expr("Const(3)", Const(3), df)

    lin1 = LinearForm(0.0, [(1.0, "alpha"), (2.0, "mu")])   # alpha + 2·mu
    lin2 = LinearForm(1.0, [(1.0, "alpha"), (-1.0, "order")])  # 1 + alpha - order
    show_expr("LinearForm(0; alpha + 2·mu)", lin1, df)
    show_expr("LinearForm(1; alpha - order)", lin2, df)

    # ----------------------------------------------------------------
    # 2) Binary ops: precedence, unicode dot, superscripts ²/³
    # ----------------------------------------------------------------
    H1("2) Binary ops: precedence, ·, superscripts")
    e1 = to_expr("alpha") + 2 * to_expr("mu") - 1
    e2 = (to_expr("alpha") + to_expr("mu")) * to_expr("size")
    e3 = to_expr("alpha") + to_expr("mu") * to_expr("size")   # precedence
    e4 = to_expr("size") ** (to_expr("alpha") ** 2)           # nested power → (alpha)²
    e5 = (to_expr("order") + 1) ** 3                          # → (…)³
    show_expr("alpha + 2*mu - 1", e1, df)
    show_expr("(alpha + mu) * size", e2, df)
    show_expr("alpha + (mu * size)", e3, df)
    show_expr("size ** (alpha ** 2)", e4, df)
    show_expr("(order + 1) ** 3", e5, df)

    # ----------------------------------------------------------------
    # 3) Unary ops: ⌊⌋, ⌈⌉, |x|, √(x), exp(x), ln / log₂ / log₁₀
    # ----------------------------------------------------------------
    H1("3) Unary ops: floor/ceil/abs/sqrt/exp/log (unicode)")
    ratio = to_expr("size") / to_expr("order")
    show_expr("size / order", ratio, df)
    show_expr("floor(size / order)", floor(ratio), df)
    show_expr("ceil(size / order)", ceil(ratio), df)
    show_expr("abs(alpha - mu)", abs_(to_expr("alpha") - to_expr("mu")), df)
    show_expr("sqrt(order)", sqrt(to_expr("order")), df)
    show_expr("exp(alpha - 2)", exp(to_expr("alpha") - 2), df)
    show_expr("log(order) (natural)", log("order"), df)
    show_expr("log2(order)", log("order", base=2), df)
    show_expr("log10(order)", log("order", base=10), df)

    # epsilon-safe log (simulate a zero present)
    H2("epsilon-safe log")
    df_eps = df.copy()
    df_eps.loc["G1", "alpha"] = 0  # alpha=0 ⇒ ln(0) would be -inf without epsilon
    show_expr("ln(alpha) without epsilon (may -inf)", log("alpha"), df_eps)
    show_expr("ln(alpha) with epsilon=1e-9 (clamped)", log("alpha", epsilon=1e-9), df_eps)

    # ----------------------------------------------------------------
    # 4) Predicates (IN/BETWEEN/IS_*), combining with &, |, ~
    # ----------------------------------------------------------------
    H1("4) Predicates: IN/BETWEEN/IS_* and composition")
    P_regular = Predicate.from_column("regular")
    P_planar  = Predicate.from_column("planar")
    show_mask("regular", P_regular, df)
    show_mask("planar", P_planar, df)
    show_mask("mu in {2,3,4}", IN("mu", {2,3,4}), df)
    show_mask("order BETWEEN [4,8]", BETWEEN("order", 4, 8, inc_lo=True, inc_hi=True), df)
    show_mask("IS_INT(order)", IS_INT("order"), df)
    show_mask("IS_NAN(alpha)", IS_NAN("alpha"), df)
    show_mask("IS_FINITE(size/order)", IS_FINITE(ratio), df)

    # ----------------------------------------------------------------
    # 5) Conjectures – math-style output with unicode floor/ceiling
    # ----------------------------------------------------------------
    H1("5) Conjectures: unicode math pretty")

    # (planar ∧ regular) ⇒ (alpha ≤ mu) ∧ (alpha ≥ ⌊order/3⌋)
    bound_le = Le("alpha", "mu")
    bound_ge = Ge("alpha", floor(to_expr("order") / 3))
    conj_bounds = Conjecture(AllOf([bound_le, bound_ge]), condition=P_planar & P_regular)
    show_conj("D: (planar ∧ regular) ⇒ bounds on alpha", conj_bounds, df)

    # TRUE ⇒ (alpha ≤ mu) ∨ (size = order - 1)
    conj_either = Conjecture(AnyOf([Le("alpha","mu"), Eq("size", to_expr("order") - 1)]), condition=TRUE)
    show_conj("E: TRUE ⇒ (alpha ≤ mu) ∨ (size = order - 1)", conj_either, df)

    print("\nDone. Tweak the DataFrame or expressions and re-run to explore.")

if __name__ == "__main__":
    main()
