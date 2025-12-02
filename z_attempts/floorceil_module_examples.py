# floorceil_module_examples.py
# Educational walkthrough of R3 floor/ceil wrappers

import pandas as pd

from txgraffiti2025.forms.floorceil import with_floor, with_ceil
from txgraffiti2025.forms.utils import to_expr
from txgraffiti2025.forms.generic_conjecture import Le, Ge, Eq, Conjecture
from txgraffiti2025.forms.predicates import Predicate

SEP = "\n" + "="*78 + "\n"

def show(title, expr, df):
    print(f"\n[{title}] {expr!r}")
    if hasattr(expr, "pretty"):
        print("pretty():", expr.pretty())
    else:
        print("pretty():", repr(expr))
    try:
        out = expr.eval(df)
        print("eval head:\n", out.head())
    except Exception as e:
        print("eval -> ERROR:", e)

def show_rel(title, rel, df):
    print(f"\n[{title}] {rel.pretty()}")
    print("repr(relation):", repr(rel))
    print("evaluate mask:\n", rel.evaluate(df).astype(int))
    print("slack head:\n", rel.slack(df).head())

def show_conj(title, conj, df):
    print(f"\n[{title}]")
    print("pretty():", conj.pretty())
    applicable, holds, failures = conj.check(df)
    print("applicable.sum():", int(applicable.sum()))
    print("holds among applicable:", bool(holds[applicable].all()))
    print("touch_count():", conj.touch_count(df))
    print("violation_count():", conj.violation_count(df))
    if not failures.empty:
        print("failures:\n", failures)

def main():
    print(SEP + "0) DataFrame" + SEP)
    df = pd.DataFrame(
        {
            "order": [3, 4, 5, 6, 8, 10, 7],
            "size":  [2, 3, 4, 5, 8, 12, 6],
            "alpha": [1, 2, 3, 3, 3, 5, 4],
            "omega": [1, 1, 3, 2, 1, 2, 1],
            "regular": [True, True, False, False, True, False, True],
        },
        index=[f"G{i}" for i in range(1, 8)],
    )
    print(df)

    print(SEP + "1) Basic floor / ceil wrapping" + SEP)

    # Simple expressions
    e_ratio = to_expr("size") / to_expr("order")
    show("ratio size/order", e_ratio, df)

    show("with_floor(size/order)", with_floor(e_ratio), df)
    show("with_ceil(size/order)", with_ceil(e_ratio), df)

    # On a plain column
    show("with_floor(order/3)", with_floor(to_expr("order") / 3), df)
    show("with_ceil(order/3)", with_ceil(to_expr("order") / 3), df)

    print(SEP + "2) Floor/Ceil inside relations (Le/Ge/Eq)" + SEP)

    # α ≤ ⌈order/3⌉
    r1 = Le("alpha", with_ceil(to_expr("order") / 3))
    show_rel("R1: alpha ≤ ⌈order/3⌉", r1, df)

    # ⌊size/2⌋ ≥ omega
    r2 = Ge(with_floor(to_expr("size") / 2), "omega")
    show_rel("R2: ⌊size/2⌋ ≥ omega", r2, df)

    # ⌊size/order⌋ == 0 (approx equality with tol)
    r3 = Eq(with_floor(e_ratio), 0, tol=1e-9)
    show_rel("R3: ⌊size/order⌋ = 0", r3, df)

    print(SEP + "3) As conjectures (with/without conditions)" + SEP)

    # With a class condition: (regular) ⇒ (alpha ≤ ⌈order/3⌉)
    C_regular = Predicate.from_column("regular")
    conj_A = Conjecture(r1, condition=C_regular)
    show_conj("A: (regular) ⇒ (alpha ≤ ⌈order/3⌉)", conj_A, df)

    # Global (no condition): ⌊size/2⌋ ≥ omega
    conj_B = Conjecture(r2)
    show_conj("B: TRUE ⇒ (⌊size/2⌋ ≥ omega)", conj_B, df)

    print(SEP + "4) Composition: floor/ceil with other algebra" + SEP)

    # α + ⌈size/3⌉ ≤ ⌈order/2⌉ + ω
    complex_left = to_expr("alpha") + with_ceil(to_expr("size") / 3)
    complex_right = with_ceil(to_expr("order") / 2) + to_expr("omega")
    r4 = Le(complex_left, complex_right)
    show_rel("R4: alpha + ⌈size/3⌉ ≤ ⌈order/2⌉ + omega", r4, df)

    # ⌊(size + omega)/order⌋ ≥ 0   (always true unless negative data / NaNs)
    r5 = Ge(with_floor((to_expr("size") + to_expr("omega")) / to_expr("order")), 0)
    show_rel("R5: ⌊(size+omega)/order⌋ ≥ 0", r5, df)

    print(SEP + "5) Edge cases & tips" + SEP)
    print("- Floor/Ceil return float Series (0.0, 1.0, …) so they align with other Expr results.")
    print("- Equality with floor/ceil is brittle; prefer inequalities unless you truly expect exact ties.")
    print("- If you use logs elsewhere, ensure inputs are positive (or add a small epsilon in your log wrapper).")
    print("- Predicates with nullable booleans are treated as False on NA by default.")

if __name__ == "__main__":
    main()
