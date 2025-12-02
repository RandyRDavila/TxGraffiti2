# scripts/demo_relations_core.py
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from txgraffiti2025.relations.core import DataModel, MaskCache
from txgraffiti2025.forms.utils import to_expr, abs_, sqrt, min_, max_
from txgraffiti2025.forms.predicates import GE, LE, BETWEEN, IN, IS_INT
from txgraffiti2025.forms.generic_conjecture import Ge, Conjecture, TRUE

def make_df() -> pd.DataFrame:
    n = 12
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "order": np.arange(4, 4 + n, dtype=float),
        "size": rng.integers(3, 20, size=n).astype(float),
        "connected": [True]*n,                         # always True
        "nontrivial": [True]*n,                        # always True
        "planar": rng.integers(0, 2, size=n),         # 0/1 integer (boolean-like)
        "tree": rng.integers(0, 2, size=n).astype(float),  # 0.0/1.0 float (boolean-like)
        "notes": ["ok" if i % 3 else "skip" for i in range(n)],  # text
        "maybe_num": [str(x) if i % 4 else "NaN" for i, x in enumerate(rng.integers(10, 99, size=n))],  # mostly numeric strings
    })
    # sprinkle a couple NA
    df.loc[1, "tree"] = np.nan
    df.loc[2, "planar"] = np.nan
    return df

def main():
    df = make_df()
    model = DataModel(df)
    cache = MaskCache(model)

    print("=== DataModel.summary() ===")
    print(model.summary())
    print()

    # Show iterators
    print("Numeric-like Exprs:", list(model.exprs.keys()))
    print("Boolean-like Preds:", list(model.preds.keys()))
    print("Text cols:", model.text_cols)
    print()

    # Build a few expressions and predicates
    order = model.expr("order")
    size = model.expr("size")
    maybe_num = to_expr("maybe_num")  # possible NaNs after coercion are fine in Expr

    # Expr arithmetic demo (pretty-printing)
    expr1 = min_(order, size) + sqrt(abs_(size - order))
    expr2 = max_(order, 2*size)
    print("Expr1:", expr1)  # Unicode pretty
    print("Expr2:", expr2)
    print()

    # Predicate demo
    P_planar = model.pred("planar")            # boolean-like column as Predicate
    P_tree   = model.pred("tree")
    P_base   = model.auto_base_from_always_true()  # conjunction of always-true cols or TRUE
    print("Auto base predicate:", repr(P_base))
    print("Support(planar):", model.support(P_planar))
    print("Support(tree):", model.support(P_tree))
    print()

    # Vectorized comparison predicates using the DSL
    P_big_order = GE(order, 10)                           # order ≥ 10
    P_size_band = BETWEEN(size, 8, 15, inc_lo=True, inc_hi=False)  # 8 ≤ size < 15
    P_subset    = IN("notes", {"ok"})                     # notes ∈ {ok}

    # MaskCache demo – cache a few masks
    m1 = cache.mask(P_big_order)
    m2 = cache.mask(P_size_band & P_subset)
    print("mask(order ≥ 10):", list(m1.astype(int)))
    print("mask(size in [8,15) ∧ notes=ok):", list(m2.astype(int)))
    print()

    # Tiny conjecture: order ≥ size/2 on the base class (TRUE or conjunction of constants)
    conj = Conjecture(Ge(order, 0.5 * size), condition=P_base)
    applicable, holds, failures = conj.check(df, auto_base=False)
    print("Conjecture:", conj.pretty())
    print("Applicable rows:", int(applicable.sum()))
    print("Violations:", int((applicable & ~holds).sum()))
    print("Touches (≈ equality rows):", conj.touch_count(df, auto_base=False))
    if not failures.empty:
        print("Failures head:\n", failures.head())
    print()

    # Use the registry
    model.record_absdiff(inv1="order", inv2="size", expr_name=str(expr1), hypothesis=repr(P_base), support=int(applicable.sum()))
    model.record_minmax(inv1="order", inv2="size", key_min=str(min_(order, size)), key_max=str(max_(order, size)),
                        hypothesis=repr(P_base), support=int(applicable.sum()))
    print("Registry counts after records:", model.summary()["registry_counts"])
    print("Registry.absdiff frame:\n", model.registry_frame("absdiff").head())

if __name__ == "__main__":
    main()
