# scripts/demo_base_model.py
from __future__ import annotations

import pandas as pd

from txgraffiti.example_data import graph_data as df
from txgraffiti2025.graffiti.core import DataModel
from txgraffiti2025.forms.utils import sqrt
from txgraffiti2025.forms.predicates import GE

def main():

    # add additional base hypothesis
    df['nontrivial'] = df['connected']

    # 0) Load data (already a DataFrame in your example_data)
    assert isinstance(df, pd.DataFrame)

    # 1) Minimal model & initial scan
    model = DataModel(df)

    print("=== After base scan ===")
    print("invariants:", model.invariant_names[:8], "..." if len(model.invariant_names) > 8 else "")
    print("booleans  :", model.boolean_names)

    # 2) Add a derived invariant (Expr) and a mined boolean (Predicate)
    #    - invariant: √(diameter)
    #    - boolean  : (order ≥ 10)
    inv_key = model.add_invariant(sqrt("diameter"))
    bool_key = model.add_boolean(GE("order", 10))

    print("\n[add] added invariant:", inv_key)
    print("[add] added boolean  :", bool_key)

    # 3) Show current names (just to confirm registry/lookup)
    print("\n=== Current symbols ===")
    print("invariants:", model.invariant_names[:10], "..." if len(model.invariant_names) > 10 else "")
    print("booleans  :", model.boolean_names)

    # 4) Resolve the derived invariant by its pretty name and evaluate a few rows
    e = model.invariant(inv_key)           # same as model.to_expr(inv_key)
    vals = e.eval(model.df).head(5)
    print("\n√(diameter) head(5):")
    print(vals.to_string(index=False))

    # 5) Basic support count for the mined boolean
    pred = model.boolean(bool_key)
    supp = model.support(pred)
    print(f"\nSupport for '{bool_key}': {supp} rows")

    # 6) Optional: auto base (always-true conjunction) preview
    auto_base = model.auto_base_from_always_true()
    print("\nAuto-base from always-true booleans:", auto_base)

if __name__ == "__main__":
    main()
