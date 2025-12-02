# Run with:
#   PYTHONPATH=src python conclusions_module_examples.py
from __future__ import annotations
import numpy as np
import pandas as pd

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
)
from txgraffiti2025.processing.pre.conclusions import (
    generate_all_conclusions,
    list_numeric_targets,
)

def make_df():
    # Toy dataset mixing discrete and numeric invariants
    idx = [f"G{i}" for i in range(1, 11)]
    df = pd.DataFrame(
        {
            "connected": [True]*10,
            "simple":    [True]*10,
            "planar":    [True, True, True, False, False, False, True, True, False, False],
            "regular":   [True, True, False, False, True, False, True, False, False, False],
            "order":     [3,4,5,6,8,10,7,9,12,15],
            "size":      [2,3,4,5,8,12,6,10,18,24],
            "alpha":     [1,2,2,3,3,5,4,4,6,7],
            "omega":     [1,1,3,2,1,2,1,3,4,5],
            "mu":        [1,2,2,3,4,5,3,4,6,7],
        },
        index=idx
    )
    return df

def show_top(conjs, df, k=5):
    print(f"Top {k} sample conjectures (pretty + quick check):")
    picked = 0
    for Cj in conjs:
        if picked >= k:
            break
        applicable, holds, failures = Cj.check(df)
        rel_pretty = Cj.pretty(unicode_ops=True)
        status = "OK" if bool(holds[applicable].all()) else "FAIL"
        print(f" - {rel_pretty}   [{status}, applicable={int(applicable.sum())}, violations={int((applicable & ~holds).sum())}]")
        picked += 1

def main():
    print("\n" + "="*78)
    print("0) DataFrame")
    print("="*78)
    df = make_df()
    print(df)

    # Hypotheses (base, singles, pairs) using the earlier module
    hyps = enumerate_boolean_hypotheses(df, include_pairs=True, skip_always_false=True)
    print("\n" + "="*78)
    print("1) Hypotheses discovered")
    print("="*78)
    for i, H in enumerate(hyps, 1):
        print(f"[H{i:02d}] {repr(H)}  support={int(H.mask(df).sum())}")

    # Numeric targets
    targets = list_numeric_targets(df)
    print("\n" + "="*78)
    print("2) Numeric targets")
    print("="*78)
    print(targets)

    # Generate conclusions for all (H, target) pairs using ratio + affine tightening
    results = generate_all_conclusions(
        df,
        hyps,
        targets=targets,
        transforms=("id", "sqrt", "log", "square"),
        log_bases=(None, 10),
        tol=1e-9,
    )

    # Demonstrate: pick a hypothesis and a target and show a few results
    sample_key = next(iter(results.keys()))
    H_repr, tgt = sample_key
    conjs = results[sample_key]

    print("\n" + "="*78)
    print(f"3) Sample: hypothesis = {H_repr}, target = {tgt}")
    print("="*78)
    show_top(conjs, df, k=10)

    # Optionally, check a stricter hypothesis/target combo
    for (Hk, Tk), conjs2 in results.items():
        if "planar" in Hk and Tk == "alpha":
            print("\n" + "="*78)
            print(f"4) Sample (planar hyp): hypothesis = {Hk}, target = {Tk}")
            print("="*78)
            show_top(conjs2, df, k=10)
            break

if __name__ == "__main__":
    main()
