# scripts/demo_relations_core.py
from __future__ import annotations

import numpy as np
import pandas as pd

from txgraffiti2025.relations.core import DataModel, MaskCache
from txgraffiti.example_data import graph_data as df


def main() -> None:
    # 0) Example data + a convenience boolean column
    df_local = df.copy()
    # Many of your demos use this convention:
    df_local["nontrivial"] = df_local["connected"]

    # 1) Build the model + mask cache
    m = DataModel(df_local)
    mc = MaskCache(m)

    # 2) Show column partition
    print("─" * 72)
    print("DataModel column partition")
    print("Boolean-like:", m.boolean_cols)
    print("Numeric     :", m.numeric_cols[:12], "..." if len(m.numeric_cols) > 12 else "")
    print("Rows        :", len(m.df))

    # 3) Base hypothesis H: connected ∧ nontrivial
    #    (Predicate supports &, |, and .mask(df) under the hood)
    H = m.pred("connected") & m.pred("nontrivial")
    H_mask = mc.mask(H)
    support = int(H_mask.sum())
    print("\nHypothesis: H := connected ∧ nontrivial")
    print("Support(H):", support)

    # 4) Peek at a couple of numeric columns under H to prove alignment
    if len(m.numeric_cols) >= 2 and support > 0:
        x, y = m.numeric_cols[0], m.numeric_cols[1]
        xs = m.df.loc[H_mask, x].astype(float)
        ys = m.df.loc[H_mask, y].astype(float)

        print("\nSample numeric columns under H:")
        print(f"  {x}: count={xs.size}, mean≈{xs.mean():.4g}, min={xs.min()}, max={xs.max()}")
        print(f"  {y}: count={ys.size}, mean≈{ys.mean():.4g}, min={ys.min()}, max={ys.max()}")

        # 5) Record a couple of artifacts in the registry (just to show API)
        # 5a) If x is (near-)constant under H, record it
        TOL = 1e-9
        if xs.size > 0 and float(xs.max() - xs.min()) <= TOL:
            m.record_constant(inv=x, value=float(xs.iloc[0]), hypothesis="H", support=int(xs.size), tol=TOL)

        # 5b) Record an |x - y| summary under H
        valid = xs.notna() & ys.notna()
        if valid.any():
            d = (xs[valid] - ys[valid]).abs().to_numpy()
            m.record_absdiff(
                inv1=x,
                inv2=y,
                expr_name=f"|{x}-{y}|",
                hypothesis="H",
                support=int(valid.sum()),
                mean=float(d.mean()),
                median=float(np.median(d)),
                max=float(d.max()),
                count=int(d.size),
            )

    # 6) Print registry snapshots
    def show(kind: str):
        frame = m.registry_frame(kind)
        if frame.empty:
            print(f"(registry[{kind}] is empty)")
        else:
            print(frame.head().reset_index(drop=True))

    print("\n─" * 72)
    print("Registry: constants")
    show("constants")

    print("\nRegistry: absdiff")
    show("absdiff")

    print("\nRegistry: pairs")
    show("pairs")

    print("\nRegistry: minmax")
    show("minmax")

    print("\nRegistry: conjectures")
    show("conjectures")

    print("\nDone.")


if __name__ == "__main__":
    main()
