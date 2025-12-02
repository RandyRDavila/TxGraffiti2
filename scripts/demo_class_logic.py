# scripts/demo_class_logic.py
from __future__ import annotations

import numpy as np
import pandas as pd

from txgraffiti2025.relations.core import DataModel, MaskCache
from txgraffiti2025.relations.class_logic import ClassLogic

def make_df() -> pd.DataFrame:
    n = 14
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "order": np.arange(6, 6 + n, dtype=float),
        "size": rng.integers(3, 22, size=n).astype(float),
        "connected": [True]*n,
        "nontrivial": [True]*n,
        "planar": rng.integers(0, 2, size=n),                  # 0/1
        "tree": rng.integers(0, 2, size=n).astype(float),      # 0.0/1.0
        "triangle_free": rng.integers(0, 2, size=n),           # 0/1
        "notes": ["ok" if i % 3 else "skip" for i in range(n)],
    })
    # a couple NA for realism
    df.loc[2, "tree"] = np.nan
    df.loc[5, "planar"] = np.nan
    return df

def main():
    # df = make_df()
    from txgraffiti.example_data import graph_data as df
    df['nontrivial'] = df['connected']
    model = DataModel(df)
    cache = MaskCache(model)

    print("=== DataModel ===")
    print(model.summary())
    print()

    logic = ClassLogic(model, cache)
    print("Base name:", logic.base_name())
    print("Base parts:", logic.base_parts())
    print("Non-base boolean pool:", sorted(list(logic.base_predicates().keys())))
    print()

    print("=== Enumerate (max_arity=2, min_support=2, include_base=True) ===")
    enum = logic.enumerate(max_arity=2, min_support=2, include_base=True)
    for name, _ in enum[:8]:
        print("  ", name)
    if len(enum) > 8:
        print("  ... total:", len(enum))
    print()

    print("=== Normalize (nonredundant / redundant / equivalence-groups) ===")
    nonred, red, equiv = logic.normalize()
    print("Nonredundant (top 8):")
    for name, _ in nonred[:8]:
        print("  ", name)
    if len(nonred) > 8:
        print("  ... total:", len(nonred))
    print("Redundant (count):", len(red))
    print("Equiv groups (count):", len(equiv))
    if equiv[:1]:
        print("  Example equiv group:", [n for (n, _) in equiv[0]])
    print()

    print("=== Sort by generality (descending) ===")
    sorted_nonred = logic.sort_by_generality()
    for name, _ in sorted_nonred[:8]:
        print("  ", name, "| support=", int(cache.mask(dict(sorted_nonred)[name]).sum()))
    if len(sorted_nonred) > 8:
        print("  ... total:", len(sorted_nonred))
    print()

    print("=== Canonicalize external set (with deliberate duplicates) ===")
    # Build a small external list containing duplicates/equivalents
    ext = []
    pool = logic.base_predicates()
    keys = sorted(pool.keys())
    if len(keys) >= 2:
        a, b = keys[:2]
        # B ∧ a, B ∧ b, B ∧ a ∧ b, and an exact duplicate name for B ∧ a
        # Note: we only need names for display; predicates drive semantics
        ext.append((f"{logic.base_name()} ∧ {a}", logic.base_predicate() & pool[a]))
        ext.append((f"{logic.base_name()} ∧ {a}", logic.base_predicate() & pool[a]))  # duplicate
        ext.append((f"{logic.base_name()} ∧ {b}", logic.base_predicate() & pool[b]))
        ext.append((f"{logic.base_name()} ∧ {a} ∧ {b}", logic.base_predicate() & pool[a] & pool[b]))

    kept, merged, dominated = logic.canonicalize(ext)
    print("Kept:", [n for (n, _) in kept])
    print("Merged groups (#):", len(merged))
    if merged[:1]:
        print("  Example merged group:", [n for (n, _) in merged[0]])
    print("Dominated:", [(d, "⊂", by) for (d, _, by) in dominated])

    print()
    print("=== Write nonredundant (sorted) to model.registry['classes'] ===")
    logic.to_registry(sorted_by_generality=True)

    # safer: no walrus operator
    registry_counts = model.summary()["registry_counts"]
    print("Registry counts:", registry_counts)  # shows that 'classes' was added
    classes_df = model.registry_frame("classes")
    print("registry['classes'] head:")
    print(classes_df[["name", "arity", "support", "extras", "base_parts"]].head(8))

if __name__ == "__main__":
    main()
