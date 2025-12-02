# class_relations_module_examples.py
"""
Demonstration of txgraffiti2025.forms.class_relations

This script showcases:
    - ClassInclusion (A ⊆ B)
    - ClassEquivalence (A ≡ B)

using small synthetic DataFrames of boolean graph properties.

Each example shows:
    1. The base DataFrame of classes (planar, bipartite, tree, etc.)
    2. The logical inclusion/equivalence relations
    3. Their .mask(df), .violations(df), and pretty() outputs

Run with:
    PYTHONPATH=src python class_relations_module_examples.py
"""

import pandas as pd
from txgraffiti2025.forms.predicates import Predicate, AndPred
from txgraffiti2025.forms.class_relations import ClassInclusion, ClassEquivalence


# ------------------------------------------------------------------------------
# Utility display helpers
# ------------------------------------------------------------------------------

def show_header(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_relation(label: str, rel, df: pd.DataFrame):
    """Pretty-print a relation’s outputs."""
    print(f"\n[{label}] {rel.pretty()}")
    print("-" * 78)
    print("mask(df):")
    print(rel.mask(df).astype(int).to_string())
    print("\nviolations(df):")
    v = rel.violations(df)
    if v.empty:
        print("  (none)")
    else:
        print(v)
    print(f"\nviolation_count: {rel.violation_count(df)}")
    print(f"holds_all:       {rel.holds_all(df)}")


# ------------------------------------------------------------------------------
# Demo setup
# ------------------------------------------------------------------------------

def main():
    # === Base data: pretend these are boolean graph properties ===
    show_header("0) DataFrame of graph class features")
    df = pd.DataFrame({
        "planar":      [True, True, False, False, True],
        "bipartite":   [True, False, True, True, True],
        "tree":        [True, False, False, False, True],
        "triangle_free": [True, True, True, False, True],
    }, index=["G1", "G2", "G3", "G4", "G5"])
    print(df)

    # Predicates
    P_planar = Predicate.from_column("planar")
    P_bip = Predicate.from_column("bipartite")
    P_tree = Predicate.from_column("tree")
    P_trifree = Predicate.from_column("triangle_free")

    # ------------------------------------------------------------------------------
    # 1) Inclusion examples
    # ------------------------------------------------------------------------------

    show_header("1) ClassInclusion examples")

    # (A) Every tree is planar
    incl_A = ClassInclusion(P_tree, P_planar)
    show_relation("A: trees ⊆ planar", incl_A, df)

    # (B) Every planar & triangle-free graph is bipartite
    incl_B = ClassInclusion(AndPred(P_planar, P_trifree), P_bip)
    show_relation("B: (planar ∧ triangle_free) ⊆ bipartite", incl_B, df)

    # (C) Counterexample: bipartite ⊆ planar (false for G3)
    incl_C = ClassInclusion(P_bip, P_planar)
    show_relation("C: bipartite ⊆ planar", incl_C, df)

    # ------------------------------------------------------------------------------
    # 2) Equivalence examples
    # ------------------------------------------------------------------------------

    show_header("2) ClassEquivalence examples")

    # (D) Define a derived class: “planar and tree” vs “planar ∧ tree”
    eqv_D = ClassEquivalence(P_tree, AndPred(P_planar, P_tree))
    show_relation("D: tree ≡ (planar ∧ tree)", eqv_D, df)

    # (E) False equivalence: planar ≡ bipartite
    eqv_E = ClassEquivalence(P_planar, P_bip)
    show_relation("E: planar ≡ bipartite", eqv_E, df)

    # ------------------------------------------------------------------------------
    # 3) Unicode vs ASCII display
    # ------------------------------------------------------------------------------

    show_header("3) Unicode vs ASCII pretty()")

    print("Unicode :", incl_A.pretty(unicode_ops=True))
    print("ASCII   :", incl_A.pretty(unicode_ops=False))
    print("Unicode :", eqv_E.pretty(unicode_ops=True))
    print("ASCII   :", eqv_E.pretty(unicode_ops=False))


if __name__ == "__main__":
    main()
