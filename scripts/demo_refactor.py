# scripts/exercise_graffiti_relations_on_df.py
from __future__ import annotations

import pandas as pd

# Import the façade (which delegates to the internal classes)
from txgraffiti2025.graffiti_relations import GraffitiClassRelations, TRUE


def exercise_graffiti_relations(df: pd.DataFrame, show: int = 8) -> None:
    """
    Run the main capabilities of GraffitiClassRelations on an existing DataFrame `df`,
    printing concise outputs for each internal component (init/partition, conjunctions,
    constants & classes, ratio mining, equality-class mining, integration, incomparability).
    """
    print("\n=== 0) Instantiate & Initialize ===")
    g = GraffitiClassRelations(df)
    _ = g.print_initialization_report(max_list=show)

    # ─────────────────────────────────────────────────────────────────────
    # 1) Partitioning & Base Hypothesis  ( _PartitionAndInit )
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== 1) Partitioning & Base Hypothesis ===")
    print("Boolean-like predicates:", list(g.get_base_predicates().keys())[:show])
    print("Numeric expr columns:", g.get_expr_columns()[:show])

    # Demonstrate accessors
    base_name = getattr(g, "_S").base_hypothesis_name  # type: ignore[attr-defined]
    print("Base hypothesis name:", base_name)

    # ─────────────────────────────────────────────────────────────────────
    # 2) Conjunction Enumeration & Redundancy  ( _Conjunctions )
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== 2) Conjunctions: enumerate, redundancy, sorting ===")
    g.enumerate_conjunctions(max_arity=2)
    nonred, red, equiv = g.find_redundant_conjunctions()
    print(f"Enumerated: {len(getattr(g, '_S').base_conjunctions)}")  # type: ignore[attr-defined]
    print(f"Nonredundant: {len(nonred)}   Redundant: {len(red)}   Equiv groups: {len(equiv)}")
    top_conjs = g.sort_conjunctions_by_generality()[:show]
    print("Top conjunctions by support:")
    for name, _pred in top_conjs:
        print(" •", name)

    # ─────────────────────────────────────────────────────────────────────
    # 3) Constant Detection & Class Characterization  ( _ConstantsAndClasses )
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== 3) Constants & Class Characterization ===")
    const_map = g.find_constant_exprs(tol=0.0, require_finite=True)
    seen = 0
    for hyp_name, consts in const_map.items():
        if consts:
            print(f"• {hyp_name}: {consts}")
            seen += 1
            if seen >= show:
                break
    if seen == 0:
        print("(no constants found)")
    char = g.characterize_constant_classes(tol=0.0, group_per_hypothesis=True)
    g.print_class_characterization_summary()
    print(f"Equivalences: {len(char['equivalences'])}, "
          f"Inclusions (eq⇒class): {len(char['inclusions_AB'])}, "
          f"Inclusions (class⇒eq): {len(char['inclusions_BA'])}")

    # ─────────────────────────────────────────────────────────────────────
    # 4) Ratio Bounds on Base  ( _RatiosAndMining )
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== 4) Ratio Bounds on Base ===")
    ratio_summary, ratio_conjs = g.analyze_ratio_bounds_on_base(
        min_support=0.05,
        positive_denominator=True,
        require_finite=True,
        rationalize=True,
        max_denom=30,
        touch_atol=0.0,
        touch_rtol=0.0,
        emit_conjectures=True,
    )
    if not ratio_summary.empty:
        print(ratio_summary.head(show).to_string(index=False))
    else:
        print("(no ratio pairs passed the filters)")
    print("\nSample conjecture groups:")
    for cj in ratio_conjs[:show]:
        cname = getattr(cj, "name", "(unnamed)")
        cond = cj.condition
        cond_repr = "TRUE" if (cond is None or cond is TRUE) else repr(cond)
        print(f"• {cname}  |  condition={cond_repr}")

    # Show spawning one equality-class predicate from a ratio row & analyze it
    if not ratio_summary.empty:
        print("\nSpawn an equality class from the strongest 'touch' row and analyze it:")
        row0 = ratio_summary.iloc[0]
        name, pred = g.spawn_equality_classes_from_ratio_row(row0, which="auto", tol=0.0)
        print("Spawned hypothesis:", name)
        per_class = g.analyze_ratio_bounds_on_condition(
            pred,
            min_support=0.05,
            positive_denominator=True,
            require_finite=True,
            rationalize=True,
            max_denom=30,
            touch_atol=0.0,
            touch_rtol=0.0,
        )
        if not per_class.empty:
            print(per_class.head(show).to_string(index=False))
        else:
            print("(no per-class ratio pairs passed the filters)")

    # ─────────────────────────────────────────────────────────────────────
    # 5) Mine Equality Classes & Integrate into Conjunctions ( _RatiosAndMining )
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== 5) Mine Equality Classes & Integrate ===")
    mined = g.use_top_equality_classes_as_hypotheses(
        min_support=0.05,
        min_touch_count=1,
        min_touch_rate=0.10,
        which="auto",
        touch_atol=0.0,
        touch_rtol=0.0,
        rationalize=True,
        max_denom=30,
        top_k=20,
        conjoin_base=True,
        analyze_bounds_per_class=False,
        per_class_min_support=0.05,
    )
    cand_df = mined["candidates_df"]
    kept = mined["kept"]
    merged = mined["merged"]
    dominated = mined["dominated"]
    print("Candidate equality-classes:")
    print(cand_df.head(show).to_string(index=False) if not cand_df.empty else "(none)")
    print("\nKept (after equivalence merge & dominance):")
    if kept:
        for n, _p in kept[:show]:
            print(" •", n)
    else:
        print("(none)")
    if merged:
        print("\nMerged groups (equivalents):")
        for group in merged[:show]:
            print(" •", [n for (n, _p) in group])
    if dominated:
        print("\nDominated:")
        for (n, _p, by) in dominated[:show]:
            print(f" • {n}  ⊆  {by}")

    # integrate and re-rank conjunctions
    g._integrate_derived_hypotheses(kept)
    print("\nTop conjunctions after integrating derived classes:")
    for n, _p in g.sort_conjunctions_by_generality()[:show]:
        print(" •", n)

    # ─────────────────────────────────────────────────────────────────────
    # 6) Incomparability & |x−y| synthesis ( _IncomparabilityAndReport )
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== 6) Incomparability (unordered pairs with both x<y and x>y) ===")
    inc = g.analyze_incomparability(include_all_pairs=False)
    print(inc.head(show).to_string(index=False) if not inc.empty else "(no incomparable pairs found)")

    # Show the synthesized |x−y| exprs kept during init
    S = getattr(g, "_S")  # internal state for quick inspection
    abs_examples = getattr(S, "abs_exprs", [])
    print("\nTop synthesized |x−y| exprs by mean gap (names only):")
    if abs_examples:
        for name, _expr in abs_examples[:show]:
            print(" •", name)
    else:
        print("(none)")

    print("\nDone.\n")

from txgraffiti.example_data import graph_data as df
df['nontrivial'] = df['connected']
exercise_graffiti_relations(df)

# Usage:
# Call the function below wherever your `df` already exists.
# Example (in your notebook / REPL):
#     from scripts.exercise_graffiti_relations_on_df import exercise_graffiti_relations
#     exercise_graffiti_relations(df)
