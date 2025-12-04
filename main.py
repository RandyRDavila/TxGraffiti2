# # if __name__ == "__main__":
# #     from txgraffiti.example_data import graph_data as df
# #     from txgraffiti2025.graffiti_base import GraffitiBase
# #     from txgraffiti2025.graffiti_class_logic import GraffitiClassLogic
# #     from txgraffiti2025.graffiti_ratios import GraffitiRatios

# #     from txgraffiti2025.graffiti_base import GraffitiBase
# #     from txgraffiti2025.graffiti_class_logic import GraffitiClassLogic
# #     from txgraffiti2025.graffiti_comparable import GraffitiComparable

# #     df.drop(columns=['vertex_cover_number'], inplace=True)
# #     df['nontrivial'] = df['connected']

# #     gb = GraffitiClassLogic(df)
# #     gl = GraffitiClassLogic(gb, run_pipeline=True, max_arity=2)

# #     gc = GraffitiComparable(gl)  # inherits df/partitions and sees gl.sorted_conjunctions_
# #     # top_pairs = gc.top_incomparable_pairs_on_base(top_k=15, tol=1e-9)

# #     # # 2) Build & register |x−y| exprs for those pairs:
# #     # abs_exprs = gc.build_abs_exprs_for_top_incomparables(top_k=15, tol=1e-9, register=True)

# #     # Quick base-scope summary
# #     # gc.summary_noncomparables(scope="base", tol=0.0, max_pairs=20)

# #     # Include per-class scopes discovered by logic (top 6 conjunctions)
# #     # gc.summary_noncomparables(scope="base", tol=0.0,
# #     #                         use_sorted_conjunctions=True,
# #     #                         conjunction_limit=6)

# #     gc.summary_abs_exprs()
# #     # g.print_class_characterization_summary()
# #     #     g.summary_conjectures()


# #     # gb = GraffitiBase(df)
# #     # gl = GraffitiClassLogic(gb)              # provides base + sorted_conjunctions_
# #     # gr = GraffitiRatios(gl, config=RatioConfig(min_support=0.02, touch_atol=1e-9))

# #     # # Compute base ratios
# #     # base_summary, base_conjs = gr.generate_on_base()

# #     # # Show a tidy, conjectures-only summary (base + a few top classes)
# #     # gr.summary(show_top_k_base=15, per_class_limit=6)

# #     H = gc.base_hypothesis & gc.pred("bipartite")

# #     m = gc.surprising_measure_pair(
# #         "total_domination_number",
# #         "diameter",
# #         hypothesis=H,
# #         tol=1e-9,
# #         score="ratio",
# #     )
# #     print(m)

# #     top = gc.surprising_top_pairs(hypothesis=H, score="ratio", tol=1e-9, top_k=20)
# #     print(top.head(10))

# #     rat = GraffitiRatios(gl)  # or pass a DataFrame / GraffitiBase
# #     res = rat.run_pipeline("total_domination_number", weight=0.5)
# #     rat.summarize(res, max_per_bucket=8)

# import pandas as pd
# import numpy as np

# from txgraffiti.example_data import graph_data as df
# from txgraffiti2025.relations.core import DataModel, MaskCache
# from txgraffiti2025.relations.class_logic import ClassLogic

# model = DataModel(df)
# cache = MaskCache(model)

# print("=== DataModel.summary() ===")
# print(model.summary())
# print()


# logic = ClassLogic(model, cache)
# print("Base name:", logic.base_name())
# print("Base parts:", logic.base_parts())
# print("Non-base boolean pool:", sorted(list(logic.base_predicates().keys())))
# print()

# print("=== Enumerate (max_arity=2, min_support=2, include_base=True) ===")
# enum = logic.enumerate(max_arity=2, min_support=2, include_base=True)
# for name, _ in enum[:8]:
#     print("  ", name)
# if len(enum) > 8:
#     print("  ... total:", len(enum))
# print()

# print("=== Normalize (nonredundant / redundant / equivalence-groups) ===")
# nonred, red, equiv = logic.normalize()
# print("Nonredundant (top 8):")
# for name, _ in nonred[:8]:
#     print("  ", name)
# if len(nonred) > 8:
#     print("  ... total:", len(nonred))
# print("Redundant (count):", len(red))
# print("Equiv groups (count):", len(equiv))
# if equiv[:1]:
#     print("  Example equiv group:", [n for (n, _) in equiv[0]])
# print()

# print("=== Sort by generality (descending) ===")
# sorted_nonred = logic.sort_by_generality()
# for name, _ in sorted_nonred:
#     print("  ", name, "| support=", int(cache.mask(dict(sorted_nonred)[name]).sum()))
# if len(sorted_nonred) > 8:
#     print("  ... total:", len(sorted_nonred))
# print()


# print("=== Canonicalize external set (with deliberate duplicates) ===")
# # Build a small external list containing duplicates/equivalents
# ext = []
# pool = logic.base_predicates()
# keys = sorted(pool.keys())
# if len(keys) >= 2:
#     a, b = keys[:2]
#     # B ∧ a, B ∧ b, B ∧ a ∧ b, and an exact duplicate name for B ∧ a
#     # Note: we only need names for display; predicates drive semantics
#     ext.append((f"{logic.base_name()} ∧ {a}", logic.base_predicate() & pool[a]))
#     ext.append((f"{logic.base_name()} ∧ {a}", logic.base_predicate() & pool[a]))  # duplicate
#     ext.append((f"{logic.base_name()} ∧ {b}", logic.base_predicate() & pool[b]))
#     ext.append((f"{logic.base_name()} ∧ {a} ∧ {b}", logic.base_predicate() & pool[a] & pool[b]))

# kept, merged, dominated = logic.canonicalize(ext)
# print("Kept:", [n for (n, _) in kept])
# print("Merged groups (#):", len(merged))
# if merged[:1]:
#     print("  Example merged group:", [n for (n, _) in merged[0]])
# print("Dominated:", [(d, "⊂", by) for (d, _, by) in dominated])

# print()
# print("=== Write nonredundant (sorted) to model.registry['classes'] ===")
# logic.to_registry(sorted_by_generality=True)

# # safer: no walrus operator
# registry_counts = model.summary()["registry_counts"]
# print("Registry counts:", registry_counts)  # shows that 'classes' was added
# classes_df = model.registry_frame("classes")
# print("registry['classes'] head:")
# print(classes_df[["name", "arity", "support", "extras", "base_parts"]].head(8))


# print()
# print("=== DataModel.summary() ===")
# print(model.summary())
# print()
