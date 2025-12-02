# run_txg_pipeline_with_lp_transforms.py
"""
Complete end-to-end pipeline:
hypotheses → ratios(Expr) + LP(Expr) → generalize → refine → Hazel→Morgan→Dalmatian
"""

from __future__ import annotations
import pandas as pd
from txgraffiti.example_data import graph_data as df

# --- A) hypotheses → simplify/dedup (base-aware) ---
from txgraffiti2025.processing.pre.hypotheses import enumerate_boolean_hypotheses
from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses

hyps = enumerate_boolean_hypotheses(df, include_base=True, include_pairs=True, skip_always_false=True)
min_support = int(0.05 * len(df))
kept, eqs = simplify_and_dedup_hypotheses(df, hyps, min_support=min_support)

print("=== kept (base-aware, simplified) ===")
for h in kept: print(h)
print("\n=== saved equivalence conjectures ===")
for c in eqs: print(c)

# --- B) Targets/columns & feature makers ---
TARGET = "domination_number"
numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != TARGET]

from txgraffiti2025.forms.utils import to_expr, sqrt, log

def make_ratio_feature_sets(x_expr):
    # Each inner list is multiplied to build "other"
    return [
        [x_expr],        # y / x
        [sqrt(x_expr)],  # y / sqrt(x)
        [x_expr**2],     # y / x^2
        # Example product: uncomment to allow y / (x1*x2) later
        # [x_expr, to_expr("matching_number")],
    ]

def make_lp_feature_sets(x_expr):
    # Each inner list is a linear predictor basis
    return [
        [x_expr],            # y vs [x]
        [x_expr, sqrt(x_expr)],  # y vs [x, sqrt(x)]
        [x_expr, log(x_expr)],   # y vs [x, log(x)]
        [x_expr**2],             # y vs [x^2]
    ]

# --- B1) Generators (Expr-based) ---
from txgraffiti2025.generators.ratios_exprs import ratios_bounds, RatiosConfig
from txgraffiti2025.generators.lp_exprs import lp_bounds, LPConfig

conjectures = []
min_support_any = max(8, int(0.05 * len(df)))

# Single pass over (H, other)
for H in kept:
    for other in ["order", "maximum_degree", "minimum_degree"]: #numeric_cols:
        x = to_expr(other)

        # Ratios: c_min <= y/other <= c_max
        for feats in make_ratio_feature_sets(x):
            rcfg = RatiosConfig(
                features=feats,
                target=TARGET,
                direction="both",
                max_denominator=50,
                q_clip=None,                # set e.g. 0.01 for robustness
                min_support=min_support_any,
                min_valid_frac=0.20,
                allow_mixed_sign_splits=True,  # MaskPredicate-based splits for composite Exprs
                warn=False,
                name_prefix="ratiox",
            )
            conjectures.extend(ratios_bounds(df, hypothesis=H, config=rcfg))

        # LP: y vs feature bases
        for feats in make_lp_feature_sets(x):
            lpcfg = LPConfig(
                features=feats,
                target=TARGET,
                direction="both",
                max_denominator=50,
                tol=1e-9,
                min_support=min_support_any,
                min_valid_frac=0.20,
                warn=False,
                name_prefix="lp",
            )
            conjectures.extend(lp_bounds(df, hypothesis=H, config=lpcfg))
        # conjectures = list(set(conjectures))

# print(f"\nGenerated {len(ratiox)} ratio(Expr) conjectures.")
# print(f"Generated {len(lp_conjs)} LP-based conjectures (Expr features).")

print(f"Generated {len(conjectures)}.")


from txgraffiti2025.processing.post import hazel_rank, morgan_filter
from txgraffiti2025.processing.post import dalmatian_filter

hazel_res = hazel_rank(df, conjectures, drop_frac=0.25, atol=1e-9)
morgan_res = morgan_filter(df, hazel_res.kept_sorted)
print(f"Hazel → Morgan kept {len(conjectures)}")

dalmatian_kept = dalmatian_filter(morgan_res.kept, df)

print("\n=== saved conjectures (Hazel → Morgan → Dalmatian, incl. ratios/LP/generalizations/refinements) ===")
for c in dalmatian_kept:
    print(c.pretty(arrow='⇒'))

# # --- C) precompute constants for {base} ∪ {base∧each_boolean} ---
# from txgraffiti2025.forms.generic_conjecture import Conjecture
# base = Conjecture(relation=None, condition=None)._auto_base(df)

# from txgraffiti2025.processing.pre.constants_cache import precompute_constant_ratios_pairs
# const_cache = precompute_constant_ratios_pairs(
#     df,
#     base=base,
#     numeric_cols=numeric_cols,
#     shifts=(-1, 0, 1),
#     atol=1e-9, rtol=1e-9,
#     min_support=min_support_any,
#     max_denominator=50,
#     skip_identity=True,
# )

# # --- D) generalize from constants ---
# from txgraffiti2025.processing.post.generalize_from_constants import propose_generalizations_from_constants

# def _is_subset(df, A, B):
#     a = (A.mask(df) if A is not None else pd.Series(True, index=df.index)).astype(bool)
#     b = (B.mask(df) if B is not None else pd.Series(True, index=df.index)).astype(bool)
#     return not (a & ~b).any()

# gens_all = []
# for conj in conjectures:
#     supersets = [H for H in kept if _is_subset(df, conj.condition, H)]
#     gens_all.extend(
#         propose_generalizations_from_constants(df, conj, const_cache, candidate_hypotheses=supersets)
#     )
# generalized_from_constants = [g.new_conjecture for g in gens_all]
# print(f"Proposed {len(generalized_from_constants)} generalized conjectures from constants.")

# # --- D2) joint generalizer: slope (constants/reciprocals) + intercept ---
# from txgraffiti2025.processing.post.generalize import propose_joint_generalizations

# joint_generals = []
# for conj in conjectures:
#     supersets = [H for H in kept if _is_subset(df, conj.condition, H)]
#     joint_generals.extend(
#         propose_joint_generalizations(
#             df,
#             conj,
#             cache=const_cache,
#             candidate_hypotheses=supersets,
#             candidate_intercepts=None,
#             relaxers_Z=None,
#             require_superset=True,
#             atol=1e-9,
#         )
#     )
# print(f"Proposed {len(joint_generals)} joint generalizations (slope/intercept/reciprocals).")

# # --- D3) (OPTIONAL) numeric refinement ---
# USE_REFINER = True
# from txgraffiti2025.processing.post.refine_numeric import refine_numeric_bounds, RefinementConfig

# refined = []
# if USE_REFINER:
#     cfg = RefinementConfig(
#         try_whole_rhs_floor=True,
#         try_whole_rhs_ceil=False,
#         try_prod_floor=False,
#         try_coeff_round_const=True,
#         try_intercept_round_const=True,
#         try_sqrt_intercept=True,
#         try_log_intercept=False,
#         require_tighter=True,
#     )
#     for conj in (conjectures + generalized_from_constants + joint_generals):
#         refined.extend(refine_numeric_bounds(df, conj, config=cfg))
# print(f"Refined {len(refined)} conjectures numerically.")

# # --- E) Hazel → Morgan → Dalmatian ---


# merged = (
#     conjectures
#     + generalized_from_constants
#     + joint_generals
#     + (refined if USE_REFINER else [])
# )

# hazel_res = hazel_rank(df, merged, drop_frac=0.25, atol=1e-9)
# morgan_res = morgan_filter(df, hazel_res.kept_sorted)
# print(f"Hazel → Morgan kept {len(morgan_res.kept)} / {len(merged)}")

# dalmatian_kept = dalmatian_filter(morgan_res.kept, df)

# print("\n=== saved conjectures (Hazel → Morgan → Dalmatian, incl. ratios/LP/generalizations/refinements) ===")
# for c in dalmatian_kept:
#     print(c.pretty(arrow='⇒'))
