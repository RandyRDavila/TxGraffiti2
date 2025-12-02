
# import pandas as pd
# from txgraffiti.example_data import graph_data as df

# # --- A) hypotheses → simplify/dedup (base-aware) ---
# from txgraffiti2025.processing.pre.hypotheses import enumerate_boolean_hypotheses
# from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses

# hyps = enumerate_boolean_hypotheses(df, include_base=True, include_pairs=True, skip_always_false=True)
# min_support = int(0.05 * len(df))
# kept, eqs = simplify_and_dedup_hypotheses(df, hyps, min_support=min_support)

# print("=== kept (base-aware, simplified) ===")
# for h in kept: print(h)
# print("\n=== saved equivalence conjectures ===")
# for c in eqs: print(c)

# # --- B) generate type-one (ratio) conjectures: target vs others ---
# TARGET = "independence_number"
# numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
# numeric_cols = [c for c in numeric_cols if c != TARGET]

# from txgraffiti2025.generators import ratios
# type_one = []
# for other in numeric_cols:
#     for H in kept:
#         type_one.extend(ratios(df, features=[other], target=TARGET, hypothesis=H))
# print(f"\nGenerated {len(type_one)} raw type-one conjectures.")

# # --- C) precompute constants for {base} ∪ {base∧each_boolean}, allow shifted same-col, skip identity ---
# from txgraffiti2025.forms.generic_conjecture import Conjecture
# base = Conjecture(relation=None, condition=None)._auto_base(df)

# from txgraffiti2025.processing.pre.constants_cache import precompute_constant_ratios_pairs
# const_cache = precompute_constant_ratios_pairs(
#     df,
#     base=base,
#     numeric_cols=numeric_cols,
#     shifts=(-1, 0, 1),
#     atol=1e-9, rtol=1e-9,
#     min_support=max(8, int(0.05 * len(df))),
#     max_denominator=50,
#     skip_identity=True,   # keeps (N+1)/(N-1), skips (N+a)/(N+a)
# )

# # --- D) generalize from constants onto supersets (kept) ---
# from txgraffiti2025.processing.post.generalize_from_constants import propose_generalizations_from_constants

# def _is_subset(df, A, B):
#     a = (A.mask(df) if A is not None else pd.Series(True, index=df.index)).astype(bool)
#     b = (B.mask(df) if B is not None else pd.Series(True, index=df.index)).astype(bool)
#     return not (a & ~b).any()

# gens_all = []
# for conj in type_one:
#     supersets = [H for H in kept if _is_subset(df, conj.condition, H)]
#     gens_all.extend(
#         propose_generalizations_from_constants(df, conj, const_cache, candidate_hypotheses=supersets)
#     )
# generalized_from_constants = [g.new_conjecture for g in gens_all]
# print(f"Proposed {len(generalized_from_constants)} generalized conjectures from constants.")

# # --- D2) (NEW) joint generalizer: slope (constants/reciprocals) + intercept ---
# # Uses the modules you pasted: generalize_from_constants, reciprocal_generalizer, intercept_generalizer
# from txgraffiti2025.processing.post.generalize import propose_joint_generalizations

# joint_generals = []
# for conj in type_one:
#     supersets = [H for H in kept if _is_subset(df, conj.condition, H)]
#     joint_generals.extend(
#         propose_joint_generalizations(
#             df,
#             conj,
#             cache=const_cache,
#             candidate_hypotheses=supersets,
#             candidate_intercepts=None,  # or a list of Expr if you want to seed intercepts
#             relaxers_Z=None,            # or a list of Expr relaxers if you have them
#             require_superset=True,
#             atol=1e-9,
#         )
#     )
# print(f"Proposed {len(joint_generals)} joint generalizations (slope/intercept/reciprocals).")

# # --- D3) (OPTIONAL) numeric refinement of RHS for any conjectures you want ---
# # If you want this off, set USE_REFINER = False.
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
#     for conj in (type_one + generalized_from_constants + joint_generals):
#         refined.extend(refine_numeric_bounds(df, conj, config=cfg))
# print(f"Refined {len(refined)} conjectures numerically.")

# # --- E) Hazel → Morgan → Dalmatian (Hazel first, as requested) ---
# from txgraffiti2025.processing.post import hazel_rank, morgan_filter
# from txgraffiti2025.processing.post import dalmatian_filter

# merged = (
#     type_one
#     + generalized_from_constants
#     + joint_generals
#     + (refined if USE_REFINER else [])
# )

# # Hazel first (ranking/filtering by touch)
# hazel_res = hazel_rank(df, merged, drop_frac=0.25, atol=1e-9)

# # Morgan next (dedupe by conclusion; prefer most-general hypothesis)
# morgan_res = morgan_filter(df, hazel_res.kept_sorted)
# print(f"Hazel → Morgan kept {len(morgan_res.kept)} / {len(merged)}")

# # Dalmatian last (significance; compares within same target/direction & hypothesis group)
# dalmatian_kept = dalmatian_filter(morgan_res.kept, df)

# print("\n=== saved type-one conjectures (Hazel → Morgan → Dalmatian, incl. generalizations/refinements) ===")
# for c in dalmatian_kept:
#     print(c.pretty(arrow='⇒'))


# run_txg_pipeline_with_lp_transforms.py
# Complete end-to-end pipeline: hypotheses → ratios → LP(+sqrt/log/square) → generalize → refine → Hazel→Morgan→Dalmatian

############
###########
##########
########
######

# from __future__ import annotations
# import numpy as np
# import pandas as pd

# # --- Example data ---
# from txgraffiti.example_data import graph_data as df

# # ─────────────────────────────────────────────────────────────────────────────
# # A) Hypotheses: enumerate → simplify/dedup (base-aware)
# # ─────────────────────────────────────────────────────────────────────────────
# from txgraffiti2025.processing.pre.hypotheses import enumerate_boolean_hypotheses
# from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses

# hyps = enumerate_boolean_hypotheses(
#     df,
#     include_base=True,
#     include_pairs=True,
#     skip_always_false=True,
# )
# min_support = int(0.05 * len(df))
# kept, eqs = simplify_and_dedup_hypotheses(df, hyps, min_support=min_support)

# print("=== kept (base-aware, simplified) ===")
# for h in kept:
#     print(h)
# print("\n=== saved equivalence conjectures ===")
# for c in eqs:
#     print(c)

# # ─────────────────────────────────────────────────────────────────────────────
# # B) Type-one (ratio) conjectures: TARGET vs other numerics
# # ─────────────────────────────────────────────────────────────────────────────
# TARGET = "independence_number"

# numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
# numeric_cols = [c for c in numeric_cols if c != TARGET]
# # Optional: drop strictly-binary numeric columns
# numeric_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 2]

# from txgraffiti2025.generators import ratios
# type_one = []
# for other in numeric_cols:
#     for H in kept:
#         type_one.extend(ratios(df, features=[other], target=TARGET, hypothesis=H))
# print(f"\nGenerated {len(type_one)} raw type-one conjectures.")

# # ─────────────────────────────────────────────────────────────────────────────
# # B+) LP (R2 family) with transforms: plain + [base, sqrt/log/square(base)]
# # ─────────────────────────────────────────────────────────────────────────────
# from txgraffiti2025.generators.lp import lp_bounds, LPConfig

# LP_MIN_SUPPORT = max(8, int(0.05 * len(df)))
# LP_DIRECTION = "both"      # "upper", "lower", or "both"
# LP_TOL = 1e-9
# LP_MAX_DEN = 50

# def _ensure_augmented_cols(df: pd.DataFrame, col: str) -> list[str]:
#     """
#     Adds domain-safe transformed columns for `col`:
#       - sqrt(col)   (col >= 0)
#       - log(col)    (col > 0) [natural log]
#       - square(col) (always)
#     Returns the new/ensured column names.
#     """
#     created = []

#     # sqrt
#     c_sqrt = f"{col}__sqrt"
#     if c_sqrt not in df.columns:
#         s = pd.to_numeric(df[col], errors="coerce")
#         s = s.where(s >= 0)
#         df[c_sqrt] = np.sqrt(s)
#     created.append(c_sqrt)

#     # log
#     c_log = f"{col}__log"
#     if c_log not in df.columns:
#         s = pd.to_numeric(df[col], errors="coerce")
#         s = s.where(s > 0)
#         df[c_log] = np.log(s)
#     created.append(c_log)

#     # square
#     c_sq = f"{col}__square"
#     if c_sq not in df.columns:
#         s = pd.to_numeric(df[col], errors="coerce")
#         df[c_sq] = s * s
#     created.append(c_sq)

#     return created

# lp_conjs = []
# try:
#     for H in kept:
#         m = (H.mask(df) if H is not None else pd.Series(True, index=df.index)).astype(bool)
#         if int(m.sum()) < LP_MIN_SUPPORT:
#             continue

#         for base_col in numeric_cols:
#             # 1) single-feature fit
#             cfg_single = LPConfig(
#                 features=[base_col],
#                 target=TARGET,
#                 direction=LP_DIRECTION,
#                 max_denominator=LP_MAX_DEN,
#                 tol=LP_TOL,
#                 min_support=LP_MIN_SUPPORT,
#             )
#             lp_conjs.extend(list(lp_bounds(df, hypothesis=H, config=cfg_single)))

#             # 2) transformed pairs: [base, transform(base)]
#             aug_cols = _ensure_augmented_cols(df, base_col)
#             for aug in aug_cols:
#                 cfg_pair = LPConfig(
#                     features=[base_col, aug],
#                     target=TARGET,
#                     direction=LP_DIRECTION,
#                     max_denominator=LP_MAX_DEN,
#                     tol=LP_TOL,
#                     min_support=LP_MIN_SUPPORT,
#                 )
#                 lp_conjs.extend(list(lp_bounds(df, hypothesis=H, config=cfg_pair)))

#     print(f"Generated {len(lp_conjs)} LP (R2) conjectures (single + sqrt/log/square pairs).")
# except RuntimeError as e:
#     if "No LP solver found" in str(e):
#         print("LP stage skipped: install CBC or GLPK to enable R2 generator.")
#     else:
#         raise

# # ─────────────────────────────────────────────────────────────────────────────
# # C) Precompute constants cache for {base} ∪ {base∧each_boolean}; allow shifts
# # ─────────────────────────────────────────────────────────────────────────────
# from txgraffiti2025.forms.generic_conjecture import Conjecture
# base = Conjecture(relation=None, condition=None)._auto_base(df)

# from txgraffiti2025.processing.pre.constants_cache import precompute_constant_ratios_pairs
# const_cache = precompute_constant_ratios_pairs(
#     df,
#     base=base,
#     numeric_cols=numeric_cols,
#     shifts=(-1, 0, 1),
#     atol=1e-9,
#     rtol=1e-9,
#     min_support=max(8, int(0.05 * len(df))),
#     max_denominator=50,
#     skip_identity=True,   # keeps (N+1)/(N-1), skips (N+a)/(N+a)
# )

# # ─────────────────────────────────────────────────────────────────────────────
# # D) Generalize from constants onto supersets (kept)
# # ─────────────────────────────────────────────────────────────────────────────
# from txgraffiti2025.processing.post.generalize_from_constants import propose_generalizations_from_constants

# def _is_subset(df: pd.DataFrame, A, B) -> bool:
#     a = (A.mask(df) if A is not None else pd.Series(True, index=df.index)).astype(bool)
#     b = (B.mask(df) if B is not None else pd.Series(True, index=df.index)).astype(bool)
#     # Align indexes defensively (in case)
#     a = a.reindex(df.index, fill_value=False)
#     b = b.reindex(df.index, fill_value=False)
#     return not (a & ~b).any()

# gens_all = []
# for conj in type_one:
#     supersets = [H for H in kept if _is_subset(df, conj.condition, H)]
#     gens_all.extend(
#         propose_generalizations_from_constants(
#             df, conj, const_cache, candidate_hypotheses=supersets
#         )
#     )
# generalized_from_constants = [g.new_conjecture for g in gens_all]
# print(f"Proposed {len(generalized_from_constants)} generalized conjectures from constants.")

# # ─────────────────────────────────────────────────────────────────────────────
# # D2) Joint generalizer: slope (constants/reciprocals) + intercept
# # ─────────────────────────────────────────────────────────────────────────────
# from txgraffiti2025.processing.post.generalize import propose_joint_generalizations

# joint_generals = []
# for conj in type_one:
#     supersets = [H for H in kept if _is_subset(df, conj.condition, H)]
#     joint_generals.extend(
#         propose_joint_generalizations(
#             df,
#             conj,
#             cache=const_cache,
#             candidate_hypotheses=supersets,
#             candidate_intercepts=None,  # seed if desired with Exprs
#             relaxers_Z=None,            # seed if desired with Exprs
#             require_superset=True,
#             atol=1e-9,
#         )
#     )
# print(f"Proposed {len(joint_generals)} joint generalizations (slope/intercept/reciprocals).")

# # ─────────────────────────────────────────────────────────────────────────────
# # D3) (Optional) numeric refinement of RHS
# # ─────────────────────────────────────────────────────────────────────────────
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
#     for conj in (type_one + lp_conjs + generalized_from_constants + joint_generals):
#         refined.extend(refine_numeric_bounds(df, conj, config=cfg))
# print(f"Refined {len(refined)} conjectures numerically.")

# # ─────────────────────────────────────────────────────────────────────────────
# # E) Hazel → Morgan → Dalmatian
# # ─────────────────────────────────────────────────────────────────────────────
# from txgraffiti2025.processing.post import hazel_rank, morgan_filter
# from txgraffiti2025.processing.post import dalmatian_filter

# merged = (
#     type_one
#     + lp_conjs
#     + generalized_from_constants
#     + joint_generals
#     + (refined if USE_REFINER else [])
# )

# # Hazel first (ranking/filtering by touch)
# hazel_res = hazel_rank(df, merged, drop_frac=0.25, atol=1e-9)

# # Morgan next (dedupe by conclusion; prefer most-general hypothesis)
# morgan_res = morgan_filter(df, hazel_res.kept_sorted)
# print(f"Hazel → Morgan kept {len(morgan_res.kept)} / {len(merged)}")

# # Dalmatian last (significance; compares within same target/direction & hypothesis group)
# dalmatian_kept = dalmatian_filter(morgan_res.kept, df)

# print("\n=== saved conjectures (Type-One + LP + generalizations/refinements; Hazel→Morgan→Dalmatian) ===")
# for c in dalmatian_kept:
#     print(c.pretty(arrow='⇒'))

# run_independence_number_demo.py
from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import combinations

# --- Example data ---
from txgraffiti.example_data import graph_data as df

# =========================
# A) Hypotheses
# =========================
from txgraffiti2025.processing.pre.hypotheses import enumerate_boolean_hypotheses
from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses

hyps = enumerate_boolean_hypotheses(
    df, include_base=True, include_pairs=True, skip_always_false=True
)
min_support = int(0.05 * len(df))
kept, eqs = simplify_and_dedup_hypotheses(df, hyps, min_support=min_support)

print("=== kept (base-aware, simplified) ===")
for h in kept:
    print(h)
print("\n=== saved equivalence conjectures ===")
for c in eqs:
    print(c)

# =========================
# B) Type-one ratios
# =========================
TARGET = "domination_number"
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != TARGET and df[c].nunique(dropna=True) > 2]

from txgraffiti2025.generators import ratios
type_one = []
for other in numeric_cols:
    for H in kept:
        type_one.extend(ratios(df, features=[other], target=TARGET, hypothesis=H))
print(f"\nGenerated {len(type_one)} raw type-one conjectures.")

# =========================
# Safe transforms (no warnings)
# =========================
def safe_square(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    out = s * s
    return out

def safe_sqrt(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s >= 0
    out.loc[mask] = np.sqrt(s.loc[mask])
    return out

def safe_log(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s > 0
    out.loc[mask] = np.log(s.loc[mask])
    return out

# =========================
# PERFORMANCE-SAFE AUGMENTER
# =========================
class Augmenter:
    """Batch-create derived columns and concat to df in chunks to avoid fragmentation."""
    def __init__(self, df: pd.DataFrame, *, batch_size: int = 512):
        self.df = df
        self.pending: dict[str, pd.Series] = {}
        self.batch_size = batch_size

    def _ensure(self, name: str, series: pd.Series) -> str:
        if name in self.df.columns or name in self.pending:
            return name
        # Align index and dtype
        series = series.reindex(self.df.index)
        self.pending[name] = series
        if len(self.pending) >= self.batch_size:
            self.flush()
        return name

    def flush(self) -> None:
        if not self.pending:
            return
        new = pd.DataFrame(self.pending, index=self.df.index)
        self.df = pd.concat([self.df, new], axis=1)
        self.pending.clear()

    # --- unary transforms ---
    def ensure_unary(self, col: str) -> list[str]:
        s = pd.to_numeric(self.df[col], errors="coerce")

        c_sq = f"{col}__square"
        c_sqrt = f"{col}__sqrt"
        c_log = f"{col}__log"

        self._ensure(c_sq, safe_square(s))
        self._ensure(c_sqrt, safe_sqrt(s))
        self._ensure(c_log, safe_log(s))

        return [c_sq, c_sqrt, c_log]

    # --- pair features ---
    def ensure_absdiff(self, x1: str, x2: str) -> str:
        s1 = pd.to_numeric(self.df[x1], errors="coerce")
        s2 = pd.to_numeric(self.df[x2], errors="coerce")
        name = f"{x1}__absdiff__{x2}"
        return self._ensure(name, (s1 - s2).abs())

    def ensure_sqrt_of(self, base_col: str) -> str:
        s = pd.to_numeric(self.df[base_col], errors="coerce")
        name = f"{base_col}__sqrt"
        return self._ensure(name, safe_sqrt(s))

    def ensure_minmax(self, x1: str, x2: str) -> tuple[str, str]:
        s1 = pd.to_numeric(self.df[x1], errors="coerce")
        s2 = pd.to_numeric(self.df[x2], errors="coerce")
        cmin, cmax = f"min__{x1}__{x2}", f"max__{x1}__{x2}"
        self._ensure(cmin, np.minimum(s1, s2))
        self._ensure(cmax, np.maximum(s1, s2))
        return cmin, cmax

aug = Augmenter(df, batch_size=512)

def valid_support(df: pd.DataFrame, feats: list[str], mask: pd.Series, min_support: int) -> bool:
    sub = df.loc[mask, feats].apply(pd.to_numeric, errors="coerce")
    ok = int((~sub.isna().any(axis=1)).sum())
    return ok >= min_support

# =========================
# B+) LP with 1-var + transforms and 2-var combos
# =========================
from txgraffiti2025.generators.lp import lp_bounds, LPConfig

LP_MIN_SUPPORT = max(8, int(0.05 * len(df)))
LP_DIRECTION   = "both"
LP_TOL         = 1e-9
LP_MAX_DEN     = 50

lp_conjs = []
try:
    for H in kept:
        mask = (H.mask(aug.df) if H is not None else pd.Series(True, index=aug.df.index)).astype(bool)
        if int(mask.sum()) < LP_MIN_SUPPORT:
            continue

        # (i) 1-var: [x] and [x, transform(x)]
        for x in numeric_cols:
            # ensure transforms (buffered; no warnings)
            aug.ensure_unary(x)
            aug.flush()

            feats = [x]
            if valid_support(aug.df, feats, mask, LP_MIN_SUPPORT):
                cfg = LPConfig(features=feats, target=TARGET, direction=LP_DIRECTION,
                               max_denominator=LP_MAX_DEN, tol=LP_TOL, min_support=LP_MIN_SUPPORT)
                lp_conjs.extend(list(lp_bounds(aug.df, hypothesis=H, config=cfg)))

            for tcol in [f"{x}__square", f"{x}__sqrt", f"{x}__log"]:
                feats = [x, tcol]
                if not valid_support(aug.df, feats, mask, LP_MIN_SUPPORT):
                    continue
                cfg = LPConfig(features=feats, target=TARGET, direction=LP_DIRECTION,
                               max_denominator=LP_MAX_DEN, tol=LP_TOL, min_support=LP_MIN_SUPPORT)
                lp_conjs.extend(list(lp_bounds(aug.df, hypothesis=H, config=cfg)))

        # (ii) 2-var: [x1, x2], + absdiff (and sqrt), + minmax (+ sqrt(min))
        for x1, x2 in combinations(numeric_cols, 2):
            # base pair
            feats = [x1, x2]
            if valid_support(aug.df, feats, mask, LP_MIN_SUPPORT):
                cfg = LPConfig(features=feats, target=TARGET, direction=LP_DIRECTION,
                               max_denominator=LP_MAX_DEN, tol=LP_TOL, min_support=LP_MIN_SUPPORT)
                lp_conjs.extend(list(lp_bounds(aug.df, hypothesis=H, config=cfg)))

            # absdiff and sqrt(absdiff)
            absd = aug.ensure_absdiff(x1, x2)
            aug.flush()
            feats = [x1, x2, absd]
            if valid_support(aug.df, feats, mask, LP_MIN_SUPPORT):
                cfg = LPConfig(features=feats, target=TARGET, direction=LP_DIRECTION,
                               max_denominator=LP_MAX_DEN, tol=LP_TOL, min_support=LP_MIN_SUPPORT)
                lp_conjs.extend(list(lp_bounds(aug.df, hypothesis=H, config=cfg)))

            absd_sqrt = aug.ensure_sqrt_of(absd)
            aug.flush()
            feats = [x1, x2, absd, absd_sqrt]
            if valid_support(aug.df, feats, mask, LP_MIN_SUPPORT):
                cfg = LPConfig(features=feats, target=TARGET, direction=LP_DIRECTION,
                               max_denominator=LP_MAX_DEN, tol=LP_TOL, min_support=LP_MIN_SUPPORT)
                lp_conjs.extend(list(lp_bounds(aug.df, hypothesis=H, config=cfg)))

            # min/max and sqrt(min) (light)
            cmin, cmax = aug.ensure_minmax(x1, x2)
            aug.flush()
            feats = [cmin, cmax]
            if valid_support(aug.df, feats, mask, LP_MIN_SUPPORT):
                cfg = LPConfig(features=feats, target=TARGET, direction=LP_DIRECTION,
                               max_denominator=LP_MAX_DEN, tol=LP_TOL, min_support=LP_MIN_SUPPORT)
                lp_conjs.extend(list(lp_bounds(aug.df, hypothesis=H, config=cfg)))

            min_sqrt = aug.ensure_sqrt_of(cmin)
            aug.flush()
            feats = [cmin, cmax, min_sqrt]
            if valid_support(aug.df, feats, mask, LP_MIN_SUPPORT):
                cfg = LPConfig(features=feats, target=TARGET, direction=LP_DIRECTION,
                               max_denominator=LP_MAX_DEN, tol=LP_TOL, min_support=LP_MIN_SUPPORT)
                lp_conjs.extend(list(lp_bounds(aug.df, hypothesis=H, config=cfg)))

    print(f"Generated {len(lp_conjs)} LP (R2) conjectures (batched augments; 1-var + 2-var combos).")
except RuntimeError as e:
    if "No LP solver found" in str(e):
        print("LP stage skipped: install CBC or GLPK to enable R2 generator.")
    else:
        raise

# IMPORTANT: use the augmented df going forward
df = aug.df  # defragmented, with all new columns materialized

# =========================
# C) constants cache
# =========================
from txgraffiti2025.forms.generic_conjecture import Conjecture
base = Conjecture(relation=None, condition=None)._auto_base(df)

from txgraffiti2025.processing.pre.constants_cache import precompute_constant_ratios_pairs
const_cache = precompute_constant_ratios_pairs(
    df,
    base=base,
    numeric_cols=[c for c in numeric_cols if c in df.columns],
    shifts=(-1, 0, 1),
    atol=1e-9, rtol=1e-9,
    min_support=max(8, int(0.05 * len(df))),
    max_denominator=50,
    skip_identity=True,
)

# =========================
# D) generalize + joint + refine
# =========================
from txgraffiti2025.processing.post.generalize_from_constants import propose_generalizations_from_constants

def _is_subset(df: pd.DataFrame, A, B) -> bool:
    a = (A.mask(df) if A is not None else pd.Series(True, index=df.index)).astype(bool).reindex(df.index, fill_value=False)
    b = (B.mask(df) if B is not None else pd.Series(True, index=df.index)).astype(bool).reindex(df.index, fill_value=False)
    return not (a & ~b).any()

gens_all = []
for conj in type_one:
    supersets = [H for H in kept if _is_subset(df, conj.condition, H)]
    gens_all.extend(propose_generalizations_from_constants(df, conj, const_cache, candidate_hypotheses=supersets))
generalized_from_constants = [g.new_conjecture for g in gens_all]
print(f"Proposed {len(generalized_from_constants)} generalized conjectures from constants.")

from txgraffiti2025.processing.post.generalize import propose_joint_generalizations
joint_generals = []
for conj in type_one:
    supersets = [H for H in kept if _is_subset(df, conj.condition, H)]
    joint_generals.extend(
        propose_joint_generalizations(
            df, conj, cache=const_cache, candidate_hypotheses=supersets,
            candidate_intercepts=None, relaxers_Z=None,
            require_superset=True, atol=1e-9,
        )
    )
print(f"Proposed {len(joint_generals)} joint generalizations (slope/intercept/reciprocals).")

USE_REFINER = True
from txgraffiti2025.processing.post.refine_numeric import refine_numeric_bounds, RefinementConfig
refined = []
if USE_REFINER:
    cfg = RefinementConfig(
        try_whole_rhs_floor=True, try_whole_rhs_ceil=False, try_prod_floor=False,
        try_coeff_round_const=True, try_intercept_round_const=True,
        try_sqrt_intercept=True, try_log_intercept=False, require_tighter=True,
    )
    for conj in (type_one + lp_conjs + generalized_from_constants + joint_generals):
        refined.extend(refine_numeric_bounds(df, conj, config=cfg))
print(f"Refined {len(refined)} conjectures numerically.")

# =========================
# E) Hazel → Morgan → Dalmatian
# =========================
from txgraffiti2025.processing.post import hazel_rank, morgan_filter
from txgraffiti2025.processing.post import dalmatian_filter

merged = type_one + lp_conjs + generalized_from_constants + joint_generals + (refined if USE_REFINER else [])

hazel_res = hazel_rank(df, merged, drop_frac=0.25, atol=1e-9)
morgan_res = morgan_filter(df, hazel_res.kept_sorted)
print(f"Hazel → Morgan kept {len(morgan_res.kept)} / {len(merged)}")

dalmatian_kept = dalmatian_filter(morgan_res.kept, df)

# =========================
# Pretty output
# =========================
from txgraffiti2025.forms.tidy import tidy_conjecture

print("\n=== saved conjectures (Type-One + LP + generalizations/refinements; Hazel→Morgan→Dalmatian) ===")
for c in dalmatian_kept:
    print(tidy_conjecture(c.pretty(arrow='⇒'), unicode=True))
