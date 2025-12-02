#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import combinations
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ========= Load data =========

def load_dataset(csv_path: str) -> pd.DataFrame:
    # Treat empty strings as NaN so numeric columns keep numeric dtype.
    df = pd.read_csv(csv_path, na_values=[""])
    # Optional: ensure all-bool-ish integer columns (0/1) become bool.
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_integer_dtype(s):
            vals = pd.unique(s.dropna())
            try:
                ints = set(int(v) for v in vals)
            except Exception:
                continue
            if len(ints) <= 2 and ints.issubset({0, 1}):
                df[c] = s.astype(bool)
    return df

# ========= TxGraffiti imports (consolidated & consistent) =========

from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
    detect_base_hypothesis,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_and_dedup_hypotheses,
)

from txgraffiti2025.forms.utils import (
    Expr,
    Const,
    BinOp,
    to_expr,
    floor,
    ceil,
    sqrt,
)
from txgraffiti2025.forms.nonlinear import square
from txgraffiti2025.forms.generic_conjecture import (
    Eq,
    Le,
    Ge,
    AllOf,
    AnyOf,
    Conjecture,
    TRUE,
)

from txgraffiti2025.generators.lp import lp_bounds, LPConfig
from txgraffiti2025.processing.post import morgan_filter

from txgraffiti2025.processing.post.constant_ratios import (
    extract_ratio_pattern,
    find_constant_ratios_over_hypotheses,
)
from txgraffiti2025.processing.post.generalize_from_constants import (
    constants_from_ratios_hits,
    propose_generalizations_from_constants,
)
from txgraffiti2025.processing.post.intercept_generalizer import (
    propose_generalizations_from_intercept,
)
from txgraffiti2025.processing.post.refine_numeric import (
    refine_numeric_bounds,
    RefinementConfig,
)

# ========= Utilities =========

def to_frac_const(val: float, max_denom: int = 30) -> Const:
    """Convert float -> Const(Fraction) with bounded denominator."""
    return Const(Fraction(val).limit_denominator(max_denom))

def _is_add(e: Expr) -> bool:
    return isinstance(e, BinOp) and getattr(e, "fn", None) is np.add

def _split_add(e: Expr) -> Optional[Tuple[Expr, Expr]]:
    return (e.left, e.right) if _is_add(e) else None

def _touch_count(conj: Conjecture, df: pd.DataFrame) -> int:
    """Count rows (under conj.condition) where lhs == rhs within tolerance."""
    try:
        lhs = conj.relation.left.eval(df)
        rhs = conj.relation.right.eval(df)
        mask = conj.condition.mask(df) if conj.condition else np.ones(len(df), dtype=bool)
        lhs = np.asarray(lhs, dtype=float)
        rhs = np.asarray(rhs, dtype=float)
        m = mask & np.isfinite(lhs) & np.isfinite(rhs)
        if not np.any(m):
            return 0
        touch = np.isclose(lhs[m], rhs[m], rtol=1e-8, atol=1e-8)
        return int(np.sum(touch))
    except Exception:
        return 0

def _all_le_on_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> bool:
    """Check a <= b on rows where mask & both finite."""
    m = mask & np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return False
    return bool(np.all(a[m] <= b[m]))

def _caro_trivial_masked(W_arr, x_arr, y_arr, z_arr, mask) -> bool:
    """
    Trivial if (W<=y & x<=z) OR (W<=z & x<=y) on rows where mask & all four finite.
    """
    common = mask & np.isfinite(W_arr) & np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
    if not np.any(common):
        return False
    c1 = _all_le_on_mask(W_arr, y_arr, common) and _all_le_on_mask(x_arr, z_arr, common)
    if c1:
        return True
    c2 = _all_le_on_mask(W_arr, z_arr, common) and _all_le_on_mask(x_arr, y_arr, common)
    return c2

class EvalCache:
    """
    Cache arrays for columns and their transforms, scoped to a filtered dataframe (df_temp).
    """
    def __init__(self, df_temp: pd.DataFrame, cols: Iterable[str]):
        self.df = df_temp
        self.cols = list(cols)
        self.arr: dict[str, np.ndarray] = {}
        self.sqrt_arr: dict[str, np.ndarray] = {}
        self.sq_arr: dict[str, np.ndarray] = {}

    def col(self, name: str) -> np.ndarray:
        a = self.arr.get(name)
        if a is None:
            s = to_expr(name).eval(self.df)
            a = np.asarray(s, dtype=float)
            self.arr[name] = a
        return a

    def sqrt_col(self, name: str) -> np.ndarray:
        a = self.sqrt_arr.get(name)
        if a is None:
            x = self.col(name)
            # callers ensure positivity
            a = np.sqrt(x.astype(float, copy=False))
            self.sqrt_arr[name] = a
        return a

    def sq_col(self, name: str) -> np.ndarray:
        a = self.sq_arr.get(name)
        if a is None:
            x = self.col(name)
            a = np.square(x.astype(float, copy=False))
            self.sq_arr[name] = a
        return a

def _pick_best_ge(t_arr: np.ndarray, rhs_variants, strength='mean'):
    """
    target >= rhs. rhs_variants: list of (label, rhs_array, make_expr_fn).
    Choose the strongest that is true on finite rows.
    """
    best = None
    best_score = -np.inf
    for lab, rhs, make_expr in rhs_variants:
        rhs = np.asarray(rhs, dtype=float)
        m = np.isfinite(t_arr) & np.isfinite(rhs)
        if not np.any(m):
            continue
        ok = np.all(t_arr[m] >= rhs[m])
        if not ok:
            continue
        score = np.mean(rhs[m]) if strength == 'mean' else np.median(rhs[m])
        if score > best_score:
            best = (lab, make_expr)
            best_score = score
    return best

def _pick_best_le(t_arr: np.ndarray, rhs_variants, strength='mean'):
    """
    target <= rhs. Prefer smaller rhs among those true.
    """
    best = None
    best_score = -np.inf
    for lab, rhs, make_expr in rhs_variants:
        rhs = np.asarray(rhs, dtype=float)
        m = np.isfinite(t_arr) & np.isfinite(rhs)
        if not np.any(m):
            continue
        ok = np.all(t_arr[m] <= rhs[m])
        if not ok:
            continue
        score = -np.mean(rhs[m]) if strength == 'mean' else -np.median(rhs[m])
        if score > best_score:
            best = (lab, make_expr)
            best_score = score
    return best

# ========= TxGraffiti wrapper =========

class TxGraffiti:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.base = detect_base_hypothesis(df)
        self.hyps_all = enumerate_boolean_hypotheses(
            df,
            treat_binary_ints=True,
            include_base=True,
            include_pairs=True,
            skip_always_false=True,
        )
        self._simplify_and_dedup_hypotheses()
        self._set_columns()

    def _simplify_and_dedup_hypotheses(self):
        self.hyps_kept, _ = simplify_and_dedup_hypotheses(
            self.df,
            self.hyps_all,
            min_support=10,
            treat_binary_ints=True,
        )

    def _set_columns(self):
        bool_cols: list[str] = []
        for c in self.df.columns:
            s = self.df[c]
            if s.dtype == bool:
                bool_cols.append(c)
            elif pd.api.types.is_integer_dtype(s):
                vals = pd.unique(s.dropna())
                try:
                    ints = set(int(v) for v in vals)
                except Exception:
                    continue
                if len(ints) <= 2 and ints.issubset({0, 1}):
                    bool_cols.append(c)
        self.bool_columns = bool_cols
        self.numeric_columns = [
            c for c in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[c]) and c not in bool_cols
        ]

# ========= Mining logic =========

def mine_linear_and_mixed(ai: TxGraffiti, df: pd.DataFrame, target_name: str):
    target = to_expr(target_name)

    upper_conjectures: list[Conjecture] = []
    lower_conjectures: list[Conjecture] = []

    for hypothesis in tqdm(ai.hyps_kept, desc="Hypotheses", unit="hyp"):
        mask = hypothesis.mask(df)
        df_temp = df[mask]
        if df_temp.empty:
            continue

        cache = EvalCache(df_temp, ai.numeric_columns)
        t_arr = target.eval(df_temp).values.astype(float, copy=False)

        for other in tqdm(ai.numeric_columns, desc="Primary columns", unit="col", leave=False):
            if other == target_name:
                continue
            x_arr = cache.col(other)
            m = np.isfinite(t_arr) & np.isfinite(x_arr)
            if not np.any(m):
                continue
            if np.min(x_arr[m]) <= 0:
                continue

            rx = t_arr[m] / x_arr[m]
            cmin_f = float(np.min(rx))
            cmax_f = float(np.max(rx))
            cmin_c = to_frac_const(cmin_f)
            cmax_c = to_frac_const(cmax_f)

            # y ≥ cmin * x
            lower_conjectures.append(Conjecture(Ge(target, cmin_c * to_expr(other)), hypothesis))

            # y ≤ cmax * x
            base_upper_expr = cmax_c * to_expr(other)
            upper_conjectures.append(Conjecture(Le(target, base_upper_expr), hypothesis))

            # floor / ceil refinements (array-guarded)
            floor_arr = np.floor(cmax_f * x_arr)
            mf = m & np.isfinite(floor_arr)
            if np.any(mf) and np.all(t_arr[mf] <= floor_arr[mf]) and cmax_c.value.denominator > 1:
                upper_conjectures.append(Conjecture(Le(target, floor(base_upper_expr)), hypothesis))

            ceil_arr = np.ceil(cmin_f * x_arr)
            mc = m & np.isfinite(ceil_arr)
            if np.any(mc) and np.all(t_arr[mc] >= ceil_arr[mc]) and cmin_c.value.denominator > 1:
                lower_conjectures.append(Conjecture(Ge(target, ceil(cmin_c * to_expr(other))), hypothesis))

            # ---------- TWO-FEATURE MIXES (x with sqrt(x2), and x with (x2)^2) ----------
            for other2 in tqdm(ai.numeric_columns, desc="Secondary columns", unit="col", leave=False):
                if other2 == target_name:
                    continue

                x2_arr = cache.col(other2)
                m2 = m & np.isfinite(x2_arr)
                if not np.any(m2):
                    continue

                # ==== sqrt mix ====
                # Require positivity for sqrt
                if np.min(x2_arr[m2]) > 0:
                    sqrt_x2_arr = cache.sqrt_col(other2)
                    m_s = m2 & np.isfinite(sqrt_x2_arr)
                    if np.any(m_s) and np.min(sqrt_x2_arr[m_s]) > 0:
                        r_sqrt = t_arr[m_s] / sqrt_x2_arr[m_s]
                        s_cmin_f = float(np.min(r_sqrt))
                        s_cmax_f = float(np.max(r_sqrt))

                        mix_lower_arr = 0.5 * (cmin_f * x_arr + s_cmin_f * sqrt_x2_arr)
                        mix_upper_arr = 0.5 * (cmax_f * x_arr + s_cmax_f * sqrt_x2_arr)

                        # ≥ variants (choose best)
                        lower_mix_variants = [
                            ("base",
                             mix_lower_arr,
                             lambda: (to_frac_const(0.5 * cmin_f) * to_expr(other)
                                      + to_frac_const(0.5 * s_cmin_f) * sqrt(to_expr(other2)))),

                            ("ceil whole",
                             np.ceil(mix_lower_arr),
                             lambda: ceil(
                                 to_frac_const(0.5 * cmin_f) * to_expr(other)
                                 + to_frac_const(0.5 * s_cmin_f) * sqrt(to_expr(other2))
                             )),

                            ("ceil-split-1",
                             np.ceil(0.5 * cmin_f * x_arr) + np.ceil(0.5 * s_cmin_f * sqrt_x2_arr) - 1.0,
                             lambda: (ceil(to_frac_const(0.5 * cmin_f) * to_expr(other))
                                      + ceil(to_frac_const(0.5 * s_cmin_f) * sqrt(to_expr(other2)))
                                      - Const(1))),
                        ]
                        choice = _pick_best_ge(t_arr, lower_mix_variants)
                        if choice is not None:
                            _, make_expr = choice
                            lower_conjectures.append(Conjecture(Ge(target, make_expr()), hypothesis))

                        # ≤ variants (choose best)
                        upper_mix_variants = [
                            ("base",
                             mix_upper_arr,
                             lambda: (to_frac_const(0.5 * cmax_f) * to_expr(other)
                                      + to_frac_const(0.5 * s_cmax_f) * sqrt(to_expr(other2)))),

                            ("floor whole",
                             np.floor(mix_upper_arr),
                             lambda: floor(
                                 to_frac_const(0.5 * cmax_f) * to_expr(other)
                                 + to_frac_const(0.5 * s_cmax_f) * sqrt(to_expr(other2))
                             )),

                            ("floor-split",
                             np.floor(0.5 * cmax_f * x_arr) + np.floor(0.5 * s_cmax_f * sqrt_x2_arr),
                             lambda: (floor(to_frac_const(0.5 * cmax_f) * to_expr(other))
                                      + floor(to_frac_const(0.5 * s_cmax_f) * sqrt(to_expr(other2))))),
                        ]
                        choice = _pick_best_le(t_arr, upper_mix_variants)
                        if choice is not None:
                            _, make_expr = choice
                            upper_conjectures.append(Conjecture(Le(target, make_expr()), hypothesis))

                # ==== square mix ====
                sq_x2_arr = cache.sq_col(other2)
                m_sq = m2 & np.isfinite(sq_x2_arr)
                if np.any(m_sq) and np.min(sq_x2_arr[m_sq]) > 0:
                    r_sq = t_arr[m_sq] / sq_x2_arr[m_sq]
                    sq_cmin_f = float(np.min(r_sq))
                    sq_cmax_f = float(np.max(r_sq))

                    mix_lower_sq_arr = 0.5 * (cmin_f * x_arr + sq_cmin_f * sq_x2_arr)
                    mix_upper_sq_arr = 0.5 * (cmax_f * x_arr + sq_cmax_f * sq_x2_arr)

                    # ≥ variants
                    lower_sq_variants = [
                        ("base",
                         mix_lower_sq_arr,
                         lambda: (to_frac_const(0.5 * cmin_f) * to_expr(other)
                                  + to_frac_const(0.5 * sq_cmin_f) * square(to_expr(other2)))),

                        ("ceil whole",
                         np.ceil(mix_lower_sq_arr),
                         lambda: ceil(
                             to_frac_const(0.5 * cmin_f) * to_expr(other)
                             + to_frac_const(0.5 * sq_cmin_f) * square(to_expr(other2))
                         )),
                    ]
                    choice = _pick_best_ge(t_arr, lower_sq_variants)
                    if choice is not None:
                        _, make_expr = choice
                        lower_conjectures.append(Conjecture(Ge(target, make_expr()), hypothesis))

                    # ≤ variants
                    upper_sq_variants = [
                        ("base",
                         mix_upper_sq_arr,
                         lambda: (to_frac_const(0.5 * cmax_f) * to_expr(other)
                                  + to_frac_const(0.5 * sq_cmax_f) * square(to_expr(other2)))),

                        ("floor whole",
                         np.floor(mix_upper_sq_arr),
                         lambda: floor(
                             to_frac_const(0.5 * cmax_f) * to_expr(other)
                             + to_frac_const(0.5 * sq_cmax_f) * square(to_expr(other2))
                         )),
                    ]
                    choice = _pick_best_le(t_arr, upper_sq_variants)
                    if choice is not None:
                        _, make_expr = choice
                        upper_conjectures.append(Conjecture(Le(target, make_expr()), hypothesis))

    return lower_conjectures, upper_conjectures

def mine_product_caro(ai: TxGraffiti, df: pd.DataFrame) -> list[Conjecture]:
    """
    Search product inequalities of the form W*x ≤ y*z under each hypothesis,
    skipping trivial “Caro” cases and guarding for finiteness.
    """
    product_upper_conjectures: list[Conjecture] = []

    for hypothesis in tqdm(ai.hyps_kept, desc="Hypotheses (Caro)", unit="hyp"):
        hyp_mask = hypothesis.mask(df)
        df_temp = df[hyp_mask]
        if df_temp.empty:
            continue

        cand_cols = list(ai.numeric_columns)
        arrays = {c: to_expr(c).eval(df_temp).values.astype(float, copy=False) for c in cand_cols}
        finite = {c: np.isfinite(arrays[c]) for c in cand_cols}

        for (W, x) in tqdm(list(combinations(cand_cols, 2)), desc="Product LHS pairs", unit="pair", leave=False):
            W_arr = arrays[W]; x_arr = arrays[x]
            L_arr = W_arr * x_arr
            L_fin = finite[W] & finite[x] & np.isfinite(L_arr)
            if not np.any(L_fin):
                continue

            for (y, z) in tqdm(list(combinations(cand_cols, 2)), desc="Product RHS pairs", unit="pair", leave=False):
                y_arr = arrays[y]; z_arr = arrays[z]
                R_arr = y_arr * z_arr
                R_fin = finite[y] & finite[z] & np.isfinite(R_arr)

                both_fin = L_fin & R_fin
                if not np.any(both_fin):
                    continue

                if _caro_trivial_masked(W_arr, x_arr, y_arr, z_arr, both_fin):
                    continue

                if _all_le_on_mask(L_arr, R_arr, both_fin):
                    lhs = to_expr(W) * to_expr(x)
                    rhs = to_expr(y) * to_expr(z)
                    product_upper_conjectures.append(Conjecture(Le(lhs, rhs), hypothesis))

    return product_upper_conjectures

# ========= Main =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="integers_dataset.csv", help="Input CSV path.")
    ap.add_argument("--target", type=str, default="", help="Name of numeric target column.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (only used if target not specified).")
    ap.add_argument("--max-print", type=int, default=100, help="Max conjectures to print per bucket.")
    args = ap.parse_args()

    df = load_dataset(args.csv)
    ai = TxGraffiti(df)

    # Choose a target deterministically if none provided.
    target_name = args.target.strip()
    if not target_name:
        # stable choice: the first numeric column that is not obviously an ID
        candidates = [c for c in ai.numeric_columns if c.lower() not in {"n", "id", "index"}]
        if not candidates:
            raise ValueError("No numeric target candidates found.")
        target_name = candidates[0]
    if target_name not in df.columns:
        raise ValueError(f"Target '{target_name}' not found in columns.")

    print(f"[info] Using target: {target_name}")

    # Mine conjectures (linear + mixed)
    lower_conjectures, upper_conjectures = mine_linear_and_mixed(ai, df, target_name)

    # Rank + basic Morgan filter (upper)
    upper_conjectures.sort(key=lambda c: _touch_count(c, df), reverse=True)
    upper_conjectures = [c for c in upper_conjectures if _touch_count(c, df) > 2]
    morgan_filtered_upper = morgan_filter(df, upper_conjectures)

    # Rank + basic Morgan filter (lower)
    lower_conjectures.sort(key=lambda c: _touch_count(c, df), reverse=True)
    lower_conjectures = [c for c in lower_conjectures if _touch_count(c, df) > 2]
    morgan_filtered_lower = morgan_filter(df, lower_conjectures)

    # Product (Caro) inequalities
    product_conjs = mine_product_caro(ai, df)
    product_conjs.sort(key=lambda c: _touch_count(c, df), reverse=True)
    product_conjs = [c for c in product_conjs if _touch_count(c, df) > 2]
    morgan_filtered_product = morgan_filter(df, product_conjs)

    # ===== Output =====
    max_print = max(1, int(args.max_print))

    print("\n---------- Caro Product Conjectures ----------")
    kept = getattr(morgan_filtered_product, "kept", product_conjs)
    for i, c in enumerate(kept[:max_print], 1):
        print(f"Conjecture Caro.{i}. {c.pretty()}\n")

    print("\n----------- Non-Linear Upper Conjectures ----------")
    kept_u = getattr(morgan_filtered_upper, "kept", upper_conjectures)
    for i, c in enumerate(kept_u[:max_print], 1):
        print(f"Conjecture NLU.{i}. {c.pretty()}\n")

    print("\n----------- Non-Linear Lower Conjectures ----------")
    kept_l = getattr(morgan_filtered_lower, "kept", lower_conjectures)
    for i, c in enumerate(kept_l[:max_print], 1):
        print(f"Conjecture NLL.{i}. {c.pretty()}\n")

if __name__ == "__main__":
    main()
