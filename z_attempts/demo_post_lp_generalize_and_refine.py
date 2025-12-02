# demos/demo_post_lp_generalize_and_refine.py
from __future__ import annotations
import numpy as np
import pandas as pd

from txgraffiti.example_data import graph_data as df

# Hypotheses
from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
    detect_base_hypothesis,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_and_dedup_hypotheses,
)

# Forms
from txgraffiti2025.forms.utils import to_expr, Expr, Const
from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.logexp import log_base, sqrt
from txgraffiti2025.forms.nonlinear import square

# Generators
from txgraffiti2025.generators.lp import lp_bounds, LPConfig

# Post: constant finder + generalizers + refinement
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
    refine_numeric_bounds, RefinementConfig
)

# ----------------------------- small helpers -----------------------------

def _header(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78 + "\n")

def _boolish_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        s = df[c]
        if s.dtype == bool:
            cols.append(c)
        elif pd.api.types.is_integer_dtype(s):
            vals = pd.unique(s.dropna())
            try: ints = set(int(v) for v in vals)
            except Exception: continue
            if len(ints) <= 2 and ints.issubset({0,1}):
                cols.append(c)
    return cols

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in _boolish_cols(df)]

def _mask(df: pd.DataFrame, H: Predicate | None) -> pd.Series:
    if H is None:
        return pd.Series(True, index=df.index)
    return H.mask(df).reindex(df.index, fill_value=False).astype(bool)

def _domain_ok_pos(series: pd.Series, mask: pd.Series, min_support: int = 3) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    m = mask & s.notna() & (s > 0)
    return int(m.sum()) >= min_support

def build_feature_exprs_for_hypothesis(
    df: pd.DataFrame,
    hypothesis: Predicate | None,
    raw_features: list[str],
    *,
    include_sqrt=True,
    include_log=True,
    include_square=True,
    min_support: int = 3,
) -> list[Expr]:
    Hmask = _mask(df, hypothesis)
    feats: list[Expr] = []
    seen: set[str] = set()
    for col in raw_features:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        φ = to_expr(col)
        r = repr(φ)
        if r not in seen:
            feats.append(φ); seen.add(r)
        if include_sqrt and _domain_ok_pos(x, Hmask, min_support=min_support):
            φs = sqrt(col); rs = repr(φs)
            if rs not in seen: feats.append(φs); seen.add(rs)
        if include_log and _domain_ok_pos(x, Hmask, min_support=min_support):
            φl = log_base(col); rl = repr(φl)
            if rl not in seen: feats.append(φl); seen.add(rl)
        if include_square:
            φq = square(col); rq = repr(φq)
            if rq not in seen: feats.append(φq); seen.add(rq)
    return feats

def _finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _touch_count(conj: Conjecture, df: pd.DataFrame, tol: float = 1e-8) -> int:
    try:
        lhs = _finite(conj.relation.left.eval(df))
        rhs = _finite(conj.relation.right.eval(df))
        m = _mask(df, conj.condition)
        v = np.isclose(lhs[m], rhs[m], rtol=tol, atol=tol)
        return int(v.sum())
    except Exception:
        return 0

def _pick_hypotheses(df: pd.DataFrame, *, min_support: int) -> tuple[Predicate | None, list[Predicate]]:
    base = detect_base_hypothesis(df)
    hyps_all = enumerate_boolean_hypotheses(
        df, treat_binary_ints=True, include_base=True, include_pairs=True, skip_always_false=True
    )
    kept, _ = simplify_and_dedup_hypotheses(df, hyps_all, min_support=min_support, treat_binary_ints=True)
    pairs = [H for H in kept if "AndPred" in type(H).__name__]
    singles = [H for H in kept if "AndPred" not in type(H).__name__]
    pairs.sort(key=lambda H: int(H.mask(df).sum()), reverse=True)
    singles.sort(key=lambda H: int(H.mask(df).sum()), reverse=True)
    H0 = pairs[0] if pairs else (singles[0] if singles else None)
    return H0, kept

# ---- NEW: strict vetting for LP (no NaN/inf in y or in kept features on used rows)

def vet_features_for_lp(
    df: pd.DataFrame,
    hypothesis: Predicate | None,
    target_col: str,
    features: list[Expr],
    *,
    min_rows: int = 8,
    verbose: bool = True,
) -> tuple[pd.Series, list[Expr], pd.Series]:
    """
    Returns (row_mask, vetted_features, y) where:
      - row_mask: rows under hypothesis with finite target and finite values for EVERY kept feature
      - vetted_features: only those Expr whose values are finite on row_mask
      - y: finite target on row_mask
    """
    Hmask = _mask(df, hypothesis)
    y_raw = _finite(df[target_col] if target_col in df.columns else pd.Series(np.nan, index=df.index))
    # Start from rows with finite target under H
    row_mask = Hmask & y_raw.notna()

    # Evaluate all candidate features once
    feat_vals = []
    dropped = []
    for φ in features:
        try:
            s = _finite(φ.eval(df))
        except Exception:
            s = pd.Series(np.nan, index=df.index)
        feat_vals.append(s)

    # Keep features that are finite everywhere on current row_mask
    kept_feats: list[Expr] = []
    for φ, s in zip(features, feat_vals):
        ok = row_mask & s.notna()
        # Require: on the rows we plan to use, this feature must be finite everywhere
        if ok.sum() == row_mask.sum() and row_mask.sum() >= min_rows:
            kept_feats.append(φ)
        else:
            dropped.append((repr(φ), int((~s.notna() & row_mask).sum())))

    # Rebuild row_mask so that **all** kept features are finite there
    if kept_feats:
        for s in feat_vals:
            # only intersect with s if its corresponding feature was kept
            # (match by repr to avoid re-evals)
            pass
        # Build a mask requiring finiteness for every kept feature
        for φ in kept_feats:
            s = _finite(φ.eval(df))
            row_mask &= s.notna()
    # Final y on the vetted mask
    y = y_raw[row_mask]

    if verbose:
        print(f"LP vetting: start rows under H0 = {int(Hmask.sum())}")
        print(f"LP vetting: rows with finite target = {int((Hmask & y_raw.notna()).sum())}")
        print(f"LP vetting: rows kept after feature finiteness = {int(row_mask.sum())}")
        print(f"LP vetting: kept features = {len(kept_feats)} / {len(features)}")
        if dropped:
            shown = ", ".join(f"{name} (nan_on_mask={cnt})" for name, cnt in dropped[:8])
            more = "" if len(dropped) <= 8 else f", +{len(dropped)-8} more"
            print(f"LP vetting: dropped examples: {shown}{more}")

    return row_mask, kept_feats, y

# ----------------------------- demo main -----------------------------

def main():
    _header("LP generation → generalize (coeff/intercept) → refine → sort by touch")
    print(f"Dataset: {df.shape}")

    # 1) Hypotheses
    _, kept = _pick_hypotheses(df, min_support=max(30, int(0.10 * len(df))))
    for H0 in kept:
        print("Base (narrow) hypothesis H0:", repr(H0) if H0 else "None")
        if H0 is None:
            print("No suitable H0; exiting.")
            return

        # 2) Choose target & candidate features (exclude target)
        target = "domination_number"
        num_cols = _numeric_cols(df)
        raw_features = [c for c in num_cols if c != target]
        print("Target:", target)
        print("Raw features (first 10):", raw_features[:10])

        # 3) Build φ features under H0
        _header("LP bounds on H0 (sum-of-slacks)")
        φs = build_feature_exprs_for_hypothesis(
            df, H0, raw_features, include_sqrt=True, include_log=True, include_square=True, min_support=3
        )
        print(f"Feature count under H0 (pre-vet): {len(φs)}")

        # ---- Strict vetting: ensure no NaN/inf in the rows used by LP
        row_mask, φs_kept, y_fin = vet_features_for_lp(
            df, H0, target, φs, min_rows=8, verbose=True
        )
        if len(φs_kept) == 0 or int(row_mask.sum()) < 8:
            print("After vetting, not enough clean features or rows for LP; exiting.")
            return
        print(f"Feature count after vet: {len(φs_kept)}")

        # 4) Run LP on the **same hypothesis**; lp_bounds will re-evaluate features,
        #    but our vetting makes sure they’re finite everywhere on the mask.
        # cfg = LPConfig(
        #     features=φs_kept,
        #     target=target,
        #     direction="both",       # get ≤ and ≥
        #     max_denominator=50,
        #     tol=1e-9,
        #     min_support=8,
        # )

        seeds: list[Conjecture] = []
        for other in φs_kept:
            cfg = LPConfig(
                features=[other],
                target=target,
                direction="both",       # get ≤ and ≥
                max_denominator=50,
                tol=1e-9,
                min_support=8,
            )
            try:
                for conj in lp_bounds(df, hypothesis=H0, config=cfg):
                    seeds.append(conj)
                    app, holds, _ = conj.check(df)
                    eq = _touch_count(conj, df)
                    print("  LP:", conj.pretty(unicode_ops=True),
                        f"| applicable={int(app.sum())}, satisfied={int((app & holds).sum())}, eq={eq}")
            except RuntimeError as e:
                print("LP solver unavailable:", e)
                return

        if not seeds:
            print("No LP seeds; exiting.")
            return

        # 5) Mine structural K candidates across subclasses (for coefficient generalizer)
        _header("Mining constant ratios (for coefficient lifts)")
        hits = find_constant_ratios_over_hypotheses(
            df,
            hypotheses=kept,
            shifts=tuple(range(-2, 3)),
            constancy="cv",
            cv_tol=0.10,
            min_support=max(12, int(0.08 * len(df))),
            max_denominator=30,
            top_k_per_hypothesis=20,
        )
        cands = constants_from_ratios_hits(hits)
        print(f"Constant-ratio candidates: {len(cands)}")

        # 6) For each LP seed:
        #    - If ratio pattern, try coefficient generalization (c → K)
        #    - Always try intercept generalization
        #    - Run numeric refinement on each accepted result
        _header("Generalization + refinement")
        kept_rows = []  # (kind, conj, superset, meta, align)

        for s in seeds:
            kept_rows.append(("original", s, s.condition, {}, None))

            if extract_ratio_pattern(s) is not None:
                gens_coeff = propose_generalizations_from_constants(
                    df,
                    s,
                    candidate_hypotheses=kept,
                    constant_candidates=cands,
                    constants_cache=None,
                    require_strict_superset=True,
                    drop_near_one_coeffs=True,
                    near_one_med_tol=1e-2,
                    near_one_cv_tol=2e-2,
                    atol=1e-6,
                    rtol=5e-2,
                )
                for g in gens_coeff:
                    kept_rows.append(("generalized-coeff", g.new_conjecture, g.witness_superset,
                                    g.meta.get("candidate_info", {}), g.meta.get("align", None)))

            gens_inter = propose_generalizations_from_intercept(
                df,
                s,
                cache=None,
                candidate_hypotheses=kept,
                candidate_intercepts=None,
                relaxers_Z=None,
                require_superset=True,
                tol=1e-9,
            )
            for g in gens_inter:
                kept_rows.append(("generalized-intercept", g.new_conjecture, g.new_conjecture.condition, {}, None))

            ref_cfg = RefinementConfig(
                try_whole_rhs_floor=True,
                try_whole_rhs_ceil=False,
                try_prod_floor=False,
                try_coeff_round_const=True,
                try_intercept_round_const=True,
                try_sqrt_intercept=True,
                try_log_intercept=False,
                require_tighter=True,
            )
            refined = refine_numeric_bounds(df, s, config=ref_cfg)
            for rconj in refined:
                kept_rows.append(("refined", rconj, rconj.condition, {}, None))

        # 7) Dedup by structure + mask
        uniq = []
        seen = set()
        for kind, conj, Hsup, meta, align in kept_rows:
            key = (repr(conj), tuple(_mask(df, conj.condition).values))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((kind, conj, Hsup, meta, align))

        # 8) Sort by touch count and print
        print(f"\nTotal kept (orig + generalized + refined, deduped): {len(uniq)}")
        scored = [(kind, conj, Hsup, meta, align, _touch_count(conj, df)) for kind, conj, Hsup, meta, align in uniq]
        scored.sort(key=lambda t: t[5], reverse=True)

        _header("Top 15 by touch count")
        for j, (kind, conj, Hsup, meta, align, t) in enumerate(scored[:15], 1):
            app, holds, _ = conj.check(df)
            print(f"[#{j:02d}] ({kind})  {conj.pretty(unicode_ops=True)}")
            print(f"     touch={t} | applicable={int(app.sum())} | satisfied={int((app & holds).sum())}")
            if align:
                print(f"     align: K@base_median={align['K@base_median']:.4g}, |median-c|={align['abs_err_to_c']:.4g}")
            print()

        _header("Done")


if __name__ == "__main__":
    main()
