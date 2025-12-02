# conclusion_generators_examples.py
# -----------------------------------------------------------------------------
# Educational demo:
#  - enumerate "good" hypotheses from boolean-like columns
#  - pick a numeric target
#  - build feature set (raw, sqrt, log, square) per hypothesis when domains allow
#  - generate conjectures via ratio and LP generators
#  - print pretty forms + quick diagnostics
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Sequence

import math
import pandas as pd
import numpy as np

# Hypothesis discovery / simplification
from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
    detect_base_hypothesis,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_and_dedup_hypotheses,
)

# Conjecture forms & utils
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.forms.utils import to_expr, Expr
from txgraffiti2025.forms.logexp import log_base, sqrt
from txgraffiti2025.forms.nonlinear import square

# Generators
from txgraffiti2025.generators.ratios import ratios
from txgraffiti2025.generators.lp import lp_bounds, LPConfig


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _header(title: str):
    print("\n" + "="*78)
    print(title)
    print("="*78 + "\n")

def _show_df(df: pd.DataFrame, title="DataFrame"):
    _header(f"0) {title}")
    print(df, "\n")

def _bool_cols_readable(df: pd.DataFrame) -> List[str]:
    # For explanation/printing only: heuristic boolean-ish columns
    cols = []
    for c in df.columns:
        s = df[c]
        if s.dtype == bool:
            cols.append(c)
        elif pd.api.types.is_integer_dtype(s):
            vals = pd.unique(s.dropna())
            try:
                ints = set(int(v) for v in vals)
            except Exception:
                continue
            if len(ints) <= 2 and ints.issubset({0, 1}):
                cols.append(c)
    return cols

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and c not in _bool_cols_readable(df):
            out.append(c)
    return out

def _domain_ok_pos(series: pd.Series, mask: pd.Series, min_support: int = 2) -> bool:
    """True if series > 0 on enough masked rows to safely use sqrt/log."""
    s = pd.to_numeric(series, errors="coerce")
    m = mask & s.notna() & (s > 0)
    return int(m.sum()) >= min_support

def build_feature_exprs_for_hypothesis(
    df: pd.DataFrame,
    hypothesis: Predicate,
    raw_features: Sequence[str],
    *,
    include_sqrt=True,
    include_log=True,
    include_square=True,
    min_support: int = 2,
) -> List[Expr]:
    """
    From a list of raw numeric columns, build a richer feature set under hypothesis:
      - φ = x
      - √x  (only if x>0 has enough support under H)
      - log(x)  (only if x>0 has enough support under H)
      - x²
    """
    Hmask = hypothesis.mask(df).reindex(df.index, fill_value=False).astype(bool)
    feats: List[Expr] = []
    seen_reprs: set[str] = set()

    for col in raw_features:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")
        # identity
        φ = to_expr(col)
        r = repr(φ)
        if r not in seen_reprs:
            feats.append(φ); seen_reprs.add(r)

        # sqrt/log only if positive domain under H has sufficient support
        if include_sqrt and _domain_ok_pos(x, Hmask, min_support=min_support):
            φs = sqrt(col)
            rs = repr(φs)
            if rs not in seen_reprs:
                feats.append(φs); seen_reprs.add(rs)

        if include_log and _domain_ok_pos(x, Hmask, min_support=min_support):
            φl = log_base(col)  # natural log
            rl = repr(φl)
            if rl not in seen_reprs:
                feats.append(φl); seen_reprs.add(rl)

        if include_square:
            φq = square(col)
            rq = repr(φq)
            if rq not in seen_reprs:
                feats.append(φq); seen_reprs.add(rq)

    return feats

def summarize_conjecture(df: pd.DataFrame, conj: Conjecture, label: str = ""):
    """Pretty-print a conjecture and a tiny diagnostic on the demo df."""
    ptxt = conj.pretty(unicode_ops=True)
    print((f"{label} " if label else "") + ptxt)
    applicable, holds, failures = conj.check(df)
    total_app = int(applicable.sum())
    good = int((applicable & holds).sum())
    bad = int((applicable & ~holds).sum())
    print(f"    applicable: {total_app} | satisfied: {good} | violations: {bad}")
    if bad:
        # show up to 3 failures with slack if present
        head = failures.head(3)
        cols_to_show = [c for c in ["__slack__"] if c in head.columns]
        print("    sample violations:", list(head.index))
        if cols_to_show:
            print("    slack (first few):")
            print(head[cols_to_show])
    print()


# -----------------------------------------------------------------------------
# Demo dataset
# -----------------------------------------------------------------------------

def demo_df() -> pd.DataFrame:
    # Small, slightly varied toy dataset
    df = pd.DataFrame(
        {
            "connected": [True, True, True, True, True, True, True],
            "simple":    [True, True, True, True, True, True, True],
            "planar":    [True, True, True, False, False, False, True],
            "regular":   [True, True, False, False, True, False, True],
            "order":     [3, 4, 5, 6, 8, 10, 7],
            "size":      [2, 3, 4, 5, 8, 12, 6],
            "alpha":     [1, 2, 3, 3, 3, 5, 4],   # target candidate
            "omega":     [1, 1, 3, 2, 1, 2, 1],
            "mu":        [1, 2, 2, 3, 4, 4, 3],
        },
        index=[f"G{i}" for i in range(1, 8)],
    )
    return df


# -----------------------------------------------------------------------------
# Main walkthrough
# -----------------------------------------------------------------------------

def main():
    df = demo_df()
    _show_df(df, "DataFrame")

    # 1) Enumerate boolean hypotheses and simplify/dedup
    _header("1) Enumerate + simplify 'good' hypotheses")
    base = detect_base_hypothesis(df)
    print("Detected base:", repr(base), "\n")

    hyps_all = enumerate_boolean_hypotheses(
        df,
        treat_binary_ints=True,
        include_base=True,
        include_pairs=True,
        skip_always_false=True,
    )
    # Simplify, enforce min_support, dedup by mask
    kept_hyps, eq_witnesses = simplify_and_dedup_hypotheses(
        df, hyps_all, min_support=3, treat_binary_ints=True
    )

    print(f"Kept hypotheses: {len(kept_hyps)}")
    for i, H in enumerate(kept_hyps, 1):
        print(f"[H{i:02d}] {repr(H)}   mask support={int(H.mask(df).sum())}")
    if eq_witnesses:
        print("\nEquivalence witnesses (renamings):")
        for w in eq_witnesses:
            print(" -", repr(w))
    print()

    # 2) Choose a target and define raw numeric features (exclude target)
    _header("2) Choose target and prepare feature families per hypothesis")
    target = "alpha"
    numeric_cols = _numeric_cols(df)
    raw_features = [c for c in numeric_cols if c != target]
    print(f"Target: {target}")
    print("Raw numeric features:", raw_features, "\n")

    # 3) For each hypothesis, build domain-safe transformed features (φ set)
    #    and run ratio + LP generators
    _header("3) Generate conjectures (ratios first, then LP)")
    all_conjectures: List[Conjecture] = []

    for i, H in enumerate(kept_hyps, 1):
        print(f"--- Hypothesis H{i:02d}: {repr(H)} ---")
        φs = build_feature_exprs_for_hypothesis(
            df, H, raw_features,
            include_sqrt=True, include_log=True, include_square=True, min_support=2
        )
        if not φs:
            print("  (No usable features under this hypothesis.)\n")
            continue

        print("  Features under H:")
        for φ in φs:
            print("   •", repr(φ))
        print()

        # 3a) Ratio-based conjectures
        print("  Ratio bounds:")
        for conj in ratios(
            df,
            features=φs,           # Expr or str accepted
            target=target,
            hypothesis=H,
            max_denominator=50,
            direction="both",
            q_clip=None,           # no clipping for demo
            min_support=2,
            simplify_condition=True,
        ):
            all_conjectures.append(conj)
            summarize_conjecture(df, conj)

        # 3b) LP-based conjectures (multi-feature plane)
        print("  LP bounds (sum-of-slacks fit):")
        try:
            cfg = LPConfig(
                features=φs,       # multivariate
                target=target,
                direction="both",
                max_denominator=50,
                tol=1e-9,
                min_support=3,
            )
            for conj in lp_bounds(df, hypothesis=H, config=cfg):
                all_conjectures.append(conj)
                summarize_conjecture(df, conj)
        except RuntimeError as e:
            # No solver found: keep the script educational and non-failing
            print("   (LP skipped:", str(e), ")")
        print()

    # 4) Epilogue
    _header("4) Summary")
    print(f"Total conjectures generated: {len(all_conjectures)}\n")
    # Show a compact list (pretty)
    for j, c in enumerate(all_conjectures[:20], 1):  # cap to first 20 for brevity
        summarize_conjecture(df, c, label=f"[C{j:02d}]")

if __name__ == "__main__":
    main()
