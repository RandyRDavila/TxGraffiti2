"""
Diversity metrics for hypotheses (Predicate masks over a DataFrame).

Given a hypothesis H (Predicate), this module measures how "diverse" the
subset S = { rows : H.mask(df) == True } is, along multiple axes:

- Boolean columns: balance (entropy-like) under S vs globally
- Categorical columns: normalized entropy under S
- Numeric columns: robust spread (IQR_sub / IQR_global)
- Coverage: |S| / |df|

All components are normalized to [0, 1] and combined by an unweighted mean
(you can change weights in `combine_scores` if desired).

The output includes a top-level DiversityScore plus a rich per-column breakdown.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_categorical_dtype,
    is_object_dtype,
)

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.processing.pre.hypotheses import list_boolean_columns

# ---------------------------
# Utilities
# ---------------------------
def _is_categorical(s: pd.Series) -> bool:
    # treat object & categorical as categorical; exclude pure booleans
    return is_categorical_dtype(s) or (is_object_dtype(s) and not is_bool_dtype(s))

def _is_numeric(s: pd.Series) -> bool:
    # numeric but not boolean
    return is_numeric_dtype(s) and not is_bool_dtype(s)


def _safe_iqr(x: pd.Series) -> float:
    # Force float so quantile never sees boolean dtype
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    x = pd.to_numeric(x, errors="coerce").astype(float)
    if x.empty:
        return 0.0
    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)
    if pd.isna(q25) or pd.isna(q75):
        return 0.0
    return float(q75 - q25)

def _shannon_entropy(probs: np.ndarray) -> float:
    # natural log; define 0*log(0)=0 by masking
    p = probs[(probs > 0) & np.isfinite(probs)]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())

def _normalized_entropy(counts: pd.Series) -> float:
    """H/H_max in [0,1], H_max=log(k) where k = #non-empty categories under S."""
    counts = counts[counts > 0]
    k = len(counts)
    if k <= 1:
        return 0.0
    probs = counts.to_numpy(dtype=float)
    probs /= probs.sum()
    H = _shannon_entropy(probs)
    Hmax = np.log(k)
    return float(H / Hmax) if Hmax > 0 else 0.0

def _boolean_balance(p: float) -> float:
    """
    Symmetric balance score in [0,1] with max at p=0.5.
    Uses 4*p*(1-p) (the Gini impurity scaled to [0,1]).
    """
    if not np.isfinite(p):
        return 0.0
    p = float(p)
    p = max(0.0, min(1.0, p))
    return 4.0 * p * (1.0 - p)

def _ratio_clip(num: float, den: float) -> Optional[float]:
    """Return num/den clipped to [0,1]; None if den==0."""
    if den == 0:
        return None
    return float(max(0.0, min(1.0, num / den)))


# ---------------------------
# Config dataclass
# ---------------------------

@dataclass
class DiversityConfig:
    treat_binary_ints_as_bool: bool = True
    include_coverage: bool = True
    include_boolean: bool = True
    include_categorical: bool = True
    include_numeric: bool = True

    # Optional explicit column lists (if None, auto-detect)
    boolean_cols: Optional[Sequence[str]] = None
    categorical_cols: Optional[Sequence[str]] = None
    numeric_cols: Optional[Sequence[str]] = None

    # Weighting of components when combining
    w_coverage: float = 1.0
    w_boolean: float = 1.0
    w_categorical: float = 1.0
    w_numeric: float = 1.0


# ---------------------------
# Main metric
# ---------------------------

def compute_diversity(
    df: pd.DataFrame,
    hypothesis: Predicate,
    *,
    config: Optional[DiversityConfig] = None,
) -> Dict[str, object]:
    """
    Compute a diversity score for the subset defined by `hypothesis.mask(df)`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing boolean/categorical/numeric features (invariants).
    hypothesis : Predicate
        Hypothesis predicate (e.g., (connected), (connected ∧ planar), ...).
    config : DiversityConfig, optional
        Configuration toggles and weights.

    Returns
    -------
    dict
        {
            "DiversityScore": float in [0,1],
            "components": {
                "coverage": float or None,
                "boolean": float or None,
                "categorical": float or None,
                "numeric": float or None,
            },
            "details": {
                "coverage": {"n": int, "N": int, "fraction": float},
                "boolean": {col: {"p": float, "balance": float, "balance_vs_global": float}},
                "categorical": {col: {"norm_entropy": float}},
                "numeric": {col: {"iqr_sub": float, "iqr_all": float, "relative_iqr": float or None}},
            }
        }

    Notes
    -----
    - Boolean component averages a symmetric balance score 4*p*(1-p) per boolean-like column
      (p = prevalence under the hypothesis). Optionally you can compare to global by looking at
      `balance_vs_global` in details.
    - Categorical component averages normalized entropy over categorical columns.
    - Numeric component averages IQR_sub / IQR_all (robust spread vs global).
    - Coverage is |S|/|df| and acts as a “support” factor if enabled.
    """
    cfg = config or DiversityConfig()

    mask = hypothesis.mask(df).reindex(df.index, fill_value=False)
    sub = df.loc[mask]
    n, N = len(sub), len(df)

    # -------- coverage
    coverage_score = None
    coverage_details = {"n": n, "N": N, "fraction": float(n / N) if N else 0.0}
    if cfg.include_coverage:
        coverage_score = coverage_details["fraction"]

    # -------- auto-detect columns
    if cfg.boolean_cols is None:
        bool_cols = list_boolean_columns(df, treat_binary_ints=cfg.treat_binary_ints_as_bool)
    else:
        bool_cols = list(cfg.boolean_cols)

    if cfg.categorical_cols is None:
        categorical_cols = [c for c in df.columns if _is_categorical(df[c])]
    else:
        categorical_cols = list(cfg.categorical_cols)

    if cfg.numeric_cols is None:
        numeric_cols = [c for c in df.columns if _is_numeric(df[c])]
    else:
        numeric_cols = list(cfg.numeric_cols)

    # -------- boolean diversity
    boolean_score = None
    boolean_details: Dict[str, Dict[str, float]] = {}
    if cfg.include_boolean and n > 0 and bool_cols:
        balances = []
        for c in bool_cols:
            s_all = df[c]
            s_sub = sub[c]
            p_sub = float(s_sub.astype(bool).mean()) if len(s_sub) else 0.0
            bal_sub = _boolean_balance(p_sub)
            # Optional reference against global balance (not used in score, but useful):
            p_all = float(s_all.astype(bool).mean()) if len(s_all) else 0.0
            bal_all = _boolean_balance(p_all)
            bal_vs_global = None if bal_all == 0 else bal_sub / bal_all
            boolean_details[c] = {"p": p_sub, "balance": bal_sub, "balance_vs_global": bal_vs_global}
            balances.append(bal_sub)
        boolean_score = float(np.mean(balances)) if balances else None

    # -------- categorical diversity
    categorical_score = None
    categorical_details: Dict[str, Dict[str, float]] = {}
    if cfg.include_categorical and n > 0 and categorical_cols:
        ents = []
        for c in categorical_cols:
            norm_H = _normalized_entropy(sub[c].value_counts(dropna=False))
            categorical_details[c] = {"norm_entropy": norm_H}
            ents.append(norm_H)
        categorical_score = float(np.mean(ents)) if ents else None

    # -------- numeric diversity
    numeric_score = None
    numeric_details: Dict[str, Dict[str, Optional[float]]] = {}
    if cfg.include_numeric and n > 0 and numeric_cols:
        rels: List[float] = []
        for c in numeric_cols:
            iqr_all = _safe_iqr(df[c])
            iqr_sub = _safe_iqr(sub[c])
            rel = _ratio_clip(iqr_sub, iqr_all)
            numeric_details[c] = {"iqr_sub": iqr_sub, "iqr_all": iqr_all, "relative_iqr": rel}
            if rel is not None:
                rels.append(rel)
        numeric_score = float(np.mean(rels)) if rels else None

    # -------- combine
    comps = {
        "coverage": coverage_score,
        "boolean": boolean_score,
        "categorical": categorical_score,
        "numeric": numeric_score,
    }
    details = {
        "coverage": coverage_details,
        "boolean": boolean_details,
        "categorical": categorical_details,
        "numeric": numeric_details,
    }
    final = combine_scores(comps, cfg)

    return {"DiversityScore": final, "components": comps, "details": details}


def combine_scores(components: Dict[str, Optional[float]], cfg: DiversityConfig) -> float:
    """Combine component scores into a single number using simple weighted mean."""
    vals, wts = [], []
    for key, val in components.items():
        if val is None:
            continue
        w = {
            "coverage": cfg.w_coverage,
            "boolean": cfg.w_boolean,
            "categorical": cfg.w_categorical,
            "numeric": cfg.w_numeric,
        }[key]
        if w <= 0:
            continue
        vals.append(val)
        wts.append(w)
    if not vals:
        return 0.0
    wts = np.asarray(wts, dtype=float)
    vals = np.asarray(vals, dtype=float)
    return float((vals * wts).sum() / wts.sum())


# ---------------------------
# Convenience: batch scoring
# ---------------------------

def score_hypotheses(
    df: pd.DataFrame,
    hypotheses: Sequence[Predicate],
    *,
    config: Optional[DiversityConfig] = None,
) -> pd.DataFrame:
    """
    Score many hypotheses and return a tidy DataFrame with per-hypothesis scores.

    Returns columns:
        ["hypothesis", "DiversityScore", "coverage", "boolean", "categorical", "numeric", "n", "N"]

    The "hypothesis" column is `repr(hypothesis)` so your pretty names render.
    """
    rows = []
    for H in hypotheses:
        res = compute_diversity(df, H, config=config)
        comps = res["components"]
        n = res["details"]["coverage"]["n"]
        N = res["details"]["coverage"]["N"]
        rows.append({
            "hypothesis": repr(H),
            "DiversityScore": res["DiversityScore"],
            "coverage": comps.get("coverage"),
            "boolean": comps.get("boolean"),
            "categorical": comps.get("categorical"),
            "numeric": comps.get("numeric"),
            "n": n, "N": N, "frac": (n / N) if N else 0.0,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["DiversityScore", "frac", "hypothesis"], ascending=[False, False, True]).reset_index(drop=True)
