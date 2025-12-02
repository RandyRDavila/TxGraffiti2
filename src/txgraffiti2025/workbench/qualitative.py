"""
qualitative.py
==============

Tools for discovering qualitative (monotone) relationships between numerical
columns under given logical hypotheses.

Each hypothesis defines a Boolean slice of the dataframe, and for each slice,
pairwise monotone relations are evaluated (e.g., Spearman or Kendall rank
correlation) to identify strong increasing or decreasing trends.

The output is a ranked list of :class:`QualResult` objects that record the
relation, its strength, and supporting metadata.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd

from txgraffiti2025.forms.qualitative import MonotoneRelation, Method as CorrMethod
from txgraffiti2025.forms.predicates import Predicate


@dataclass
class QualResult:
    """Container for a discovered qualitative (monotone) relation.

    Attributes
    ----------
    relation : MonotoneRelation
        The underlying monotone relation describing the direction and method
        of correlation between `x` and `y`.
    condition : Predicate
        The Boolean predicate (hypothesis) defining the dataframe slice
        where this relation holds.
    rho : float
        The correlation coefficient computed on the slice (positive for
        increasing, negative for decreasing).
    n : int
        Number of non-missing data points used in the correlation.
    support : int
        Total number of rows satisfying the hypothesis, regardless of missing values.
    x : str
        Name of the explanatory variable (independent feature).
    y : str
        Name of the response variable (dependent or target feature).
    """
    relation: MonotoneRelation
    condition: Predicate
    rho: float
    n: int
    support: int
    x: str
    y: str



def generate_qualitative_relations(
    df: pd.DataFrame,
    *,
    y_targets: Iterable[str],
    x_candidates: Iterable[str],
    hyps: Iterable[Predicate],
    method: CorrMethod = "spearman",
    min_abs_rho: float = 0.35,
    min_n: int = 12,
    drop_constant: bool = True,
    top_k_per_hyp: Optional[int] = None,
) -> List[QualResult]:
    """Discover monotone (qualitative) relations among numerical features.

    For each Boolean predicate in `hyps`, this function filters the dataframe
    and computes pairwise correlations between every target `y` and each
    candidate `x` in that slice. Only pairs with sufficiently strong
    monotone relationships (|ρ| ≥ `min_abs_rho`) and sample size ≥ `min_n`
    are retained.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing numeric columns and possible Boolean features.
    y_targets : Iterable[str]
        Column names to treat as dependent variables (`y`) in correlations.
    x_candidates : Iterable[str]
        Column names to treat as independent variables (`x`).
    hyps : Iterable[Predicate]
        Hypotheses defining logical slices of the dataframe.
    method : {"spearman", "kendall"}, optional
        Correlation method used to assess monotonicity. Default is "spearman".
    min_abs_rho : float, optional
        Minimum absolute correlation magnitude required to keep a relation.
        Default is 0.35.
    min_n : int, optional
        Minimum number of valid (non-NaN) samples required for computation.
        Default is 12.
    drop_constant : bool, optional
        Whether to skip columns that are constant (or nearly constant)
        within the filtered slice. Default is True.
    top_k_per_hyp : int or None, optional
        If given, keep only the top-k strongest relations per hypothesis
        (ranked by |ρ| and support). If None, keep all.

    Returns
    -------
    List[QualResult]
        A list of qualitative relation results, sorted by descending |ρ| and
        support.

    Notes
    -----
    * This function is qualitative—it captures monotone tendencies, not
      explicit functional forms.
    * The resulting :class:`MonotoneRelation` objects can later be used to
      generate symbolic conjectures (e.g., "as `x` increases, `y` tends to increase").
    * To interpret the results, inspect ``result.relation.direction`` and
      ``result.rho``.

    Examples
    --------
    >>> from txgraffiti2025.forms.predicates import Where
    >>> hyp = Where(fn=lambda df: df["connected"] == 1, name="(connected)")
    >>> res = generate_qualitative_relations(
    ...     df, y_targets=["independence_number"],
    ...     x_candidates=["order","size","radius"],
    ...     hyps=[hyp], method="spearman"
    ... )
    >>> for r in res[:3]:
    ...     print(f"{r.condition.name}: {r.x} → {r.y} ({r.relation.direction}, ρ={r.rho:.2f})")
    """
    results: List[QualResult] = []
    for H in hyps:
        mask = H.mask(df).astype(bool)
        support = int(mask.sum())
        if support < min_n:
            continue
        num_df = df.loc[mask].apply(pd.to_numeric, errors="coerce")

        hyp_results: List[QualResult] = []
        for y in y_targets:
            ys = num_df.get(y)
            if ys is None or (drop_constant and ys.nunique(dropna=True) <= 1):
                continue
            for x in x_candidates:
                if x == y:
                    continue
                xs = num_df.get(x)
                if xs is None or (drop_constant and xs.nunique(dropna=True) <= 1):
                    continue
                valid = xs.notna() & ys.notna()
                n = int(valid.sum())
                if n < min_n:
                    continue
                mr = MonotoneRelation(
                    x=x,
                    y=y,
                    direction="increasing",
                    method=method,
                    min_abs_rho=min_abs_rho,
                    min_n=min_n,
                )
                rho = mr._corr(xs[valid].to_numpy(), ys[valid].to_numpy())
                if not np.isfinite(rho) or abs(rho) < float(min_abs_rho):
                    continue
                mr.direction = "increasing" if rho >= 0 else "decreasing"
                hyp_results.append(
                    QualResult(
                        relation=mr,
                        condition=H,
                        rho=float(rho),
                        n=n,
                        support=support,
                        x=x,
                        y=y,
                    )
                )

        if top_k_per_hyp and top_k_per_hyp > 0:
            hyp_results.sort(key=lambda r: (abs(r.rho), r.support), reverse=True)
            hyp_results = hyp_results[:top_k_per_hyp]
        results.extend(hyp_results)

    results.sort(key=lambda r: (abs(r.rho), r.support), reverse=True)
    return results
