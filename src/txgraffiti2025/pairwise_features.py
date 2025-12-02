from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import Expr, to_expr, min_, max_, abs_


@dataclass(frozen=True)
class NonComparablePair:
    """Representation of a non-comparable numeric pair (x, y)."""
    left: str
    right: str

    def as_tuple(self) -> Tuple[str, str]:
        return (self.left, self.right)


def find_noncomparable_pairs(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    *,
    min_support: int = 20,
    eps: float = 0.0,
    max_pairs: int | None = None,
) -> List[NonComparablePair]:
    """
    Identify pairs of numeric columns that are not (approximately) ordered.

    A pair (x, y) is considered comparable if one dominates the other on at
    least (1 - eps) fraction of rows where both are defined:

        frac(x >= y) >= 1 - eps    or    frac(y >= x) >= 1 - eps.

    Otherwise, the pair is marked non-comparable.

    Parameters
    ----------
    df : DataFrame
        The data table containing numeric invariants.
    numeric_cols : sequence of str
        Column names to consider as numeric invariants.
    min_support : int, default=20
        Minimum number of rows with finite values in both columns to even
        consider the pair.
    eps : float, default=0.0
        Tolerance for “almost everywhere” dominance; e.g., eps=0.01 allows
        up to 1% violations while still calling a pair comparable.
    max_pairs : int or None, default=None
        Hard cap on the number of non-comparable pairs to return. If None,
        no cap is applied.

    Returns
    -------
    list of NonComparablePair
        All detected non-comparable pairs (x, y) with x < y in lexicographic
        order to avoid duplicates.
    """
    pairs: List[NonComparablePair] = []

    ncols = len(numeric_cols)
    for i in range(ncols):
        a = numeric_cols[i]
        s_a = pd.to_numeric(df[a], errors="coerce")
        for j in range(i + 1, ncols):
            b = numeric_cols[j]
            s_b = pd.to_numeric(df[b], errors="coerce")

            mask = s_a.notna() & s_b.notna()
            if mask.sum() < min_support:
                continue

            x = s_a[mask].to_numpy()
            y = s_b[mask].to_numpy()
            n = len(x)

            if n == 0:
                continue

            frac_x_ge_y = float(np.mean(x >= y))
            frac_y_ge_x = float(np.mean(y >= x))

            # If one side dominates almost everywhere, treat as comparable
            if frac_x_ge_y >= 1.0 - eps or frac_y_ge_x >= 1.0 - eps:
                continue

            # Otherwise, genuinely non-comparable
            pairs.append(NonComparablePair(left=a, right=b))

            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs

    return pairs


def build_pairwise_expr_features_for_target(
    pairs: Iterable[NonComparablePair],
    *,
    target_name: str,
    include_abs: bool = True,
) -> List[Expr]:
    """
    Build Expr features min(x, y), max(x, y) and optionally |x - y| from
    non-comparable pairs, excluding any pair that involves the target.

    Parameters
    ----------
    pairs : iterable of NonComparablePair
        Non-comparable numeric pairs.
    target_name : str
        Name of the column serving as the target invariant in this run.
        Any pair that includes this column is ignored.
    include_abs : bool, default=True
        If True, also create |x - y| as an Expr.

    Returns
    -------
    list of Expr
        Expression features usable by both LP and other Expr-based engines.
    """
    feats: List[Expr] = []
    seen: Set[str] = set()

    for p in pairs:
        x, y = p.left, p.right

        if x == target_name or y == target_name:
            continue

        e_min = min_(x, y)
        e_max = max_(x, y)

        for e in (e_min, e_max):
            s = repr(e)
            if s not in seen:
                feats.append(e)
                seen.add(s)

        if include_abs:
            e_abs = abs_(to_expr(x) - to_expr(y))
            s = repr(e_abs)
            if s not in seen:
                feats.append(e_abs)
                seen.add(s)

    return feats


def add_pairwise_columns(
    df: pd.DataFrame,
    pairs: Iterable[NonComparablePair],
    *,
    include_abs: bool = True,
    include_min: bool = True,
    include_max: bool = True,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Materialize min(x, y), max(x, y), and |x - y| as concrete numeric
    columns in a copy of `df`.

    This is intended for engines that work column-wise (e.g., Intricate),
    while LP can still use Expr features if desired.

    Column naming convention
    ------------------------
    For a pair (x, y) with x < y, we create up to three columns:

        min:  min_x_y[_suffix]
        max:  max_x_y[_suffix]
        abs:  abs_x_minus_y[_suffix]

    Parameters
    ----------
    df : DataFrame
        Input data frame. It is *not* modified in-place.
    pairs : iterable of NonComparablePair
        Non-comparable numeric pairs to materialize.
    include_abs : bool, default=True
        Whether to create |x - y| columns.
    include_min : bool, default=True
        Whether to create min(x, y) columns.
    include_max : bool, default=True
        Whether to create max(x, y) columns.
    suffix : str, default=""
        Optional suffix appended to each generated column name.

    Returns
    -------
    DataFrame
        A new DataFrame with added numeric columns.
    """
    df_new = df.copy()

    for p in pairs:
        x, y = p.left, p.right
        if x not in df_new.columns or y not in df_new.columns:
            continue

        s_x = pd.to_numeric(df_new[x], errors="coerce")
        s_y = pd.to_numeric(df_new[y], errors="coerce")

        base = f"{x}_{y}"
        sfx = f"_{suffix}" if suffix else ""

        if include_min:
            col_min = f"min_{base}{sfx}"
            if col_min not in df_new.columns:
                df_new[col_min] = np.minimum(s_x, s_y)

        if include_max:
            col_max = f"max_{base}{sfx}"
            if col_max not in df_new.columns:
                df_new[col_max] = np.maximum(s_x, s_y)

        if include_abs:
            col_abs = f"abs_{x}_minus_{y}{sfx}"
            if col_abs not in df_new.columns:
                df_new[col_abs] = (s_x - s_y).abs()

    return df_new
