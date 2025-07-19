import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from itertools import combinations
from typing import List, Callable, Sequence

from txgraffiti.logic import Property, Predicate, Conjecture
from txgraffiti.generators import linear_programming, convex_hull
from txgraffiti.processing import (
    remove_duplicates,
    filter_with_dalmatian,
    filter_with_morgan,
    sort_by_touch_count,
)
from txgraffiti.heuristics.delavina import sophie_accept

__all__ = [
    "christine",
]

def auto_wrap(df: pd.DataFrame):
    """
    Turn each boolean column into a Predicate and each numeric
    column into a Property (skipping 'name' and 'Unnamed: 0').
    """
    numeric_props: List[Property] = []
    bool_preds:   List[Predicate] = []

    for col in df.columns:
        if col in ("name", "Unnamed: 0"):
            continue
        if is_bool_dtype(df[col]):
            bool_preds.append(Predicate(col, lambda df, c=col: df[c]))
        elif is_numeric_dtype(df[col]):
            numeric_props.append(Property(col, lambda df, c=col: df[c]))

    return numeric_props, bool_preds

def christine(
    df: pd.DataFrame,
    target_column: str,
    generators: Sequence[Callable] = (linear_programming, convex_hull),
    feature_size: int = 1
) -> List[Conjecture]:
    """
    Full conjecture‐generation pipeline:
      1) auto_wrap → numeric_props, bool_preds
      2) for each generator G, each size‑`feature_size` combo of numeric_props
         (excluding the target), and each boolean predicate H,
           generate G(df, features, target, hypothesis=~H)
      3) dedupe, Dalmatian‑filter, Morgan‑filter, sort by touch count
      4) for each candidate C: if sophie_accept(C.contrapositive(), accepted, df)
         then add C.contrapositive() to accepted

    Returns
    -------
    List[Conjecture]
        The contrapositives of the conjectures accepted by Sophie.
    """
    # 1) wrap
    numeric_props, bool_preds = auto_wrap(df)

    # build target Property and drop it from numeric_props
    target_prop = Property(
        target_column,
        lambda df, c=target_column: df[c],
    )
    features_candidates = [
        p for p in numeric_props
        if p.name != target_prop.name
    ]

    # 2) generate all raw conjectures
    raw: List[Conjecture] = []
    for gen in generators:
        for feat_tuple in combinations(features_candidates, feature_size):
            feats = list(feat_tuple)
            for bp in bool_preds:
                # we use the negated boolean as hypothesis per your example
                hy = ~bp
                conjs = gen(
                    df=df,
                    features=feats,
                    target=target_prop,
                    hypothesis=hy,
                )
                raw.extend(conjs)

    # 3) post‑processing
    raw = remove_duplicates(raw, df)
    raw = filter_with_dalmatian(raw, df)
    raw = filter_with_morgan(raw, df)
    raw = sort_by_touch_count(raw, df)

    # 4) Sophie on contrapositives
    accepted: List[Conjecture] = []
    for c in raw:
        cp = c.contrapositive()
        if sophie_accept(cp, accepted, df):
            accepted.append(cp)

    return accepted
