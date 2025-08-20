import pandas as pd
from itertools import combinations
from typing import List, Callable, Sequence

from txgraffiti.logic import Property, Conjecture
from txgraffiti.generators import linear_programming, convex_hull, ratios
from txgraffiti.processing import (
    remove_duplicates,
    filter_with_dalmatian,
    filter_with_morgan,
    sort_by_touch_count,
    extract_equalities,
    auto_wrap,

)

from typing import List, Callable
import pandas as pd

from txgraffiti.logic import Conjecture

__all__ = [
    "txgraffiti2",
]

def txgraffiti2(
    df: pd.DataFrame,
    target_column: str,
    generators: Sequence[Callable] = (linear_programming, convex_hull, ratios),
    feature_size: int = 1
) -> List[Conjecture]:

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

    for feat_tuple in combinations(features_candidates, feature_size):
        for gen in generators:
            feats = list(feat_tuple)
            for hyp in bool_preds:
                conjs = gen(
                    df=df,
                    features=feats,
                    target=target_prop,
                    hypothesis=hyp,
                )
                raw.extend(conjs)

    # 3) post‑processing
    raw = remove_duplicates(raw, df)
    raw = filter_with_dalmatian(raw, df)
    raw = filter_with_morgan(raw, df)
    raw = sort_by_touch_count(raw, df)

    # 4) extract inequalities and equalities
    eqs, ineqs = extract_equalities(raw, df)
    ineqs.extend(eqs)
    conjectures = sort_by_touch_count(ineqs, df)

    return conjectures
