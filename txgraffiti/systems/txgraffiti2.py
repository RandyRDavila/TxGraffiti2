import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from itertools import combinations
from typing import List, Callable, Sequence

from txgraffiti.logic import Property, Predicate, Conjecture, KnowledgeTable
from txgraffiti.generators import linear_programming, convex_hull
from txgraffiti.processing import (
    remove_duplicates,
    filter_with_dalmatian,
    filter_with_morgan,
    sort_by_touch_count,
)
from txgraffiti.heuristics.delavina import sophie_accept

from typing import List, Tuple
import pandas as pd

from txgraffiti.logic import Conjecture
from txgraffiti.logic import Inequality

__all__ = [
    "txgraffiti2",
]

def extract_equalities(
    conjs: List[Conjecture],
    df: pd.DataFrame
) -> Tuple[List[Conjecture], List[Conjecture]]:
    """
    Given conjectures whose conclusions are Inequalities, check each one:
      – if on all rows where the hypothesis holds the slack == 0,
        convert it to an equality Conjecture (lhs == rhs);
      – otherwise keep the original inequality Conjecture.

    Returns
    -------
    (equalities, inequalities)
    """
    eqs: List[Conjecture] = []
    ineqs: List[Conjecture] = []

    for c in conjs:
        concl = c.conclusion
        # only handle Inequality conclusions
        if isinstance(concl, Inequality):
            hyp_mask = c.hypothesis(df)
            # only consider it an equality if there's at least one row
            # and slack is zero everywhere under the hypothesis
            slack = concl.slack(df)
            if hyp_mask.any() and (slack[hyp_mask] == 0).all():
                # build a new equality predicate
                eq_pred = Inequality(concl.lhs, "==", concl.rhs)
                eqs.append(Conjecture(c.hypothesis, eq_pred))
            else:
                ineqs.append(c)
        else:
            # leave any non-inequality conjecture untouched
            ineqs.append(c)

    return eqs, ineqs

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

def txgraffiti2(
    df: pd.DataFrame,
    target_column: str,
    generators: Sequence[Callable] = (linear_programming, convex_hull),
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
    for gen in generators:
        for feat_tuple in combinations(features_candidates, feature_size):
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


    eqs, ineqs = extract_equalities(raw, df)
    return eqs, ineqs
