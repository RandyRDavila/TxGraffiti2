"""
Post‐processing functions for TxGraffiti conjecture pipelines.

Each function is registered under a short name via the
`@register_post` decorator and can be applied in a
ConjecturePlayground’s `post_processors` step.
"""

import pandas as pd
from typing import List, Tuple
from txgraffiti.processing.registry import register_post
from txgraffiti.logic import Conjecture, Inequality

__all__ = [
    'remove_duplicates',
    'sort_by_accuracy',
    'sort_by_touch_count',
    'extract_equalities',
]

@register_post("remove_duplicates")
def remove_duplicates(conjs: List[Conjecture], df: pd.DataFrame) -> List[Conjecture]:
    """
    Remove duplicate conjectures based on hypothesis and conclusion names.

    This post‐processor walks through the list of conjectures in order
    and only keeps the first occurrence of each unique
    (hypothesis.name, conclusion.name) pair.

    Parameters
    ----------
    conjs : List[Conjecture]
        The list of conjectures to filter.
    df : pandas.DataFrame
        The DataFrame on which these conjectures were evaluated
        (not used in this function).

    Returns
    -------
    List[Conjecture]
        A new list containing only the first instance of each unique
        hypothesis/conclusion combination.
    """
    seen = set()
    out  = []
    for c in conjs:
        key = (c.hypothesis.name, c.conclusion.name)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


@register_post("sort_by_accuracy")
def sort_by_accuracy(conjs: List[Conjecture], df: pd.DataFrame) -> List[Conjecture]:
    """
    Sort conjectures by descending accuracy on the DataFrame.

    Parameters
    ----------
    conjs : List[Conjecture]
        The list of conjectures to sort.
    df : pandas.DataFrame
        The DataFrame on which to compute each conjecture’s accuracy.

    Returns
    -------
    List[Conjecture]
        A new list of conjectures sorted so that the highest‐accuracy
        conjecture comes first.
    """
    # highest accuracy first
    return sorted(conjs, key=lambda c: c.accuracy(df), reverse=True)


@register_post("sort_by_touch_count")
def sort_by_touch_count(conjs: List[Conjecture], df: pd.DataFrame) -> List[Conjecture]:
    """
    Sort conjectures by descending touch count (slack‐zero instances).

    The touch count of an inequality is the number of rows where
    its conclusion holds with equality (zero slack).  This
    post‐processor brings conjectures with *more* tight instances to front.

    Parameters
    ----------
    conjs : List[Conjecture]
        The list of conjectures to sort.
    df : pandas.DataFrame
        The DataFrame on which to compute each conjecture’s touch count.

    Returns
    -------
    List[Conjecture]
        A new list of conjectures sorted so that those with the highest
        touch counts appear first.
    """
    # lowest touch count first
    return sorted(
        conjs,
        key=lambda c: c.conclusion.touch_count(df),
        reverse=True
    )

@register_post("extract_equalities")
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
