"""
Post‐processing functions for TxGraffiti conjecture pipelines.

Each function is registered under a short name via the
`@register_post` decorator and can be applied in a
ConjecturePlayground’s `post_processors` step.
"""

import pandas as pd
from typing import List
from txgraffiti.processing.registry import register_post
from txgraffiti.logic import Conjecture


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
