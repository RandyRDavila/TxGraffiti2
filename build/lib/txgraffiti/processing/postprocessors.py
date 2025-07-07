# postprocessors.py
import pandas as pd

from txgraffiti.processing.registry import register_post
from txgraffiti.logic import Conjecture

@register_post("remove_duplicates")
def remove_duplicates(conjs: list[Conjecture], df: pd.DataFrame) -> list[Conjecture]:
    seen = set()
    out  = []
    for c in conjs:
        key = (c.hypothesis.name, c.conclusion.name)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

@register_post("sort_by_accuracy")
def sort_by_accuracy(conjs: list[Conjecture], df: pd.DataFrame) -> list[Conjecture]:
    # highest accuracy first
    return sorted(conjs, key=lambda c: c.accuracy(df), reverse=True)


@register_post("sort_by_touch_count")
def sort_by_touch_count(conjs: list[Conjecture], df: pd.DataFrame) -> list[Conjecture]:
    # lowest touch count first
    return sorted(conjs, key=lambda c: c.conclusion.touch_count(df), reverse=True)
