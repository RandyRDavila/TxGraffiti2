import numpy as np
from txgraffiti2025.workbench.ranking import touch_count, rank_and_filter
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge
from txgraffiti2025.forms.utils import to_expr
from txgraffiti2025.forms.predicates import Where

def TRUE_where():
    return Where(fn=lambda df: [True]*len(df), name="TRUE")

def test_touch_count_and_rank(toy_df):
    # y >= 2*x is tight everywhere in toy_df
    c = Conjecture(Ge(to_expr("y"), (to_expr("x")*2)), TRUE_where())
    t = touch_count(c, toy_df)
    assert t == len(toy_df)
    kept = rank_and_filter(toy_df, [c], min_touch=len(toy_df))
    assert kept and kept[0] is c
