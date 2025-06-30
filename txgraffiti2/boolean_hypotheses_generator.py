import pandas as pd
import itertools
from functools import reduce
from typing import List, Optional
from pandas.api.types import is_bool_dtype

from txgraffiti2.conjecture_logic import Property, Predicate, Inequality, Conjecture

# assumes you have your Predicate class in scope

def generate_boolean_hypotheses(
    df: pd.DataFrame,
    lower_bound: int = 1,
    upper_bound: Optional[int] = None
) -> List[Predicate]:
    """
    From all boolean columns in `df`, build every hypothesis of the form
      P1 ∧ P2 ∧ … ∧ Pr
    where lower_bound ≤ r ≤ upper_bound, and each Pi is either the column
    or its negation ¬column.
    
    Returns a list of Predicate objects.
    """
    # 1) find all boolean columns
    bool_cols = [c for c in df.columns if is_bool_dtype(df[c])]
    if upper_bound is None:
        upper_bound = len(bool_cols)
    assert 1 <= lower_bound <= upper_bound <= len(bool_cols), \
        "bounds must satisfy 1 ≤ lower ≤ upper ≤ #bool_cols"

    # 2) wrap each column as a Predicate
    base_preds = [Predicate(col, lambda df, c=col: df[c])
                  for col in bool_cols]

    all_hyps: List[Predicate] = []

    # 3) for each size r in [lower_bound..upper_bound]
    for r in range(lower_bound, upper_bound+1):
        # choose r distinct base predicates
        for combo in itertools.combinations(base_preds, r):
            # for each choice of negating each one (2^r possibilities)
            for signs in itertools.product([True, False], repeat=r):
                terms = []
                for pred, keep_positive in zip(combo, signs):
                    terms.append(pred if keep_positive else ~pred)
                # conjoin them into one Predicate
                hyp = reduce(lambda a,b: a & b, terms)
                all_hyps.append(hyp)

    return all_hyps
