import pandas as pd
from typing import List

from pandas.api.types import is_bool_dtype, is_numeric_dtype

from txgraffiti.logic import Property, Predicate

__all__ = [
    'auto_wrap',
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
