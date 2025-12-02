# utils_feature_types.py
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_integer_dtype, is_float_dtype

def is_boolish_series(s: pd.Series) -> bool:
    """
    Treat as boolean if:
      - dtype is bool; OR
      - all non-NaN values are in {0,1} (ints or floats).
    """
    if is_bool_dtype(s):
        return True
    t = s.dropna()
    if t.empty:
        return False
    # fast checks
    uniq = pd.unique(t)
    # integer-like 0/1
    if is_integer_dtype(t) and set(map(int, uniq)) <= {0,1}:
        return True
    # float-like but only 0.0/1.0
    if is_float_dtype(t) and set(map(float, uniq)) <= {0.0, 1.0}:
        return True
    # generic numeric but strictly 0/1
    if is_numeric_dtype(t):
        try:
            vals = pd.to_numeric(t, errors="coerce")
            u = set(pd.unique(vals))
            if u <= {0, 1, 0.0, 1.0}:
                return True
        except Exception:
            pass
    return False

def numeric_nonboolean_columns(df: pd.DataFrame) -> list[str]:
    """
    Return numeric columns excluding boolean-ish (dtype bool or set {0,1} ignoring NaN).
    """
    import numpy as np
    import pandas as pd

    out = []
    for c in df.columns:
        s = df[c]
        if not (np.issubdtype(s.dtype, np.number) or s.dtype == "Int64"):
            continue
        if s.dtype == bool:
            continue
        vals = pd.Series(s.dropna().unique())
        if len(vals) and vals.isin([0, 1]).all():
            continue
        out.append(c)
    return out
