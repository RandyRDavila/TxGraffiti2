import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def toy_df():
    # Small numeric + boolean dataset
    # y = 2*x is exact; also provide another numeric to allow mixes
    df = pd.DataFrame({
        "x":  [1, 2, 3, 4, 5, 6],
        "y":  [2, 4, 6, 8, 10, 12],  # 2*x
        "z":  [1, 1, 2, 3, 5, 8],    # fibonacci-like
        "b1": [1, 1, 1, 0, 0, 0],    # binary-int treated as bool
        "b2": [0, 1, 1, 0, 0, 0],
    })
    return df
