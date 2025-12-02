import os, sys
import pandas as pd
import numpy as np
import pytest

# Ensure `src/` is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

@pytest.fixture
def df_basic():
    # A small, clean numeric DF for most tests
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 2.0, 2.0],
        "c": [1.5, 3.0, 4.5, 6.0],
        "n": [5, 6, 7, 8],
        "m": [4, 8, 12, 16],
        "cat": ["x", "y", "x", "z"],
    })

@pytest.fixture
def df_mixed():
    # Includes NaNs and non-numeric; good for log/qualitative tests
    return pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0, 5.0],
        "y": [0.0, 1.0, 1.0, 2.0, 2.5],
        "z": [1, 1, 1, 1, 1],  # constant
        "cat": ["A", "B", "A", "C", None],
    })

@pytest.fixture
def df_objects():
    class Obj:
        def __init__(self, val): self.val = val
        def has_even(self): return self.val % 2 == 0
    return pd.DataFrame({
        "object": [Obj(1), Obj(2), Obj(3), Obj(4), None]
    })
