import pandas as pd
from txgraffiti2025.workbench.engine import WorkbenchEngine
from txgraffiti2025.workbench.config import GenerationConfig

def test_engine_instantiates_and_splits_columns(toy_df):
    eng = WorkbenchEngine(toy_df, config=GenerationConfig())
    # Ensure columns split roughly as expected
    assert "x" in eng.numeric_columns and "y" in eng.numeric_columns
    assert "b1" in eng.bool_columns and "b2" in eng.bool_columns
