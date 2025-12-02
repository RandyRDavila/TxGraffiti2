import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms import MonotoneRelation, GEQ
from txgraffiti2025.forms.predicates import Predicate


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def df_mixed():
    # x increasing, y mostly increasing with some noise and a NaN
    return pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [0.0, 0.8, np.nan, 2.2, 2.9, 3.8],
    })

@pytest.fixture
def df_decreasing():
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [12, 10, 8, 6, 4, 2],  # strictly decreasing in x
    })

@pytest.fixture
def df_constant_y():
    return pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 5],
        "y": [7, 7, 7, 7, 7, 7],  # constant → correlation should be 0.0
    })


# -----------------------
# Basic smoke (your originals, with fixture)
# -----------------------

def test_monotone_increasing_spearman(df_mixed):
    mono = MonotoneRelation(x="x", y="y", direction="increasing", method="spearman")
    res = mono.evaluate_global(df_mixed)
    assert set(res.keys()) >= {"ok", "rho", "direction", "method", "n", "x", "y"}
    assert res["n"] >= 2
    assert isinstance(res["ok"], bool)

def test_monotone_decreasing_with_mask(df_mixed):
    # Keep only rows with x >= 2.0
    mask = GEQ("x", 2.0).mask(df_mixed)
    mono = MonotoneRelation(x="x", y="y", direction="decreasing", method="pearson", min_abs_rho=0.0)
    res = mono.evaluate_global(df_mixed, mask=mask)
    assert set(res.keys()) >= {"ok", "rho", "direction", "method", "n", "x", "y"}
    # Count valid rows after mask and NaN drop (matches implementation)
    expected_n = (
        pd.to_numeric(df_mixed.loc[mask, "x"], errors="coerce")
          .to_frame("x")
          .assign(y=pd.to_numeric(df_mixed.loc[mask, "y"], errors="coerce"))
          .dropna()
          .shape[0]
    )
    assert res["n"] == expected_n


# -----------------------
# Additional coverage
# -----------------------

def test_strict_decreasing_detected_by_both_methods(df_decreasing):
    m_s = MonotoneRelation("x", "y", "decreasing", "spearman", min_abs_rho=0.9)
    m_p = MonotoneRelation("x", "y", "decreasing", "pearson",  min_abs_rho=0.9)
    rs = m_s.evaluate_global(df_decreasing)
    rp = m_p.evaluate_global(df_decreasing)
    assert rs["ok"] is True and rp["ok"] is True
    assert rs["rho"] <= 0.0 and rp["rho"] <= 0.0
    assert abs(rs["rho"]) >= 0.9 and abs(rp["rho"]) >= 0.9

def test_constant_series_yields_zero_rho_and_ok_false(df_constant_y):
    m1 = MonotoneRelation("x", "y", "increasing", "spearman", min_abs_rho=0.1)
    m2 = MonotoneRelation("x", "y", "increasing", "pearson",  min_abs_rho=0.1)
    r1 = m1.evaluate_global(df_constant_y)
    r2 = m2.evaluate_global(df_constant_y)
    assert r1["rho"] == 0.0 and r2["rho"] == 0.0
    assert r1["ok"] is False and r2["ok"] is False

def test_threshold_controls_success(df_decreasing):
    # Strong negative correlation: decreasing direction with high threshold succeeds
    ok_high = MonotoneRelation("x", "y", "decreasing", "spearman", min_abs_rho=0.95).evaluate_global(df_decreasing)["ok"]
    # But the same data with increasing direction should fail
    bad_dir = MonotoneRelation("x", "y", "increasing", "spearman", min_abs_rho=0.0).evaluate_global(df_decreasing)["ok"]
    assert ok_high is True
    assert bad_dir is False

def test_mask_restriction_applies_before_nan_drop():
    # Build a frame with NaNs that would normally be dropped, but mask to a clean subset
    df = pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 5],
        "y": [np.nan, np.nan, 2, 3, 4, 5],
        "use": [False, False, True, True, True, True],
    })
    mask = Predicate.from_column("use").mask(df)
    res = MonotoneRelation("x", "y", "increasing", "spearman", 0.8).evaluate_global(df, mask=mask)
    assert res["n"] == 4  # rows 2..5 after mask; none are NaN now
    assert res["ok"] is True

def test_repr_smoke():
    s = repr(MonotoneRelation("x", "y", "increasing", "spearman", 0.5))
    assert "Monotone(" in s and "x→y" in s and "min|rho|=0.5" in s
