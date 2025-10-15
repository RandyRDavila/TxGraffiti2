import numpy as np
from txgraffiti2025.forms import MonotoneRelation, GEQ

def test_monotone_increasing_spearman(df_mixed):
    mono = MonotoneRelation(x="x", y="y", direction="increasing", method="spearman")
    res = mono.evaluate_global(df_mixed)
    assert set(res.keys()) >= {"ok","rho","direction","method","n","x","y"}
    assert res["n"] >= 2
    assert isinstance(res["ok"], bool)

def test_monotone_decreasing_with_mask(df_mixed):
    mask = GEQ("x", 2.0).mask(df_mixed)
    mono = MonotoneRelation(x="x", y="y", direction="decreasing", method="pearson", min_abs_rho=0.0)
    res = mono.evaluate_global(df_mixed, mask=mask)
    assert set(res.keys()) >= {"ok","rho","direction","method","n","x","y"}
