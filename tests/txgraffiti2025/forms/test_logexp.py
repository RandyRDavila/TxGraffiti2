import numpy as np
import pandas as pd
import pytest
from txgraffiti2025.forms import log_base, exp_e, sqrt, to_expr
from txgraffiti2025.forms.generic_conjecture import Le, Ge

def test_log_exp_sqrt(df_basic):
    lg = log_base("a").eval(df_basic)
    ex = exp_e(0 * to_expr("a")).eval(df_basic)
    rt = sqrt("c").eval(df_basic)
    assert np.isfinite(lg).all()
    assert (ex == 1.0).all()
    assert (rt >= 0.0).all()

def test_log_base_change(df_basic):
    # log_b(x) = ln(x)/ln(b)
    lg10 = log_base("a", base=10).eval(df_basic)
    lge = log_base("a").eval(df_basic)
    ratio = (lge / lg10).replace([np.inf, -np.inf], np.nan)
    finite = ratio.notna()
    # ratio should equal ln(10) wherever defined (exclude x==1 where both logs are 0)
    assert np.allclose(ratio[finite].to_numpy(), np.log(10.0), rtol=1e-6, atol=1e-6)

def test_log_on_nonpositive_values_behaviour():
    # Just documents numpy/pandas semantics: log(0)->-inf, log(neg)->nan
    df = pd.DataFrame({"x": [0.0, -1.0, 1.0]})
    out = log_base("x").eval(df).to_numpy()
    assert np.isneginf(out[0])         # log(0) = -inf
    assert np.isnan(out[1])            # log(-1) = nan
    assert np.isfinite(out[2])         # log(1) = 0

def test_exp_and_sqrt_relation_integration(df_basic):
    # sqrt(a)^2 ≤ a (within floating tolerance)
    r1 = Le(sqrt("a") * sqrt("a"), "a")
    # Le slack = a - (sqrt(a)*sqrt(a)); allow tiny negative due to FP error
    assert (r1.slack(df_basic) >= -1e-12).all()

    # exp(0*x) >= 1
    r2 = Ge(exp_e(0 * to_expr("a")), 1.0)
    assert r2.evaluate(df_basic).all()

def test_log_base_equivalence_to_natural_log(df_basic):
    # log_b(x) = ln(x)/ln(b), so for b=e we match natural log exactly
    ln = log_base("a")              # natural log
    lne = log_base("a", base=np.e)  # base-e log
    np.testing.assert_allclose(ln.eval(df_basic), lne.eval(df_basic), rtol=0, atol=1e-12)
