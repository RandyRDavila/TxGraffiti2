import numpy as np
from txgraffiti2025.forms import log_base, exp_e, sqrt, to_expr

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
