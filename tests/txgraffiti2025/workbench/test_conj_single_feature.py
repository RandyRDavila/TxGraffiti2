from txgraffiti2025.workbench.conj_single_feature import generate_single_feature_bounds
from txgraffiti2025.workbench.config import GenerationConfig
from txgraffiti2025.forms.predicates import Where

def TRUE_where():
    return Where(fn=lambda df: [True]*len(df), name="TRUE")

def test_single_feature_emits_bounds_for_linear_relation(toy_df):
    cfg = GenerationConfig(max_denom=8, use_floor_ceil_if_true=True)
    H = TRUE_where()
    lowers, uppers = generate_single_feature_bounds(
        toy_df, "y", hyps=[H], numeric_columns=["x", "z", "y"], config=cfg
    )

    # Numeric detector for y ? 2*x regardless of pretty formatting
    import numpy as np
    def is_two_x(conj, df):
        if conj.relation.left.pretty() != "y":
            return False
        rhs = conj.relation.right.eval(df).to_numpy(dtype=float, copy=False)
        x = df["x"].to_numpy(dtype=float, copy=False)
        m = np.isfinite(rhs) & np.isfinite(x) & (x != 0)
        if not np.any(m):
            return False
        ratios = rhs[m] / x[m]
        return np.allclose(ratios, 2.0, rtol=1e-9, atol=1e-9)

    has_lb_2x = any(is_two_x(c, toy_df) for c in lowers)
    has_ub_2x = any(is_two_x(c, toy_df) for c in uppers)
    assert has_lb_2x and has_ub_2x
