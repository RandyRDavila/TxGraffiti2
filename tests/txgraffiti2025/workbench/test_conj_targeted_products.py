from txgraffiti2025.workbench.conj_targeted_products import generate_targeted_product_bounds
from txgraffiti2025.forms.predicates import Where

def TRUE_where():
    return Where(fn=lambda df: [True]*len(df), name="TRUE")

def test_targeted_products_smoke(toy_df):
    H = TRUE_where()
    lows, ups = generate_targeted_product_bounds(
        toy_df,
        target_col="x",         # treat x as the T in T*x ? No, function multiplies target * x; so set target to 'x'
        hyps=[H],
        x_candidates=["z"],     # T*x = x*z
        yz_candidates=["y","z"]
    )
    assert isinstance(lows, list) and isinstance(ups, list)
