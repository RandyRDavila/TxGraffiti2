from txgraffiti2025.workbench.conj_mixed_bounds import generate_mixed_bounds
from txgraffiti2025.workbench.config import GenerationConfig
from txgraffiti2025.forms.predicates import Where

def TRUE_where():
    return Where(fn=lambda df: [True]*len(df), name="TRUE")

def test_mixed_bounds_smoke(toy_df):
    cfg = GenerationConfig()
    H = TRUE_where()
    lows, ups = generate_mixed_bounds(
        toy_df, "y",
        hyps=[H],
        primary=["x"],
        secondary=["z"],
        config=cfg,
        weight=0.5
    )
    # Should produce at least some bounds
    assert len(lows) > 0
    assert len(ups) > 0
