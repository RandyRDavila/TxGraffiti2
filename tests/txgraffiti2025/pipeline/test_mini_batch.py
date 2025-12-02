import numpy as np
import pandas as pd

from txgraffiti2025.forms.predicates import Where
from txgraffiti2025.processing.pipeline.mini_batch import (
    generate_with_local_pruning, MiniBatchConfig
)

def df_basic():
    # Construct a small df where y ~ 2x, plus a weakly related feature w
    return pd.DataFrame({
        "H_all": [True]*8,
        "x":     [1,2,3,4,5,6,7,8],
        "w":     [1,1,1,2,3,5,8,13],     # noisy / poor predictor
        "y":     [2,4,6,8,10,12,14,16],  # exactly 2x
    })


def test_caps_respected_local_beam():
    df = df_basic()
    H = Where(lambda d: d["H_all"])
    cfg = MiniBatchConfig(
        direction="both",
        generators=frozenset({"ratios", "lp"}),
        context_mode="one",
        drop_frac_local=0.50,
        beam_k_per_H=2,  # very small cap to test
        max_seeds_per_generator_per_pair=10,
    )
    numeric_cols = ["x","w"]  # exclude target "y"
    seeds = generate_with_local_pruning(
        df, target="y", hypothesis=H, numeric_columns=numeric_cols, config=cfg
    )
    assert len(seeds) <= 2


def test_none_hypothesis_normalized_to_TRUE_for_generators():
    df = df_basic()
    H = None  # we’ll rely on TRUE inside the controller where needed
    cfg = MiniBatchConfig(
        direction="both",
        generators=frozenset({"ratios"}),  # ratios requires non-None predicate
        context_mode="one",
        drop_frac_local=0.50,
        beam_k_per_H=10,
    )
    numeric_cols = ["x","w"]
    seeds = generate_with_local_pruning(
        df, target="y", hypothesis=H, numeric_columns=numeric_cols, config=cfg
    )
    # We expect at least one valid ratio conjecture (y <= 2*x or y >= 2*x)
    assert len(seeds) >= 1
    # All results should be evaluable and true on their masks
    for c in seeds:
        assert c.is_true(df)


def test_local_hazel_drops_low_touch_with_aggressive_drop():
    """
    Create two feature contexts:
      - x gives exact y == 2x → high touch
      - w is weak/noisy → low touch
    With drop_frac_local=0.5 and small beam, the high-touch variant should survive.
    """
    df = df_basic()
    H = Where(lambda d: d["H_all"])
    cfg = MiniBatchConfig(
        direction="upper",
        generators=frozenset({"ratios"}),
        context_mode="one",
        drop_frac_local=0.50,
        beam_k_per_H=1,  # force single survivor
        max_seeds_per_generator_per_pair=10,
        ratios_q_clip=None,  # exact min/max
    )
    numeric_cols = ["x","w"]
    seeds = generate_with_local_pruning(
        df, target="y", hypothesis=H, numeric_columns=numeric_cols, config=cfg
    )
    assert len(seeds) == 1
    # The surviving conjecture should involve 'x' rather than 'w'
    # (touches many points with equality for y <= 2*x)
    txt = seeds[0].pretty(arrow="⇒")
    assert ("x" in txt) or ("x" in repr(seeds[0].relation))
