import pandas as pd
import numpy as np
import pytest

from txgraffiti2025.forms import Conjecture, Le, to_expr, Where
from txgraffiti2025.processing.post.generalize import generalize, generalize_one

@pytest.fixture
def df_connected_regular():
    # Build rows where "regular" implies max_deg == min_deg, hence m == 1 = (max_deg)/(min_deg)
    return pd.DataFrame({
        "connected": [True, True, True, False],
        "regular":   [True, True, False, False],
        "max_deg":   [3, 5, 7, 2],
        "min_deg":   [3, 5, 1, 1],
        "alpha":     [2, 3, 4, 1],    # target
        "mu":        [2, 3, 6, 2],    # other
    })

def P(col):
    return Where(lambda df: df[col].astype(bool))

def test_generalize_from_m_equals_1_to_ratio(df_connected_regular):
    # Seed: (connected & regular) -> alpha <= 1 * mu + 0
    H = P("connected") & P("regular")
    conj = Conjecture(Le("alpha", to_expr("mu")), H, name="alpha_le_mu_on_conn_reg")
    out = generalize([conj], df_connected_regular)
    new = out[0]
    # Expect the hypothesis to weaken to connected (dropping regular)
    assert new.condition.mask(df_connected_regular).sum() >= df_connected_regular["connected"].sum() - 0
    # Verify right-hand side uses a ratio of max_deg/min_deg (or shifted variant) multiplying mu
    rhs_str = new.name  # we appended ratio descriptor into name
    assert "max_deg" in rhs_str and "min_deg" in rhs_str

def test_generalize_cubic_three_fourths(df_connected_regular):
    # Modify df so that on "cubic" rows, max_deg=3, and ratio max_deg/(max_deg+1) = 3/4
    df = df_connected_regular.copy()
    df["cubic"] = (df["max_deg"] == 3)
    df.loc[:, "alpha"] = [2.25, 3.75, 4.0, 1.0]  # arbitrary
    df.loc[:, "n"] = [4, 8, 12, 4]

    H = P("connected") & P("cubic")
    # Seed slope = 3/4: target <= (3/4)*n
    conj = Conjecture(Le("alpha", (to_expr("n") * 0.75)), H, name="alpha_le_3_4_n_cubic")
    new = generalize([conj], df)[0]
    # Expect generalized ratio includes max_deg and (max_deg+1) in descriptor
    assert "max_deg" in new.name and "+1" in new.name

def test_no_generalization_when_no_ratio_matches(df_connected_regular):
    df = df_connected_regular
    H = P("connected") & P("regular")
    # Use slope that doesn't match any ratio from shifts {-2..2}
    conj = Conjecture(Le("alpha", (to_expr("mu") * 0.63)), H, name="alpha_le_weird_slope")
    res = generalize([conj], df)[0]
    # Should return original unchanged
    assert res.name == "alpha_le_weird_slope"
