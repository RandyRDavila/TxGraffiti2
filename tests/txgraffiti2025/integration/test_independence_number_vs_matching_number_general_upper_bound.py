import pytest
import pandas as pd

from txgraffiti.example_data import graph_data
from txgraffiti2025.forms.utils import to_expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le
from txgraffiti2025.forms.predicates import Where
from txgraffiti2025.processing.pre.constants_cache import precompute_constant_ratios
from txgraffiti2025.processing.post.generalize_from_constants import (
    propose_generalizations_from_constants,
)

@pytest.mark.integration
def test_regular_alpha_le_mu_generalizes_to_connected_with_Delta_over_delta():
    """
    Coefficient generalization via cached ratios:

    ((connected) ∧ (regular)) ⇒ α ≤ 1·μ
    should generalize to
    (connected) ⇒ α ≤ (Δ/δ)·μ

    Mechanism:
      - Under regular graphs, Δ == δ, so Δ/δ == 1.
      - The constants cache built on the (connected ∧ regular) slice
        will contain the ratio (maximum_degree + 0)/(minimum_degree + 0)
        with value ≈ 1.
      - propose_generalizations_from_constants replaces the numeric
        coefficient (1) by that structural ratio and tests it on the
        superset hypothesis (connected).
    """
    df = graph_data.copy()

    # Sanity: required columns exist
    needed = {
        "connected", "regular",
        "independence_number", "matching_number",
        "maximum_degree", "minimum_degree",
    }
    assert needed.issubset(df.columns)

    # Hypotheses: H1 = connected ∧ regular, H2 = connected
    H1 = Where(lambda d: d["connected"] & d["regular"])
    H2 = Where(lambda d: d["connected"])

    # Base conjecture on regular graphs: α ≤ 1·μ
    base = Conjecture(
        relation=Le("independence_number", 1.0 * to_expr("matching_number")),
        condition=H1,
        name="alpha_le_mu_regular"
    )

    # Build constants cache on H1 that can discover Δ/δ == 1
    cache = precompute_constant_ratios(
        df,
        hypotheses=[H1],
        numeric_cols=["maximum_degree", "minimum_degree"],
        shifts=(0,),           # Δ/δ exactly
        min_support=3,         # modest support
        max_denominator=200    # nicer display (not required)
    )

    # Ask for generalization to connected graphs
    props = propose_generalizations_from_constants(
        df,
        conj=base,
        cache=cache,
        candidate_hypotheses=[H2],
        atol=1e-9,
    )

    # We expect at least one valid generalization
    assert len(props) >= 1

    # And it should explicitly use the Δ/δ coefficient
    found = any(
        ("maximum_degree" in repr(p.new_conjecture.relation)) and
        ("minimum_degree" in repr(p.new_conjecture.relation))
        for p in props
    )
    assert found, "Expected a proposal with (maximum_degree/minimum_degree)·matching_number"
