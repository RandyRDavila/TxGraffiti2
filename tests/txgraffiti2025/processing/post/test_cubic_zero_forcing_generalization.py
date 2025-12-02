import pytest
import numpy as np
import pandas as pd

from txgraffiti.example_data import graph_data as df
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le
from txgraffiti2025.forms.predicates import Where
from txgraffiti2025.forms.utils import to_expr
from txgraffiti2025.processing.pre.constants_cache import (
    precompute_constant_ratios,
)
from txgraffiti2025.processing.post.generalize_from_constants import (
    propose_generalizations_from_constants,
)

@pytest.mark.integration
def test_cubic_upper_bound_generalizes_to_delta_over_delta_plus_one():
    """
    Check that:
        (connected ∧ cubic) ⇒ ZF ≤ (3/4)·n
    generalizes to:
        (connected) ⇒ ZF ≤ (Δ/(Δ+1))·n

    Implementation note:
    - We rely on the constants-cache to discover the ratio (Δ + 0) / (Δ + 1)
      under the cubic+connected slice, where Δ = 3 ⇒ 3/4.
    - This requires the constants-cache to allow numerator/denominator built
      from the SAME numeric column (with different shifts).
    """
    # Columns expected in example graph data:
    #   connected (bool), cubic (bool), order (n), zero_forcing_number (ZF), maximum_degree (Δ)
    assert {"connected", "cubic", "order", "zero_forcing_number", "maximum_degree"}.issubset(df.columns)

    H_cubic = Where(lambda d: d["connected"] & d["cubic"])
    H_conn  = Where(lambda d: d["connected"])

    # Base conjecture on cubic connected graphs: ZF ≤ (3/4)·n
    base = Conjecture(
        relation=Le("zero_forcing_number", (3.0/4.0) * to_expr("order")),
        condition=H_cubic,
        name="zf_upper_cubic_connected"
    )

    # Precompute constants under H_cubic, focusing on Δ (so we can form Δ/(Δ+1))
    cache = precompute_constant_ratios(
        df,
        hypotheses=[H_cubic],
        numeric_cols=["maximum_degree"],   # only need Δ
        shifts=(-2, -1, 0, 1, 2),
        min_support=3,                     # keep this modest for the slice
        max_denominator=200
    )

    # Ask the generalizer to replace 3/4 by a structural ratio and try the superset (connected)
    props = propose_generalizations_from_constants(
        df,
        base,
        cache,
        candidate_hypotheses=[H_conn],
        atol=1e-9,
    )

    # We expect at least one generalized conjecture on the connected class
    assert len(props) >= 1

    # Verify at least one proposal is on connected and holds globally
    ok = False
    for g in props:
        applicable, holds, failures = g.new_conjecture.check(df, auto_base=False)
        if g.new_conjecture.condition is H_conn and applicable.all() and holds.all() and failures.empty:
            ok = True
            break

    assert ok, "Did not find a true generalized conjecture on (connected). " \
               "If this fails, ensure constants_cache allows i == j (same column) " \
               "so it can form Δ/(Δ+1) via shifts."
