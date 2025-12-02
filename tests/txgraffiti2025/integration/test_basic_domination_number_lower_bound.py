# tests/unit/txgraffiti/integration/test_basic_domination_number_lower_bound.py

import pytest
import numpy as np
import pandas as pd

from txgraffiti.example_data import graph_data as df
from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge
from txgraffiti2025.forms.predicates import Predicate, LEQ, AndPred, Where
from txgraffiti2025.processing.post.reciprocal_generalizer import (
    propose_generalizations_from_reciprocals,
)

@pytest.mark.integration
def test_generalizes_subcubic_lower_bound_to_connected_symbolic_reciprocal():
    """
    Check that:
      ((connected) ∧ (subcubic)) ⇒ γ ≥ (1/4)·n
    generalizes to the symbolic
      (connected) ⇒ γ ≥ (1/(Δ+1))·n

    using the reciprocal generalizer on example graph data.
    """
    # Column names expected in example data:
    #   - "connected" (bool)
    #   - "maximum_degree" (Δ)
    #   - "order" (n)
    #   - "domination_number" (γ)
    assert {"connected", "maximum_degree", "order", "domination_number"} <= set(df.columns)

    # Base hypothesis: connected & subcubic (Δ ≤ 3)
    H_conn = Predicate.from_column("connected")
    H_subcubic = LEQ("maximum_degree", 3.0)
    H_base = AndPred(H_conn, H_subcubic)

    # Base conjecture on H_base: γ ≥ (1/4)·n
    base = Conjecture(
        relation=Ge("domination_number", (1/4) * to_expr("order")),
        condition=H_base,
        name="base_subcubic_lower",
    )

    # Ask the reciprocal generalizer to try replacing 1/4 by 1/(Δ+1) and
    # test on the superset (connected) (and also on the base, harmless).
    props = propose_generalizations_from_reciprocals(
        df,
        base,
        candidate_hypotheses=[H_conn, H_base],
        candidate_cols=["maximum_degree"],
        shifts=(1,),        # we want 1/(Δ + 1)
        min_support=5,      # enough rows on the slice
        atol=1e-9,
    )

    # We expect *at least one* proposal, and among them the symbolic:
    #   (connected) ⇒ γ ≥ (1/(Δ+1))·n
    assert len(props) >= 1

    # Build the expected symbolic relation for matching via repr
    coeff_symbolic = Const(1) / (to_expr("maximum_degree") + Const(1))
    expected_rel_repr = repr(Ge("domination_number", coeff_symbolic * to_expr("order")))

    # Find a candidate that (a) uses the connected mask and (b) matches the relation structurally
    found = False
    conn_mask = H_conn.mask(df)
    for c in props:
        # same condition mask as connected?
        if not c.condition.mask(df).equals(conn_mask):
            continue
        # relation matches the symbolic "1/(Δ+1) * order" (structural repr match)
        if repr(c.relation) == expected_rel_repr:
            # sanity: the proposed conjecture should actually hold on df
            assert c.is_true(df)
            found = True
            break

    assert found, "Did not find (connected) ⇒ γ ≥ (1/(Δ+1))·n among proposals"
