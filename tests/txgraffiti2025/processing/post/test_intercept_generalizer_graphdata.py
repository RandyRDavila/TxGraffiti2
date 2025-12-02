import pytest

from txgraffiti2025.forms.predicates import Where
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le
from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.processing.post.intercept_generalizer import propose_generalizations_from_intercept
from txgraffiti2025.forms.pretty import format_relation

# Use the example dataset shipped with txgraffiti
from txgraffiti.example_data import graph_data as df


@pytest.mark.integration
def test_tree_upper_bound_generalizes_to_order_minus_min_degree_on_connected():
    """
    Check that:
        (connected ∧ tree) ⇒ α ≤ n - 1
    generalizes to:
        (connected) ⇒ α ≤ n - δ

    where α = independence_number, n = order, δ = minimum_degree.
    """
    # Hypotheses
    H_tree = Where(lambda d: d["connected"] & d["tree"])
    H_conn = Where(lambda d: d["connected"])

    # Base conjecture on trees: independence_number <= order - 1
    base = Conjecture(
        relation=Le("independence_number", to_expr("order") + Const(-1.0)),
        condition=H_tree,
        name="alpha_upper_tree_connected",
    )

    # Sanity: base should be true on the tree slice
    assert base.is_true(df)

    # Ask the intercept generalizer to swap the numeric -1 with -minimum_degree
    # i.e., RHS = order + ( - minimum_degree ) = order - minimum_degree
    candidate_intercepts = [-to_expr("minimum_degree")]
    props = propose_generalizations_from_intercept(
        df,
        base,
        cache=None,                               # rely on user-supplied candidate
        candidate_hypotheses=[H_conn],
        candidate_intercepts=candidate_intercepts,
        require_superset=True,
    )

    # We expect at least one generalized conjecture on (connected)
    assert len(props) >= 1

    # And it should literally be "independence_number <= order - minimum_degree"
    pretty_hits = [
        p for p in props
        if p.new_conjecture.condition is H_conn
        and "minimum_degree" in format_relation(p.new_conjecture.relation, unicode_ops=False, strip_ones=True)
    ]
    assert pretty_hits, "No generalized conjecture used 'minimum_degree' in the intercept."

    # Finally, verify the chosen generalization is true on (connected)
    for g in pretty_hits:
        assert g.new_conjecture.is_true(df)
