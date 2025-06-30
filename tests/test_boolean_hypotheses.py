import pandas as pd
import pytest
import itertools

from txgraffiti2.conjecture_logic import Property, Predicate, Inequality, Conjecture
from txgraffiti2.boolean_hypotheses_generator import generate_boolean_hypotheses

def make_simple_df():
    # 3 boolean columns A,B,C and one numeric
    return pd.DataFrame({
        "A": [ True,  False, True ],
        "B": [ False, True,  True ],
        "C": [ True,  True,  False],
        "x": [ 1, 2, 3 ]
    })

def test_all_valid_combinations_of_two_booleans():
    df = make_simple_df()[["A", "B"]]  # only use A and B

    hyps = generate_boolean_hypotheses(df, lower_bound=1, upper_bound=2)

    expected_fragments = {
        "A",
        "¬(A)",
        "B",
        "¬(B)",
        "(A) ∧ (B)",
        "(A) ∧ (¬(B))",
        "(¬(A)) ∧ (B)",
        "(¬(A)) ∧ (¬(B))",
    }

    generated_names = {h.name for h in hyps}

    # Every generated hypothesis must be one of the expected ones
    for name in generated_names:
        assert name in expected_fragments, f"Unexpected hypothesis: {name}"

def test_count_size_2():
    df = make_simple_df()
    # all combinations of 2 columns, each negated or not:
    # C(3,2)=3 pairs, each has 2^2=4 sign‐choices → 12 total
    hyps = generate_boolean_hypotheses(df, lower_bound=2, upper_bound=2)
    assert len(hyps) == 3 * 4

    # Check that a known combo appears, e.g. A ∧ ¬B
    target = next(h for h in hyps if "(A)" in h.name and "(¬(B))" in h.name)
    # Its mask should equal df["A"] & ~df["B"]
    expected_mask = df["A"] & ~df["B"]
    pd.testing.assert_series_equal(target(df), expected_mask)

def test_count_sizes_1_to_3():
    df = make_simple_df()
    # lower=1, upper=3 → sum_{r=1..3} C(3,r)*2^r = 3*2 + 3*4 + 1*8 = 6 + 12 + 8 = 26
    hyps = generate_boolean_hypotheses(df, lower_bound=1, upper_bound=3)
    assert len(hyps) == 26

def test_invalid_bounds():
    df = make_simple_df()
    # upper bound too large
    with pytest.raises(AssertionError):
        generate_boolean_hypotheses(df, lower_bound=1, upper_bound=4)
    # lower > upper
    with pytest.raises(AssertionError):
        generate_boolean_hypotheses(df, lower_bound=3, upper_bound=2)
        