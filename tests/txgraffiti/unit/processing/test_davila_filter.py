import pandas as pd
from txgraffiti.logic import Predicate, Property, Conjecture
from txgraffiti.processing.davila import filter_with_morgan

def test_filter_with_morgan_basic():
    # Create test DataFrame
    df = pd.DataFrame({
        'alpha':     [1, 2, 3],
        'beta':      [3, 2, 1],
        'connected': [True, True, True],
        'tree':      [False, True, False],
    })

    # Define predicates and properties
    P_conn = Predicate('connected', lambda df: df['connected'])
    P_tree = Predicate('tree',      lambda df: df['tree'])

    A = Property('alpha', lambda df: df['alpha'])
    B = Property('beta',  lambda df: df['beta'])

    # Conjecture 1: more specific
    c1 = P_tree >> (A <= B)
    # Conjecture 2: more general, same conclusion
    c2 = P_conn >> (A <= B)

    # Conjecture 3: different conclusion, should not be removed
    c3 = P_tree >> (A >= B)

    # Apply Morgan filtering
    accepted = filter_with_morgan([c1, c2, c3], df)

    # Only c2 and c3 should be retained
    assert c2 in accepted
    assert c3 in accepted
    assert c1 not in accepted
