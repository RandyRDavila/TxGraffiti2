from txgraffiti.logic import *
from txgraffiti.processing import filter_with_dalmatian
import pandas as pd


def test_filter_with_dalmatian_basic():
    df = pd.DataFrame({
        'alpha':     [1, 2, 3],
        'beta':      [2, 3, 4],
        'connected': [True, True, True],
    })

    P = Predicate('connected', lambda df: df['connected'])
    A = Property('alpha', lambda df: df['alpha'])
    B = Property('beta',  lambda df: df['beta'])

    c1 = P >> (A <= B + 3)   # Loose
    c2 = P >> (A <= B + 1)   # Tighter
    c3 = P >> (A <= B)       # Best

    c4 = P >> (B <= 4) # Noncomparible to c1, c2, and c3
    c5 = P >> (B <= 5) # Comparible to c4, but worse
    c6 = P >> (B <= 3) # Comparible to, and better than c4, but false

    # Only c2 and c3 should remain
    accepted = filter_with_dalmatian([c1, c2, c3, c4, c5], df)

    assert c3 in accepted
    assert c2 not in accepted
    assert c1 not in accepted

    assert c4 in accepted
    assert c5 not in accepted
    assert c6 not in accepted
