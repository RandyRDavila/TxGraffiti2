import pandas as pd
import pytest

from txgraffiti.logic.properties import Property, Constant
from txgraffiti.logic.tables import KnowledgeTable

# ——————— Fixtures ———————
@pytest.fixture
def df():
    # simple DataFrame with two rows
    return pd.DataFrame({
        'alpha': [1, 2, 3],
        'beta': [3, 1, 1],
        'connected': [True, True, True],
        'K_n': [True, False, False],
        'tree': [False, False, True],
    })

@pytest.fixture
def kt():
    return KnowledgeTable({
        'alpha': [1, 2, 3],
        'beta': [3, 1, 1],
        'connected': [True, True, True],
        'K_n': [True, False, False],
        'tree': [False, False, True],
    })


def test_constant_property_arithmetic():
    alpha = Property('alpha', lambda df: df['alpha'])
    beta = Property('beta', lambda df: df['beta'])

    assert repr(alpha) == '<Property alpha>'
    assert repr(beta) == '<Property beta>'

    # addition
    assert repr(alpha + beta) == '<Property (alpha + beta)>'
    assert repr(alpha + 1) == '<Property (1 + alpha)>' # constants always move left for convience
    # additive identity
    assert alpha + 0 is alpha
    assert alpha - 0 is alpha

    # multiplication
    assert repr(alpha * beta) == '<Property (alpha * beta)>'
    assert repr(alpha * 2) == '<Property (2 * alpha)>' # constants always move left for convience
    # additive identity
    assert alpha * 1 is alpha
    assert alpha / 1 is alpha

    # zero out a property
    assert repr(alpha - alpha) == '<Property 0>'
    # assert repr(alpha + (- alpha)) == '<Property 0>' # TODO: This fails and needs fixing
    assert repr(alpha * 0) == '<Property 0>'
    assert repr(0 * alpha) == '<Property 0>'

    # unit outcomes
    assert repr(alpha / alpha) == '<Property 1>'
    assert repr(alpha ** 0) == '<Property 1>'

def test_property_on_dataframe(df):
    alpha = Property('alpha', lambda df: df['alpha'])
    beta = Property('beta', lambda df: df['beta'])

    assert alpha(df).tolist() == [1, 2, 3]
    assert beta(df).tolist() == [3, 1, 1]

    # addition
    assert (alpha + beta)(df).tolist() == [4, 3, 4]
    assert (alpha + 1)(df).tolist() == [2, 3, 4]
    assert (1 + alpha)(df).tolist() == [2, 3, 4]
    # additive identity
    assert (alpha + 0)(df).tolist() == [1, 2, 3]
    assert (0 + alpha)(df).tolist() == [1, 2, 3]

    # zero out the Property
    assert (alpha - alpha)(df).tolist() == [0, 0, 0]
    assert (alpha * 0)(df).tolist() == [0, 0, 0]
    assert (0 * alpha)(df).tolist() == [0, 0, 0]

    # unit outcomes
    assert (alpha / alpha)(df).tolist() == [1, 1, 1]
    assert (alpha ** 0)(df).tolist() == [1, 1, 1]

def test_property_on_knowledgetable(kt):
    alpha = kt.alpha
    beta = kt.beta

    assert alpha(kt).tolist() == [1, 2, 3]
    assert beta(kt).tolist() == [3, 1, 1]

    # addition
    assert (alpha + beta)(kt).tolist() == [4, 3, 4]
    assert (alpha + 1)(kt).tolist() == [2, 3, 4]
    assert (1 + alpha)(kt).tolist() == [2, 3, 4]
    # additive identity
    assert (alpha + 0)(kt).tolist() == [1, 2, 3]
    assert (0 + alpha)(kt).tolist() == [1, 2, 3]

    # zero out the Property
    assert (alpha - alpha)(kt).tolist() == [0, 0, 0]
    assert (alpha * 0)(kt).tolist() == [0, 0, 0]
    assert (0 * alpha)(kt).tolist() == [0, 0, 0]

    # unit outcomes
    assert (alpha / alpha)(kt).tolist() == [1, 1, 1]
    assert (alpha ** 0)(kt).tolist() == [1, 1, 1]
