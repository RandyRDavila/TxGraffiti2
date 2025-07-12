import pandas as pd
import pytest

from txgraffiti.logic.properties import Constant
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
    const1 = Constant(1)
    const2 = Constant(2)

    assert repr(const1) == '<Property 1>'
    assert repr(const2) == '<Property 2>'

    # addition
    assert repr(const1 + const2) == '<Property (1 + 2)>'

    # addition with number
    assert repr(const1 + 1) == '<Property (1 + 1)>'

    # subtraction
    assert repr(const1 - const2) == '<Property (1 - 2)>'
    assert repr(const2 - const1) == '<Property (2 - 1)>'

    # subtraction with number
    assert repr(const1 - 2) == '<Property (1 - 2)>'
    assert repr(const2 - 1) == '<Property (2 - 1)>'

    # additive identity
    assert const1 + 0 is const1

    # multiplicative identity
    assert const1 * 1 is const1

    # multiplication
    assert repr(const2 * const2) == '<Property (2 * 2)>'

def test_constant_property_dataframe(df):
    const1 = Constant(1)
    const2 = Constant(2)

    assert const1(df).tolist() == [1, 1, 1]
    assert const2(df).tolist() == [2, 2, 2]

    # addition
    assert (const1 + const2)(df).tolist() == [3, 3, 3]
    assert (1 + const2)(df).tolist() == [3, 3, 3]
    assert (const2 + const1)(df).tolist() == [3, 3, 3]
    assert (const1 + 2)(df).tolist() == [3, 3, 3]
    # additive identity
    assert (const1 + 0)(df).tolist() == [1, 1, 1]
    assert (const2 + 0)(df).tolist() == [2, 2, 2]

    # multiplication
    assert (const1 * const2)(df).tolist() == [2, 2, 2]
    assert (1 * const2)(df).tolist() == [2, 2, 2]
    assert (const2 * const1)(df).tolist() == [2, 2, 2]
    assert (const1 * 2)(df).tolist() == [2, 2, 2]
    # multiplicative identity
    assert (const1 * 1)(df).tolist() == [1, 1, 1]
    assert (const2 * 1)(df).tolist() == [2, 2, 2]

def test_constant_property_knowledgetable(kt):
    const1 = Constant(1)
    const2 = Constant(2)

    assert const1(kt).tolist() == [1, 1, 1]
    assert const2(kt).tolist() == [2, 2, 2]

    # addition
    assert (const1 + const2)(kt).tolist() == [3, 3, 3]
    assert (1 + const2)(kt).tolist() == [3, 3, 3]
    assert (const2 + const1)(kt).tolist() == [3, 3, 3]
    assert (const1 + 2)(kt).tolist() == [3, 3, 3]
    # additive identity
    assert (const1 + 0)(kt).tolist() == [1, 1, 1]
    assert (const2 + 0)(kt).tolist() == [2, 2, 2]

    # multiplication
    assert (const1 * const2)(kt).tolist() == [2, 2, 2]
    assert (1 * const2)(kt).tolist() == [2, 2, 2]
    assert (const2 * const1)(kt).tolist() == [2, 2, 2]
    assert (const1 * 2)(kt).tolist() == [2, 2, 2]
    # multiplicative identity
    assert (const1 * 1)(kt).tolist() == [1, 1, 1]
    assert (const2 * 1)(kt).tolist() == [2, 2, 2]
