import numpy as np
from txgraffiti2025.workbench.arrays import support, same_mask, includes, all_le_on_mask, strictly_pos_on

def test_support_and_same_mask():
    a = np.array([True, False, True, False])
    b = np.array([True, False, True, False])
    c = np.array([True, True, False, False])
    assert support(a) == 2
    assert same_mask(a, b)
    assert not same_mask(a, c)

def test_includes_and_comparisons():
    a = np.array([True,  True,  False, True ])
    b = np.array([False, True,  False, False])
    assert includes(a, b)   # when b, a must be True
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.5, 3.5, 10.0])
    m = np.array([True, True, True, False])
    assert all_le_on_mask(x, y, m)
    assert strictly_pos_on(x, m)
