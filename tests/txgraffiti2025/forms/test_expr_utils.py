import numpy as np
import pytest

from txgraffiti2025.forms.expr_utils import (
    expr_depends_on,
    cancel_feature_in_product,
    simplify_coeff_times_feature,
    structurally_equal,
)
from txgraffiti2025.forms.utils import to_expr, sqrt
from txgraffiti2025.forms.nonlinear import ratio


def test_expr_depends_on_binary_and_unary():
    # e = a + 2*b
    e = to_expr("a") + (to_expr(2) * to_expr("b"))
    assert expr_depends_on(e, "a") is True
    assert expr_depends_on(e, ["b", "c"]) is True
    assert expr_depends_on(e, "c") is False

    # Unary path: sqrt(a)
    u = sqrt("a")
    assert expr_depends_on(u, "a") is True
    assert expr_depends_on(u, {"b", "c"}) is False


def test_cancel_feature_in_product_matches_ratio_denominator():
    # coeff = num / a  -> cancel_feature_in_product(..., "a") = num
    num = to_expr("num")
    coeff = ratio(num, "a")  # BinOp(np.divide)
    out = cancel_feature_in_product(coeff, "a")
    assert out is not None
    assert structurally_equal(out, num)

    # wrong denominator: num / b  with feature "a" -> None
    coeff2 = ratio(num, "b")
    assert cancel_feature_in_product(coeff2, "a") is None


def test_simplify_coeff_times_feature_cancels_or_multiplies():
    # (num / a) * a  -> num
    num = to_expr("num")
    coeff = ratio(num, "a")
    simp = simplify_coeff_times_feature(coeff, "a")
    assert structurally_equal(simp, num)

    # Otherwise: 3 * a
    coeff2 = to_expr(3)
    simp2 = simplify_coeff_times_feature(coeff2, "a")
    expected = to_expr(3) * to_expr("a")
    assert structurally_equal(simp2, expected)


def test_structurally_equal_compares_tree_shape_not_algebra():
    # Same tree -> True
    e1 = to_expr("a") + to_expr("b")
    e2 = to_expr("a") + to_expr("b")
    assert structurally_equal(e1, e2) is True

    # Different operand order -> should differ structurally
    e3 = to_expr("b") + to_expr("a")
    assert structurally_equal(e1, e3) is False

    # Different association with different operators: (a + (b * c)) vs ((a + b) * c) -> False
    # This avoids addition-flattening in your repr, ensuring a structural diff.
    e_left = to_expr("a") + (to_expr("b") * to_expr("c"))
    e_right = (to_expr("a") + to_expr("b")) * to_expr("c")
    assert structurally_equal(e_left, e_right) is False
