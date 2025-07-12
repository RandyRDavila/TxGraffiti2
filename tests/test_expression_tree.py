import pandas as pd
import numpy as np
import pytest

from txgraffiti2.conjecture_logic import Property, Predicate, Inequality, Conjecture
from txgraffiti2.expression_tree_generator import (
    ExprNode, PropertyLeaf, ConstLeaf, BinaryOpNode, is_trivial_expr, generate_binary_exprs, generate_conjectures_by_expr_search
)   

def test_property_leaf_evaluation():
    df = pd.DataFrame({'x': [1, 2, 3]})
    prop = Property('x', lambda df: df['x'])
    leaf = PropertyLeaf(prop)
    evaluated = leaf.evaluate()
    expected = pd.Series([1, 2, 3], name='x')  # Match the name
    pd.testing.assert_series_equal(evaluated(df), expected)

def test_const_leaf_evaluation():
    df = pd.DataFrame(index=range(3))
    const = ConstLeaf(42)
    evaluated = const.evaluate()
    pd.testing.assert_series_equal(evaluated(df), pd.Series([42, 42, 42]))

def test_binary_op_add():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    pa = Property('a', lambda df: df['a'])
    pb = Property('b', lambda df: df['b'])
    node = BinaryOpNode('+', PropertyLeaf(pa), PropertyLeaf(pb))
    result = node.evaluate()(df)
    expected = pd.Series([4, 6])
    pd.testing.assert_series_equal(result, expected)

def test_binary_op_divide_by_zero():
    df = pd.DataFrame({'a': [1, 2], 'b': [0, 1]})
    pa = Property('a', lambda df: df['a'])
    pb = Property('b', lambda df: df['b'])
    node = BinaryOpNode('/', PropertyLeaf(pa), PropertyLeaf(pb))
    result = node.evaluate()(df)
    expected = pd.Series([np.nan, 2.0])
    pd.testing.assert_series_equal(result, expected)


def test_trivial_sub_same_var():
    x = ConstLeaf(5)
    node = BinaryOpNode('-', x, x)
    assert is_trivial_expr(node)

def test_nontrivial_add_different():
    a = ConstLeaf(2)
    b = ConstLeaf(3)
    node = BinaryOpNode('+', a, b)
    assert not is_trivial_expr(node)

# NOT CURRENTLY WORKING BUT WILL BE FIXED
# def test_generate_binary_exprs_no_duplicate_commutative():
#     x = ConstLeaf(1)
#     y = PropertyLeaf(Property("y", lambda df: df["x"]))
#     exprs = [x, y]
#     expr_list = generate_binary_exprs(exprs, ['+'])
#     assert len(expr_list) == 1  # Only one of (x + y) or (y + x)


# def test_generate_conjectures_simple_case():
#     df = pd.DataFrame({
#         'is_simple': [True, True, True],
#         'f0': [1, 2, 3],
#         'f1': [2, 3, 4],
#     })

#     f0 = Property('f0', lambda df: df['f0'])
#     f1 = Property('f1', lambda df: df['f1'])
#     is_simple = Predicate('is_simple', lambda df: df['is_simple'])

#     conjs = generate_conjectures_by_expr_search(
#         df=df,
#         features=[f0],
#         target=f1,
#         hypothesis=is_simple,
#         max_depth=2,
#         constants=[1],
#         operators=['+', '-', '*', '/']
#     )
#     assert any('≥' in c.conclusion.op or '≤' in c.conclusion.op for c in conjs)
