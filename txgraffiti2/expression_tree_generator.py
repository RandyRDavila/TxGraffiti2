import numpy as np
from typing import Union
from itertools import product
from typing import List
import pandas as pd

from txgraffiti2.conjecture_logic import Property, Predicate, Conjecture


# Expression Tree Generation of Conjectures
class ExprNode:
    def evaluate(self) -> Union["Property", "Predicate", "Conjecture"]:
        raise NotImplementedError()

class PropertyLeaf(ExprNode):
    def __init__(self, prop: Property):
        self.prop = prop

    def evaluate(self):
        return self.prop
    
    def __eq__(self, other):
        return isinstance(other, PropertyLeaf) and self.prop.name == other.prop.name

    def __hash__(self):
        return hash(('PropertyLeaf', self.prop.name))

class ConstLeaf(ExprNode):
    def __init__(self, value: float):
        self.value = value

    def evaluate(self):
        return Property(str(self.value), lambda df, v=self.value: pd.Series(v, index=df.index))
    
    def __eq__(self, other):
        return isinstance(other, ConstLeaf) and self.value == other.value
    
    def __hash__(self):
        return hash(('ConstLeaf', self.value))

class BinaryOpNode(ExprNode):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __eq__(self, other):
        return (
            isinstance(other, BinaryOpNode)
            and self.op == other.op
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self):
        return hash((self.op, self.left, self.right))


    def evaluate(self):
        lval = self.left.evaluate()
        rval = self.right.evaluate()

        if self.op == '+': return lval + rval
        if self.op == '-': return lval - rval
        if self.op == '*': return lval * rval
        if self.op == '/':
            # Create a Property that handles divide-by-zero safely during evaluation
            new_name = f"({lval.name} / {rval.name})"
            return Property(new_name, lambda df: lval(df) / rval(df).replace(0, np.nan))

        if self.op == '**': return lval ** rval
        if self.op == '>=': return lval >= rval
        if self.op == '<=': return lval <= rval
        if self.op == '>': return lval > rval
        if self.op == '<': return lval < rval
        if self.op == '==': return lval == rval
        if self.op == '!=': return lval != rval
        raise ValueError(f"Unknown operator {self.op}")

class ConjectureNode(ExprNode):
    def __init__(self, hypothesis: ExprNode, conclusion: ExprNode):
        self.hypothesis = hypothesis
        self.conclusion = conclusion

    def evaluate(self):
        hyp = self.hypothesis.evaluate()
        concl = self.conclusion.evaluate()
        return Conjecture(hyp, concl)

def is_trivial_expr(node: ExprNode) -> bool:
    if isinstance(node, BinaryOpNode):
        op = node.op
        left = node.left
        right = node.right

        # x - x → 0
        if op == '-' and repr(left) == repr(right):
            return True

        # x / x → 1 (when x ≠ 0)
        if op == '/' and repr(left) == repr(right):
            return True

        # x / 1 or 1 * x → same as x
        if op == '/' and isinstance(right, ConstLeaf) and right.value == 1:
            return True

        # 0 / x → 0
        if op == '/' and isinstance(left, ConstLeaf) and left.value == 0:
            return True

        # x * 1 or x + 0 → same as x
        if isinstance(right, ConstLeaf):
            if (op == '*' and right.value == 1) or (op == '+' and right.value == 0):
                return True
        if isinstance(left, ConstLeaf):
            if (op == '*' and left.value == 1) or (op == '+' and left.value == 0):
                return True

    return False

COMMUTATIVE_OPS = {"+", "*"}

def generate_binary_exprs(
    exprs: List[ExprNode],
    ops: List[str]
) -> List[ExprNode]:
    """
    Generate all valid binary expressions from exprs × exprs using given ops.
    Handles commutativity: generates only one of (a + b) and (b + a), but both (a - b) and (b - a).
    """
    expressions = []
    seen = set()

    for a, b in product(exprs, exprs):
        for op in ops:
            if op in COMMUTATIVE_OPS:
                key = (op, tuple(sorted([hash(a), hash(b)])))
                if key in seen:
                    continue
                seen.add(key)
                expr = BinaryOpNode(op, a, b)
                if not is_trivial_expr(expr):
                    expressions.append(expr)
            else:
                expr = BinaryOpNode(op, a, b)
                if not is_trivial_expr(expr):
                    expressions.append(expr)
    return expressions


def generate_conjectures_by_expr_search(
    df: pd.DataFrame,
    features: List[Property],
    target: Property,
    hypothesis: Predicate,
    max_depth: int = 2,
    constants: List[float] = [1, 2],
    operators: List[str] = ['+', '-', '*', '/'],
    threshold: float = 1.0,
) -> List[Conjecture]:
    """
    Generate Conjectures using expression trees:
    for all expr in generated_exprs:
        if hypothesis(df): target <= expr or target >= expr (if high match)
    """

    def generate_expr_trees(depth):
        if depth == 1:
            return [PropertyLeaf(f) for f in features] + [ConstLeaf(c) for c in constants]

        smaller = generate_expr_trees(depth - 1)
        new_exprs = generate_binary_exprs(smaller, operators)
        return smaller + new_exprs

    expr_nodes = generate_expr_trees(max_depth)

    # Step 2: Evaluate and filter meaningful conjectures
    conjectures = []
    target_vals = target(df)
    mask = hypothesis(df)

    for node in expr_nodes:
      try:
        expr_prop = node.evaluate()
        expr_vals = expr_prop(df)
        if expr_vals.shape != target_vals.shape:
            continue

        # Check on subset where hypothesis holds
        t = target_vals[mask]
        e = expr_vals[mask]

        eps = 1e-8

        if (t <= e + eps).all() and (np.abs(t - e) < eps).any():
            ineq = Inequality(target, '≤', expr_prop)
        elif (t >= e - eps).all() and (np.abs(t - e) < eps).any():
            ineq = Inequality(target, '≥', expr_prop)
        else:
            continue  # Not a universal bound or not sharp anywhere

        conj = Conjecture(hypothesis, ineq)
        conjectures.append(conj)

      except Exception:
        continue

    return conjectures
