'''
conjecture_logic module

Provides classes to build and evaluate conjectures over pandas.DataFrame rows.

Classes:
- Property: numeric column or lifted scalar, supports arithmetic ops.
- Predicate: boolean test on rows, supports logical ops.
- Inequality: comparison between Properties, subclass of Predicate.
- Conjecture: implication from a hypothesis (Predicate) to a conclusion (Predicate).

Usage:
    from conjecture_logic import Property, Predicate, Inequality, Conjecture

Example:
    df = pd.DataFrame(...)
    deg = Property('degree', lambda df: df['degree'])
    const = Property('3', lambda df: 3)
    pred = deg >= const
    conj = Conjecture(pred, deg < 10)
    valid = conj.is_true(df)

TODO: This needs to be fleshed out with more examples and documentation. Also, conjecture_logic.py should be split into multiple files for better organization.
'''
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Union
from numbers import Number

# ───────── Property ─────────
@dataclass(frozen=True)
class Property:
    """
    A numeric column or expression on a DataFrame.
    Supports +, -, *, /, ** with auto-lifting of scalars.
    """
    name: str
    func: Callable[[pd.DataFrame], pd.Series]

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.func(df)

    def __repr__(self):
        return f"<Property {self.name}>"

    def __eq__(self, other):
        return isinstance(other, Property) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def _lift(self, other: Union['Property', Number]) -> 'Property':
        if isinstance(other, Property):
            return other
        if isinstance(other, Number):
            # constant property
            return Property(str(other), lambda df, v=other: pd.Series(v, index=df.index))
        raise TypeError(f"Cannot lift {other!r} into Property")

    def _binop(self, other, op_symbol: str, op_func):
        other = self._lift(other)

        # identity eliminations
        if op_symbol == "+":
            if other.name == "0": return self
            if self.name == "0": return other
        if op_symbol == "-" and other.name == "0":
            return self
        if op_symbol == "*":
            if other.name == "1": return self
            if self.name == "1": return other
            if other.name == "0" or self.name == "0":
                return Property("0", lambda df: pd.Series(0, index=df.index))
        if op_symbol == "/" and other.name == "1":
            return self
        if op_symbol == "**":
            if other.name == "1": return self
            if other.name == "0":
                return Property("1", lambda df: pd.Series(1, index=df.index))

        new_name = f"({self.name} {op_symbol} {other.name})"
        return Property(
            name=new_name,
            func=lambda df: op_func(self(df), other(df))
        )

    # arithmetic operators
    def __add__(self,  o): return self._binop(o, "+", pd.Series.add)
    def __sub__(self,  o): return self._binop(o, "-", pd.Series.sub)
    def __mul__(self,  o): return self._binop(o, "*", pd.Series.mul)
    def __truediv__(self, o): return self._binop(o, "/", pd.Series.div)
    def __pow__(self,  o): return self._binop(o, "**", pd.Series.pow)

    # reflections for scalars
    __radd__     = __add__
    __rsub__     = lambda s,o: s._lift(o)._binop(s, "-", pd.Series.sub)
    __rmul__     = __mul__
    __rtruediv__ = lambda s,o: s._lift(o)._binop(s, "/", pd.Series.div)
    __rpow__     = lambda s,o: s._lift(o)._binop(s, "**", pd.Series.pow)

    # comparisons → Inequality
    def __le__(self, o): return Inequality(self, "<=", self._lift(o))
    def __lt__(self,  o): return Inequality(self, "<",  self._lift(o))
    def __ge__(self,  o): return Inequality(self, ">=", self._lift(o))
    def __gt__(self,  o): return Inequality(self, ">",  self._lift(o))
    def __eq__(self,  o): return Inequality(self, "==", self._lift(o))
    def __ne__(self,  o): return Inequality(self, "!=", self._lift(o))


# ───────── Predicate ─────────
@dataclass(frozen=True)
class Predicate:
    """
    A boolean test on each row of a DataFrame.
    Supports ∧, ∨, and ¬ for combining predicates.
    """
    name: str
    func: Callable[[pd.DataFrame], pd.Series]

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.func(df)

    def __and__(self, other: 'Predicate') -> 'Predicate':
        return Predicate(
            name=f"({self.name}) ∧ ({other.name})",
            func=lambda df: self(df) & other(df),
        )

    def __or__(self, other: 'Predicate') -> 'Predicate':
        return Predicate(
            name=f"({self.name}) ∨ ({other.name})",
            func=lambda df: self(df) | other(df),
        )

    def __invert__(self) -> 'Predicate':
        return Predicate(
            name=f"¬({self.name})",
            func=lambda df: ~self(df),
        )

    def __repr__(self):
        return f"<Predicate {self.name}>"

    def __eq__(self, other):
        return isinstance(other, Predicate) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


# ──────── Inequality ─────────
class Inequality(Predicate):
    """
    A predicate of the form `lhs op rhs` on Properties.
    """
    def __init__(self, lhs: Property, op: str, rhs: Property):
        name = f"{lhs.name} {op} {rhs.name}"
        def func(df: pd.DataFrame) -> pd.Series:
            L, R = lhs(df), rhs(df)
            return {
                "<":  L <  R,
                "<=": L <= R, "≤": L <= R,
                ">":  L >  R,
                ">=": L >= R, "≥": L >= R,
                "==": L == R,
                "!=": L != R,
            }[op]
        super().__init__(name, func)
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "op", op)

    def __repr__(self):
        return f"<Ineq {self.name}>"

    def slack(self, df: pd.DataFrame) -> pd.Series:
        L, R = self.lhs(df), self.rhs(df)
        return (R - L) if self.op in ("<","<=","≤") else (L - R)

    def touch_count(self, df: pd.DataFrame) -> int:
        return int((self.slack(df) == 0).sum())

    def __eq__(self, other):
        return (
            isinstance(other, Inequality)
            and self.lhs == other.lhs
            and self.op  == other.op
            and self.rhs == other.rhs
        )

    def __hash__(self):
        return hash((self.lhs, self.op, self.rhs))


# ──────── Conjecture ─────────
@dataclass(frozen=True)
class Conjecture:
    hypothesis: Predicate
    conclusion: Predicate

    def __repr__(self):
        return f"({self.hypothesis.name}) → ({self.conclusion.name})"

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return (~self.hypothesis(df)) | self.conclusion(df)

    def is_true(self, df: pd.DataFrame) -> bool:
        return bool(self.evaluate(df).all())

    def accuracy(self, df: pd.DataFrame) -> float:
        return float(self.evaluate(df).mean())

    def counterexamples(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[~self.evaluate(df)]

    def __eq__(self, other):
        return (
            isinstance(other, Conjecture)
            and self.hypothesis == other.hypothesis
            and self.conclusion == other.conclusion
        )

    def __hash__(self):
        return hash((self.hypothesis, self.conclusion))

# expose public API
__all__ = ['Property', 'Predicate', 'Inequality', 'Conjecture']
