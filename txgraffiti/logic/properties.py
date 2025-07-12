"""
Logical components for symbolic reasoning over dataframes.

This module defines core classes used for automated conjecturing,
including:

- `Property`: symbolic numeric expressions over DataFrame columns.
- `Predicate`: boolean-valued expressions that support logical algebra.
- `Inequality`: a comparison between `Property` objects.
- `Conjecture`: logical implications between `Predicate` expressions.

All expressions can be evaluated on a pandas DataFrame row-wise.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Callable, Union
from numbers import Number

# ───────── Property ─────────

__all__ = [
    'Property',
    'Constant',
]

@dataclass(frozen=True)
class Property:
    """
    A symbolic property representing a real-valued function on a pandas DataFrame.

    Properties can be combined with arithmetic operators (`+`, `-`, `*`, `/`, etc.)
    and compared using inequality operators (`<`, `<=`, `==`, `!=`, `>=`, `>`).

    Parameters
    ----------
    name : str
        A symbolic name for the property.
    func : Callable[[pd.DataFrame], pd.Series]
        A function that computes the property row-wise from a DataFrame.

    Examples
    --------
    >>> deg = Property("deg", lambda df: df["degree"])
    >>> 2 * deg + 3
    <Property (2 * deg) + 3>
    """
    name: str
    func: Callable[[pd.DataFrame], pd.Series]

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.func(df)

    def __repr__(self):
        return f"<Property {self.name}>"

    def _lift(self, other: Union['Property', Number]) -> 'Property':
        if isinstance(other, Property):
            return other
        if isinstance(other, Number):
            return Property(str(other),
                            lambda df, v=other: pd.Series(v, index=df.index))
        raise TypeError(f"Cannot lift {other!r} into Property")

    def _binop(self, other, op_symbol: str, op_func):
        other = self._lift(other)
        a, b = self, other

        # ── identity eliminations ──
        if op_symbol == "+":
            if b.name == "0": return a
            if a.name == "0": return b
        if op_symbol == "-":
            if b.name == "0":
                return a
            if b.name == a.name:
                return Constant(0)
        if op_symbol == "*":
            if b.name == "1": return a
            if a.name == "1": return b
            if b.name == "0" or a.name == "0":
                return Constant(0)
        if op_symbol == "/":
            if b.name == "1":
                return a
            if b.name == a.name:
                return Constant(1)
        if op_symbol == "**":
            if b.name == "1": return a
            if b.name == "0":
                return Constant(1)

        # ── commutative normalization ──
        if op_symbol in ("+", "*"):
            # force a.name <= b.name so names always sort in the same order
            if b.name < a.name:
                a, b = b, a

        # ── build the canonical name ──
        name = f"({a.name} {op_symbol} {b.name})"
        return Property(name, lambda df: op_func(a(df), b(df)))

    # arithmetic
    __add__      = lambda self, o: self._binop(o, "+", pd.Series.add)
    __sub__      = lambda self, o: self._binop(o, "-", pd.Series.sub)
    __mul__      = lambda self, o: self._binop(o, "*", pd.Series.mul)
    __truediv__  = lambda self, o: self._binop(o, "/", pd.Series.div)
    __pow__      = lambda self, o: self._binop(o, "**", pd.Series.pow)
    __mod__      = lambda self, o: self._binop(o, "%", pd.Series.mod)

    __radd__     = __add__
    __rsub__     = lambda self, o: self._lift(o)._binop(self, "-", pd.Series.sub)
    __rmul__     = __mul__
    __rtruediv__ = lambda self, o: self._lift(o)._binop(self, "/", pd.Series.div)
    __rpow__     = lambda self, o: self._lift(o)._binop(self, "**", pd.Series.pow)
    __rmod__     = lambda self, o: self._lift(o)._binop(self, "%", pd.Series.mod)

    # comparisons → Inequality
    def __lt__(self,  o):
        from txgraffiti.logic import Inequality
        return Inequality(self, "<",  self._lift(o))
    def __le__(self,  o):
        from txgraffiti.logic import Inequality
        return Inequality(self, "<=", self._lift(o))
    def __gt__(self,  o):
        from txgraffiti.logic import Inequality
        return Inequality(self, ">",  self._lift(o))
    def __ge__(self,  o):
        from txgraffiti.logic import Inequality
        return Inequality(self, ">=", self._lift(o))
    def __eq__(self,  o):
        from txgraffiti.logic import Inequality
        return Inequality(self, "==", self._lift(o))
    def __ne__(self,  o):
        from txgraffiti.logic import Inequality
        return Inequality(self, "!=", self._lift(o))


def Constant(c: Number) -> Property:
    """
    Create a constant-valued Property.

    Parameters
    ----------
    c : Number
        The constant value to use.

    Returns
    -------
    Property
        A Property that returns `c` for every row in the DataFrame.
    """
    return Property(str(c), lambda df, v=c: pd.Series(v, index=df.index))
