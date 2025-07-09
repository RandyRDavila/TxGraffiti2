# txgraffiti.logic.conjecture_logic.py
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
import functools

# ───────── Property ─────────

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
        if op_symbol == "-" and b.name == "0":
            return a
        if op_symbol == "*":
            if b.name == "1": return a
            if a.name == "1": return b
            if b.name == "0" or a.name == "0":
                return Property("0", lambda df: pd.Series(0, index=df.index))
        if op_symbol == "/" and b.name == "1":
            return a
        if op_symbol == "**":
            if b.name == "1": return a
            if b.name == "0":
                return Property("1", lambda df: pd.Series(1, index=df.index))

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
    def __lt__(self,  o): return Inequality(self, "<",  self._lift(o))
    def __le__(self,  o): return Inequality(self, "<=", self._lift(o))
    def __gt__(self,  o): return Inequality(self, ">",  self._lift(o))
    def __ge__(self,  o): return Inequality(self, ">=", self._lift(o))
    def __eq__(self,  o): return Inequality(self, "==", self._lift(o))
    def __ne__(self,  o): return Inequality(self, "!=", self._lift(o))


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


# ───────── Predicate ─────────

@dataclass(frozen=True)
class Predicate:
    """
    A boolean-valued expression on a DataFrame.

    Predicates support logical operations including AND (`&`), OR (`|`),
    XOR (`^`), NOT (`~`), and implication via `.implies()` or `>>`.

    Parameters
    ----------
    name : str
        The symbolic name of the predicate.
    func : Callable[[pd.DataFrame], pd.Series]
        A function that evaluates to a boolean Series row-wise.

    Attributes
    ----------
    _and_terms : list[Predicate], optional
        Flattened AND operands, used internally.
    _or_terms : list[Predicate], optional
        Flattened OR operands, used internally.
    _neg_operand : Predicate, optional
        The negated operand, if this predicate is a negation.

    Examples
    --------
    >>> even = Predicate("even", lambda df: df["n"] % 2 == 0)
    >>> gt_5 = Predicate(">5", lambda df: df["n"] > 5)
    >>> even & gt_5
    <Predicate (even) ∧ (>5)>
    """
    name: str
    func: Callable[[pd.DataFrame], pd.Series]

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.func(df)

    def __and__(self, other: "Predicate") -> "Predicate":
        # Complement rule:  A ∧ ¬A → False
        if getattr(other, "_neg_operand", None) is self or \
           getattr(self,  "_neg_operand", None) is other:
            return FALSE
        # Absorption:  A ∧ (A ∨ B) → A
        # If 'other' is an OR-expression whose terms include self, return self.
        if hasattr(other, "_or_terms") and self in other._or_terms:
            return self
        # Similarly if 'self' is an OR-expression containing other:
        if hasattr(self,  "_or_terms") and other in self._or_terms:
            return other
        # Identity with constants
        if other is TRUE:
            return self
        if self is TRUE:
            return other
        if other is FALSE or self is FALSE:
            return FALSE

        # Idempotence
        if self == other:
            return self

        # Flatten nested AND
        left_terms  = getattr(self,  "_and_terms", [self])
        right_terms = getattr(other, "_and_terms", [other])
        terms: list[Predicate] = []
        for t in (*left_terms, *right_terms):
            if t not in terms:
                terms.append(t)

        # If only one term remains, return it
        if len(terms) == 1:
            return terms[0]

        name = " ∧ ".join(f"({t.name})" for t in terms)
        func = lambda df, terms=terms: functools.reduce(
            lambda a, b: a & b, (t(df) for t in terms)
        )
        p = Predicate(name, func)
        object.__setattr__(p, "_and_terms", terms)
        return p

    def __or__(self, other: "Predicate") -> "Predicate":
        # Complement rule:  A ∨ ¬A → True
        if getattr(other, "_neg_operand", None) is self or \
           getattr(self,  "_neg_operand", None) is other:
            return TRUE
        # Absorption:  A ∨ (A ∧ B) → A
        if hasattr(other, "_and_terms") and self in other._and_terms:
            return self
        if hasattr(self,  "_and_terms") and other in self._and_terms:
            return other
        # Identity with constants
        if other is FALSE:
            return self
        if self is FALSE:
            return other
        if other is TRUE or self is TRUE:
            return TRUE

        # Idempotence
        if self == other:
            return self

        # Flatten nested OR
        left_terms  = getattr(self,  "_or_terms", [self])
        right_terms = getattr(other, "_or_terms", [other])
        terms: list[Predicate] = []
        for t in (*left_terms, *right_terms):
            if t not in terms:
                terms.append(t)

        # If only one term remains, return it
        if len(terms) == 1:
            return terms[0]

        name = " ∨ ".join(f"({t.name})" for t in terms)
        func = lambda df, terms=terms: functools.reduce(
            lambda a, b: a | b, (t(df) for t in terms)
        )
        p = Predicate(name, func)
        object.__setattr__(p, "_or_terms", terms)
        return p

    def __xor__(self, other: "Predicate") -> "Predicate":
        """
        Logical XOR with:
          P ⊕ P     → False
          P ⊕ ¬P    → True
          P ⊕ False → P
          False ⊕ P → P
          P ⊕ True  → ¬P
          True ⊕ P  → ¬P
        """
        # Complement rule: P ⊕ ¬P → True, and ¬P ⊕ P → True
        if getattr(other, "_neg_operand", None) is self or \
           getattr(self,  "_neg_operand", None) is other:
            return TRUE

        # Same‐operand → False
        if self == other:
            return FALSE

        # XOR‐identity:  P ⊕ False → P; False ⊕ P → P
        if other is FALSE:
            return self
        if self  is FALSE:
            return other

        # XOR‐with‐True:  P ⊕ True → ¬P; True ⊕ P → ¬P
        if other is TRUE:
            return ~self
        if self is TRUE:
            return ~other

        # Otherwise build a new XOR predicate
        return Predicate(
            name=f"({self.name}) ⊕ ({other.name})",
            func=lambda df, a=self, b=other: a(df) ^ b(df)
        )

    # allow scalar on left (though not needed for Predicate–Predicate):
    __rxor__ = __xor__

    def __invert__(self) -> "Predicate":
        # Double‐negation
        orig = getattr(self, "_neg_operand", None)
        if orig is not None:
            return orig

        # Negation of constants
        if self is TRUE:
            return FALSE
        if self is FALSE:
            return TRUE

        # Build ¬(self)
        neg = Predicate(
            name=f"¬({self.name})",
            func=lambda df, p=self: ~p(df)
        )
        object.__setattr__(neg, "_neg_operand", self)
        return neg

    def implies(self, other: "Predicate", *, as_conjecture: bool = False) -> "Predicate":
        """
        Logical implication: self → other.

        Parameters
        ----------
        other : Predicate
            The consequence.
        as_conjecture : bool, optional
            If True, returns a `Conjecture`. If False, returns a `Predicate`
            equivalent to ¬self ∨ other.

        Returns
        -------
        Predicate or Conjecture
            The implication formula.

        Examples
        --------
        >>> P.implies(Q)                # returns Predicate
        >>> P.implies(Q, as_conjecture=True)  # returns Conjecture
        """
        if as_conjecture:
            return Conjecture(self, other)

        name = f"({self.name} → {other.name})"
        return Predicate(name, lambda df, a=self, b=other: (~a(df)) | b(df))

    # -----------------------------------------------------------------------
    #  Syntactic sugar: P >> Q  → Conjecture(P, Q)
    # -----------------------------------------------------------------------
    def __rshift__(self, other: "Predicate") -> "Conjecture":
        """
        Use the bit-shift operator ‘>>’ as a readable implication that
        *always* returns a Conjecture:

            conj = hypothesis >> conclusion
        """
        if not isinstance(other, Predicate):
            raise TypeError("Right operand of >> must be a Predicate")
        return Conjecture(self, other)

    def __repr__(self):
        return f"<Predicate {self.name}>"

    def __eq__(self, other):
        return isinstance(other, Predicate) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


# Module‐level constants for logical identities
TRUE  = Predicate("True",  lambda df: pd.Series(True,  index=df.index))
FALSE = Predicate("False", lambda df: pd.Series(False, index=df.index))


# ───────── Inequality ─────────

class Inequality(Predicate):
    """
    A comparison between two `Property` expressions.

    Constructed automatically when using operators like `p1 < p2`.

    Parameters
    ----------
    lhs : Property
        The left-hand side of the inequality.
    op : str
        The comparison operator. One of {"<", "<=", ">", ">=", "==", "!="}.
    rhs : Property
        The right-hand side of the inequality.

    Attributes
    ----------
    lhs : Property
        Left operand.
    rhs : Property
        Right operand.
    op : str
        The operator used.

    Examples
    --------
    >>> p1 < p2
    <Inequality (p1 < p2)>
    """
    def __init__(self, lhs: Property, op: str, rhs: Property):
        name = f"{lhs.name} {op} {rhs.name}"
        def func(df: pd.DataFrame) -> pd.Series:
            L, R = lhs(df), rhs(df)
            return {
                "<":  L <  R, "<=": L <= R,
                ">":  L >  R, ">=": L >= R,
                "==": L == R, "!=": L != R,
            }[op]
        super().__init__(name, func)
        object.__setattr__(self, 'lhs', lhs)
        object.__setattr__(self, 'rhs', rhs)
        object.__setattr__(self, 'op',  op)

    def slack(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the slack of the inequality on a DataFrame.

        Slack is defined as:
        - rhs - lhs for "<", "<=", "≤"
        - lhs - rhs for ">", ">="

        Parameters
        ----------
        df : pd.DataFrame
            The data on which to evaluate the slack.

        Returns
        -------
        pd.Series
            The row-wise slack values.
        """
        L, R = self.lhs(df), self.rhs(df)
        return (R - L) if self.op in ("<","<=","≤") else (L - R)

    def touch_count(self, df: pd.DataFrame) -> int:
        """
        Count how many rows satisfy the inequality with equality.

        Parameters
        ----------
        df : pd.DataFrame
            The data to evaluate.

        Returns
        -------
        int
            The number of rows where slack is exactly zero.
        """
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


# ───────── Conjecture ─────────

class Conjecture(Predicate):
    """
    A logical implication between two predicates.

    Represents a rule of the form: (hypothesis) → (conclusion).

    Parameters
    ----------
    hypothesis : Predicate
        The antecedent of the implication.
    conclusion : Predicate
        The consequent of the implication.

    Examples
    --------
    >>> conj = P >> Q
    >>> conj.is_true(df)
    True
    """
    def __init__(self, hypothesis: Predicate, conclusion: Predicate):
        name = f"({hypothesis.name}) → ({conclusion.name})"
        func = lambda df: (~hypothesis(df)) | conclusion(df)
        super().__init__(name, func)
        object.__setattr__(self, 'hypothesis',  hypothesis)
        object.__setattr__(self, 'conclusion',  conclusion)

    def is_true(self, df: pd.DataFrame) -> bool:
        """
        Check if the conjecture holds on all rows of the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The data to evaluate.

        Returns
        -------
        bool
            True if all rows satisfy the implication.
        """
        return bool(self(df).all())

    def accuracy(self, df: pd.DataFrame) -> float:
        """
        Compute the conditional accuracy of the conjecture.

        This is defined as the fraction of rows satisfying the conclusion
        among those satisfying the hypothesis.

        Parameters
        ----------
        df : pd.DataFrame
            The data to evaluate.

        Returns
        -------
        float
            The accuracy of the conjecture.
        """
        hyp = self.hypothesis(df)
        if not hyp.any():
            return 0.0
        return float(self(df)[hyp].mean())

    def counterexamples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the rows that violate the conjecture.

        Parameters
        ----------
        df : pd.DataFrame
            The data to search.

        Returns
        -------
        pd.DataFrame
            Subset of rows where the implication fails.
        """
        return df[~self(df)]

    def __repr__(self):
        return f"<Conj {self.name}>"
