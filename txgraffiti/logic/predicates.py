# ───────── Predicate ─────────
from dataclasses import dataclass
from typing import Callable, Union
import pandas as pd
import functools

__all__ = [
    'Predicate',
    'TRUE',
    'FALSE',
]

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
        from txgraffiti.logic import Conjecture
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
        from txgraffiti.logic import Conjecture
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
