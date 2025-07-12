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
from txgraffiti.logic.properties import Property
from txgraffiti.logic.predicates import Predicate

# ───────── Inequality ─────────

__all__ = [
    'Inequality'
]

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
