"""
R₅-type conjectures: structural or sub-object existence forms.

General statements like:
    "Object x has (or does not have) a sub-object satisfying property P"
can be expressed without assuming a specific field (graph, polytope, etc.).

Usage:
    # existence (user fn returns bool)
    C = SubstructurePredicate(lambda obj: obj is not None and obj.has_property_X())

    # non-existence:
    # from txgraffiti2025.forms.predicates import NotPred
    # C_not = ~C
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Literal
import pandas as pd
from .predicates import Predicate

__all__ = [
    "SubstructurePredicate",
]

OnError = Literal["raise", "false"]


@dataclass
class SubstructurePredicate(Predicate):
    """
    Boolean predicate based on an existence test applied to each object.

    fn : Callable[[Any], bool]
        User-supplied function receiving the row's object (graph, polytope, group,
        integer structure, etc.) and returning True/False.

    object_col : str
        Name of the DataFrame column holding the structured object.

    on_error : "raise" | "false"
        What to do if fn(obj) raises: re-raise the exception, or treat as False.
    """
    fn: Callable[[Any], bool]
    object_col: str = "object"
    on_error: OnError = "raise"
    name: str = "SubstructurePredicate"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        if self.object_col not in df.columns:
            raise KeyError(f"Column '{self.object_col}' not found in DataFrame.")

        def _safe_apply(obj: Any) -> bool:
            try:
                return bool(self.fn(obj))
            except Exception:
                if self.on_error == "false":
                    return False
                raise

        out = df[self.object_col].apply(_safe_apply)
        # Ensure boolean Series aligned to df.index
        return out.astype(bool).reindex(df.index, fill_value=False)
