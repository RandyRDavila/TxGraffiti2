# src/txgraffiti2025/forms/substructure.py
from __future__ import annotations

"""
R₅-type conjectures: structural / sub-object existence forms.

This module lets you express qualitative statements like:
  • “Object x has a sub-object with property P.”
  • “No sub-object of x satisfies property Q.”  (via logical negation)

Use cases span graphs, polytopes, groups, sequences—anything where each row
contains a structured object in a column (default: "object").

Design
------
- SubstructurePredicate(fn, object_col="object", on_error="raise"|"false")
  evaluates fn over each row's object, producing a boolean mask.

- fn(obj) may also be row-aware with accepts_row=True: fn(obj, row) -> bool

- on_error="false" converts exceptions to False (robust for mixed/dirty data).

- Repr is compact and human-friendly, e.g.:
      (∃ substructure: has_even)   or    (∃ substructure: object test)
  and negation (~) composes via the standard Predicate combinators.

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.substructure import SubstructurePredicate
>>> class Dummy:
...     def __init__(self, values): self.values = values
...     def has_even(self): return any(v % 2 == 0 for v in self.values)
...
>>> df = pd.DataFrame({"object": [Dummy([1,3,5]), Dummy([1,2,3])]})
>>> C_even = SubstructurePredicate(lambda obj: obj.has_even(), name="has_even", on_error="false")
>>> C_even.mask(df).tolist()
[False, True]
>>> (~C_even).mask(df).tolist()   # non-existence
[True, False]
"""

from dataclasses import dataclass
from typing import Callable, Any, Literal, Optional

import pandas as pd

from .predicates import Predicate

__all__ = [
    "SubstructurePredicate",
    "exists_sub",
    "not_exists_sub",
]

OnError = Literal["raise", "false"]


@dataclass
class SubstructurePredicate(Predicate):
    """
    Boolean predicate based on a per-row substructure test.

    Parameters
    ----------
    fn : Callable[[Any], bool] or Callable[[Any, pd.Series], bool]
        Function that checks the property on the object in `object_col`.
        If `accepts_row=True`, the function must accept two arguments:
        `fn(obj, row)`.
    object_col : str, default "object"
        Name of the DataFrame column holding the structured object.
    on_error : {"raise", "false"}, default "raise"
        - "raise": propagate exceptions from `fn`.
        - "false": treat exceptions as False.
    accepts_row : bool, default False
        If True, call `fn(obj, row)` instead of `fn(obj)`.
    name : Optional[str], default None
        Label used in pretty printing / repr. If None, use fn.__name__ if
        available, else "object test".

    Notes
    -----
    - Implements **R₅ structural/existence** style statements.
    - Combine with logical negation (~) for non-existence forms.
    - Safe handling of NA/missing objects: underlying call receives the
      (possibly None) object value; exceptions can be coerced to False via
      on_error="false".
    """
    fn: Callable[..., bool]
    object_col: str = "object"
    on_error: OnError = "raise"
    accepts_row: bool = False
    name: Optional[str] = None

    # Predicate.name (from base) is shown by __repr__; we keep a friendly one
    # but compute it dynamically to reflect fn/label.
    @property
    def _label(self) -> str:
        if isinstance(self.name, str) and self.name.strip():
            return self.name.strip()
        return getattr(self.fn, "__name__", "object test")

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate the existence test row-wise.

        Returns
        -------
        pd.Series[bool]
            Boolean mask aligned to df.index.
        """
        if self.object_col not in df.columns:
            raise KeyError(f"Column '{self.object_col}' not found in DataFrame.")

        def _safe_call(obj: Any, row: pd.Series) -> bool:
            try:
                if self.accepts_row:
                    return bool(self.fn(obj, row))
                return bool(self.fn(obj))
            except Exception:
                if self.on_error == "false":
                    return False
                raise

        out = df.apply(lambda row: _safe_call(row[self.object_col], row), axis=1)
        return out.reindex(df.index, fill_value=False).astype(bool)

    # Nice, mathy display
    def __repr__(self) -> str:
        return f"(∃ substructure: {self._label})"

    # Pretty identical to __repr__ (kept for symmetry with other modules)
    def pretty(self, *, unicode_ops: bool = True) -> str:
        arrow = "∃" if unicode_ops else "exists"
        return f"({arrow} substructure: {self._label})"


# ---------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------

def exists_sub(
    fn: Callable[..., bool],
    *,
    object_col: str = "object",
    on_error: OnError = "raise",
    accepts_row: bool = False,
    name: Optional[str] = None,
) -> SubstructurePredicate:
    """
    Shorthand to build a SubstructurePredicate (“∃ substructure ...”).

    Examples
    --------
    >>> exists_sub(lambda obj: hasattr(obj, "edges"), name="has_edges")
    """
    return SubstructurePredicate(
        fn=fn,
        object_col=object_col,
        on_error=on_error,
        accepts_row=accepts_row,
        name=name,
    )


def not_exists_sub(
    fn: Callable[..., bool],
    *,
    object_col: str = "object",
    on_error: OnError = "raise",
    accepts_row: bool = False,
    name: Optional[str] = None,
) -> Predicate:
    """
    Shorthand for the non-existence form:  ~exists_sub(...).
    """
    return ~exists_sub(
        fn,
        object_col=object_col,
        on_error=on_error,
        accepts_row=accepts_row,
        name=name,
    )
