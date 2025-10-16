# src/txgraffiti2025/forms/substructure.py

"""
R₅-type conjectures: structural or sub-object existence forms.

These capture qualitative statements about the presence (or absence) of a
substructure satisfying a given property, without assuming any particular
domain such as graphs, polytopes, or groups.

Typical verbal forms:
    “Object x has a sub-object satisfying property P.”
    “No sub-object of x satisfies property Q.”

Such conjectures often appear in topological, combinatorial, or geometric
contexts where “existence of a substructure” replaces explicit numerical
relations.

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.substructure import SubstructurePredicate
>>> class Dummy:
...     def __init__(self, values): self.values = values
...     def has_even(self): return any(v % 2 == 0 for v in self.values)
...
>>> df = pd.DataFrame({"object": [Dummy([1,3,5]), Dummy([1,2,3])]})
>>> # existence: “object has even element”
>>> C_even = SubstructurePredicate(lambda obj: obj.has_even())
>>> C_even.mask(df).tolist()
[False, True]
>>> # non-existence form (~C_even): “object has no even element”
>>> from txgraffiti2025.forms.predicates import NotPred
>>> (~C_even).mask(df).tolist()
[True, False]
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
    Boolean predicate based on an existence test applied to each structured object.

    Parameters
    ----------
    fn : Callable[[Any], bool]
        User-supplied function that receives the row’s structured object
        (graph, polytope, group, integer configuration, etc.) and returns
        ``True`` if the desired substructure property holds.
    object_col : str, default "object"
        Name of the DataFrame column containing the structured object to test.
    on_error : {"raise", "false"}, default "raise"
        How to handle exceptions raised by ``fn(obj)``:
        - ``"raise"`` : propagate the exception.
        - ``"false"`` : treat errors as ``False`` results.
    name : str, default "SubstructurePredicate"
        Display name for introspection or printing.

    Returns
    -------
    SubstructurePredicate
        A :class:`Predicate` whose :meth:`mask` evaluates the user function on
        each row of a DataFrame.

    Notes
    -----
    - Implements the **R₅ structural/existence** conjecture type.
    - Often combined with logical negation (``~C``) to express non-existence.
    - Safe handling of missing or malformed objects via ``on_error="false"``.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.substructure import SubstructurePredicate
    >>> df = pd.DataFrame({"object": [{"edges": [1,2]}, None, {"edges": []}]})
    >>> # Example: “object has nonempty edge set”
    >>> C_has_edges = SubstructurePredicate(lambda obj: bool(obj and obj["edges"]), on_error="false")
    >>> C_has_edges.mask(df).tolist()
    [True, False, False]
    >>> # Logical negation gives the non-existence form
    >>> (~C_has_edges).mask(df).tolist()
    [False, True, True]
    """
    fn: Callable[[Any], bool]
    object_col: str = "object"
    on_error: OnError = "raise"
    name: str = "SubstructurePredicate"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate the existence test row-wise on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a column ``object_col`` with structured
            objects to test.

        Returns
        -------
        pd.Series of bool
            Boolean mask indicating where ``fn(obj)`` returned ``True``.
            Missing objects or errors are handled according to ``on_error``.

        Raises
        ------
        KeyError
            If the specified ``object_col`` is not found in the DataFrame.
        Exception
            If ``fn`` raises and ``on_error="raise"``.

        Examples
        --------
        >>> import pandas as pd
        >>> from txgraffiti2025.forms.substructure import SubstructurePredicate
        >>> df = pd.DataFrame({"object": [[1,2,3], [], [5]]})
        >>> has_elements = SubstructurePredicate(lambda obj: len(obj) > 0)
        >>> has_elements.mask(df).tolist()
        [True, False, True]
        """
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
        return out.astype(bool).reindex(df.index, fill_value=False)

    def __repr__(self) -> str:
        fn = getattr(self.fn, "__name__", "fn")
        return f"SubstructurePredicate(fn={fn}, object_col={self.object_col!r}, on_error={self.on_error!r})"
