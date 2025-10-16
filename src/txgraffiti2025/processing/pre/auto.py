"""
Auto-base wrappers for conjectures and relation-logic.

If `condition` is None, these classes detect a natural base hypothesis from
the DataFrame at evaluation time (using `detect_base_hypothesis`).

Usage
-----
>>> from txgraffiti2025.forms.generic_conjecture import Le
>>> from txgraffiti2025.processing.pre.auto import AutoConjecture
>>> C = AutoConjecture(Le("alpha", "mu"))     # condition=None -> auto-detected
>>> applicable, holds, failures = C.check(df)  # uses ((connected) ∧ ...) or TRUE
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.forms.implication import Implication as _Imp, Equivalence as _Eqv
from txgraffiti2025.processing.pre.hypotheses import detect_base_hypothesis

__all__ = [
    'AutoConjecture',
    'AutoImplication',
    'AutoEquivalence',
]

@dataclass
class AutoConjecture(Conjecture):
    """
    Conjecture with automatic base hypothesis.

    If `condition` is None, `check(df)` will call `detect_base_hypothesis(df)`
    and use the result as the class condition; otherwise it respects the
    provided `condition`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Le
    >>> from txgraffiti2025.processing.pre.auto import AutoConjecture
    >>> df = pd.DataFrame({"alpha":[2,3], "mu":[2,5], "connected":[True,True]})
    >>> C = AutoConjecture(Le("alpha","mu"))  # auto condition -> (connected)
    >>> _, holds, _ = C.check(df)
    >>> holds.tolist()
    [True, True]
    """
    def check(self, df: pd.DataFrame):
        if self.condition is None:
            cond = detect_base_hypothesis(df)
            # Temporarily evaluate with detected base (don’t mutate self)
            original = self.condition
            try:
                object.__setattr__(self, "condition", cond)
                return super().check(df)
            finally:
                object.__setattr__(self, "condition", original)
        return super().check(df)


@dataclass
class AutoImplication(_Imp):
    """
    Relation implication with automatic base hypothesis.

    If `condition` is None, `check(df)` uses `detect_base_hypothesis(df)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Le, Ge
    >>> from txgraffiti2025.processing.pre.auto import AutoImplication
    >>> df = pd.DataFrame({"a":[1,3], "b":[2,1], "connected":[True,True]})
    >>> impl = AutoImplication(Le("a","b"), Ge("b","a"))
    >>> _, holds, _ = impl.check(df)  # auto uses (connected)
    >>> holds.tolist()
    [True, False]
    """
    def check(self, df: pd.DataFrame):
        if self.condition is None:
            cond = detect_base_hypothesis(df)
            original = self.condition
            try:
                object.__setattr__(self, "condition", cond)
                return super().check(df)
            finally:
                object.__setattr__(self, "condition", original)
        return super().check(df)


@dataclass
class AutoEquivalence(_Eqv):
    """
    Relation equivalence with automatic base hypothesis.

    If `condition` is None, `check(df)` uses `detect_base_hypothesis(df)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Le, Ge
    >>> from txgraffiti2025.processing.pre.auto import AutoEquivalence
    >>> df = pd.DataFrame({"a":[1,2], "b":[1,3], "connected":[True,True]})
    >>> eqv = AutoEquivalence(Le("a","b"), Ge("b","a"))
    >>> _, holds, _ = eqv.check(df)  # auto uses (connected)
    >>> holds.tolist()
    [True, True]
    """
    def check(self, df: pd.DataFrame):
        if self.condition is None:
            cond = detect_base_hypothesis(df)
            original = self.condition
            try:
                object.__setattr__(self, "condition", cond)
                return super().check(df)
            finally:
                object.__setattr__(self, "condition", original)
        return super().check(df)
