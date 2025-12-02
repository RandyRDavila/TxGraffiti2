from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple, Union, overload, Literal, Callable
import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.processing.post.morgan import morgan_filter


# ───────────────────────────── Internal caching ───────────────────────────── #

class _EvalCaches:
    """
    Per-call caches to avoid repeated work across many conjectures on the same df.

    Parameters
    ----------
    expr_key_mode : {"id", "repr"}, default "id"
        How to key expression evaluations:
        - "id": cache by object identity (fast, safest; default).
        - "repr": cache by string representation (useful if generators
          create distinct but structurally-equal Expr objects).
          Only use if Expr.pretty()/__repr__ is stable and canonical.
    """
    __slots__ = ("mask_cache", "expr_cache", "expr_key_mode")

    def __init__(self, expr_key_mode: Literal["id", "repr"] = "id"):
        self.mask_cache: Dict[Predicate, np.ndarray] = {}
        # Key: Expr or str  -> np.ndarray (float)
        self.expr_cache: Dict[Union[object, str], np.ndarray] = {}
        self.expr_key_mode = expr_key_mode

    def key_expr(self, expr: object) -> Union[object, str]:
        if self.expr_key_mode == "repr":
            # Prefer pretty() if present; fall back to repr()
            if hasattr(expr, "pretty"):
                try:
                    return str(expr.pretty())  # type: ignore[attr-defined]
                except Exception:
                    pass
            return repr(expr)
        return expr  # identity-based


# ───────────────────────────── Touch counting ────────────────────────────── #

def _get_mask(H: Optional[Predicate], df: pd.DataFrame, caches: _EvalCaches) -> np.ndarray:
    if H is None:
        return np.ones(len(df), dtype=bool)
    m = caches.mask_cache.get(H)
    if m is None:
        # tolerant boolean coercion
        m = H.mask(df).astype(bool).to_numpy()
        caches.mask_cache[H] = m
    return m


def _eval_expr(expr, df: pd.DataFrame, caches: _EvalCaches) -> np.ndarray:
    k = caches.key_expr(expr)
    arr = caches.expr_cache.get(k)
    if arr is None:
        # normalize to float ndarray aligned with df.index
        s = expr.eval(df)  # expected to be a Series
        arr = s.to_numpy(dtype=float, copy=False)
        caches.expr_cache[k] = arr
    return arr


def touch_count(
    c: Conjecture,
    df: pd.DataFrame,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    caches: Optional[_EvalCaches] = None,
) -> int:
    """
    Count the number of 'touch' rows (LHS == RHS within tolerances) for a conjecture.

    Uses optional caches to avoid re-evaluating the same predicate masks and
    expressions across multiple conjectures on the same DataFrame.

    Parameters
    ----------
    c : Conjecture
        Conjecture with attributes ``relation.left``, ``relation.right``, and optional
        ``condition`` (Predicate with ``mask(df)``).
    df : pandas.DataFrame
        Data on which to evaluate the conjecture.
    rtol, atol : float, optional
        Tolerances forwarded to ``np.isclose``. Defaults are 1e-8.
    caches : _EvalCaches or None, optional
        Per-call caches. If None, a throwaway cache is created.

    Returns
    -------
    int
        Count of rows where LHS and RHS are numerically equal (after excluding
        non-finite values and slicing by the conjecture's predicate).
    """
    # Fast path if Conjecture offers its own
    try:
        return int(c.is_touch(df))  # type: ignore[attr-defined]
    except Exception:
        pass

    caches = caches or _EvalCaches()
    try:
        lhs = _eval_expr(c.relation.left, df, caches)
        rhs = _eval_expr(c.relation.right, df, caches)
        mask = _get_mask(getattr(c, "condition", None), df, caches)

        valid = mask & np.isfinite(lhs) & np.isfinite(rhs)
        if not np.any(valid):
            return 0

        eq = np.isclose(lhs[valid], rhs[valid], rtol=rtol, atol=atol)
        return int(eq.sum())
    except Exception:
        return 0


# ───────────────────────────── Ranking & filtering ────────────────────────── #

@overload
def rank_and_filter(
    conjs: List[Conjecture],
    df: pd.DataFrame,
    *,
    min_touch: Optional[int] = ...,
    rtol: float = ...,
    atol: float = ...,
    expr_key_mode: Literal["id", "repr"] = ...,
) -> List[Conjecture]: ...
# (The overload is just for editor help; implementation below.)

def rank_and_filter(
    conjs: List[Conjecture],
    df: pd.DataFrame,
    *,
    min_touch: Optional[int] = None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    expr_key_mode: Literal["id", "repr"] = "id",
) -> List[Conjecture]:
    """
    Rank conjectures by touch count (descending), drop those below threshold,
    then apply Morgan post-processing.

    This is the canonical entry point used by the workbench engine after each
    generation pass. It is **pure** with respect to inputs: caches exist only
    for the duration of the call.

    Parameters
    ----------
    conjs : list of Conjecture
        Candidate conjectures to score.
    df : pandas.DataFrame
        Data on which to evaluate.
    min_touch : int or None, optional
        Minimum touch count to keep a conjecture. Defaults to 0 if None.
    rtol, atol : float, optional
        Tolerances forwarded to ``np.isclose`` during touch counting.
    expr_key_mode : {"id", "repr"}, default "id"
        Keying mode for the expression evaluation cache:
        - "id": Use object identity. Fast and safest (default).
        - "repr": Use string representation (e.g., ``expr.pretty()`` or ``repr(expr)``).
          Useful if structurally-equal expressions are created as distinct objects
          by your generator. Only enable if your expression stringification is
          stable and canonical across the run.

    Returns
    -------
    list of Conjecture
        Morgan-filtered conjectures, in the order produced by your Morgan pipeline
        (``morgan_filter(df, kept).kept``).

    Notes
    -----
    - Non-finite rows (NaN/Inf) are excluded from equality checks.
    - Predicate masks are cached once per Predicate object; expression evaluations
      are cached per expression key (see ``expr_key_mode``).
    - Argument order is ``(conjs, df, ...)``. Passing ``(df, conjs, ...)`` will
      crash inside Morgan or the touch logic because non-Conjecture objects won’t
      have the required attributes.
    """
    threshold = 0 if min_touch is None else int(min_touch)
    caches = _EvalCaches(expr_key_mode=expr_key_mode)

    # Score by (cached) touch count
    scored: List[Tuple[Conjecture, int]] = []
    for c in conjs:
        sc = touch_count(c, df, rtol=rtol, atol=atol, caches=caches)
        scored.append((c, sc))

    scored.sort(key=lambda t: t[1], reverse=True)

    # Threshold
    kept = [c for (c, sc) in scored if sc >= threshold]

    # Morgan post-processing (dedupe/canonicalization/prioritization)
    mf = morgan_filter(df, kept)
    return list(mf.kept)
