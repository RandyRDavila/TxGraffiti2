# txgraffiti2025/finalization.py
from __future__ import annotations

from typing import Sequence, Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, TRUE


# ───────────────────── basic helpers ───────────────────── #

def print_full_result(result: TxGraffitiResult) -> None:
    print_bank(result.final_bank, k_per_bucket=None)
    if "sophie_conditions" in result.extras:
        print_sophie_conditions(result.extras["sophie_conditions"], top_n=len(result.extras["sophie_conditions"]))


def _as_number(v: Any, default: Any = None) -> Any:
    """
    Best-effort conversion of v into a plain Python number (int/float),
    falling back to `default` if anything goes wrong.
    """
    if v is None:
        return default
    try:
        v = v() if callable(v) else v
        # Handle numpy scalars
        if hasattr(v, "item") and callable(getattr(v, "item")):
            v = v.item()
        if isinstance(v, (int, bool)):
            return int(v)
        return float(v)
    except Exception:
        return default


def _mask_for(df: pd.DataFrame, H: Any) -> np.ndarray:
    """
    Convert a hypothesis/predicate H into a boolean numpy mask over df.

    Assumes:
      * H is None or TRUE  → full mask of True
      * H has a .mask(df) method   OR
      * H is directly callable on df (H(df)).
    """
    if H is None or H is TRUE:
        return np.ones(len(df), dtype=bool)
    if hasattr(H, "mask"):
        s = H.mask(df)
    else:
        s = H(df)
    return np.asarray(s, dtype=bool)


def _compute_touch_support_batch(
    df: pd.DataFrame,
    conjs: Sequence[Conjecture],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> Dict[int, Tuple[int, int, float]]:
    """
    Batch compute (touch_count, support_n, touch_rate) for all conjectures
    in `conjs`, evaluated on the rows of `df`.

    We group by (H, L, R) so each triplet is evaluated once, then share
    the statistics across all conjectures in that group.

    Returns
    -------
    stats : dict
        Mapping index -> (touch_count, support_n, touch_rate).
    """
    if not conjs:
        return {}

    # local key helpers
    def _pred_key(p: Any) -> str:
        if p is None or p is TRUE:
            return "TRUE"
        n = getattr(p, "name", None)
        return f"name:{n}" if n else f"repr:{repr(p)}"

    def _expr_key(e: Any) -> str:
        return repr(e)

    # Group conjectures by (predicate, left_expr, right_expr)
    groups: Dict[Tuple[str, str, str], List[int]] = {}
    for i, c in enumerate(conjs):
        groups.setdefault(
            (_pred_key(c.condition),
             _expr_key(c.relation.left),
             _expr_key(c.relation.right)),
            [],
        ).append(i)

    # Caches for masks/arrays
    mcache: Dict[str, np.ndarray] = {}
    arrcache: Dict[str, np.ndarray] = {}

    def _mask(pred: Any) -> np.ndarray:
        k = _pred_key(pred)
        if k in mcache:
            return mcache[k]
        m = _mask_for(df, pred)
        mcache[k] = m
        return m

    def _arr(expr: Any) -> np.ndarray:
        k = _expr_key(expr)
        a = arrcache.get(k)
        if a is None:
            s = expr.eval(df)
            if hasattr(s, "to_numpy"):
                a = s.to_numpy(dtype=float, copy=False)
            else:
                a = np.asarray(s, dtype=float)
                if a.ndim == 0:
                    # broadcast scalar to full column
                    a = np.full(len(df), float(a), dtype=float)
            arrcache[k] = a
        return a

    out: Dict[int, Tuple[int, int, float]] = {}
    for (_, _, _), idxs in groups.items():
        rep = conjs[idxs[0]]

        Hm = _mask(rep.condition)
        if not np.any(Hm):
            for j in idxs:
                out[j] = (0, 0, 0.0)
            continue

        L = _arr(rep.relation.left)
        R = _arr(rep.relation.right)

        ok = np.isfinite(L) & np.isfinite(R) & Hm
        sup = int(ok.sum())
        if sup == 0:
            tc, rate = 0, 0.0
        else:
            eq = np.isclose(L[ok], R[ok], rtol=rtol, atol=atol)
            tc = int(eq.sum())
            rate = float(tc / sup)

        for j in idxs:
            out[j] = (tc, sup, rate)

    return out


def _dedup_by_string(conjs: Sequence[Conjecture]) -> List[Conjecture]:
    """
    Remove duplicates based on the string representation of the conjecture.
    """
    seen: set[str] = set()
    out: List[Conjecture] = []
    for c in conjs:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        out.append(c)
    return out


def _annotate_touch_support(df: pd.DataFrame, conjs: List[Conjecture]) -> None:
    """
    Attach touch_count, support_n, and touch_rate attributes to each conjecture
    in-place, using a single batch evaluation over df.
    """
    stats = _compute_touch_support_batch(df, conjs)

    for i, c in enumerate(conjs):
        tc, sup, rate = stats.get(i, (0, 0, 0.0))
        # Preserve existing values if present; otherwise use computed ones.
        setattr(
            c,
            "touch_count",
            int(_as_number(getattr(c, "touch_count", tc), default=tc) or tc),
        )
        setattr(
            c,
            "support_n",
            int(_as_number(getattr(c, "support_n", sup), default=sup) or sup),
        )
        setattr(
            c,
            "touch_rate",
            float(_as_number(getattr(c, "touch_rate", rate), default=rate) or rate),
        )


def _sort_by_touch_support(conjs: List[Conjecture]) -> List[Conjecture]:
    """
    Sort conjectures in descending order of (touch_count, support_n).
    """
    def key(c: Conjecture) -> Tuple[int, int]:
        return (
            int(getattr(c, "touch_count", 0) or 0),
            int(getattr(c, "support_n", 0) or 0),
        )

    return sorted(conjs, key=key, reverse=True)


def _pretty_safe(c: Conjecture) -> str:
    """
    Robust pretty printer: fall back to str(c) if .pretty() fails.
    """
    try:
        return c.pretty(show_tol=False)
    except Exception:
        return str(c)


# ───────────────────── public helpers ───────────────────── #

def print_bank(
    bank: Dict[str, List[Conjecture]],
    *,
    k_per_bucket: Optional[int] = 10,
    title: str = "FULL CONJECTURE LIST",
) -> None:
    """
    Pretty-print a conjecture bank of the form:
        {"lowers": [...], "uppers": [...], "equals": [...]}

    Parameters
    ----------
    bank : dict
        "lowers", "uppers", "equals" -> lists of Conjecture.
    k_per_bucket : int or None, default=10
        If an integer, print at most this many conjectures per bucket.
        If None, print *all* conjectures in each bucket.
    title : str
        Section title to print.
    """
    def section(title: str) -> None:
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80 + "\n")

    section(title)

    for name in ("lowers", "uppers", "equals"):
        lst = bank.get(name, [])
        print(f"[{name.upper()}] total={len(lst)}\n")

        if k_per_bucket is None:
            # Print all
            to_show = lst
        else:
            to_show = lst[:k_per_bucket]

        for c in to_show:
            print("•", _pretty_safe(c))
            print(
                f"    touches={getattr(c, 'touch_count', '?')}, "
                f"support={getattr(c, 'support_n', '?')}"
            )
        print()


def finalize_conjecture_bank(
    df: pd.DataFrame,
    all_lowers: List[Conjecture],
    all_uppers: List[Conjecture],
    all_equals: List[Conjecture],
    *,
    top_k_per_bucket: Optional[int] = 100,
    apply_morgan: bool = True,
) -> Dict[str, List[Conjecture]]:
    """
    Finalize a collection of conjectures into a ranked bank.

    Steps
    -----
    1. Deduplicate per bucket using string representation.
    2. Annotate touches/support (batch evaluation on df).
    3. Sort by (touch_count, support_n).
    4. (Optional) Apply Morgan filter per bucket.
    5. Re-annotate + re-sort after Morgan (if applied).
    6. (Optional) Truncate to top_k_per_bucket per bucket.
    7. Return a dict with "lowers", "uppers", "equals".

    Notes
    -----
    If top_k_per_bucket is None, no truncation is performed and
    *all* conjectures that survive Morgan (if applied) are returned.
    """
    # Local import to avoid circular deps at module import time.
    from txgraffiti2025.processing.post import morgan_filter

    lowers = _dedup_by_string(all_lowers)
    uppers = _dedup_by_string(all_uppers)
    equals = _dedup_by_string(all_equals)

    # Initial annotation
    _annotate_touch_support(df, lowers)
    _annotate_touch_support(df, uppers)
    _annotate_touch_support(df, equals)

    # Initial sort
    lowers = _sort_by_touch_support(lowers)
    uppers = _sort_by_touch_support(uppers)
    equals = _sort_by_touch_support(equals)

    lowers = lowers[:20]
    uppers = uppers[:20]
    equals = equals[:20]

    # Optional Morgan filter
    if apply_morgan:
        lowers = list(morgan_filter(df, lowers).kept)
        uppers = list(morgan_filter(df, uppers).kept)
        equals = list(morgan_filter(df, equals).kept)

        # Re-annotate after Morgan
        _annotate_touch_support(df, lowers)
        _annotate_touch_support(df, uppers)
        _annotate_touch_support(df, equals)

        # Re-sort
        lowers = _sort_by_touch_support(lowers)
        uppers = _sort_by_touch_support(uppers)
        equals = _sort_by_touch_support(equals)

    # Optional truncation
    if top_k_per_bucket is not None:
        lowers = lowers[:top_k_per_bucket]
        uppers = uppers[:top_k_per_bucket]
        equals = equals[:top_k_per_bucket]

    bank: Dict[str, List[Conjecture]] = {
        "lowers": lowers,
        "uppers": uppers,
        "equals": equals,
    }
    return bank
