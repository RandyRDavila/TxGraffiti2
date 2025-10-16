# """
# Hazel heuristic (post-processing)

# Rule:
#     Discard the bottom 25% of conjectures by touch count (closeness).
#     Keep only inequalities (Le, Ge). Sort survivors by touch desc.

# Touch definition:
#     touches = # of rows where |lhs - rhs| <= eps, with finite values.

# Edge handling:
#     - Ignore equalities and non-inequalities entirely.
#     - If a conjecture errors during eval, exclude it from scoring.
#     - If all exclusions lead to no valid scores, return [].
# """

# from __future__ import annotations
# import numpy as np
# import pandas as pd
# from typing import List, Optional, Tuple

# from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge
# from txgraffiti2025.processing.utils import log_event

# def _safe_touch_count(conj: Conjecture, df: pd.DataFrame, eps: float) -> Optional[int]:
#     """Return touch count or None if evaluation fails or relation is not inequality."""
#     rel = conj.relation
#     if not isinstance(rel, (Le, Ge)):
#         return None
#     try:
#         lhs = rel.left.eval(df)
#         rhs = rel.right.eval(df)
#         lhs = np.asarray(lhs, dtype=float)
#         rhs = np.asarray(rhs, dtype=float)
#         good = np.isfinite(lhs) & np.isfinite(rhs)
#         if not np.any(good):
#             return 0
#         diff = np.abs(lhs[good] - rhs[good])
#         return int((diff <= eps).sum())
#     except Exception:
#         return None

# def hazel_filter(conjectures: List[Conjecture], df: pd.DataFrame, eps: float = 1e-6) -> List[Conjecture]:
#     """
#     Keep only conjectures in the top 75% by touch count (i.e., drop bottom quartile).
#     Only considers inequalities (Le, Ge). Survivors sorted by touch desc.
#     """
#     if not conjectures:
#         return []

#     # Filter to inequalities up front
#     ineqs = [c for c in conjectures if isinstance(c.relation, (Le, Ge))]
#     if not ineqs:
#         return []

#     scored: List[Tuple[Conjecture, int]] = []
#     for c in ineqs:
#         t = _safe_touch_count(c, df, eps)
#         if t is not None:
#             scored.append((c, t))

#     if not scored:
#         return []

#     touches = np.array([t for _, t in scored], dtype=float)
#     cutoff = np.percentile(touches, 25)  # bottom quartile threshold

#     # Strictly drop bottom quartile (strict > to avoid keeping all on ties at cutoff)
#     survivors = [(c, t) for (c, t) in scored if t > cutoff]

#     # If strict cut drops everything (e.g., all equal touches), keep them all
#     if not survivors:
#         survivors = scored

#     survivors.sort(key=lambda x: x[1], reverse=True)
#     log_event(f"Hazel: kept {len(survivors)}/{len(ineqs)} (strictly above 25th percentile unless tied)")
#     return [c for c, _ in survivors]


"""
Hazel heuristic: rank conjectures by 'touch number' and prune the bottom tail.

Definition
----------
Touch number = number of rows where the conjecture holds with equality on its
own hypothesis (class condition). For inequalities, "equality" means zero
slack within a small tolerance. For equalities (R == L), "equality" is just
the relation holding within its own tolerance.

Procedure
---------
1) Compute touch for each conjecture on the provided DataFrame.
2) Drop the bottom `drop_frac` quantile of touch numbers (default 25%).
3) Return the remaining conjectures sorted by non-increasing touch.

Outputs
-------
- A scoreboard DataFrame (touch counts, support, etc.)
- A kept list (sorted)
- A dropped list with reasons

Usage
-----
>>> from txgraffiti2025.processing.post.hazel import hazel_rank
>>> res = hazel_rank(df, morgan_result.kept, drop_frac=0.25)
>>> res.scoreboard.head()
>>> for c in res.kept_sorted:
...     print(c.pretty(arrow="⇒"))
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Le, Ge
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.pretty import format_conjecture


# -----------------------------
# equality / touch computation
# -----------------------------

def _applicable_mask(df: pd.DataFrame, cond: Optional[Predicate]) -> pd.Series:
    if cond is None:
        return pd.Series(True, index=df.index)
    m = cond.mask(df)
    return m.reindex(df.index, fill_value=False).astype(bool)

def _holds_mask_under_applicable(df: pd.DataFrame, conj: Conjecture) -> pd.Series:
    app = _applicable_mask(df, conj.condition)
    eval_mask = conj.relation.evaluate(df).reindex(df.index, fill_value=False).astype(bool)
    return app & eval_mask

def _touch_mask(df: pd.DataFrame, conj: Conjecture, *, atol: float) -> pd.Series:
    """
    Rows where the conjecture 'touches' (holds with equality) under its hypothesis.

    - For Eq: equality = |L - R| <= tol (uses the relation's own tol).
    - For Le/Ge: equality = |slack| <= atol, under applicability & satisfaction.
    """
    app = _applicable_mask(df, conj.condition)
    holds = _holds_mask_under_applicable(df, conj)
    if isinstance(conj.relation, Eq):
        # Use the Eq tolerance for equality
        rel = conj.relation
        l = rel.left.eval(df)
        r = rel.right.eval(df)
        eq = pd.Series(np.isclose(l - r, 0.0, atol=float(rel.tol)), index=df.index)
        return app & holds & eq
    else:
        slack = conj.relation.slack(df)
        eq = pd.Series(np.isclose(slack, 0.0, atol=float(atol)), index=df.index)
        return app & holds & eq

def compute_touch_number(df: pd.DataFrame, conj: Conjecture, *, atol: float = 1e-9) -> Dict[str, object]:
    """
    Return touch statistics for a single conjecture.

    Returns
    -------
    dict with keys:
        touch : int
        applicable_n : int
        holds_n : int  (within applicable)
        touch_frac_app : float
        touch_frac_holds : float
    """
    app = _applicable_mask(df, conj.condition)
    holds = _holds_mask_under_applicable(df, conj)
    touch_mask = _touch_mask(df, conj, atol=atol)

    applicable_n = int(app.sum())
    holds_n = int(holds.sum())
    touch = int((touch_mask).sum())

    touch_frac_app = float(touch / applicable_n) if applicable_n else 0.0
    touch_frac_holds = float(touch / holds_n) if holds_n else 0.0

    return {
        "touch": touch,
        "applicable_n": applicable_n,
        "holds_n": holds_n,
        "touch_frac_app": touch_frac_app,
        "touch_frac_holds": touch_frac_holds,
    }


# -----------------------------
# Hazel ranking / pruning
# -----------------------------

@dataclass
class HazelResult:
    kept_sorted: List[Conjecture]
    dropped: List[Tuple[Conjecture, str]]
    scoreboard: pd.DataFrame  # includes a 'conjecture' column with the object


def hazel_rank(
    df: pd.DataFrame,
    conjectures: Iterable[Conjecture],
    *,
    drop_frac: float = 0.25,
    atol: float = 1e-9,
) -> HazelResult:
    """
    Apply the Hazel heuristic:
    - compute touch numbers
    - drop the bottom `drop_frac` quantile
    - sort remaining in non-increasing order of touch

    Parameters
    ----------
    df : pd.DataFrame
    conjectures : iterable of Conjecture
    drop_frac : float in [0,1], default 0.25
        Fraction to prune from the bottom by touch number.
    atol : float, default 1e-9
        Equality tolerance for inequalities (Le/Ge). Eq uses its own tol.

    Returns
    -------
    HazelResult
    """
    rows = []
    conjs_list = list(conjectures)
    for c in conjs_list:
        stats = compute_touch_number(df, c, atol=atol)
        rows.append({
            "conjecture": c,
            "touch": stats["touch"],
            "applicable_n": stats["applicable_n"],
            "holds_n": stats["holds_n"],
            "touch_frac_app": stats["touch_frac_app"],
            "touch_frac_holds": stats["touch_frac_holds"],
            "pretty": c.pretty(arrow="⇒"),  # requires pretty module import somewhere earlier
            "relation_text": format_conjecture(c, show_condition=False),
            "condition_text": format_conjecture(c, show_condition=True).split("⇒")[0].strip() if "⇒" in c.pretty(arrow="⇒") else "",
        })

    scoreboard = pd.DataFrame(rows)
    if len(scoreboard) == 0:
        return HazelResult(kept_sorted=[], dropped=[], scoreboard=scoreboard)

    # Determine pruning threshold by quantile
    q = float(scoreboard["touch"].quantile(drop_frac))
    keep_mask = scoreboard["touch"] > q  # strictly greater than the quantile
    # If all ties at q get dropped (potentially too aggressive for small sets), consider a fallback:
    if keep_mask.sum() == 0:
        # keep the top ceil((1-drop_frac)*n) by sorting
        k = int(np.ceil((1.0 - drop_frac) * len(scoreboard)))
        keep_mask.iloc[scoreboard["touch"].sort_values(ascending=False).index[:k]] = True

    kept_df = (scoreboard[keep_mask]
               .sort_values(["touch", "holds_n", "applicable_n", "pretty"], ascending=[False, False, False, True])
               .reset_index(drop=True))
    dropped_df = scoreboard[~keep_mask].copy()

    kept_sorted = kept_df["conjecture"].tolist()
    dropped = [(row["conjecture"], f"low touch (touch={row['touch']}, threshold={q})")
               for _, row in dropped_df.iterrows()]

    return HazelResult(kept_sorted=kept_sorted, dropped=dropped, scoreboard=scoreboard)
