# """
# Shared utilities for conjecture processing (pre and post heuristics).

# These functions are DataFrame-agnostic and operate on Conjecture objects
# from `txgraffiti2025.forms.generic_conjecture`.  They provide:
# - truth evaluation and slack summarization
# - deduplication and triviality checks
# - safe functional composition helpers
# """

# from __future__ import annotations
# import hashlib
# import numpy as np
# import pandas as pd
# from typing import Callable, Iterable, List, Tuple, Any

# from txgraffiti2025.forms.generic_conjecture import Conjecture


# # ============================================================
# # --- Conjecture evaluation helpers
# # ============================================================

# def truth_mask(conj: Conjecture, df: pd.DataFrame) -> pd.Series:
#     """Return boolean mask where the conjecture holds."""
#     _, holds, _ = conj.check(df)
#     return holds


# def touch_count(conj: Conjecture, df: pd.DataFrame) -> int:
#     """Number of rows where conjecture holds with equality (slack == 0)."""
#     try:
#         applicable, _, _ = conj.check(df)
#         slack = conj.relation.slack(df)
#         m = applicable & np.isclose(slack, 0.0, atol=1e-9)
#         return int(m.sum())
#     except Exception:
#         return 0


# def slack_summary(conj: Conjecture, df: pd.DataFrame) -> Tuple[float, float]:
#     """Return (min_slack, mean_slack) among applicable rows."""
#     try:
#         applicable, _, _ = conj.check(df)
#         s = conj.relation.slack(df)[applicable]
#         if len(s) == 0:
#             return (np.nan, np.nan)
#         return (float(np.nanmin(s)), float(np.nanmean(s)))
#     except Exception:
#         return (np.nan, np.nan)


# # ============================================================
# # --- Structural / symbolic utilities
# # ============================================================

# def hash_conjecture(conj: Conjecture) -> str:
#     """
#     Return a hash string for identifying duplicate conjectures.
#     Based on string representations of the relation and condition.
#     """
#     h = hashlib.sha1()
#     payload = f"{conj.condition!r}|{conj.relation!r}|{conj.name}"
#     h.update(payload.encode("utf-8"))
#     return h.hexdigest()


# def is_trivial_conjecture(conj: Conjecture, df: pd.DataFrame) -> bool:
#     """
#     Detect trivial conjectures such as always-true inequalities
#     or empty applicable sets.
#     """
#     applicable, holds, _ = conj.check(df)
#     if not applicable.any():
#         return True
#     if holds.all():
#         # Holds everywhere → possibly tautology
#         sl = conj.relation.slack(df).loc[applicable]
#         if np.all(sl >= 0):
#             return True
#     return False


# def compare_conjectures(a: Conjecture, b: Conjecture) -> bool:
#     """Loose equality check based on structural hashes."""
#     return hash_conjecture(a) == hash_conjecture(b)


# # ============================================================
# # --- Functional helpers for pipelines
# # ============================================================

# def safe_apply(conjs: Iterable[Conjecture], fn: Callable[[Conjecture], Any]) -> List[Any]:
#     """Apply a function safely to each conjecture, skipping failures."""
#     out = []
#     for c in conjs:
#         try:
#             out.append(fn(c))
#         except Exception:
#             continue
#     return out


# def describe_conjectures(conjs: Iterable[Conjecture], df: pd.DataFrame) -> pd.DataFrame:
#     """Produce a quick summary table for a list of conjectures."""
#     rows = []
#     for c in conjs:
#         t = touch_count(c, df)
#         mn, avg = slack_summary(c, df)
#         rows.append({
#             "name": getattr(c, "name", "Conjecture"),
#             "hash": hash_conjecture(c)[:8],
#             "touch": t,
#             "min_slack": mn,
#             "mean_slack": avg,
#         })
#     return pd.DataFrame(rows)


# # ============================================================
# # --- Light logging helpers (for heuristics)
# # ============================================================

# def log_event(msg: str):
#     """Minimal consistent log printer for processing steps."""
#     print(f"[processing] {msg}")


# # src/txgraffiti2025/processing/utils/inspect.py

# # from __future__ import annotations
# from typing import Iterable, List, Dict, Any

# import numpy as np
# import pandas as pd

# from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge, Eq
# from txgraffiti2025.forms.pretty import format_conjecture
# from txgraffiti2025.processing.post.hazel import compute_touch_number

# def conjectures_dataframe(df: pd.DataFrame, conjs: Iterable[Conjecture]) -> pd.DataFrame:
#     """
#     Return a DataFrame summarizing a set of conjectures:
#     - name
#     - kind (Le/Ge/Eq)
#     - condition_pretty
#     - relation_pretty
#     - holds (boolean)
#     - touch, applicable_n, holds_n
#     - repr_condition, repr_relation (for exact matching/debug)
#     """
#     rows: List[Dict[str, Any]] = []
#     for c in conjs:
#         try:
#             rel = c.relation
#             if isinstance(rel, Le):
#                 kind = "Le"
#             elif isinstance(rel, Ge):
#                 kind = "Ge"
#             elif isinstance(rel, Eq):
#                 kind = "Eq"
#             else:
#                 kind = type(rel).__name__

#             applicable_mask, holds_mask, _fails = c.check(df, auto_base=False)
#             holds_all = bool(holds_mask[applicable_mask].all())

#             # Touch stats (Hazel-compatible)
#             stats = compute_touch_number(df, c, atol=1e-9)

#             rows.append({
#                 "name": getattr(c, "name", None),
#                 "kind": kind,
#                 "condition_pretty": format_conjecture(c, show_condition=True),
#                 "relation_pretty": format_conjecture(c, show_condition=False),
#                 "holds": holds_all,
#                 "touch": stats["touch"],
#                 "applicable_n": stats["applicable_n"],
#                 "holds_n": stats["holds_n"],
#                 "repr_condition": repr(c.condition),
#                 "repr_relation": repr(c.relation),
#             })
#         except Exception as e:
#             rows.append({
#                 "name": getattr(c, "name", None),
#                 "kind": "ERR",
#                 "condition_pretty": "<error>",
#                 "relation_pretty": f"<error: {e}>",
#                 "holds": False,
#                 "touch": 0,
#                 "applicable_n": 0,
#                 "holds_n": 0,
#                 "repr_condition": repr(getattr(c, "condition", None)),
#                 "repr_relation": repr(getattr(c, "relation", None)),
#             })

#     df_out = pd.DataFrame(rows)
#     if not df_out.empty:
#         df_out = df_out.sort_values(
#             by=["holds", "touch", "holds_n", "applicable_n", "name"],
#             ascending=[False, False, False, False, True],
#             kind="mergesort",
#         ).reset_index(drop=True)
#     return df_out


# def print_conjectures_summary(df: pd.DataFrame, conjs: Iterable[Conjecture], top: int = 50) -> None:
#     """
#     Pretty print the top-N conjectures by (holds desc, touch desc).
#     """
#     table = conjectures_dataframe(df, conjs)
#     n = min(top, len(table))
#     for i in range(n):
#         row = table.iloc[i]
#         print(f"[{i+1:02d}] {row['relation_pretty']}   {row['condition_pretty']}")
#         print(f"     holds={row['holds']}  touch={row['touch']}  support={row['applicable_n']}  holds_n={row['holds_n']}")
#         if row["name"]:
#             print(f"     name={row['name']}")
#         print()


"""
Shared utilities for conjecture processing (pre and post heuristics).

These functions are DataFrame-agnostic and operate on Conjecture objects
from `txgraffiti2025.forms.generic_conjecture`. They provide:
- truth evaluation and slack summarization
- deduplication and triviality checks
- safe functional composition helpers
"""

from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from typing import Callable, Iterable, List, Tuple, Any

from txgraffiti2025.forms.generic_conjecture import Conjecture


# ============================================================
# --- Conjecture evaluation helpers
# ============================================================

def truth_mask(conj: Conjecture, df: pd.DataFrame) -> pd.Series:
    """Return boolean mask where the conjecture holds."""
    _, holds, _ = conj.check(df)
    return holds


def touch_count(conj: Conjecture, df: pd.DataFrame) -> int:
    """Number of rows where conjecture holds with equality (slack == 0)."""
    try:
        applicable, _, _ = conj.check(df)
        slack = conj.relation.slack(df)
        eq = pd.Series(np.isclose(slack.values, 0.0, atol=1e-9), index=slack.index)
        m = applicable & eq
        return int(m.sum())
    except Exception:
        return 0


def slack_summary(conj: Conjecture, df: pd.DataFrame) -> Tuple[float, float]:
    """Return (min_slack, mean_slack) among applicable rows."""
    try:
        applicable, _, _ = conj.check(df)
        s = conj.relation.slack(df)[applicable]
        if len(s) == 0:
            return (np.nan, np.nan)
        return (float(np.nanmin(s)), float(np.nanmean(s)))
    except Exception:
        return (np.nan, np.nan)


# ============================================================
# --- Structural / symbolic utilities
# ============================================================

def hash_conjecture(conj: Conjecture) -> str:
    """
    Return a hash string for identifying duplicate conjectures.
    Based on string representations of the relation and condition.
    """
    h = hashlib.sha1()
    payload = f"{conj.condition!r}|{conj.relation!r}|{conj.name}"
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def is_trivial_conjecture(conj: Conjecture, df: pd.DataFrame) -> bool:
    """
    Detect trivial conjectures such as always-true inequalities
    or empty applicable sets.
    """
    applicable, holds, _ = conj.check(df)
    if not applicable.any():
        return True
    if holds.all():
        sl = conj.relation.slack(df).loc[applicable]
        if np.all(sl >= 0):
            return True
    return False


def compare_conjectures(a: Conjecture, b: Conjecture) -> bool:
    """Loose equality check based on structural hashes."""
    return hash_conjecture(a) == hash_conjecture(b)


# ============================================================
# --- Functional helpers for pipelines
# ============================================================

def safe_apply(conjs: Iterable[Conjecture], fn: Callable[[Conjecture], Any]) -> List[Any]:
    """Apply a function safely to each conjecture, skipping failures."""
    out = []
    for c in conjs:
        try:
            out.append(fn(c))
        except Exception:
            continue
    return out


def describe_conjectures(conjs: Iterable[Conjecture], df: pd.DataFrame) -> pd.DataFrame:
    """Produce a quick summary table for a list of conjectures."""
    rows = []
    for c in conjs:
        t = touch_count(c, df)
        mn, avg = slack_summary(c, df)
        rows.append({
            "name": getattr(c, "name", "Conjecture"),
            "hash": hash_conjecture(c)[:8],
            "touch": t,
            "min_slack": mn,
            "mean_slack": avg,
        })
    return pd.DataFrame(rows)


# ============================================================
# --- Light logging helpers (for heuristics)
# ============================================================

def log_event(msg: str):
    """Minimal consistent log printer for processing steps."""
    print(f"[processing] {msg}")
