# src/txgraffiti2025/processing/utils/inspect.py

from typing import Iterable, List, Dict, Any
import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge, Eq
from txgraffiti2025.forms.pretty import format_conjecture
from txgraffiti2025.processing.post.hazel import compute_touch_number

__all__ = ["conjectures_dataframe", "print_conjectures_summary"]

def conjectures_dataframe(df: pd.DataFrame, conjs: Iterable[Conjecture]) -> pd.DataFrame:
    """
    Return a DataFrame summarizing a set of conjectures:
    - name
    - kind (Le/Ge/Eq)
    - condition_pretty
    - relation_pretty
    - holds (boolean)
    - touch, applicable_n, holds_n
    - repr_condition, repr_relation (for exact matching/debug)
    """
    rows: List[Dict[str, Any]] = []
    for c in conjs:
        try:
            rel = c.relation
            if isinstance(rel, Le):
                kind = "Le"
            elif isinstance(rel, Ge):
                kind = "Ge"
            elif isinstance(rel, Eq):
                kind = "Eq"
            else:
                kind = type(rel).__name__

            applicable_mask, holds_mask, _fails = c.check(df, auto_base=False)
            holds_all = bool(holds_mask[applicable_mask].all())

            # Touch stats (Hazel-compatible)
            stats = compute_touch_number(df, c, atol=1e-9)

            rows.append({
                "name": getattr(c, "name", None),
                "kind": kind,
                "condition_pretty": format_conjecture(c, show_condition=True),
                "relation_pretty": format_conjecture(c, show_condition=False),
                "holds": holds_all,
                "touch": stats["touch"],
                "applicable_n": stats["applicable_n"],
                "holds_n": stats["holds_n"],
                "repr_condition": repr(c.condition),
                "repr_relation": repr(c.relation),
            })
        except Exception as e:
            rows.append({
                "name": getattr(c, "name", None),
                "kind": "ERR",
                "condition_pretty": "<error>",
                "relation_pretty": f"<error: {e}>",
                "holds": False,
                "touch": 0,
                "applicable_n": 0,
                "holds_n": 0,
                "repr_condition": repr(getattr(c, "condition", None)),
                "repr_relation": repr(getattr(c, "relation", None)),
            })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(
            by=["holds", "touch", "holds_n", "applicable_n", "name"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return df_out


def print_conjectures_summary(df: pd.DataFrame, conjs: Iterable[Conjecture], top: int = 50) -> None:
    """
    Pretty print the top-N conjectures by (holds desc, touch desc).
    """
    table = conjectures_dataframe(df, conjs)
    n = min(top, len(table))
    for i in range(n):
        row = table.iloc[i]
        print(f"[{i+1:02d}] {row['relation_pretty']}   {row['condition_pretty']}")
        print(f"     holds={row['holds']}  touch={row['touch']}  support={row['applicable_n']}  holds_n={row['holds_n']}")
        if row["name"]:
            print(f"     name={row['name']}")
        print()
