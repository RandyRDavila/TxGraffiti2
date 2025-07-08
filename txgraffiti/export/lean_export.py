"""
lean_export.py
==============

Tools for turning Conjecture names into Lean-4 propositions.

Key public entry points
-----------------------
conjecture_to_lean(conj, df, *, theorem_name=None) -> str
    Translate Conjecture *conj* into a Lean statement, using *df* to discover
    which variable names exist.  If *theorem_name* is given, wrap it in a
    `theorem ... :` block with a final `:= sorry`.

auto_var_map(df, *, skip=("name",)) -> dict[str, str]
    Build the {python_name: lean_name} map automatically from the columns of
    *df*.  Numeric or Boolean – doesn’t matter.
"""

from __future__ import annotations
import re
from collections.abc import Mapping
import pandas as pd
from typing import Any

from txgraffiti.logic.conjecture_logic import *

__all__ = [
    "conjecture_to_lean",
    "conjecture_to_lean4",
    "auto_var_map",
    "LEAN_SYMBOLS",
    "LEAN_SYMBOLS",
]

# ---------------------------------------------------------------------------
# 1. Lean-friendly replacements for operators & symbols
# ---------------------------------------------------------------------------
LEAN_SYMBOLS: Mapping[str, str] = {
    "∧": "∧",
    "∨": "∨",
    "¬": "¬",
    "→": "→",
    "≥": "≥",
    "<=": "≤",
    ">=": "≥",
    "==": "=",
    "=": "=",
    "!=": "≠",
    "<": "<",
    ">": ">",
    "/": "/",
    "**": "^",
}

# ---------------------------------------------------------------------------
# 2. Automatic variable-map builder
# ---------------------------------------------------------------------------
def auto_var_map(df: pd.DataFrame, *, skip: tuple[str, ...] = ("name",)) -> dict[str, str]:
    """
    Build a mapping  {column_name -> "column_name G"}   (for Lean)
    while skipping columns in *skip* (default: "name").
    """
    return {c: f"{c} G" for c in df.columns if c not in skip}


# ---------------------------------------------------------------------------
# 3. The main translator
# ---------------------------------------------------------------------------
def _translate(expr: str, var_map: Mapping[str, str]) -> str:
    # 3a. longest variable names first so 'order' doesn't clobber 'total_order'
    for var in sorted(var_map, key=len, reverse=True):
        expr = re.sub(rf"\b{re.escape(var)}\b", var_map[var], expr)

    # 3b. symbolic replacements (do ** after >= / <= replacements)
    for sym, lean_sym in LEAN_SYMBOLS.items():
        expr = expr.replace(sym, lean_sym)

    # tidy whitespace
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


def conjecture_to_lean(
    conj: "Conjecture | str",           # accepts object or raw string
    df: pd.DataFrame,
    *,
    theorem_name: str | None = None,
    extra_var_map: Mapping[str, str] | None = None,
) -> str:
    """
    Parameters
    ----------
    conj : Conjecture | str
        Either a Conjecture instance or a raw name string to translate.
    df : pandas.DataFrame
        DataFrame whose columns define which variables exist.
    theorem_name : str, optional
        If provided, wrap the result in a `theorem … :` Lean block.
    extra_var_map : dict, optional
        Add/override variable translations.

    Returns
    -------
    str : Lean 4 proposition (or theorem block).
    """
    name_str = conj.name if hasattr(conj, "name") else str(conj)
    var_map = {**auto_var_map(df), **(extra_var_map or {})}
    proposition = _translate(name_str, var_map)
    proposition = f"∀ G : SimpleGraph V, {proposition}"

    if theorem_name:
        proposition = f"theorem {theorem_name} : {proposition} := by\n  -- sketch proof\n  sorry"
    return proposition

# txgraffiti/utils/lean_export.py


# txgraffiti/utils/lean_export.py

def conjecture_to_lean4(
    conj: Conjecture,
    name: str,
    object_symbol: str = "G",
    object_decl: str = "SimpleGraph V"
) -> str:
    # 1) extract hypothesis Predicates
    terms = getattr(conj.hypothesis, "_and_terms", [conj.hypothesis])
    binds = []
    for idx, p in enumerate(terms, start=1):
        lean_pred = p.name
        binds.append(f"(h{idx} : {lean_pred} {object_symbol})")

    # 2) extract conclusion
    ineq = conj.conclusion
    lhs, op, rhs = ineq.lhs.name, ineq.op, ineq.rhs.name
    lean_rel = {"<=":"≤", "<":"<", ">=":"≥", ">":">", "==":"=", "!=":"≠"}[op]

    # 3) assemble
    bind_str = "\n    ".join(binds)
    return (
        f"theorem {name} ({object_symbol} : {object_decl})\n"
        f"    {bind_str} : {lhs} {object_symbol} {lean_rel} {rhs} {object_symbol} :=\n"
        f"sorry \n"
    )

