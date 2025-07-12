"""
Module for exporting TxGraffiti conjectures to Lean 4 syntax.

This module provides functions to convert symbolic conjectures into
Lean-compatible theorems or propositions. It includes Lean-friendly
symbol mappings, automatic variable mappings, and translators from
Conjecture objects to Lean 4 strings.
"""

from __future__ import annotations
import re
from collections.abc import Mapping
import pandas as pd


from txgraffiti.logic import *

__all__ = [
    "conjecture_to_lean4",
    # "auto_var_map",
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
    Build a variable mapping for Lean 4 translation.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe from which to extract column names.
    skip : tuple of str, optional
        Column names to skip in the output (default is ('name',)).

    Returns
    -------
    dict of str to str
        A mapping from column names to Lean variable expressions.
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

def conjecture_to_lean4(
    conj: Conjecture,
    name: str,
    object_symbol: str = "G",
    object_decl: str = "SimpleGraph V"
) -> str:
    """
    Convert a Conjecture object into a Lean 4 theorem with explicit hypotheses.

    Parameters
    ----------
    conj : Conjecture
        The conjecture object to convert.
    name : str
        Name of the theorem in Lean.
    object_symbol : str, optional
        Symbol representing the graph (default is 'G').
    object_decl : str, optional
        Lean type declaration for the object (default is 'SimpleGraph V').

    Returns
    -------
    str
        A Lean 4 theorem string with bound hypotheses and a conclusion.
    """

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
