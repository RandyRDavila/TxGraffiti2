"""
Shared utilities for conjecture processing (pre and post heuristics).

These functions are DataFrame-agnostic and operate on Conjecture objects
from `txgraffiti2025.forms.generic_conjecture`.  They provide:
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
        m = applicable & np.isclose(slack, 0.0, atol=1e-9)
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
        # Holds everywhere → possibly tautology
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
