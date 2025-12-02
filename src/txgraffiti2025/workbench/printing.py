# src/txgraffiti2025/workbench/printing.py

from __future__ import annotations
from typing import Iterable, Sequence, Optional
import numpy as np
import pandas as pd

from txgraffiti2025.forms.predicates import Predicate
from .textparse import predicate_to_conjunction, predicate_to_if_then


# ────────────────────────────── Internals ────────────────────────────── #

def _cond_str(H: Optional[Predicate]) -> str:
    """
    Return a stable, human-friendly name for a predicate.

    Prefers `H.pretty()` if present, then `H.name`, finally `repr(H)`.
    """
    if H is None:
        return "TRUE"
    if hasattr(H, "pretty"):
        try:
            return H.pretty()
        except Exception:
            pass
    if getattr(H, "name", None):
        return str(H.name)
    return repr(H)


def _impl_symbols(ascii_ops: bool) -> tuple[str, str]:
    """
    Return implication/equivalence symbols (⇒/⇔ or ASCII).
    """
    impl  = "->" if ascii_ops else "⇒"
    equiv = "<->" if ascii_ops else "⇔"
    return impl, equiv


def _rel_symbol(rel_obj, *, ascii_ops: bool) -> str:
    """
    Map a relation object (e.g., Ge/Le) to a printable symbol.
    """
    name = getattr(rel_obj, "__class__", type(rel_obj)).__name__
    if name == "Le":
        return "<=" if ascii_ops else "≤"
    if name == "Ge":
        return ">=" if ascii_ops else "≥"
    # Fallback (rare): use its own pretty or name
    return getattr(rel_obj, "pretty", lambda: name)()


def _fmt_support(mask: np.ndarray, total: int) -> str:
    """
    'support/total = pct%'
    """
    sup = int(mask.sum())
    pct = 100.0 * sup / total if total else 0.0
    return f"[{sup}/{total} = {pct:4.1f}%]"


def _mask_for(H: Optional[Predicate], df: pd.DataFrame) -> np.ndarray:
    if H is None:
        return np.ones(len(df), dtype=bool)
    return H.mask(df).astype(bool).to_numpy()


# ─────────────────────────── Conjecture printers ─────────────────────────── #

def pretty_conjectures(
    title: str,
    conjs: Sequence,
    *,
    df: Optional[pd.DataFrame] = None,
    max_items: int = 100,
    ascii_ops: bool = False,
    show_support: bool = False,
    show_name: bool = False,
) -> None:
    """
    Print a standardized block of conjectures.

    Parameters
    ----------
    title : str
        Section title to display.
    conjs : sequence
        Iterable of Conjecture-like objects. Each item is expected to have:
        - `.relation.left` / `.relation.right` expressions with `.pretty()`
        - `.relation` with class name 'Le' or 'Ge'
        - optional `.condition` (Predicate)
        - optional `.name`
        - optional `.pretty(arrow=...)` (we ignore it for uniformity)
    df : pandas.DataFrame, optional
        If provided and `show_support=True`, we compute and print support
        for each conjecture’s hypothesis condition.
    max_items : int, default=100
        Max number of conjectures to print.
    ascii_ops : bool, default=False
        Use ASCII operators (->, <=, >=) instead of Unicode (⇒, ≤, ≥).
    show_support : bool, default=False
        If True, print hypothesis support statistics for each conjecture.
    show_name : bool, default=False
        If True and a conjecture has `name`, print it.

    Notes
    -----
    - This function standardizes output across single-feature, mixed, LP,
      and product conjecture generators.
    """
    impl, _ = _impl_symbols(ascii_ops)
    print(f"\n=== {title} ===")
    for i, c in enumerate(conjs[:max_items], 1):
        H = getattr(c, "condition", None)
        cond = _cond_str(H)
        lhs = getattr(c.relation, "left", None)
        rhs = getattr(c.relation, "right", None)
        lhs_s = lhs.pretty() if hasattr(lhs, "pretty") else str(lhs)
        rhs_s = rhs.pretty() if hasattr(rhs, "pretty") else str(rhs)
        rel_s = _rel_symbol(getattr(c, "relation", None), ascii_ops=ascii_ops)

        # header line: "  1. (COND) -> LHS <= RHS"
        line = f"{i:3d}. ({cond}) {impl} {lhs_s} {rel_s} {rhs_s}"
        print(line)

        # optional name + support lines
        if show_name and getattr(c, "name", None):
            print(f"     Name     : {c.name}")
        if show_support and df is not None:
            mask = _mask_for(H, df)
            print(f"     Support  : {_fmt_support(mask, len(df))}")


def pretty_block(title: str, conjs: Sequence, max_items: int = 100) -> None:
    """
    Backward-compatible compact printer using each conjecture's own `pretty()`.
    """
    print(f"\n=== {title} ===")
    for i, c in enumerate(conjs[:max_items], 1):
        try:
            print(f"{i:3d}. {c.pretty(arrow='⇒')}")
        except Exception:
            print(f"{i:3d}. {c}")


def pretty_block_with_hyp(
    title: str,
    conjs: Sequence,
    df: pd.DataFrame,
    max_items: int = 40
) -> None:
    """
    Backward-compatible detailed printer including hypothesis support.
    """
    print(f"\n=== {title} ===")
    for i, c in enumerate(conjs[:max_items], 1):
        H = getattr(c, "condition", None)
        cond = _cond_str(H)
        mask = _mask_for(H, df)
        print(f"\n{i:3d}. Hypothesis {_fmt_support(mask, len(df))}: {cond}")
        if getattr(c, "name", None):
            print(f"     Name      : {c.name}")
        lhs = c.relation.left.pretty()
        rhs = c.relation.right.pretty()
        rel_symbol = _rel_symbol(c.relation, ascii_ops=False)
        print(f"     Relation  : {lhs} {rel_symbol} {rhs}")


# ───────────────────────── Qualitative printers ───────────────────────── #

def pretty_qualitative(
    title: str,
    results: Sequence,
    *,
    max_items: int = 40,
    ascii_ops: bool = False,
    verbose: bool = False,
) -> None:
    """
    Print qualitative monotone relations in plain language.

    Parameters
    ----------
    title : str
        Section title to display.
    results : sequence of QualResult
        Each item is expected to have attributes:
        `relation` (with `.direction`), `condition`, `x`, `y`, `rho`, `n`, `support`.
    max_items : int, default=40
        Max rows to print.
    ascii_ops : bool, default=False
        If True, prints '->' instead of '⇒' in the header relation.
    verbose : bool, default=False
        If True, also prints the underlying correlation method and thresholds
        if present on the `relation` object.
    """
    impl, _ = _impl_symbols(ascii_ops)
    print(f"\n=== {title} ===")
    for i, r in enumerate(results[:max_items], 1):
        cond = _cond_str(getattr(r, "condition", None))
        direction = getattr(r.relation, "direction", "increasing")
        phrase = "increases with" if direction == "increasing" else "decreases with"
        rho = float(getattr(r, "rho", float("nan")))
        n = int(getattr(r, "n", 0))
        supp = int(getattr(r, "support", 0))
        method = getattr(r.relation, "method", None)

        # "  1. (COND) -> y increases with x   [ρ=+0.94, n=66, support=66]"
        line = (
            f"{i:3d}. ({cond}) {impl} {r.y} {phrase} {r.x}   "
            f"[ρ={rho:+.2f}, n={n}, support={supp}]"
        )
        print(line)

        if verbose and method:
            extra = f"     Method   : {method}"
            if hasattr(r.relation, "min_abs_rho"):
                extra += f" (min |ρ|={float(r.relation.min_abs_rho):.2f})"
            if hasattr(r.relation, "min_n"):
                extra += f", min n={int(r.relation.min_n)}"
            print(extra)


# ───────────────────── Class relations (unchanged API) ───────────────────── #

def pretty_class_relations_ifthen(title: str, eqs, incs, df, max_items: int = 60) -> None:
    """
    Print class relations (equivalences & inclusions) in if-then form with violation stats.
    """
    print(f"\n=== {title} ===")
    print("\n-- Equivalences --")
    for i, e in enumerate(eqs[:max_items], 1):
        print(f"{i:3d}. {predicate_to_if_then(e.A)}  ⇔  {predicate_to_if_then(e.B)}    [violations={e.violation_count(df)}]")
    print("\n-- Inclusions --")
    for i, inc in enumerate(incs[:max_items], 1):
        suppA  = int(inc.A.mask(df).sum())
        viol   = inc.violation_count(df)
        print(f"{i:3d}. {predicate_to_if_then(inc.A)}  ⇒  {predicate_to_if_then(inc.B)}    [support(A)={suppA}, violations={viol}]")


def pretty_class_relations_conj(
    title: str,
    eqs,
    incs,
    df,
    *,
    max_items: int = 60,
    ascii_ops: bool = False
) -> None:
    """
    Print class relations as conjunction formulas (Unicode or ASCII symbols).
    """
    impl, equiv = _impl_symbols(ascii_ops)
    print(f"\n=== {title} ===")
    print("\n-- Equivalences --")
    for i, e in enumerate(eqs[:max_items], 1):
        print(f"{i:3d}. {predicate_to_conjunction(e.A, ascii_ops=ascii_ops)} {equiv} {predicate_to_conjunction(e.B, ascii_ops=ascii_ops)}")
    print("\n-- Inclusions --")
    for i, inc in enumerate(incs[:max_items], 1):
        suppA  = int(inc.A.mask(df).sum())
        print(f"{i:3d}. {predicate_to_conjunction(inc.A, ascii_ops=ascii_ops)} {impl} {predicate_to_conjunction(inc.B, ascii_ops=ascii_ops)}  [support(A)={suppA}]")
