# post_constant_generalizer.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.workbench.ranking import rank_and_filter

# Generalizers & caches from your codebase
from txgraffiti2025.processing.pre.constants_cache import (
    precompute_constant_ratios,
)
from txgraffiti2025.processing.post.generalize_from_constants import (
    propose_generalizations_from_constants,
)
from txgraffiti2025.processing.post.reciprocal_generalizer import (
    propose_generalizations_from_reciprocals,
)


def _mask(df: pd.DataFrame, H: Optional[Predicate]) -> np.ndarray:
    if H is None:
        return np.ones(len(df), dtype=bool)
    return H.mask(df).astype(bool).to_numpy()


def _support(m: np.ndarray) -> int:
    return int(np.asarray(m, dtype=bool).sum())


def _supersets_of(
    df: pd.DataFrame, H0: Predicate, candidates: Iterable[Predicate], *, min_support: int
) -> List[Predicate]:
    """
    Return candidate predicates that are supersets of H0 and have at least min_support rows,
    sorted by support descending.
    """
    m0 = _mask(df, H0)
    outs: List[Tuple[Predicate, int]] = []
    for H in candidates:
        m = _mask(df, H)
        # Superset test: every row allowed by H0 must also be allowed by H
        if ((~m0) | m).all():
            s = _support(m)
            if s >= min_support:
                outs.append((H, s))
    outs.sort(key=lambda t: -t[1])
    return [H for H, _ in outs]


def _dedupe_conjs(df: pd.DataFrame, conjs: Iterable[Conjecture]) -> List[Conjecture]:
    """
    Dedupe by (condition-mask signature, relation repr), preserving order.
    """
    seen = set()
    out: List[Conjecture] = []
    for c in conjs:
        try:
            m = _mask(df, getattr(c, "condition", None))
            sig = (int(m.sum()), hash(m.tobytes()), repr(c.relation))
        except Exception:
            sig = (repr(getattr(c, "condition", None)), repr(c.relation))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


def generalize_constants_on_broader_hypotheses(
    df: pd.DataFrame,
    base_conjectures: Iterable[Conjecture],
    *,
    candidate_hypotheses: Iterable[Predicate],
    numeric_cols: Iterable[str],
    # Ratio (c -> u/v) settings
    ratio_shifts: Tuple[int, ...] = (0,),
    ratio_min_support: int = 5,
    ratio_max_denominator: int = 64,
    ratio_atol: float = 1e-9,
    # Reciprocal (c -> 1/(x+shift)) settings
    recip_shifts: Tuple[int, ...] = (1,),
    recip_min_support: int = 5,
    recip_atol: float = 1e-9,
    # Keep criteria
    require_strictly_larger: bool = True,
    # Final ranking
    final_min_touch: int = 0,
) -> List[Conjecture]:
    """
    Post-process base conjectures by replacing numeric constants with ratios or reciprocals
    of invariants, and keep a proposal iff it is true on a strictly larger hypothesis.

    Parameters
    ----------
    df : DataFrame
        Data on which hypotheses and conjectures are evaluated.
    base_conjectures : iterable of Conjecture
        Seed conjectures to attempt generalization from. Each must have a condition (hypothesis).
    candidate_hypotheses : iterable of Predicate
        Hypotheses considered as potential supersets (broader slices). You may include TRUE.
    numeric_cols : iterable of str
        Numeric invariant columns used for u/v ratios and 1/(x+shift) reciprocals.
    ratio_shifts : tuple of int, optional
        Shifts for (u / (v + shift)). Default (0,) i.e., plain u/v.
    ratio_min_support : int, optional
        Minimum support on a hypothesis to accept a ratio cache entry/proposal.
    ratio_max_denominator : int, optional
        Max denominator when rationalizing near-constant ratios for readability.
    ratio_atol : float, optional
        Absolute tolerance for ratio-based checks.
    recip_shifts : tuple of int, optional
        Shifts for reciprocals, e.g., (1,) to try 1/(Δ+1).
    recip_min_support : int, optional
        Minimum support for reciprocal proposals.
    recip_atol : float, optional
        Absolute tolerance for reciprocal-based checks.
    require_strictly_larger : bool, optional
        If True (default), only accept generalized conjectures whose condition has larger
        support than the base hypothesis. If False, accept ≥ (non-decreasing) support.
    final_min_touch : int, optional
        Minimum touch count for the final rank-and-filter stage.

    Returns
    -------
    list of Conjecture
        For each base conjecture: either the best accepted generalization (max support),
        or the original conjecture if no accepted generalization exists. Output is deduped
        and Morgan-filtered via `rank_and_filter`.
    """
    # 1) Build ratio cache once for all candidate hypotheses / columns / shifts
    ratio_cache = precompute_constant_ratios(
        df,
        hypotheses=list(candidate_hypotheses),
        numeric_cols=list(numeric_cols),
        shifts=list(ratio_shifts),
        min_support=ratio_min_support,
        max_denominator=ratio_max_denominator,
    )

    results: List[Conjecture] = []
    cand_hyps = list(candidate_hypotheses)

    for base in base_conjectures:
        H0: Optional[Predicate] = getattr(base, "condition", None)
        if H0 is None:
            # If no hypothesis on the base, treat as TRUE; only allow strictly larger if False.
            # (You can choose to skip these.)
            base_mask = _mask(df, None)
        else:
            base_mask = _mask(df, H0)

        base_support = _support(base_mask)

        # Consider supersets of H0 (including H0 itself to let generalizers validate there too)
        test_hyps = [H0] if H0 is not None else []
        if H0 is not None:
            test_hyps += _supersets_of(df, H0, cand_hyps, min_support=max(ratio_min_support, recip_min_support))
        else:
            # If H0 is None/TRUE, just test on candidate hypotheses
            test_hyps += list(cand_hyps)

        proposals: List[Conjecture] = []

        # 2A) Reciprocal proposals (c -> 1/(x+shift))
        try:
            props_recip = propose_generalizations_from_reciprocals(
                df,
                base,
                candidate_hypotheses=test_hyps,
                candidate_cols=list(numeric_cols),
                shifts=tuple(recip_shifts),
                min_support=recip_min_support,
                atol=recip_atol,
            )
            proposals.extend(props_recip)  # already Conjecture objects
        except Exception:
            pass

        # 2B) Ratio proposals (c -> u/v)
        try:
            props_ratio = propose_generalizations_from_constants(
                df,
                base,
                ratio_cache,
                candidate_hypotheses=test_hyps,
                atol=ratio_atol,
            )
            # API returns wrapper objects; take their .new_conjecture
            proposals.extend([p.new_conjecture for p in props_ratio])
        except Exception:
            pass

        # 3) Choose the best proposal that truly generalizes support
        best: Optional[Conjecture] = None
        best_support = -1

        for p in proposals:
            try:
                applicable, holds, failures = p.check(df, auto_base=False)
                if not bool(holds[applicable].all()):
                    continue
                s = int(applicable.sum())
                if require_strictly_larger:
                    if s <= base_support:
                        continue
                else:
                    if s < base_support:
                        continue
                if s > best_support:
                    best, best_support = p, s
            except Exception:
                continue

        results.append(best if best is not None else base)

    # 4) Dedupe and Morgan-filter
    results = _dedupe_conjs(df, results)
    results = rank_and_filter(results, df, min_touch=final_min_touch, expr_key_mode="repr")
    return results
