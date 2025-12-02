# src/txgraffiti2025/workbench/config.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

"""
Configuration objects for the workbench pipeline.

This module centralizes user-tunable knobs that affect bound generation,
lightweight generalization (e.g., reciprocal/constant banking), and ranking.

The primary entry point is :class:`GenerationConfig`, a small dataclass with
sane defaults. Treat it as an immutable configuration snapshot you pass into
your workbench/engine; avoid mutating it mid-run.

Examples
--------
>>> from txgraffiti2025.workbench.config import GenerationConfig
>>> cfg = GenerationConfig(min_touch_keep=5, max_denom=20)
>>> cfg.min_touch_keep
5
>>> cfg.max_denom
20
"""

__all__ = [
    'GenerationConfig',
]

@dataclass
class GenerationConfig:
    """
    Global knobs used across bound generation, generalization, and ranking.

    Parameters
    ----------
    min_touch_keep : int, default=3
        Minimum number of equality touches (instances with ``lhs == rhs`` under the
        relevant hypothesis) required to retain a candidate bound in post-filters.
        Set higher to bias toward sharper, frequently tight inequalities.
    max_denom : int, default=30
        Maximum denominator considered when searching for rational coefficients
        (e.g., scanning grids like ``p/q`` with ``1 ≤ q ≤ max_denom``).
    use_floor_ceil_if_true : bool, default=True
        If ``True``, allow wrappers such as ``floor``/``ceil`` where logically valid
        (e.g., lower bounds with ``ceil``, upper bounds with ``floor``) during refinement
        or generalization steps.

    min_support_const : int, default=5
        Minimum support (row count) required for a constant-bank candidate to be considered
        during generalization. Helps avoid overfitting to tiny slices.
    eps_const : float, default=1e-12
        Tolerance used when matching floating values to “known” constants (e.g., 1/2, φ).
        Smaller values increase precision but reduce matches.
    coeff_eps_match : float, default=1e-3
        Tolerance for coefficient matching/merging when combining proposals from multiple
        sources (LP, ratios, constant bank). Controls deduping of near-duplicate coeffs.
    shifts : tuple[int, ...], default=(-2, -1, 0, 1, 2)
        Integer shifts to apply in constant/reciprocal generalizers (e.g., trying
        ``X/(k + shift)`` or ``(X + shift)/Y``). Tuned for small explorations.
    numerators : tuple[int, ...], default=(1, 2, 3, 4)
        Small integer multipliers tried in product/ratio forms during generalization.
    coeff_sources : {"allow_symbolic_columns", "constants_only"}, default="allow_symbolic_columns"
        Controls where candidate coefficients may originate from during generalization:
        - ``"constants_only"``: restrict to a curated constant bank (e.g., small rationals).
        - ``"allow_symbolic_columns"``: in addition to constants, permit symbolic columns
          (e.g., ``1/deg_max``) to instantiate coefficients when safe.
    symbolic_col_limit : int, default=10
        Upper bound on the number of symbolic columns considered for coefficient proposals.
        Acts as a guardrail against combinatorial blow-ups.

    Attributes
    ----------
    min_touch_keep : int
    max_denom : int
    use_floor_ceil_if_true : bool
    min_support_const : int
    eps_const : float
    coeff_eps_match : float
    shifts : Tuple[int, ...]
    numerators : Tuple[int, ...]
    coeff_sources : str
    symbolic_col_limit : int

    Notes
    -----
    - The exact interpretation of *touch counts* and *support* depends on your pipeline’s
      definitions; ensure consistency with ranking/printing utilities.
    - ``coeff_sources`` is intentionally a string to keep dependencies light; consider a
      ``typing.Literal`` or Enum if you prefer stricter type checking.
    - This dataclass does not perform runtime validation; if you need guardrails
      (e.g., ``min_touch_keep >= 0``, ``max_denom >= 1``), add a ``__post_init__``.

    Examples
    --------
    >>> # Stricter settings for publication-grade runs:
    >>> GenerationConfig(
    ...     min_touch_keep=7,
    ...     max_denom=24,
    ...     min_support_const=12,
    ...     coeff_eps_match=5e-4,
    ...     coeff_sources="constants_only",
    ... )  # doctest: +ELLIPSIS
    GenerationConfig(min_touch_keep=7, max_denom=24, use_floor_ceil_if_true=True, min_support_const=12, eps_const=1e-12, coeff_eps_match=0.0005, shifts=(-2, -1, 0, 1, 2), numerators=(1, 2, 3, 4), coeff_sources='constants_only', symbolic_col_limit=10)
    """

    # Retention & coefficient search
    min_touch_keep: int = 3
    max_denom: int = 30
    use_floor_ceil_if_true: bool = True

    # Constant-bank / reciprocal generalizer
    min_support_const: int = 5
    eps_const: float = 1e-12
    coeff_eps_match: float = 1e-3
    shifts: Tuple[int, ...] = (-2, -1, 0, 1, 2)
    numerators: Tuple[int, ...] = (1, 2, 3, 4)
    coeff_sources: str = "allow_symbolic_columns"  # or "constants_only"
    symbolic_col_limit: int = 10

    def __post_init__(self):
        if self.min_touch_keep < 0:
            raise ValueError("min_touch_keep must be ≥ 0")
        if self.max_denom < 1:
            raise ValueError("max_denom must be ≥ 1")
        if self.min_support_const < 0:
            raise ValueError("min_support_const must be ≥ 0")
        if self.symbolic_col_limit < 0:
            raise ValueError("symbolic_col_limit must be ≥ 0")
