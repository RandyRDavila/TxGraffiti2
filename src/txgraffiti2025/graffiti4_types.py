# src/txgraffiti2025/graffiti4_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from txgraffiti2025.forms.utils import Expr
from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.sophie import SophieCondition


@dataclass
class HypothesisInfo:
    """
    Metadata for one hypothesis h.

    Parameters
    ----------
    name : str
        Printable name, e.g. "connected & planar".
    pred : Any
        Underlying Predicate / BoolExpr used as `Conjecture.condition`.
    mask : np.ndarray
        Boolean mask over df.index where the hypothesis holds.
    """
    name: str
    pred: Any
    mask: np.ndarray


@dataclass
class NonComparablePair:
    """
    Pair (x, y) of invariants that cross on the base universe:
    sometimes x < y and sometimes x > y.
    """
    left: Expr
    right: Expr
    left_name: str
    right_name: str


@dataclass
class Graffiti4Result:
    """
    Aggregated result of a Graffiti4.conjecture() call.
    """
    target: str
    conjectures: List[Conjecture]
    sophie_conditions: List[SophieCondition]
    stage_breakdown: Dict[str, Any]
