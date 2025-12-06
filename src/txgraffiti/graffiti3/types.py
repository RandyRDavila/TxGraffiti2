# src/txgraffiti2025/graffiti4_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from txgraffiti.graffiti3.exprs import Expr
from txgraffiti.graffiti3.relations import Conjecture

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
class SophieCondition:
    """
    A Sophie-style conjecture of the form

        (hypothesis) ⇒ (property_name)

    where the hypothesis is usually an inequality or equality between
    expressions (built from invariants), and property_name is a boolean
    property, its negation, or a conjunction of two properties.
    """
    property_name: str         # e.g. "bipartite", "¬bipartite", "bipartite & regular"
    hyp_name: str              # printed hypothesis, e.g. "independence_number < order - matching_number"
    core_hyp_name: str         # for now, same as hyp_name (used for complexity scoring)
    support_h: int             # # graphs where hypothesis holds (within base)
    coverage: int              # # target graphs covered (H ∧ P)
    target_size: int           # |P| inside base
    violations: int            # # graphs where H holds but P fails (within base)


@dataclass
class Graffiti3Result:
    """
    Aggregated result of a Graffiti4.conjecture() call.
    """
    target: str
    conjectures: List[Conjecture]
    sophie_conditions: List[SophieCondition]
    stage_breakdown: Dict[str, Any]
