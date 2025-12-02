from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Set, Any

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture


# ───────────────────────── data class ───────────────────────── #

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


# ───────────────────────── small helpers ───────────────────────── #

def _hypothesis_complexity(label: str) -> int:
    """
    Crude measure of hypothesis complexity: number of '&'-separated parts.

    For inequality hypotheses this will typically be 1, but keeping the
    helper makes scoring extensible.
    """
    if not label:
        return 0
    parts = [p.strip() for p in label.split("&") if p.strip()]
    return len(parts)


def _parse_relation_text(c: Conjecture) -> Tuple[str, str] | None:
    """
    Extract (lhs, rhs) strings from a Conjecture for labeling inequality events.

    We try to strip off the 'H ⇒' part and any trailing '[...]' metadata, and
    then look for an operator among {≤, ≥, =, <, >}. The return value is only
    used for human-readable labels; numeric evaluation uses the Expr objects.
    """
    try:
        s = c.pretty(show_tol=False)
    except Exception:
        s = str(c)

    # Remove leading hypothesis if present
    if "⇒" in s:
        _, core = s.split("⇒", 1)
        core = core.strip()
    else:
        core = s.strip()

    # Strip any trailing "[...]" block if present
    bracket_pos = core.rfind("[")
    if bracket_pos != -1:
        core = core[:bracket_pos].strip()

    for op in (" ≤ ", " ≥ ", " = ", " < ", " > "):
        if op in core:
            lhs, rhs = core.split(op, 1)
            return lhs.strip(), rhs.strip()
    return None


def _build_inequality_property_masks(
    df_num: pd.DataFrame,
    base_mask: np.ndarray,
    inequality_conjectures: Sequence[Conjecture],
    *,
    eq_tol: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """
    Construct boolean masks for inequality events derived from Conjectures.

    For each Conjecture c, we treat its relation

        L(x)  ?  R(x)

    as a *global* numeric relation over df_num and build masks inside the
    base universe:

        - L = R
        - L < R
        - L > R
        - L ≤ R  (L < R or L = R)
        - L ≥ R  (L > R or L = R)

    Each such event becomes a property-label, e.g.

        "independence_number = harmonic_index"
        "independence_number < order - matching_number"
        "independence_number ≥ (1/2) · order"

    This mirrors the inequality-based Sophie heuristic from Graffiti.pc:
    inequalities are treated as predicates on graphs, independent of the
    LP hypothesis that produced them.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping property-label → boolean mask over df_num.index.
    """
    N = len(df_num)
    base_mask = np.asarray(base_mask, dtype=bool)
    if base_mask.shape != (N,):
        raise ValueError("base_mask must have shape (len(df_num),).")

    prop_masks: Dict[str, np.ndarray] = {}

    def update_mask(label: str, mask: np.ndarray) -> None:
        """Safely append a mask to a property label."""
        if mask.any():
            existing = prop_masks.get(label)
            if existing is None:
                prop_masks[label] = mask.copy()
            else:
                existing |= mask

    for c in inequality_conjectures:
        parsed = _parse_relation_text(c)
        if parsed is None:
            continue
        lhs_text, rhs_text = parsed

        # Evaluate both sides numerically over df_num
        try:
            L_vals = c.relation.left.eval(df_num)
            R_vals = c.relation.right.eval(df_num)
        except Exception:
            # If evaluation fails for any reason, skip this conjecture
            continue

        L = np.asarray(L_vals, dtype=float)
        R = np.asarray(R_vals, dtype=float)

        if L.shape == ():
            L = np.full(N, float(L), dtype=float)
        if R.shape == ():
            R = np.full(N, float(R), dtype=float)

        if L.shape != (N,) or R.shape != (N,):
            # Unexpected shape; skip
            continue

        # Initialize masks
        eq_mask = np.zeros(N, dtype=bool)
        lt_mask = np.zeros(N, dtype=bool)
        gt_mask = np.zeros(N, dtype=bool)

        # Compute inside base universe
        bm = base_mask
        # Equality
        eq_mask[bm] = np.isclose(L[bm], R[bm], rtol=1e-9, atol=eq_tol)
        # Strict inequalities
        lt_mask[bm] = L[bm] < R[bm] - eq_tol
        gt_mask[bm] = L[bm] > R[bm] + eq_tol

        # Register events (same pattern as your old Graffiti3 code)
        update_mask(f"{lhs_text} = {rhs_text}", eq_mask)
        update_mask(f"{lhs_text} < {rhs_text}", lt_mask)
        update_mask(f"{lhs_text} > {rhs_text}", gt_mask)

        # Inclusive forms
        le_mask = eq_mask | lt_mask
        ge_mask = eq_mask | gt_mask
        update_mask(f"{lhs_text} ≤ {rhs_text}", le_mask)
        update_mask(f"{lhs_text} ≥ {rhs_text}", ge_mask)

    return prop_masks

def _property_complexity(label: str) -> int:
    """
    Complexity measure for properties P on the right-hand side:

      - count '&'-separated atoms
      - "bipartite" → 1
      - "bipartite & triangle_free" → 2
      - "¬regular" → 1 (still a single atom)

    Used to prefer simpler descriptions when two properties define the
    same subset of graphs.
    """
    if not label:
        return 0
    parts = [p.strip() for p in label.split("&") if p.strip()]
    return max(1, len(parts))


def _register_property_exprs(
    bool_df: pd.DataFrame,
    base_mask: np.ndarray,
    base_name: str,
    *,
    min_target_support: int = 5,
) -> List[Dict[str, Any]]:
    """
    Build the family of target property expressions P that Sophie will try to
    explain: single properties P, their negations ¬P, and binary conjunctions
    P1 & P2 (all restricted to the base universe).

    New: we avoid trivial conjunctions by de-duplicating by mask inside the
    base universe and keeping the *simplest* label for each mask.
    """
    base_mask = np.asarray(base_mask, dtype=bool)
    base_parts: Set[str] = {p.strip() for p in base_name.split("&") if p.strip()}

    property_exprs: List[Dict[str, Any]] = []
    atom_props: List[str] = []

    # mask → (complexity, label)
    seen_masks: Dict[bytes, Tuple[int, str]] = {}

    # Collect non-trivial boolean columns as atomic properties
    for prop in bool_df.columns:
        vals = bool_df[prop].to_numpy(dtype=bool)
        # skip universal or empty columns
        if vals.all() or (~vals).all():
            continue
        # skip columns that are already part of the base hypothesis
        if prop in base_parts:
            continue
        atom_props.append(prop)

    def _register(label: str, mask: np.ndarray, positive_atoms: Set[str]) -> None:
        # restrict to base universe
        m = np.asarray(mask, dtype=bool) & base_mask
        support = int(m.sum())
        if support < min_target_support:
            return

        key = m.tobytes()
        comp = _property_complexity(label)

        if key in seen_masks:
            old_comp, old_label = seen_masks[key]
            if comp >= old_comp:
                # existing simpler description wins
                return
            # new label is simpler: remove any old entry with old_label
            property_exprs[:] = [pe for pe in property_exprs if pe["label"] != old_label]

        seen_masks[key] = (comp, label)
        property_exprs.append(
            dict(
                label=label,
                mask=m,
                positive_atoms=set(positive_atoms),
            )
        )

    # Single properties P and negations ¬P
    for prop in atom_props:
        col = bool_df[prop].to_numpy(dtype=bool)

        # P
        _register(
            label=prop,
            mask=col,
            positive_atoms={prop},
        )

        # ¬P
        _register(
            label=f"¬{prop}",
            mask=~col,
            positive_atoms=set(),
        )

    # Binary conjunctions P1 & P2 (nontrivial only; dedup via _register)
    for i in range(len(atom_props)):
        p1 = atom_props[i]
        col1 = bool_df[p1].to_numpy(dtype=bool)
        for j in range(i + 1, len(atom_props)):
            p2 = atom_props[j]
            col2 = bool_df[p2].to_numpy(dtype=bool)
            conj_label = f"{p1} & {p2}"
            conj_mask = col1 & col2
            _register(
                label=conj_label,
                mask=conj_mask,
                positive_atoms={p1, p2},
            )

    return property_exprs


# def _register_property_exprs(
#     bool_df: pd.DataFrame,
#     base_mask: np.ndarray,
#     base_name: str,
#     *,
#     min_target_support: int = 5,
# ) -> List[Dict[str, Any]]:
#     """
#     Build the family of target property expressions P that Sophie will try to
#     explain: single properties P, their negations ¬P, and binary conjunctions
#     P1 & P2 (all restricted to the base universe).

#     This mirrors your old Graffiti3 Sophie code.
#     """
#     base_mask = np.asarray(base_mask, dtype=bool)
#     base_parts: Set[str] = {p.strip() for p in base_name.split("&") if p.strip()}

#     property_exprs: List[Dict[str, Any]] = []
#     atom_props: List[str] = []

#     # Collect non-trivial boolean columns as atomic properties
#     for prop in bool_df.columns:
#         vals = bool_df[prop].to_numpy(dtype=bool)
#         # skip universal or empty columns
#         if vals.all() or (~vals).all():
#             continue
#         # skip columns that are already part of the base hypothesis
#         if prop in base_parts:
#             continue
#         atom_props.append(prop)

#     def _add_expr(label: str, mask: np.ndarray, positive_atoms: Set[str]) -> None:
#         m = np.asarray(mask, dtype=bool) & base_mask
#         support = int(m.sum())
#         if support >= min_target_support:
#             property_exprs.append(
#                 dict(
#                     label=label,
#                     mask=m,
#                     positive_atoms=set(positive_atoms),
#                 )
#             )

#     # Single properties P and negations ¬P
#     for prop in atom_props:
#         col = bool_df[prop].to_numpy(dtype=bool)

#         # P
#         _add_expr(
#             label=prop,
#             mask=col,
#             positive_atoms={prop},
#         )

#         # ¬P
#         _add_expr(
#             label=f"¬{prop}",
#             mask=~col,
#             positive_atoms=set(),
#         )

#     # Binary conjunctions P1 & P2
#     for i in range(len(atom_props)):
#         p1 = atom_props[i]
#         col1 = bool_df[p1].to_numpy(dtype=bool)
#         for j in range(i + 1, len(atom_props)):
#             p2 = atom_props[j]
#             col2 = bool_df[p2].to_numpy(dtype=bool)
#             conj_label = f"{p1} & {p2}"
#             conj_mask = col1 & col2
#             _add_expr(
#                 label=conj_label,
#                 mask=conj_mask,
#                 positive_atoms={p1, p2},
#             )

#     return property_exprs


# ───────────────── Sophie from inequalities (original Sophie) ───────────────── #

def discover_sophie_from_inequalities(
    df_num: pd.DataFrame,
    bool_df: pd.DataFrame,
    base_mask: np.ndarray,
    base_name: str,
    inequality_conjectures: Sequence[Conjecture],
    *,
    eq_tol: float = 1e-4,
    min_target_support: int = 5,
    min_h_support: int = 3,
    max_violations: int = 0,
    min_new_coverage: int = 1,
) -> Dict[str, List[SophieCondition]]:
    """
    Original Sophie-style heuristic (DeLaViña–Waller) in the new TxGraffiti2025
    setting, using inequalities discovered by the Dalmatian / LP engines.

    Inputs
    ------
    df_num : DataFrame
        Numeric invariant table (e.g. df_eq).
    bool_df : DataFrame
        Boolean invariant table (subset of df_num columns).
    base_mask : array-like of bool
        Mask for the base universe (e.g. connected & nontrivial).
    base_name : str
        Name of the base hypothesis, used to avoid trivial properties.
    inequality_conjectures : Sequence[Conjecture]
        Conjectures whose relations L(x) ? R(x) will be turned into
        inequality events.

    Returns
    -------
    Dict[str, List[SophieCondition]]
        Mapping from property-expression label P to a list of SophieCondition
        objects of the form:

            (inequality event) ⇒ P

        The list per property is chosen greedily to cover as many instances
        of P as possible with a small set of hypotheses.
    """
    N = len(df_num)
    base_mask = np.asarray(base_mask, dtype=bool)
    if base_mask.shape != (N,):
        raise ValueError("base_mask must have shape (len(df_num),).")

    # 1️⃣ Build inequality events I: label → mask
    ineq_masks = _build_inequality_property_masks(
        df_num=df_num,
        base_mask=base_mask,
        inequality_conjectures=inequality_conjectures,
        eq_tol=eq_tol,
    )
    if not ineq_masks:
        return {}

    # 2️⃣ Build property expressions P
    property_exprs = _register_property_exprs(
        bool_df=bool_df,
        base_mask=base_mask,
        base_name=base_name,
        min_target_support=min_target_support,
    )
    if not property_exprs:
        return {}

    result: Dict[str, List[SophieCondition]] = {}

    # 3️⃣ For each property-expression P, run Sophie greedy cover
    for prop_expr in property_exprs:
        prop_label = prop_expr["label"]
        P_mask = np.asarray(prop_expr["mask"], dtype=bool)

        target_mask = P_mask & base_mask
        target_indices = np.where(target_mask)[0]
        target_size = int(target_mask.sum())
        if target_size < min_target_support:
            continue

        candidates: List[Dict[str, Any]] = []

        # Hypotheses: single inequality I
        for ineq_label, ineq_mask_full in ineq_masks.items():
            H = np.asarray(ineq_mask_full, dtype=bool) & base_mask
            support_h = int(H.sum())
            if support_h < min_h_support:
                continue

            # Violations: H ∧ ¬P
            viol_mask = H & ~target_mask
            violations = int(viol_mask.sum())
            if violations > max_violations:
                continue

            # Coverage: H ∧ P
            cover_mask = H & target_mask
            coverage = int(cover_mask.sum())
            if coverage == 0:
                continue

            candidates.append(
                dict(
                    hyp_label=ineq_label,
                    mask=H,
                    support_h=support_h,
                    coverage=coverage,
                    violations=violations,
                )
            )

        if not candidates:
            continue

        uncovered: Set[int] = set(int(i) for i in target_indices)
        accepted: List[SophieCondition] = []

        while uncovered:
            best_idx = None
            best_score = None
            best_new_covered: Set[int] = set()

            for idx, cand in enumerate(candidates):
                mask = cand["mask"] & target_mask
                cand_indices = set(int(i) for i in np.where(mask)[0])
                new_covered = cand_indices & uncovered
                new_count = len(new_covered)
                if new_count < min_new_coverage:
                    continue

                # Prefer hypotheses that cover many new targets
                # and that hold often overall (support_h).
                score = (new_count, cand["support_h"])

                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
                    best_new_covered = new_covered

            if best_idx is None:
                break

            cand = candidates[best_idx]
            accepted.append(
                SophieCondition(
                    property_name=prop_label,
                    hyp_name=cand["hyp_label"],
                    core_hyp_name=cand["hyp_label"],
                    support_h=cand["support_h"],
                    coverage=len(best_new_covered),
                    target_size=target_size,
                    violations=cand["violations"],
                )
            )
            uncovered -= best_new_covered

        if accepted:
            result[prop_label] = accepted

    return result


# ───────────────── global ranking & pretty-printing ───────────────── #

def rank_sophie_conditions_global(
    by_property: Dict[str, List[SophieCondition]],
) -> List[SophieCondition]:
    """
    Flatten a dict {property_name: [SophieCondition, ...]} into a single
    globally-ranked list.

    Ranking follows your old Graffiti3 heuristic:

      1. coverage fraction  = coverage / target_size
      2. coverage (absolute)
      3. support_h
      4. simpler core_hyp_name (fewer conjuncts) preferred
    """
    all_conds: List[SophieCondition] = []
    for conds in by_property.values():
        all_conds.extend(conds)

    # Deduplicate by (property, hyp_name, core_hyp_name)
    seen = set()
    unique: List[SophieCondition] = []
    for sc in all_conds:
        key = (sc.property_name, sc.hyp_name, sc.core_hyp_name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sc)

    def score(sc: SophieCondition):
        frac = (sc.coverage / sc.target_size) if sc.target_size > 0 else 0.0
        comp = _hypothesis_complexity(sc.core_hyp_name)
        return (frac, sc.coverage, sc.support_h, -comp)

    unique.sort(key=score, reverse=True)
    return unique


def _sophie_is_biconditional(cond: SophieCondition) -> bool:
    """
    Detect empirical biconditionals:

        P ⇔ H   if, inside the base universe,
        H holds for exactly the same rows as P.
    """
    return (
        cond.violations == 0
        and cond.coverage == cond.target_size
        and cond.support_h == cond.target_size
    )


def _sophie_arrow(cond: SophieCondition) -> str:
    """Arrow to print for a SophieCondition."""
    return "⇔" if _sophie_is_biconditional(cond) else "⇒"


def print_sophie_conditions(
    conditions: List[SophieCondition],
    top_n: int = 15,
) -> None:
    """
    Human-readable pretty-printer for a list of SophieCondition objects.

    This should produce lines like:

        Sophie Condition 1. (independence_number = harmonic_index) ⇔ bipartite & regular
        Sophie Condition 2. (independence_number < order - matching_number) ⇒ ¬bipartite
    """
    print()
    print("=== Sophie-style conditions ===\n")

    if not conditions:
        print("(none)")
        return

    for i, sc in enumerate(conditions[:top_n], 1):
        prop = sc.property_name
        cov = sc.coverage
        tgt = sc.target_size
        pct = 100.0 * cov / tgt if tgt > 0 else 0.0

        arrow = _sophie_arrow(sc)

        print(
            f"Sophie Condition {i}. ({sc.hyp_name}) {arrow} {prop}   "
            f"[coverage = {cov}/{tgt} ({pct:.1f}%), "
            f"support_H = {sc.support_h}, "
            f"violations = {sc.violations}]"
        )
        print()
