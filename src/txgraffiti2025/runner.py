
from __future__ import annotations

"""
Main public runner for the TxGraffiti 2025 conjecturing pipeline.

This module wires together:

  * GraffitiClassRelations (boolean + numeric feature management),
  * LP-based affine and k-affine bounds,
  * integer-aware lifting,
  * intricate mixed inequalities,
  * equality-class bootstrap,
  * Sophie-style inequality ⇒ Boolean conditions,
  * Morgan-upwards hypothesis generalization,
  * asymptotic and inclusion-style structure,
  * and finalization/ranking into a conjecture bank.

The primary entry point is TxGraffitiRunner.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from txgraffiti2025.graffiti_relations import GraffitiClassRelations
from txgraffiti2025.graffiti_lp import GraffitiLP, LPFitConfig, _to_expr
from txgraffiti2025.graffiti_lp_lift_integer_aware import lift_integer_aware
from txgraffiti2025.graffiti_intricate_mixed import GraffitiLPIntricate
from txgraffiti2025.graffiti_qualitative import GraffitiQualitative
from txgraffiti2025.graffiti_asymptotic_miner import (
    AsymptoticMiner,
    AsymptoticSearchConfig,
)
from txgraffiti2025.finalization import finalize_conjecture_bank, print_bank
from txgraffiti2025.reporting import tee_report
from txgraffiti2025.forms.generic_conjecture import Conjecture

# Sophie-style conditions
from txgraffiti2025.sophie import (
    SophieCondition,
    discover_sophie_from_inequalities,
    print_sophie_conditions,
    rank_sophie_conditions_global,
)

# ───────────────────── Expr helpers ───────────────────── #

def _expr_depends_on(expr, needle) -> bool:
    """
    Return True iff `expr`'s expression tree contains `needle` as a subexpression.

    We rely on structural equality (`==`) and/or identity (`is`) and recurse over
    the Expr's children (assumed to live in `expr.args` or similar).
    """
    # Direct hit
    if expr is needle or expr == needle:
        return True

    # No children → cannot contain needle
    children = getattr(expr, "args", None)
    if not children:
        return False

    # Recurse
    for child in children:
        if _expr_depends_on(child, needle):
            return True
    return False


def _is_minmax_involving_target(expr, target_expr) -> bool:
    """
    Return True iff `expr` is a min/max Expr whose subtree contains `target_expr`.
    """
    op = getattr(expr, "name", None)  # or .op/.kind depending on your Expr class
    if op not in {"min", "max"}:
        return False
    return _expr_depends_on(expr, target_expr)


import re
from txgraffiti2025.forms.utils import Expr  # if not already imported


def _expr_uses_minmax_of_target(expr: Expr, target_expr: Expr, target_col: str) -> bool:
    """
    Return True iff `expr` represents min(target_col, ·) or max(target_col, ·)
    (or the target as the second argument), either as a Func2Op or as a
    ColumnTerm/other Expr whose pretty() string is "min(x, y)" / "max(x, y)".
    """

    # 1) Func2Op-style: Expr has a 'name' and left/right attributes
    op = getattr(expr, "name", None)
    if op in {"min", "max"}:
        # recursively check if subtree contains target_expr
        def _depends_on(e: Expr) -> bool:
            if e is target_expr or e == target_expr:
                return True
            for attr in ("left", "right", "args"):
                child = getattr(e, attr, None)
                if child is None:
                    continue
                if isinstance(child, Expr):
                    if _depends_on(child):
                        return True
                elif isinstance(child, (list, tuple)):
                    if any(isinstance(c, Expr) and _depends_on(c) for c in child):
                        return True
            return False

        if _depends_on(expr):
            return True

    # 2) String-based: pretty() looks like min(a, b) or max(a, b)
    s = expr.pretty()
    m = re.match(r"^(min|max)\((.*)\)$", s)
    if not m:
        return False

    inner = m.group(2)
    if "," not in inner:
        return False
    left_str, right_str = inner.split(",", 1)
    left = left_str.strip()
    right = right_str.strip()

    # Compare to the raw target column name, e.g. "independence_number"
    return (left == target_col) or (right == target_col)


# ───────────────────────── dataclasses ───────────────────────── #

@dataclass
class TxGraffitiConfig:
    """
    Configuration for a single TxGraffiti run on a given target invariant.
    """

    # Required: which column are we conjecturing about?
    target: str

    # Equality bootstrap
    enable_eq_bootstrap: bool = True
    eq_top_k: int = 12
    eq_min_rows: int = 25
    ratio_min_support: float = 0.05
    ratio_pos_denominator: bool = True
    ratio_touch_atol: float = 0.0
    ratio_touch_rtol: float = 0.0

    # Asymptotic search
    asym_min_abs_rho: float = 0.45
    asym_tail_quantile: float = 0.75
    asym_min_support_n: int = 20

    # k-affine LP search
    k_affine_values: Tuple[int, ...] = (1, 2, 3)
    k_affine_hypotheses_limit: int = 20
    k_affine_min_touch: int = 3
    k_affine_max_denom: int = 20
    k_affine_top_m_by_variance: int = 10

    # affine LP
    affine_max_denom: int = 20

    # intricate mixed inequalities
    intricate_weight: float = 0.5
    intricate_min_touch: int = 10

    # final conjecture bank
    # If None, keep all conjectures in each bucket.
    top_k_per_bucket: Optional[int] = 100
    apply_morgan: bool = True

    # optional post-prune hook
    #   signature: post_prune(df, lowers, uppers, equals) -> (lowers, uppers, equals)
    post_prune: Optional[Any] = None

    # ── Sophie-style inequality ⇒ Boolean conditions ──
    enable_sophie: bool = True
    sophie_min_target_support: int = 5
    sophie_min_h_support: int = 3
    sophie_max_violations: int = 0
    sophie_min_new_coverage: int = 1
    # tolerance for detecting "inequality event" (e.g. near-equality)
    sophie_eq_tol: float = 1e-4
    # cap on how many Sophie conditions are kept *globally*
    sophie_max_conditions: int = 20


def _generalize_to_base(
    df: pd.DataFrame,
    conjs: List[Conjecture],
    *,
    verbose: bool = False,
    progress: bool = False,
) -> List[Conjecture]:
    """
    For each conjecture, try to weaken its hypothesis to the automatic base
    (condition=None, auto_base=True) while preserving truth.

    If the generalized conjecture is true on the base universe, we keep the
    generalized version; otherwise we keep the original.

    Touch counts and support_n are recomputed for the chosen condition.

    Parameters
    ----------
    df : DataFrame
        Data on which conjectures are checked.
    conjs : list[Conjecture]
        Conjectures to generalize.
    verbose : bool, default=False
        If True, print each successful Morgan-upwards generalization in full
        (old ⇒ new, touches, support).
    progress : bool, default=False
        If True, print a lightweight progress indicator instead of all details.
    """
    out: List[Conjecture] = []
    n = len(conjs)

    if progress and n > 0:
        print(f"[MORGAN-UPWARDS] Generalizing {n} conjectures to the base...", flush=True)

    for idx, c in enumerate(conjs, 1):
        # Optional progress ticker (no conjecture text)
        if progress and (idx % max(1, n // 10) == 0 or idx == n):
            print(f"[MORGAN-UPWARDS] {idx}/{n} conjectures processed...", end="\r", flush=True)

        # If there is no explicit condition already, nothing to weaken.
        if c.condition is None:
            applicable, _, _ = c.check(df, auto_base=True)
            Conjecture.touch_count(c, df, auto_base=True)
            setattr(c, "support_n", int(applicable.sum()))
            out.append(c)
            continue

        orig_str = c.pretty()

        # Candidate with base (auto) condition
        gen = Conjecture(
            relation=c.relation,
            condition=None,  # auto_base will infer the base predicate
            name=getattr(c, "name", "Conjecture"),
        )

        # Copy any extra metadata that downstream code might rely on
        for attr in ("coefficient_pairs", "intercept"):
            if hasattr(c, attr):
                setattr(gen, attr, getattr(c, attr))

        # Check truth on the base universe
        if gen.is_true(df, auto_base=True):
            applicable, _, _ = gen.check(df, auto_base=True)
            Conjecture.touch_count(gen, df, auto_base=True)
            setattr(gen, "support_n", int(applicable.sum()))

            if verbose:
                gen_str = gen.pretty()
                if gen_str != orig_str:
                    print(
                        "[MORGAN-UPWARDS] Generalized\n"
                        f"  {orig_str}  \n"
                        "  ↦\n"
                        f"  {gen_str}  \n"
                        f"  (touches={getattr(gen, 'touch_count', '?')}, "
                        f"support={getattr(gen, 'support_n', '?')})\n"
                    )

            out.append(gen)
        else:
            # Cannot weaken all the way to base; keep original.
            applicable, _, _ = c.check(df, auto_base=True)
            Conjecture.touch_count(c, df, auto_base=True)
            setattr(c, "support_n", int(applicable.sum()))
            out.append(c)

    if progress and n > 0:
        print(f"[MORGAN-UPWARDS] Done. {n} conjectures processed.        ")

    return out


@dataclass
class TxGraffitiResult:
    """
    Container for TxGraffiti run results.

    Attributes
    ----------
    final_bank : dict[str, list[Conjecture]]
        Ranked numeric conjectures (lowers/uppers/equals) after Morgan, Dalmatian, etc.
    all_lowers, all_uppers, all_equals : list[Conjecture]
        Full lists of numeric conjectures seen before final banking.
    df_final : DataFrame or None
        Final DataFrame used for evaluation (possibly augmented with eq-classes).
    extras : dict[str, Any]
        Structured "side channels" of the theory:
          - qualitative
          - asymptotic
          - atomic
          - equality_classes
          - class_characterization
          - sophie
          - sophie_conditions
          - etc.
    """

    final_bank: Dict[str, List[Conjecture]]
    all_lowers: List[Conjecture] = field(default_factory=list)
    all_uppers: List[Conjecture] = field(default_factory=list)
    all_equals: List[Conjecture] = field(default_factory=list)
    df_final: Optional[pd.DataFrame] = None
    extras: Dict[str, Any] = field(default_factory=dict)


# ───────────────── equality-class helpers (internal) ───────────────── #

def _top_equality_classes(
    gcr_obj: GraffitiClassRelations,
    eq_summary: pd.DataFrame,
    *,
    k: int = 12,
    min_rows: int = 25,
    side: str = "auto",
) -> List[tuple[str, Any, float, int]]:
    """
    Select top equality classes from an analyze_ratio_bounds_on_base() summary.

    Returns
    -------
    list of tuples
        (name, Predicate, best_rate, n_rows)
    """
    tmp = eq_summary.copy()
    if tmp.empty:
        return []

    tmp["best_rate"] = np.where(
        tmp["touch_lower_rate"] >= tmp["touch_upper_rate"],
        tmp["touch_lower_rate"],
        tmp["touch_upper_rate"],
    )
    tmp["best_side"] = np.where(
        tmp["touch_lower_rate"] >= tmp["touch_upper_rate"],
        "lower",
        "upper",
    )
    tmp["score"] = tmp["best_rate"] * np.log1p(tmp["n_rows"])

    tmp = tmp[tmp["n_rows"] >= min_rows]
    if tmp.empty:
        return []

    # Avoid duplicates: treat (inv1, inv2) as an unordered pair
    tmp["_pairkey"] = tmp.apply(
        lambda r: tuple(sorted([r["inv1"], r["inv2"]])),
        axis=1,
    )
    tmp = tmp.sort_values(["score"], ascending=False).drop_duplicates(
        "_pairkey",
        keep="first",
    )
    sel = tmp.head(k)

    picks: List[tuple[str, Any, float, int]] = []
    for _, row in sel.iterrows():
        which = row["best_side"] if side == "auto" else side
        name, pred = gcr_obj.spawn_equality_classes_from_ratio_row(
            row,
            which=which,
            tol=0.0,
        )
        picks.append((name, pred, float(row["best_rate"]), int(row["n_rows"])))
    return picks


def _build_df_with_eq_booleans(
    df: pd.DataFrame,
    gcr: GraffitiClassRelations,
    top_eqs,
) -> pd.DataFrame:
    """
    Return a new DataFrame with extra boolean columns for each equality-class
    predicate in `top_eqs`.
    """
    df_eq = df.copy()
    print("Selected equality classes:")
    for name, pred, rate, nrows in top_eqs:
        mask = gcr._mask_cached(pred)
        colname = name
        df_eq[colname] = mask
        print(
            f"  • {name:40s} (tightness≈{rate:.3f}, n={nrows})  "
            f"→ added as boolean '{colname}'"
        )
    return df_eq


def _print_conjs(label: str, conjs: List[Conjecture], n: int = 6) -> None:
    """
    Small helper for printing a sample of conjectures.
    """
    print(f"[{label.upper()}] total={len(conjs)}\n")
    for c in conjs[:n]:
        try:
            s = c.pretty(show_tol=False)
        except Exception:
            s = str(c)
        print("•", s)
        t = getattr(c, "touch_count", "?")
        s_n = getattr(c, "support_n", "?")
        print(f"    touches={t}, support={s_n}\n")


# ───────────────────────── main runner class ───────────────────────── #

class TxGraffitiRunner:
    """
    Public runner for the TxGraffiti conjecturing pipeline.

    This wraps the internal engines (GraffitiClassRelations, LP, qualitative,
    asymptotics, intricate mixed, equality-class bootstrap, and Sophie-style
    inequality ⇒ Boolean discovery) and produces a ranked bank of conjectures
    for a chosen target invariant, plus structured "side channels" in `extras`.
    """

    def __init__(
        self,
        config: TxGraffitiConfig,
        *,
        report_path: Optional[str] = None,
        report_title: str = "TxGraffiti • Full Program Run Report",
    ) -> None:
        self.config = config
        self.report_path = report_path
        self.report_title = report_title

    # ---- public method ----

    def run(self, df: pd.DataFrame) -> TxGraffitiResult:
        """
        Run the full pipeline on `df` and return a TxGraffitiResult.

        If report_path was provided at construction time, all prints from the
        run are mirrored into a text file at that path.
        """
        if self.report_path:
            with tee_report(self.report_path, title=self.report_title):
                return self._run_inner(df)
        else:
            return self._run_inner(df)

    # ---- internal core ----

    def _run_inner(self, df: pd.DataFrame) -> TxGraffitiResult:
        cfg = self.config
        target = cfg.target

        ALL_LOWERS: List[Conjecture] = []
        ALL_UPPERS: List[Conjecture] = []
        ALL_EQUALS: List[Conjecture] = []
        extras: Dict[str, Any] = {}

        # --- Engines on the original df ---
        gcr = GraffitiClassRelations(df)
        qual = GraffitiQualitative(gcr)

        # Qualitative relations (light preview)
        print("\n" + "-" * 80)
        print("QUALITATIVE RELATIONS (top 12)")
        print("-" * 80 + "\n")

        qual_res = qual.generate_qualitative_relations(
            y_targets=[target],
            method="spearman",
            min_abs_rho=0.4,
            min_n=12,
            top_k_per_hyp=5,
        )
        GraffitiQualitative.print_sample(qual_res, k=12)
        extras["qualitative"] = qual_res

        # Asymptotics from qualitative signals
        print("\n" + "-" * 80)
        print("ASYMPTOTIC LIMITS (qualitative → definite)")
        print("-" * 80 + "\n")

        miner = AsymptoticMiner(
            gcr,
            cfg=AsymptoticSearchConfig(
                min_abs_rho=cfg.asym_min_abs_rho,
                tail_quantile=cfg.asym_tail_quantile,
                min_support_n=cfg.asym_min_support_n,
            ),
        )
        asym_conjs = miner.generate_asymptotics_for_target(
            target=target,
            hyps=getattr(gcr, "nonredundant_conjunctions_", None),
        )
        extras["asymptotic"] = asym_conjs

        print(f"[ASYMPTOTIC] total={len(asym_conjs)}\n")
        for c in asym_conjs[:10]:
            print("•", c.pretty())

        # GCR summary
        print("\n" + "-" * 80)
        print("GRAFFITICLASSRELATIONS SUMMARY")
        print("-" * 80 + "\n")

        # ensure fresh init
        gcr = GraffitiClassRelations(df)
        print("Boolean columns:", gcr.boolean_cols)
        print("Expr columns:", gcr.expr_cols)
        print("Base hypothesis:", gcr.base_hypothesis_name)

        gcr.enumerate_conjunctions(max_arity=2)
        nonred, _, _ = gcr.find_redundant_conjunctions()
        print("\nNonredundant conjunctions:")
        for n, _ in nonred[:8]:
            print(" ", n)
        print()

        atomic = gcr.build_constant_conjectures(tol=0.0, group_per_hypothesis=False)
        extras["atomic"] = atomic

        print("Example atomic conjectures:\n")
        for c in atomic[:5]:
            from textwrap import indent
            print(indent(c.pretty(), "  "))
            print()

        # Characterizations / inclusion-ish information
        const_summary = gcr.characterize_constant_classes(
            tol=0.0,
            group_per_hypothesis=True,
            limit=50,
        )
        extras["class_characterization"] = const_summary
        gcr.print_class_characterization_summary()

        # ───────────────── AFFINE LP FITS (original df) ─────────────────
        print("\n" + "-" * 80)
        print("AFFINE LP FITS")
        print("-" * 80 + "\n")

        lp = GraffitiLP(gcr)

        # Canonical target Expr used by GraffitiLP
        target_expr = _to_expr(target, lp.exprs)
        target_repr = repr(target_expr)

        # Start by excluding the target itself as a feature
        raw_features = [e for e in lp.invariants if repr(e) != target_repr]

        # Further exclude any min/max Expr whose arguments involve the target
        # (via Func2Op or string-based ColumnTerm, depending on how the helper is defined)
        features = [
            e
            for e in raw_features
            if not _expr_uses_minmax_of_target(e, target_expr, target)
        ]

        # Make sure *all* downstream LP fits (including k-affine) use this pool
        lp.invariants = features

        fit_cfg = LPFitConfig(
            target=target_expr,           # pass Expr, not raw string
            features=features,            # Exprs with target removed
            direction="both",
            max_denom=cfg.affine_max_denom,
        )

        lowers, uppers, equals = lp.fit_affine(fit_cfg)
        ALL_LOWERS.extend(lowers)
        ALL_UPPERS.extend(uppers)
        ALL_EQUALS.extend(equals)

        # ───────────────── K-AFFINE BOUNDS (original df) ─────────────────
        print("\n" + "-" * 80)
        print("K-AFFINE BOUNDS")
        print("-" * 80 + "\n")

        res_k = lp.generate_k_affine_bounds(
            target=target_expr,
            k_values=cfg.k_affine_values,
            hypotheses_limit=cfg.k_affine_hypotheses_limit,
            min_touch=cfg.k_affine_min_touch,
            max_denom=cfg.k_affine_max_denom,
            top_m_by_variance=cfg.k_affine_top_m_by_variance,
        )
        ALL_LOWERS.extend(res_k["lowers"])
        ALL_UPPERS.extend(res_k["uppers"])
        ALL_EQUALS.extend(res_k["equals"])

        # ───────────────── INTEGER-AWARE LIFTING ─────────────────
        print("\n" + "-" * 80)
        print("INTEGER-AWARE LIFTING")
        print("-" * 80 + "\n")

        low1 = lift_integer_aware(df=lp.df, gcr=lp.gcr, conjectures=res_k["lowers"])
        up1 = lift_integer_aware(df=lp.df, gcr=lp.gcr, conjectures=res_k["uppers"])
        print(f"Before vs After (lowers): {len(res_k['lowers'])} → {len(low1)}")
        print(f"Before vs After (uppers): {len(res_k['uppers'])} → {len(up1)}\n")

        for old, new in zip(res_k["lowers"], low1):
            if old.signature() != new.signature():
                print("Refined:", new.pretty(), f"| touches={getattr(new, 'touch_count', '?')}\n")

        ALL_LOWERS.extend(low1)
        ALL_UPPERS.extend(up1)

        # ───────────────── INTRICATE MIXED INEQUALITIES (original df) ─────────────────
        print("\n" + "-" * 80)
        print("INTRICATE MIXED INEQUALITIES")
        print("-" * 80 + "\n")

        lp_intr = GraffitiLPIntricate(df)
        print("Boolean columns:", lp_intr.bool_columns)
        print("Numeric columns:", lp_intr.numeric_columns[:8], "...")
        print("Base hypothesis:", lp_intr.base_hyp, "\n")

        res_intr = lp_intr.run_intricate_mixed_pipeline(
            target_col=target,
            weight=cfg.intricate_weight,
            min_touch=cfg.intricate_min_touch,
        )
        ALL_LOWERS.extend(res_intr["lowers"])
        ALL_UPPERS.extend(res_intr["uppers"])
        ALL_EQUALS.extend(res_intr["equals"])

        # Optional post-prune before equality bootstrap
        if cfg.post_prune is not None:
            ALL_LOWERS, ALL_UPPERS, ALL_EQUALS = cfg.post_prune(
                df,
                ALL_LOWERS,
                ALL_UPPERS,
                ALL_EQUALS,
            )

        # ───────────────── Equality-class bootstrap pass (optional) ─────────────────
        df_eq = df
        gcr_eq: Optional[GraffitiClassRelations] = None

        if cfg.enable_eq_bootstrap:
            print("\n" + "-" * 80)
            print("EQUALITY-CLASS HYPOTHESES BOOTSTRAP → REBUILD ENGINES")
            print("-" * 80 + "\n")

            summary, _base_conjs = gcr.analyze_ratio_bounds_on_base(
                min_support=cfg.ratio_min_support,
                positive_denominator=cfg.ratio_pos_denominator,
                touch_atol=cfg.ratio_touch_atol,
                touch_rtol=cfg.ratio_touch_rtol,
            )
            top_eqs = _top_equality_classes(
                gcr,
                summary,
                k=cfg.eq_top_k,
                min_rows=cfg.eq_min_rows,
                side="auto",
            )
            extras["equality_classes"] = top_eqs

            df_eq = _build_df_with_eq_booleans(df, gcr, top_eqs)

            # Rebuild engines on df_eq
            gcr_eq = GraffitiClassRelations(df_eq)
            lp_eq = GraffitiLP(gcr_eq)
            lp_intr_eq = GraffitiLPIntricate(df_eq)

            # Canonical target Expr in the equality-bootstrapped frame
            target_expr_eq = _to_expr(target, lp_eq.exprs)
            target_repr_eq = repr(target_expr_eq)

            raw_features_eq = [
                e for e in lp_eq.invariants
                if repr(e) != target_repr_eq
            ]

            features_eq = [
                e
                for e in raw_features_eq
                if not _expr_uses_minmax_of_target(e, target_expr_eq, target)
            ]

            # Again, ensure all LP fits in the eq-bootstrapped frame use filtered pool
            lp_eq.invariants = features_eq

            print("\nBoolean columns in rebuilt frame (includes equality-class hypotheses):")
            print(gcr_eq.boolean_cols)

            print("\n" + "-" * 80)
            print("AFFINE LP FITS (equality-class hypotheses)")
            print("-" * 80 + "\n")

            cfg_eq = LPFitConfig(
                target=target_expr_eq,
                features=features_eq,
                direction="both",
                max_denom=cfg.affine_max_denom,
            )
            lowers_eq, uppers_eq, equals_eq = lp_eq.fit_affine(cfg_eq)
            ALL_LOWERS.extend(lowers_eq)
            ALL_UPPERS.extend(uppers_eq)
            ALL_EQUALS.extend(equals_eq)

            print("\n" + "-" * 80)
            print("K-AFFINE BOUNDS (equality-class hypotheses)")
            print("-" * 80 + "\n")

            res_eq = lp_eq.generate_k_affine_bounds(
                target=target_expr_eq,
                k_values=cfg.k_affine_values,
                hypotheses_limit=cfg.k_affine_hypotheses_limit,
                min_touch=cfg.k_affine_min_touch,
                max_denom=cfg.k_affine_max_denom,
                top_m_by_variance=cfg.k_affine_top_m_by_variance,
            )
            ALL_LOWERS.extend(res_eq["lowers"])
            ALL_UPPERS.extend(res_eq["uppers"])
            ALL_EQUALS.extend(res_eq["equals"])

            print("\n" + "-" * 80)
            print("INTRICATE MIXED INEQUALITIES (equality-class hypotheses)")
            print("-" * 80 + "\n")

            res_intr_eq = lp_intr_eq.run_intricate_mixed_pipeline(
                target_col=target,
                weight=cfg.intricate_weight,
                min_touch=cfg.intricate_min_touch,
            )
            ALL_LOWERS.extend(res_intr_eq["lowers"])
            ALL_UPPERS.extend(res_intr_eq["uppers"])
            ALL_EQUALS.extend(res_intr_eq["equals"])

            # Optional post-prune after equality bootstrap
            if cfg.post_prune is not None:
                ALL_LOWERS, ALL_UPPERS, ALL_EQUALS = cfg.post_prune(
                    df_eq,
                    ALL_LOWERS,
                    ALL_UPPERS,
                    ALL_EQUALS,
                )

        # ───────────────── Optional: global safety filter against min/max(target,·) ─────────────────
        def _conj_uses_minmax_of_target(c: Conjecture, target_col: str) -> bool:
            try:
                s = c.pretty(show_tol=False)
            except TypeError:
                s = c.pretty()
            except Exception:
                s = str(c)
            # crude but robust string test: min(… target_col …) or max(… target_col …)
            import re
            pattern = rf"(min|max)\([^)]*{re.escape(target_col)}[^)]*\)"
            return re.search(pattern, s) is not None

        ALL_LOWERS = [c for c in ALL_LOWERS if not _conj_uses_minmax_of_target(c, target)]
        ALL_UPPERS = [c for c in ALL_UPPERS if not _conj_uses_minmax_of_target(c, target)]
        ALL_EQUALS = [c for c in ALL_EQUALS if not _conj_uses_minmax_of_target(c, target)]

        # ───────────────── Sophie-style inequality ⇒ Boolean conditions ────────────────
        if cfg.enable_sophie:
            print("\n" + "-" * 80)
            print("SOPHIE-STYLE INEQUALITY ⇒ BOOLEAN CONDITIONS")
            print("-" * 80 + "\n")

            # Decide which frame Sophie should see
            if cfg.enable_eq_bootstrap and gcr_eq is not None:
                gcr_for_sophie = gcr_eq
                df_for_sophie = df_eq
            else:
                gcr_for_sophie = gcr
                df_for_sophie = df

            # Base universe + boolean property DataFrame
            base_mask_raw = gcr_for_sophie._mask_cached(gcr_for_sophie.base_hypothesis)
            base_mask = np.asarray(base_mask_raw, dtype=bool)

            bool_cols = gcr_for_sophie.boolean_cols
            bool_df = df_for_sophie[bool_cols].copy()

            inequality_conjectures = ALL_LOWERS + ALL_UPPERS + ALL_EQUALS

            sophie_by_prop = discover_sophie_from_inequalities(
                df_num=df_for_sophie,
                bool_df=bool_df,
                base_mask=base_mask,
                base_name=gcr_for_sophie.base_hypothesis_name,
                inequality_conjectures=inequality_conjectures,
                eq_tol=cfg.sophie_eq_tol,
                min_target_support=cfg.sophie_min_target_support,
                min_h_support=cfg.sophie_min_h_support,
                max_violations=cfg.sophie_max_violations,
                min_new_coverage=cfg.sophie_min_new_coverage,
            )

            extras["sophie"] = sophie_by_prop

            # Global ranking
            all_conds_sorted = rank_sophie_conditions_global(sophie_by_prop)

            if all_conds_sorted:
                max_conds = cfg.sophie_max_conditions
                if max_conds is not None and max_conds > 0:
                    all_conds_sorted = all_conds_sorted[:max_conds]

                extras["sophie_conditions"] = all_conds_sorted

                print_sophie_conditions(
                    all_conds_sorted,
                    top_n=min(15, len(all_conds_sorted)),
                )
            else:
                print("(no Sophie-style Boolean ⇒ Boolean conditions discovered)\n")

        # ───────────────── Morgan-upwards: generalize hypotheses to the base when possible ──
        print("\n" + "-" * 80)
        print("MORGAN-UPWARDS: GENERALIZING HYPOTHESES TO BASE WHEN POSSIBLE")
        print("-" * 80 + "\n")

        ALL_LOWERS = _generalize_to_base(df_eq, ALL_LOWERS, verbose=False, progress=True)
        ALL_UPPERS = _generalize_to_base(df_eq, ALL_UPPERS, verbose=False, progress=True)
        ALL_EQUALS = _generalize_to_base(df_eq, ALL_EQUALS, verbose=False, progress=True)

        # ───────────────── Finalize on df_eq (it contains all needed columns now) ─────
        final_bank = finalize_conjecture_bank(
            df_eq,
            ALL_LOWERS,
            ALL_UPPERS,
            ALL_EQUALS,
            top_k_per_bucket=cfg.top_k_per_bucket,
            apply_morgan=cfg.apply_morgan,
        )

        print_bank(
            final_bank,
            k_per_bucket=cfg.top_k_per_bucket,
            title="FULL CONJECTURE LIST (DEDUPED • RANKED)",
        )
        # Record the target so printers can annotate sections uniformly
        extras.setdefault("target", target)

        return TxGraffitiResult(
            final_bank=final_bank,
            all_lowers=ALL_LOWERS,
            all_uppers=ALL_UPPERS,
            all_equals=ALL_EQUALS,
            df_final=df_eq,
            extras=extras,
        )



from typing import Optional

from txgraffiti2025.sophie import (
    SophieCondition,
    rank_sophie_conditions_global,
    print_sophie_conditions,
)


def print_full_result(
    result: TxGraffitiResult,
    *,
    k_per_bucket: int = 20,
    top_sophie: int = 20,
    max_atomic: int = 20,
    max_asymptotic: int = 20,
    max_eq_classes: int = 20,
) -> None:
    """
    Pretty-print a TxGraffitiResult in a uniform, numbered style.

    Sections
    --------
    1. Numeric conjectures (lowers / uppers / equals) from result.final_bank,
       printed as:
           Lower Conjecture i (target = y).
             h ⇒ y ⋈ RHS
             touches=..., support=...

    2. Early atomic (constant) conjectures stored in result.extras["atomic"].

    3. Asymptotic relations stored in result.extras["asymptotic"].

    4. Equality-class hypotheses stored in result.extras["equality_classes"].

    5. Sophie-style conditions stored in result.extras["sophie_conditions"]
       (or reconstructed from result.extras["sophie"]).
    """
    target_name: str = result.extras.get("target", "<?>")

    # Helper: safe pretty string
    def _pretty(c: Conjecture) -> str:
        try:
            return c.pretty(show_tol=False)
        except TypeError:
            # if pretty() doesn't take show_tol
            return c.pretty()
        except Exception:
            return str(c)

    # 1️⃣ Numeric conjectures from final_bank
    lowers = result.final_bank.get("lowers", [])
    uppers = result.final_bank.get("uppers", [])
    equals = result.final_bank.get("equals", [])

    print("\n" + "-" * 80)
    print(f"FULL CONJECTURE LIST (DEDUPED • RANKED)  [target = {target_name}]")
    print("-" * 80 + "\n")

    # -- LOWERS --
    if lowers:
        print(f"[LOWERS on {target_name}] total={len(lowers)} (showing up to {k_per_bucket})\n")
        n_show = len(lowers) if (k_per_bucket is None or k_per_bucket <= 0) else min(len(lowers), k_per_bucket)
        for i, c in enumerate(lowers[:n_show], 1):
            s = _pretty(c)
            t = getattr(c, "touch_count", getattr(c, "touches", "?"))
            s_n = getattr(c, "support_n", getattr(c, "support", "?"))
            print(f"Lower Conjecture {i} (target = {target_name}).")
            print(f"  {s}")
            print(f"    touches={t}, support={s_n}\n")

    # -- UPPERS --
    if uppers:
        print(f"[UPPERS on {target_name}] total={len(uppers)} (showing up to {k_per_bucket})\n")
        n_show = len(uppers) if (k_per_bucket is None or k_per_bucket <= 0) else min(len(uppers), k_per_bucket)
        for i, c in enumerate(uppers[:n_show], 1):
            s = _pretty(c)
            t = getattr(c, "touch_count", getattr(c, "touches", "?"))
            s_n = getattr(c, "support_n", getattr(c, "support", "?"))
            print(f"Upper Conjecture {i} (target = {target_name}).")
            print(f"  {s}")
            print(f"    touches={t}, support={s_n}\n")

    # -- EQUALS --
    if equals:
        print(f"[EQUALITIES on {target_name}] total={len(equals)} (showing up to {k_per_bucket})\n")
        n_show = len(equals) if (k_per_bucket is None or k_per_bucket <= 0) else min(len(equals), k_per_bucket)
        for i, c in enumerate(equals[:n_show], 1):
            s = _pretty(c)
            t = getattr(c, "touch_count", getattr(c, "touches", "?"))
            s_n = getattr(c, "support_n", getattr(c, "support", "?"))
            print(f"Equality Conjecture {i} (target = {target_name}).")
            print(f"  {s}")
            print(f"    touches={t}, support={s_n}\n")

    # 2️⃣ Atomic (constant) conjectures
    atomic = result.extras.get("atomic", [])
    if atomic:
        print("\n" + "-" * 80)
        print(f"EARLY CONSTANT / ATOMIC RELATIONS (showing up to {max_atomic})")
        print("-" * 80 + "\n")
        n_show = len(atomic) if (max_atomic is None or max_atomic <= 0) else min(len(atomic), max_atomic)
        for i, c in enumerate(atomic[:n_show], 1):
            s = _pretty(c)
            print(f"Atomic Relation {i}.")
            print(f"  {s}\n")

    # 3️⃣ Asymptotic relations
    asym_conjs = result.extras.get("asymptotic", [])
    if asym_conjs:
        print("\n" + "-" * 80)
        print(f"ASYMPTOTIC RELATIONS on {target_name} (showing up to {max_asymptotic})")
        print("-" * 80 + "\n")
        n_show = len(asym_conjs) if (max_asymptotic is None or max_asymptotic <= 0) else min(len(asym_conjs), max_asymptotic)
        for i, c in enumerate(asym_conjs[:n_show], 1):
            s = _pretty(c)
            print(f"Asymptotic Relation {i} (target = {target_name}).")
            print(f"  {s}\n")

    # 4️⃣ Equality-class hypotheses
    eq_classes = result.extras.get("equality_classes", [])
    if eq_classes:
        print("\n" + "-" * 80)
        print(f"EQUALITY-CLASS HYPOTHESES (showing up to {max_eq_classes})")
        print("-" * 80 + "\n")
        n_show = len(eq_classes) if (max_eq_classes is None or max_eq_classes <= 0) else min(len(eq_classes), max_eq_classes)
        for i, (name, pred, rate, nrows) in enumerate(eq_classes[:n_show], 1):
            print(f"Equality Class {i}.")
            print(f"  {name}    (tightness≈{rate:.3f}, n={nrows})\n")

    # 5️⃣ Sophie-style inequality ⇒ Boolean conditions
    conds_sorted: Optional[list[SophieCondition]] = result.extras.get("sophie_conditions")

    if conds_sorted is None:
        sophie_by_prop = result.extras.get("sophie")
        if not sophie_by_prop:
            print("\n(no Sophie-style inequality ⇒ Boolean conditions in result.extras['sophie'])\n")
            return
        conds_sorted = rank_sophie_conditions_global(sophie_by_prop)

    if not conds_sorted:
        print("\n(no Sophie-style inequality ⇒ Boolean conditions discovered)\n")
        return

    if (top_sophie is None) or (top_sophie <= 0):
        top_n = len(conds_sorted)
    else:
        top_n = min(top_sophie, len(conds_sorted))

    print()
    print_sophie_conditions(conds_sorted, top_n=top_n)
