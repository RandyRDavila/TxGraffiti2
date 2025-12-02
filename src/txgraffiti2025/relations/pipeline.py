# src/txgraffiti2025/relations/pipeline.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .core import DataModel, MaskCache
from .class_logic import ClassLogic
from .incomparability import IncomparabilityAnalyzer
from .equality import EqualityMiner
from ..forms.predicates import Predicate, EQ, LE, GE
from ..forms.generic_conjecture import TRUE, Conjecture


__all__ = ["RelationsPipeline"]


@dataclass
class RelationsPipeline:
    """
    End-to-end relations pipeline that:
      1) Builds a DataModel/MaskCache.
      2) Detects a base hypothesis & enumerates boolean classes (ClassLogic).
      3) (Optional) Ingests external boolean rules as extra hypotheses.
      4) Mines incomparability → registers |x−y|, min(x,y), max(x,y).
      5) Mines constants and pair equalities (EqualityMiner).
      6) Builds Eq(...) conjectures per class.

    Notes
    -----
    • The pipeline keeps `model.numeric_cols` in sync with newly registered Exprs.
    • You can call `ingest_rule_candidates` at any time; the class logic is rebuilt.
    • Use `run_all()` for a simple one-shot, or call individual steps.
    """

    df: pd.DataFrame
    config: Dict[str, Any] = field(default_factory=dict)

    # constructed in __post_init__
    model: DataModel = field(init=False)
    cache: MaskCache = field(init=False)
    logic: ClassLogic = field(init=False)
    inc: IncomparabilityAnalyzer = field(init=False)
    eqm: EqualityMiner = field(init=False)

    # working state
    base_name_: str = field(init=False, default="TRUE")
    base_pred_: Predicate = field(init=False, default=TRUE)
    sorted_conjunctions_: List[Tuple[str, Predicate]] = field(init=False, default_factory=list)

    # results
    const_frames_: Dict[str, pd.DataFrame] = field(init=False, default_factory=dict)   # by class name
    pair_frames_: Dict[str, pd.DataFrame] = field(init=False, default_factory=dict)    # by class name
    eq_conjectures_: List[Conjecture] = field(init=False, default_factory=list)

    # ──────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # sensible defaults
        cfg = {
            "max_arity": 2,                 # class enumeration arity
            "min_support": 0.10,            # fraction within a domain
            "min_eq_rate": 0.985,           # for pair equalities
            "const_tol": 1e-9,              # constant tolerance
            "pair_tol": 1e-9,               # equality tolerance
            "rationalize": True,
            "max_denom": 64,
            # incomparability → |x−y|
            "abs_min_support": 0.10,
            "abs_min_side_rate": 0.10,
            "abs_min_side_count": 5,
            "abs_min_median_gap": 0.5,
            "abs_min_mean_gap": 0.5,
            "abs_top_n_store": 20,
            # often-unequal → min/max
            "mm_min_support": 0.10,
            "mm_min_neq_rate": 0.50,
            "mm_min_neq_count": 8,
            "mm_must_be_incomparable": True,
            "mm_top_n_store": 20,
            # class mining control
            "max_classes_to_mine": 32,      # cap number of conjunctions to mine
            "include_base_first": True,
        }
        cfg.update(self.config or {})
        self.config = cfg

        # Build core model/cache
        self.model = DataModel(self.df)
        self.cache = MaskCache(self.model)

        # First pass class logic
        self.logic = ClassLogic(self.model, self.cache)
        self.logic.enumerate(max_arity=self.config["max_arity"])
        self.logic.normalize()
        self.sorted_conjunctions_ = self.logic.sort_by_generality()

        # Expose base to other miners (for compatibility with getattr(model, "base_pred"))
        self.base_pred_ = self.logic.base_predicate()
        self.base_name_ = self.logic.base_name()
        # Attach to model for modules that look on model.*
        setattr(self.model, "base_pred", self.base_pred_)
        setattr(self.model, "base_name", self.base_name_)

        # Miners
        self.inc = IncomparabilityAnalyzer(self.model, self.cache)
        self.eqm = EqualityMiner(self.model, self.cache)

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────
    def _sync_numeric_cols_to_exprs(self) -> None:
        """
        Ensure model.numeric_cols contains every key in model.exprs.
        New synthesized Exprs (e.g., abs/min/max) must be visible to miners.
        """
        known = set(self.model.numeric_cols)
        for k in self.model.exprs.keys():
            if k not in known:
                self.model.numeric_cols.append(k)
                known.add(k)

    def classes_to_mine(self) -> List[Tuple[str, Predicate]]:
        """
        Pick the sequence of (name, Predicate) classes to mine.
        Includes base class (if configured) then top conjunctions by generality.
        """
        items = list(self.sorted_conjunctions_ or [])
        if self.config.get("include_base_first", True):
            base = (self.base_name_, self.base_pred_)
            # ensure base at front (dedup if also present)
            items = [base] + [(n, p) for (n, p) in items if n != base[0]]
        k = int(self.config.get("max_classes_to_mine", 32))
        return items[: max(0, k)]

    # ──────────────────────────────────────────────────────────────────────
    # Step A: Ingest external boolean rules (optional)
    # ──────────────────────────────────────────────────────────────────────
    def ingest_rule_candidates(
        self,
        rows: pd.DataFrame,
        *,
        row_to_pred: Optional[Callable[[pd.Series], Tuple[str, Predicate]]] = None,
        min_support: float = 0.15,
        name_col: str = "pretty_name",
    ) -> List[Tuple[str, Predicate]]:
        """
        Register extra boolean hypotheses discovered elsewhere (e.g., a rule miner),
        then rebuild ClassLogic so the new hypotheses are available to the pipeline.

        Expected default schema (if row_to_pred is None):
        kind ∈ {"eq","le","ge"}, inv1, inv2, support, and optional pretty_name.
        """
        if rows is None or rows.empty:
            return []

        def _auto_mapper(r: pd.Series) -> Tuple[str, Predicate]:
            needed = {"kind", "inv1", "inv2"}
            if not needed.issubset(rows.columns):
                raise ValueError(
                    "Auto-mapper requires columns: {kind, inv1, inv2} "
                    f"(got: {list(rows.columns)})"
                )
            k = str(r["kind"]).strip().lower()
            x = str(r["inv1"]); y = str(r["inv2"])
            if x not in self.model.exprs or y not in self.model.exprs:
                raise KeyError(f"Unknown invariant(s): {x!r}, {y!r}")
            ex = self.model.exprs[x]; ey = self.model.exprs[y]
            if k == "eq":
                pred = EQ(ex, ey); default_name = f"{x} = {y}"
            elif k == "le":
                pred = LE(ex, ey); default_name = f"{x} ≤ {y}"
            elif k == "ge":
                pred = GE(ex, ey); default_name = f"{x} ≥ {y}"
            else:
                raise ValueError(f"Unsupported rule kind: {k!r}")
            name = str(r.get(name_col, "") or default_name)
            return name, pred

        mapper = row_to_pred or _auto_mapper
        kept: List[Tuple[str, Predicate]] = []

        for _, r in rows.iterrows():
            sup = float(r.get("support", 1.0))
            if sup < float(min_support):
                continue
            name, pred = mapper(r)
            if name not in self.model.preds:
                self.model.preds[name] = pred
            kept.append((name, pred))

        if not kept:
            return kept

        # Rebuild class logic so new hyps participate
        self.logic = ClassLogic(self.model, self.cache)
        self.logic.enumerate(max_arity=self.config["max_arity"])
        self.logic.normalize()
        self.sorted_conjunctions_ = self.logic.sort_by_generality()

        # refresh base exposure
        self.base_pred_ = self.logic.base_predicate()
        self.base_name_ = self.logic.base_name()
        setattr(self.model, "base_pred", self.base_pred_)
        setattr(self.model, "base_name", self.base_name_)
        return kept

    # ──────────────────────────────────────────────────────────────────────
    # Step B: Incomparability → synth features
    # ──────────────────────────────────────────────────────────────────────
    def mine_incomparability_features(self) -> Dict[str, pd.DataFrame]:
        """
        Run the two synthesizers: |x−y| for “meaningful” incomparables and
        min/max for “often unequal” pairs. Keeps numeric cols in sync.
        """
        cfg = self.config
        abs_df = self.inc.register_absdiff_exprs_for_meaningful_pairs(
            condition=None,
            use_base_if_none=True,
            require_finite=True,
            min_support=cfg["abs_min_support"],
            min_side_rate=cfg["abs_min_side_rate"],
            min_side_count=cfg["abs_min_side_count"],
            min_median_gap=cfg["abs_min_median_gap"],
            min_mean_gap=cfg["abs_min_mean_gap"],
            top_n_store=cfg["abs_top_n_store"],
        )

        mm_df = self.inc.register_minmax_for_often_unequal_pairs(
            condition=None,
            use_base_if_none=True,
            require_finite=True,
            min_support=cfg["mm_min_support"],
            min_neq_rate=cfg["mm_min_neq_rate"],
            min_neq_count=cfg["mm_min_neq_count"],
            must_be_incomparable=cfg["mm_must_be_incomparable"],
            top_n_store=cfg["mm_top_n_store"],
        )

        # make newly-registered exprs visible to miners
        self._sync_numeric_cols_to_exprs()
        return {"absdiff": abs_df, "minmax": mm_df}

    # ──────────────────────────────────────────────────────────────────────
    # Step C: Equality mining (constants & pairs) per class
    # ──────────────────────────────────────────────────────────────────────
    def mine_equalities_per_class(self) -> None:
        """
        For each selected class, mine constants and pair equalities and
        assemble Eq(...) conjectures. Results are stored in:
            - const_frames_[class_name]
            - pair_frames_[class_name]
            - eq_conjectures_
        """
        cfg = self.config
        classes = self.classes_to_mine()
        self.const_frames_.clear()
        self.pair_frames_.clear()
        self.eq_conjectures_.clear()

        for cname, cpred in classes:
            const_df = self.eqm.analyze_constants(
                condition=cpred, use_base_if_none=False, require_finite=True,
                tol=cfg["const_tol"], rationalize=cfg["rationalize"], max_denom=cfg["max_denom"],
                min_support=cfg["min_support"],
            )
            pair_df = self.eqm.analyze_pair_equalities(
                condition=cpred, use_base_if_none=False, require_finite=True,
                tol=cfg["pair_tol"], min_support=cfg["min_support"], min_eq_rate=cfg["min_eq_rate"],
            )
            self.const_frames_[cname] = const_df
            self.pair_frames_[cname] = pair_df

            # Eq(...) conjectures (only selected rows)
            conjs = self.eqm.make_eq_conjectures(
                constants=const_df, pairs=pair_df, condition=cpred,
                rationalize_constants=cfg["rationalize"], max_denom=cfg["max_denom"],
                only_selected=True,
            )
            if conjs:
                self.eq_conjectures_.extend(conjs)

    # ──────────────────────────────────────────────────────────────────────
    # One-shot convenience
    # ──────────────────────────────────────────────────────────────────────
    def run_all(
        self,
        *,
        rule_table: Optional[pd.DataFrame] = None,
        min_rule_support: float = 0.15,
    ) -> "RelationsPipeline":
        """
        End-to-end default run:
          (optional) ingest rules → incomparability features → equality mining.
        """
        if rule_table is not None and not rule_table.empty:
            self.ingest_rule_candidates(rule_table, min_support=min_rule_support)

        self.mine_incomparability_features()
        self.mine_equalities_per_class()
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Light summaries (safe to print in demos)
    # ──────────────────────────────────────────────────────────────────────
    def summary(self, max_classes: int = 8, max_rows: int = 6) -> str:
        lines: List[str] = []
        lines.append("────────────────────────────────────────────────────────")
        lines.append("RelationsPipeline • Summary")
        lines.append("────────────────────────────────────────────────────────")
        lines.append(f"DataFrame: {self.model.df.shape[0]} rows × {self.model.df.shape[1]} cols")
        lines.append(f"Base: {self.base_name_}")
        lines.append("")
        lines.append("Classes to mine (top):")
        for (n, _p) in self.classes_to_mine()[:max_classes]:
            lines.append(f"  • {n}")
        lines.append("")
        tot_eq = sum(len(df) for df in self.pair_frames_.values())
        tot_const = sum(len(df) for df in self.const_frames_.values())
        lines.append(f"Eq(...) conjectures: {len(self.eq_conjectures_)}")
        lines.append(f"Pair-equality rows: {tot_eq}")
        lines.append(f"Constant rows: {tot_const}")
        lines.append("")
        # sample one class block
        if self.pair_frames_:
            cname = next(iter(self.pair_frames_.keys()))
            lines.append(f"Sample class: {cname}")
            p = self.pair_frames_[cname].head(max_rows)
            c = self.const_frames_[cname].head(max_rows)
            if not c.empty:
                lines.append("  Constants (selected):")
                for r in c[c["selected"]].head(max_rows).itertuples(index=False):
                    vf = f" ({r.value_frac})" if r.value_frac else ""
                    lines.append(f"    · {r.inv} = {r.value:g}{vf}")
            if not p.empty:
                lines.append("  Pair equalities (selected):")
                for r in p[p["selected"]].head(max_rows).itertuples(index=False):
                    lines.append(f"    · {r.inv1} = {r.inv2}  [rate={r.rate_eq:.3f}; supp={r.support:.3f}]")
        return "\n".join(lines)
