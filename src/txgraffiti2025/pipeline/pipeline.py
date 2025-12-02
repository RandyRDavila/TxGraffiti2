# src/txgraffiti2025/pipeline/pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from txgraffiti2025.pipeline.stage1_hypotheses import (
    run_stage1_hypotheses,
    HypothesisStageResult,
)

@dataclass
class PipelineConfig:
    # Stage 1 knobs
    min_support_frac: float = 0.05
    include_base: bool = True
    include_pairs: bool = True
    skip_always_false: bool = True
    precompute_constants: bool = True
    verbose: bool = True
    ui: str = "bars"       # "bars" or "spinner"
    persist_ui: bool = True

    # Future stages (placeholders; we’ll wire these in next steps)
    use_lp: bool = True
    keep_frac_after_touch: float = 0.75


class ConjecturePipeline:
    """
    High-level orchestrator for TxGraffiti2025.

    On instantiation:
      • Runs Stage 1 (boolean hypothesis discovery, simplification & equivalences)
      • Builds mask cache and constants cache
      • Stores all artifacts for later conjecturing

    Then call `run_for_target("some_numeric_col")` to generate conjectures.
    """

    def __init__(self, df: pd.DataFrame, *, config: Optional[PipelineConfig] = None):
        self.df = df
        self.cfg = config or PipelineConfig()

        # ————— Stage 1 at init —————
        s1: HypothesisStageResult = run_stage1_hypotheses(
            df,
            min_support_frac=self.cfg.min_support_frac,
            include_base=self.cfg.include_base,
            include_pairs=self.cfg.include_pairs,
            skip_always_false=self.cfg.skip_always_false,
            precompute_constants=self.cfg.precompute_constants,
            verbose=self.cfg.verbose,
            ui=self.cfg.ui,
            persist=self.cfg.persist_ui,
        )

        # Expose stage-1 artifacts
        self.bool_cols: List[str] = s1.bool_cols
        self.num_cols: List[str] = s1.num_cols
        self.hypotheses = s1.kept
        self.eq_conjectures = s1.equivalences
        self.mask_cache: Dict[str, pd.Series] = s1.mask_cache   # key: repr(h)
        self.pred_index: Dict[str, object] = s1.pred_index
        self.constants_cache = s1.constants_cache

        # Storage for later pipeline outputs
        self.conjectures_by_target: Dict[str, List[object]] = {}
        self.logs: Dict[str, List[str]] = {}

    # ————————————————————————————————————————————————
    # Stage 2+ stubs (we’ll implement these next)
    # ————————————————————————————————————————————————

    def run_for_target(self, target: str) -> List[object]:
        """
        Run the full conjecture pipeline for a single numeric target.
        (Stage 2+: raw generation, Hazel-in-generation, touch ranking,
         generalize & refine, Hazel/Morgan/Dalmatian, store results.)
        """
        if target not in self.num_cols:
            raise ValueError(f"Target '{target}' is not numeric or not found in DataFrame.")

        # TODO: implement
        results: List[object] = []
        self.conjectures_by_target[target] = results
        return results

    def summarize_results(self) -> str:
        """
        Pretty textual summary across all targets + saved equivalences.
        """
        lines: List[str] = []
        lines.append("========================================")
        lines.append("TxGraffiti2025 Pipeline Summary")
        lines.append("========================================")
        lines.append("")

        lines.append("Boolean equivalence conjectures:")
        if self.eq_conjectures:
            for c in self.eq_conjectures:
                lines.append(f" - {c}")
        else:
            lines.append(" - (none)")
        lines.append("")

        if not self.conjectures_by_target:
            lines.append("No targets run yet.")
        else:
            for t, conjs in self.conjectures_by_target.items():
                lines.append(f"Target: {t}  (n={len(conjs)})")
                for cj in conjs[:10]:  # show a few
                    try:
                        lines.append(f" - {cj}")
                    except Exception:
                        lines.append(" - (conjecture)")
                lines.append("")
        return "\n".join(lines)
