# src/txgraffiti2025/pipeline/stage1_hypotheses.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TextColumn

from txgraffiti2025.processing.pre.hypotheses import enumerate_boolean_hypotheses
from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses
# NOTE: If your file is still named constants_cashe.py, change the import below accordingly.
from txgraffiti2025.processing.pre.constants_cache import (
    precompute_constant_ratios_pairs,
    ConstantsCache,
)
from txgraffiti2025.forms.predicates import Predicate

console = Console()


def _mask(pred: Predicate, df: pd.DataFrame) -> pd.Series:
    """Robustly evaluate a predicate to a boolean Series aligned to df.index."""
    m = pred.mask(df)
    if not isinstance(m, pd.Series):
        m = pd.Series(m, index=df.index, dtype=bool)
    return m.astype(bool).reindex(df.index, fill_value=False)


def _detect_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (boolean_cols, numeric_cols) using pandas dtypes."""
    bool_cols = [c for c in df.columns if df[c].dtype == "bool"]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return bool_cols, num_cols


@dataclass
class HypothesisStageResult:
    kept: List[Predicate]
    equivalences: List  # List[ClassEquivalence]
    mask_cache: Dict[str, pd.Series]          # key: repr(h) -> mask
    pred_index: Dict[str, Predicate]          # key: repr(h) -> Predicate
    bool_cols: List[str]
    num_cols: List[str]
    constants_cache: Optional[ConstantsCache]


def _make_progress(ui: str, persist: bool) -> Progress:
    """
    Build a Rich Progress instance based on UI prefs.

    ui:
      - "bars"    -> spinner + bar + elapsed time
      - "spinner" -> spinner only
    persist:
      - True  -> keep bars on screen after completion
      - False -> auto-clear when done
    """
    columns = [SpinnerColumn(), TextColumn("[bold blue]{task.description}")]
    if ui == "bars":
        columns += [BarColumn(), TimeElapsedColumn()]
    return Progress(*columns, transient=not persist, console=console)


def run_stage1_hypotheses(
    df: pd.DataFrame,
    *,
    min_support_frac: float = 0.05,
    include_base: bool = True,
    include_pairs: bool = True,
    skip_always_false: bool = True,
    precompute_constants: bool = True,
    verbose: bool = True,
    ui: str = "bars",        # "bars" or "spinner"
    persist: bool = True,    # keep progress visible after completion
) -> HypothesisStageResult:
    """
    Stage 1 of the conjecturing pipeline with optional progress indicators.
    Enumerates, simplifies, deduplicates, caches masks, and precomputes constants.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    min_support_frac : float, default 0.05
        Minimum fraction of rows a hypothesis must cover to be kept.
    include_base : bool, default True
        Include detected base predicate.
    include_pairs : bool, default True
        Include base∧pair predicates.
    skip_always_false : bool, default True
        Skip hypotheses that are never satisfied under base.
    precompute_constants : bool, default True
        Precompute constant ratios for (base and base∧single) via pair helper.
    verbose : bool, default True
        Print summary of kept predicates and equivalences.
    ui : {"bars","spinner"}, default "bars"
        Choose progress presentation: progress bars or minimal spinners.
    persist : bool, default True
        If True, keep progress on screen when finished; else auto-clear.
    """
    n = len(df)
    min_support = max(1, int(round(min_support_frac * n)))
    bool_cols, num_cols = _detect_cols(df)

    progress = _make_progress(ui, persist)

    with progress:
        # Step 1: Enumerate boolean hypotheses
        t1 = progress.add_task("Enumerating boolean hypotheses...", total=None)
        hyps = enumerate_boolean_hypotheses(
            df,
            include_base=include_base,
            include_pairs=include_pairs,
            skip_always_false=skip_always_false,
        )
        progress.update(t1, description=f"Enumerated {len(hyps)} candidate hypotheses")
        progress.stop_task(t1)

        # Step 2: Simplify and deduplicate
        t2 = progress.add_task("Simplifying & deduplicating...", total=None)
        kept, eqs = simplify_and_dedup_hypotheses(df, hyps, min_support=min_support)
        progress.update(t2, description=f"Simplified to {len(kept)} hypotheses")
        progress.stop_task(t2)

        # Step 3: Build mask cache
        t3 = progress.add_task("Building mask cache...", total=len(kept) if kept else 1)
        mask_cache: Dict[str, pd.Series] = {}
        pred_index: Dict[str, Predicate] = {}
        for h in kept:
            k = repr(h)
            pred_index[k] = h
            mask_cache[k] = _mask(h, df)
            progress.advance(t3)
        progress.stop_task(t3)

        # Step 4: Precompute constant ratios (optional)
        constants_cache: Optional[ConstantsCache] = None
        if precompute_constants and kept:
            t4 = progress.add_task("Precomputing constant ratios...", total=None)
            base_key = max(mask_cache.keys(), key=lambda k: int(mask_cache[k].sum()))
            base = pred_index[base_key]
            constants_cache = precompute_constant_ratios_pairs(
                df,
                base=base,
                boolean_cols=bool_cols,
                numeric_cols=num_cols,
                shifts=(-2, -1, 0, 1, 2),
                # avoid spurious constants on tiny slices
                min_support=max(8, min_support),
            )
            progress.update(t4, description="Constant ratios precomputed")
            progress.stop_task(t4)
        else:
            constants_cache = None

    if verbose:
        console.print("[bold green]=== kept (simplified/dedup) ===")
        for h in kept:
            console.print(f" • {h}")
        console.print("\n[bold cyan]=== saved equivalences ===")
        for c in eqs:
            console.print(f" • {c}")
        if constants_cache is not None:
            console.print(
                f"\n[bold magenta][constants][/bold magenta] hypotheses cached: "
                f"{len(constants_cache.hyp_to_key)}"
            )

    return HypothesisStageResult(
        kept=kept,
        equivalences=eqs,
        mask_cache=mask_cache,
        pred_index=pred_index,
        bool_cols=bool_cols,
        num_cols=num_cols,
        constants_cache=constants_cache,
    )
