from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional, Dict, Any

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.forms.predicates import Predicate

# simple, dependable generators
from txgraffiti2025.generators.ratios import ratios as gen_ratios   # features=[other]
from txgraffiti2025.generators.lp import lp_bounds, LPConfig       # features=[other]

# -----------------------
# config + results
# -----------------------

@dataclass
class Stage2Config:
    target: str
    use_ratios: bool = True
    use_lp: bool = True                # LP over single features only (simple)
    min_support_frac: float = 0.05     # min rows inside hypothesis mask
    max_denominator: int = 50          # for pretty rationals in generators
    q_clip: Optional[float] = None     # e.g., 0.01 for robust ratio tails
    direction: str = "both"            # "both" | "upper" | "lower"
    progress: bool = True              # tqdm bars
    warn: bool = False                 # keep logs quiet by default

@dataclass
class Stage2Result:
    admitted: List[Conjecture]         # passed Morgan
    considered: int
    admitted_count: int
    per_hyp_counts: Dict[str, int]     # {repr(hyp): admitted_under_h}

# -----------------------
# utilities
# -----------------------

from pandas.api.types import is_numeric_dtype, is_bool_dtype

def _safe_numeric_features(df: pd.DataFrame, *, target: str) -> List[str]:
    """
    Return usable numeric feature columns for ratios/LP:
      - exclude the target column
      - require numeric dtype but not boolean dtype
      - drop near-binary columns (nunique <= 2 after numeric coercion)
    """
    out: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if not is_numeric_dtype(s) or is_bool_dtype(s):
            continue
        # Robust nunique check after numeric coercion
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.nunique(dropna=True) <= 2:
            continue
        out.append(c)
    return out


def _applicable_support(df: pd.DataFrame, hyp: Predicate) -> int:
    m = hyp.mask(df).reindex(df.index, fill_value=False).astype(bool)
    return int(m.sum())

def _morgan_admits(conj: Conjecture, df: pd.DataFrame, *, min_applicable: int) -> bool:
    """
    Morgan = “admitted if conjecture holds on applicable rows”.
    Require at least min_applicable rows inside the condition to bother checking.
    """
    applicable, holds, _ = conj.check(df, auto_base=False)
    if int(applicable.sum()) < min_applicable:
        return False
    return bool(holds[applicable].all())

# -----------------------
# main entry
# -----------------------

def run_stage2_simple(
    df: pd.DataFrame,
    good_hypotheses: Sequence[Predicate],
    *,
    cfg: Stage2Config,
) -> Stage2Result:
    """
    Minimal Stage-2:
      – choose safe numeric features
      – for each hyp in good_hypotheses:
            for each feature != target:
                run ratios (optional)
                run LP(1-feature) (optional)
                admit via Morgan
    """
    n = len(df)
    min_support = max(1, int(round(cfg.min_support_frac * max(n, 0))))

    features = _safe_numeric_features(df, target=cfg.target)

    admitted: List[Conjecture] = []
    considered = 0
    per_hyp_counts: Dict[str, int] = {}

    hyp_iter: Iterable[Predicate] = good_hypotheses
    if cfg.progress:
        hyp_iter = tqdm(hyp_iter, total=len(good_hypotheses), desc="H (stage2)")

    for H in hyp_iter:
        # quick skip if hyp has tiny support
        if _applicable_support(df, H) < min_support:
            continue

        admitted_here = 0

        feat_iter: Iterable[str] = features
        if cfg.progress:
            feat_iter = tqdm(feat_iter, total=len(features), leave=False, desc="  features")

        for other in feat_iter:
            # === Ratios: H ⇒ target ≥/≤ c * other ===
            if cfg.use_ratios:
                for conj in gen_ratios(
                    df,
                    features=[other],
                    target=cfg.target,
                    hypothesis=H,
                    max_denominator=cfg.max_denominator,
                    direction=cfg.direction,
                    q_clip=cfg.q_clip,
                    min_support=min_support,
                    simplify_condition=True,  # drop explicit sign when uniform
                ):
                    considered += 1
                    if _morgan_admits(conj, df, min_applicable=min_support):
                        admitted.append(conj)
                        admitted_here += 1

            # === LP (1 feature): H ⇒ target ≥/≤ a*other + b ===
            if cfg.use_lp:
                lp_cfg = LPConfig(
                    features=[other],
                    target=cfg.target,
                    direction=cfg.direction,
                    max_denominator=cfg.max_denominator,
                    tol=1e-9,
                    min_support=min_support,
                )
                for conj in lp_bounds(df, hypothesis=H, config=lp_cfg):
                    considered += 1
                    if _morgan_admits(conj, df, min_applicable=min_support):
                        admitted.append(conj)
                        admitted_here += 1

        per_hyp_counts[repr(H)] = admitted_here

    if cfg.progress:
        print(f"admitted {len(admitted)} / considered {considered}")

    return Stage2Result(
        admitted=admitted,
        considered=considered,
        admitted_count=len(admitted),
        per_hyp_counts=per_hyp_counts,
    )
