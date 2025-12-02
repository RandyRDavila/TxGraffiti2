# src/txgraffiti2025/generators/graffiti_qualitative.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, List, Literal, Tuple

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import TRUE
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.graffiti_relations import GraffitiClassRelations

CorrMethod = Literal["spearman", "pearson", "kendall"]


# ───────────────────────────── dataclasses ───────────────────────────── #

@dataclass
class MonotoneRelation:
    x: str
    y: str
    direction: Literal["increasing", "decreasing"]
    method: CorrMethod = "spearman"
    min_abs_rho: float = 0.35
    min_n: int = 12

    def pretty(self) -> str:
        arrow = "↗" if self.direction == "increasing" else "↘"
        return f"{self.y} {arrow} {self.x}  ({self.method})"

    # --- correlation kernels (no SciPy required) ---

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2:
            return np.nan
        a = a.astype(float, copy=False)
        b = b.astype(float, copy=False)
        a = a - np.mean(a)
        b = b - np.mean(b)
        den = np.linalg.norm(a) * np.linalg.norm(b)
        if den == 0.0:
            return np.nan
        return float(np.dot(a, b) / den)

    @staticmethod
    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        # rank via pandas (handles ties with average)
        ar = pd.Series(a).rank(method="average").to_numpy(dtype=float, copy=False)
        br = pd.Series(b).rank(method="average").to_numpy(dtype=float, copy=False)
        return MonotoneRelation._pearson(ar, br)

    @staticmethod
    def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
        # O(n^2) Kendall tau (tie-aware by skipping equal pairs)
        n = a.size
        if n < 2:
            return np.nan
        conc = 0
        disc = 0
        for i in range(n - 1):
            da = a[i+1:] - a[i]
            db = b[i+1:] - b[i]
            s = np.sign(da * db)
            conc += int((s > 0).sum())
            disc += int((s < 0).sum())
        denom = conc + disc
        if denom == 0:
            return 0.0
        return float((conc - disc) / denom)

    def _corr(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.method == "pearson":
            return self._pearson(x, y)
        elif self.method == "kendall":
            return self._kendall_tau(x, y)
        else:
            return self._spearman(x, y)  # default


@dataclass
class QualResult:
    relation: MonotoneRelation
    condition: Optional[Predicate]
    rho: float
    n: int
    support: int
    x: str
    y: str

    def pretty(self) -> str:
        cond = "TRUE" if (self.condition is None or self.condition is TRUE) else getattr(self.condition, "pretty", lambda: repr(self.condition))()
        dir_arrow = "↑" if self.relation.direction == "increasing" else "↓"
        return (f"[{cond}]  {self.y} {dir_arrow} {self.x}  "
                f"(rho={self.rho:.3f}, n={self.n}/{self.support}, method={self.relation.method})")


# ─────────────────────────── helper (mask) ─────────────────────────── #

def _mask_from_pred(df: pd.DataFrame, pred: Optional[Predicate]) -> np.ndarray:
    if pred is None or pred is TRUE:
        return np.ones(len(df), dtype=bool)
    s = pred.mask(df).reindex(df.index, fill_value=False)
    if s.dtype is not bool:
        s = s.fillna(False).astype(bool, copy=False)
    return s.to_numpy(dtype=bool, copy=False)


# ──────────────────────── main conjecturer class ─────────────────────── #

class GraffitiQualitative:
    """
    Mine qualitative (monotone) relations y ~ x under many hypotheses discovered
    by `GraffitiClassRelations`. Correlation defaults to Spearman (rank-based).

    Usage
    -----
    gcr = GraffitiClassRelations(df)
    qual = GraffitiQualitative(gcr)
    results = qual.generate_qualitative_relations(
        y_targets=["independence_number"],
        x_candidates=["order","size","radius"],
        method="spearman",
        min_abs_rho=0.4,
        min_n=12,
        top_k_per_hyp=5,
    )
    for r in results[:10]:
        print(r.pretty())
    """

    def __init__(self, gcr: GraffitiClassRelations):
        self.gcr = gcr
        self.df = gcr.df
        # numeric columns = those the GCR already treats as Expr columns
        self.numeric_columns: List[str] = list(gcr.expr_cols)
        # hypotheses = nonredundant conjunctions (includes base explicitly)
        if not hasattr(gcr, "nonredundant_conjunctions_"):
            gcr.enumerate_conjunctions()
            gcr.find_redundant_conjunctions()
        self.hyps_kept: List[Tuple[str, Predicate]] = list(gcr.nonredundant_conjunctions_)

    # ---------------------------- core API ---------------------------- #

    def generate_qualitative_relations(
        self,
        *,
        y_targets: Optional[Iterable[str]] = None,
        x_candidates: Optional[Iterable[str]] = None,
        hyps: Optional[Iterable[Predicate]] = None,
        method: CorrMethod = "spearman",
        min_abs_rho: float = 0.35,
        min_n: int = 12,
        drop_constant: bool = True,
        top_k_per_hyp: Optional[int] = None,
    ) -> List[QualResult]:
        """
        Discover monotone (increasing/decreasing) y~x tendencies under each hypothesis.

        Parameters
        ----------
        y_targets : iterable[str] or None
            Dependent variables (defaults to all numeric columns).
        x_candidates : iterable[str] or None
            Explanatory variables (defaults to all numeric columns).
        hyps : iterable[Predicate] or None
            Hypotheses to test; by default, uses `GraffitiClassRelations` nonredundant set.
        method : {"spearman","pearson","kendall"}, default="spearman"
            Correlation measure used to score monotone relation strength.
        min_abs_rho : float, default=0.35
            Minimum absolute correlation to accept a relation.
        min_n : int, default=12
            Minimum valid pair count within the hypothesis support.
        drop_constant : bool, default=True
            Skip columns that are constant (within NaN handling) on the hypothesis support.
        top_k_per_hyp : int or None, default=None
            If set, keep only the best K results per hypothesis by (|rho|, support).

        Returns
        -------
        list[QualResult]
            Results ranked by (|rho| desc, support desc) across all hypotheses.
        """
        y_cols = list(y_targets or self.numeric_columns)
        x_cols = list(x_candidates or self.numeric_columns)
        hyps_iter = list(hyps) if hyps is not None else [p for (_, p) in self.hyps_kept]

        # Coerce numeric frame once per hyp
        results: List[QualResult] = []

        for H in hyps_iter:
            mask = _mask_from_pred(self.df, H)
            support = int(mask.sum())
            if support < min_n:
                continue

            dfH = self.df.loc[mask]
            num_df = dfH.apply(pd.to_numeric, errors="coerce")

            hyp_results: List[QualResult] = []
            for y in y_cols:
                ys = num_df.get(y)
                if ys is None:
                    continue
                if drop_constant and ys.nunique(dropna=True) <= 1:
                    continue

                yv = ys.to_numpy(dtype=float, copy=False)

                for x in x_cols:
                    if x == y:
                        continue
                    xs = num_df.get(x)
                    if xs is None:
                        continue
                    if drop_constant and xs.nunique(dropna=True) <= 1:
                        continue

                    xv = xs.to_numpy(dtype=float, copy=False)
                    valid = np.isfinite(xv) & np.isfinite(yv)
                    n = int(valid.sum())
                    if n < min_n:
                        continue

                    mr = MonotoneRelation(
                        x=x, y=y, direction="increasing",
                        method=method, min_abs_rho=min_abs_rho, min_n=min_n
                    )
                    rho = mr._corr(xv[valid], yv[valid])
                    if not np.isfinite(rho):
                        continue
                    if abs(rho) < float(min_abs_rho):
                        continue

                    mr.direction = "increasing" if rho >= 0 else "decreasing"
                    hyp_results.append(
                        QualResult(relation=mr, condition=H, rho=float(rho),
                                   n=n, support=support, x=x, y=y)
                    )

            if top_k_per_hyp is not None and top_k_per_hyp > 0:
                hyp_results.sort(key=lambda r: (abs(r.rho), r.support), reverse=True)
                hyp_results = hyp_results[:int(top_k_per_hyp)]

            results.extend(hyp_results)

        results.sort(key=lambda r: (abs(r.rho), r.support), reverse=True)
        return results

    # --------------------------- tiny printers -------------------------- #

    @staticmethod
    def print_sample(results: List[QualResult], *, k: int = 10) -> None:
        print(f"=== Qualitative Relations (top {k}) ===")
        for r in results[:k]:
            print("•", r.pretty())
