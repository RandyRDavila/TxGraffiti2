# # src/txgraffiti2025/relations/class_relations_miner.py

# from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Tuple, Optional, Dict, Any, Union

# import numpy as np
# import pandas as pd

# from .core import DataModel, MaskCache
# from ..forms.predicates import Predicate
# from ..forms.class_relations import ClassInclusion, ClassEquivalence


# def _resolve_min_support(min_support: Union[int, float], N: int) -> int:
#     """
#     If float in (0,1], interpret as fraction of N; if >=1, as absolute.
#     If 0, return 0. Clamps to [0, N].
#     """
#     if isinstance(min_support, float):
#         if min_support <= 0:
#             return 0
#         if min_support <= 1:
#             return int(np.floor(min_support * N))
#         return int(min_support)
#     # int
#     return max(0, min(int(min_support), N))


# @dataclass(slots=True)
# class ClassRelationsMiner:
#     """
#     Mine logical relations between boolean hypotheses (classes):

#       • Inclusion:      H_i ⇒ H_j
#       • Equivalence:    H_i ⇔ H_j

#     Produces a diagnostics DataFrame (pairwise stats) and a list of
#     ClassInclusion / ClassEquivalence conjecture objects.

#     Thresholds allow soft acceptance with a small number of violations.
#     """

#     model: DataModel
#     cache: MaskCache

#     # ------------------------------ internal ------------------------------

#     def _mask(self, pred: Predicate) -> np.ndarray:
#         """Cached boolean mask as dense ndarray[bool]."""
#         s = self.cache.mask(pred)
#         return s.to_numpy(dtype=np.bool_, copy=False)

#     # ------------------------------ public API ----------------------------

#     def analyze(
#         self,
#         hyps: List[Tuple[str, Predicate]],
#         *,
#         min_support: Union[int, float] = 0.05,   # int (absolute) or float in (0,1]
#         max_violations: int = 0,
#         min_implication_rate: float = 1.0,       # require n(A∧B)/n(A) ≥ this
#         consider_empty: bool = False,
#         topk: Optional[int] = None,
#         return_only_strict: bool = False,        # if True, drop pairs with nI==0 or A==B masks
#     ) -> pd.DataFrame:
#         """
#         Return a DataFrame of pairwise stats with implication/equivalence flags.

#         Columns:
#           name_i, name_j,
#           nA, nB, nI, nU,
#           prec_i_to_j, rec_i_to_j, prec_j_to_i, rec_j_to_i, jaccard,
#           viol_i_to_j, viol_j_to_i,
#           imp_i_to_j, imp_j_to_i, equiv
#         """
#         if not hyps:
#             return pd.DataFrame()

#         N = len(self.model.df)
#         min_sup_abs = _resolve_min_support(min_support, N)
#         rows: List[Dict[str, Any]] = []

#         # Precompute masks/sizes
#         masks: Dict[str, np.ndarray] = {}
#         sizes: Dict[str, int] = {}
#         for name, pred in hyps:
#             m = self._mask(pred)
#             masks[name] = m
#             sizes[name] = int(m.sum())

#         # Pairwise stats
#         for i in range(len(hyps)):
#             name_i, _ = hyps[i]
#             m_i = masks[name_i]; nA = sizes[name_i]

#             if (not consider_empty) and nA == 0:
#                 continue
#             if nA < min_sup_abs:
#                 continue

#             for j in range(i + 1, len(hyps)):
#                 name_j, _ = hyps[j]
#                 m_j = masks[name_j]; nB = sizes[name_j]

#                 if (not consider_empty) and nB == 0:
#                     continue
#                 if nB < min_sup_abs:
#                     continue

#                 inter = (m_i & m_j)
#                 union = (m_i | m_j)
#                 nI = int(inter.sum())
#                 nU = int(union.sum())

#                 if return_only_strict and (nI == 0 or np.array_equal(m_i, m_j)):
#                     continue

#                 # Violations for implications
#                 viol_i_to_j = int((m_i & ~m_j).sum())
#                 viol_j_to_i = int((m_j & ~m_i).sum())

#                 # Rates (guarded by nA,nB>0 due to min_sup_abs)
#                 rec_i_to_j = (nI / nA) if nA > 0 else 1.0   # of A, how many are also B
#                 rec_j_to_i = (nI / nB) if nB > 0 else 1.0
#                 prec_i_to_j = (nI / nB) if nB > 0 else 1.0  # of B, how many are A
#                 prec_j_to_i = (nI / nA) if nA > 0 else 1.0
#                 jacc = (nI / nU) if nU > 0 else 1.0

#                 imp_i_to_j = (viol_i_to_j <= int(max_violations)) and (rec_i_to_j >= float(min_implication_rate))
#                 imp_j_to_i = (viol_j_to_i <= int(max_violations)) and (rec_j_to_i >= float(min_implication_rate))
#                 equiv = bool(imp_i_to_j and imp_j_to_i)

#                 rows.append(dict(
#                     name_i=name_i, name_j=name_j,
#                     nA=nA, nB=nB, nI=nI, nU=nU,
#                     prec_i_to_j=prec_i_to_j, rec_i_to_j=rec_i_to_j,
#                     prec_j_to_i=prec_j_to_i, rec_j_to_i=rec_j_to_i,
#                     jaccard=jacc,
#                     viol_i_to_j=viol_i_to_j, viol_j_to_i=viol_j_to_i,
#                     imp_i_to_j=imp_i_to_j, imp_j_to_i=imp_j_to_i,
#                     equiv=equiv,
#                 ))

#         cols = [
#             "name_i","name_j","nA","nB","nI","nU",
#             "prec_i_to_j","rec_i_to_j","prec_j_to_i","rec_j_to_i",
#             "jaccard","viol_i_to_j","viol_j_to_i",
#             "imp_i_to_j","imp_j_to_i","equiv",
#         ]
#         out = pd.DataFrame(rows, columns=cols)
#         if out.empty:
#             return out

#         out = out.sort_values(
#             ["equiv","imp_i_to_j","imp_j_to_i","jaccard","rec_i_to_j","rec_j_to_i","name_i","name_j"],
#             ascending=[False, False, False, False, False, False, True, True],
#         ).reset_index(drop=True)

#         if topk is not None and topk > 0:
#             out = out.head(topk).reset_index(drop=True)
#         return out

#     def make_conjectures(
#         self,
#         hyps: List[Tuple[str, Predicate]],
#         *,
#         min_support: Union[int, float] = 0.05,
#         max_violations: int = 0,
#         min_implication_rate: float = 1.0,
#         consider_empty: bool = False,
#         topk: Optional[int] = None,
#         return_only_strict: bool = False,
#     ) -> Tuple[pd.DataFrame, List[object]]:
#         """
#         Run `analyze` and construct ClassInclusion / ClassEquivalence objects
#         for accepted relations under the thresholds.

#         Returns (stats_df, conjectures)
#         """
#         stats = self.analyze(
#             hyps,
#             min_support=min_support,
#             max_violations=max_violations,
#             min_implication_rate=min_implication_rate,
#             consider_empty=consider_empty,
#             topk=topk,
#             return_only_strict=return_only_strict,
#         )

#         name_to_pred: Dict[str, Predicate] = {name: pred for name, pred in hyps}
#         conjs: List[object] = []
#         for _, row in stats.iterrows():
#             ni = row["name_i"]; nj = row["name_j"]
#             Pi = name_to_pred[ni]; Pj = name_to_pred[nj]

#             if bool(row["equiv"]):
#                 conjs.append(ClassEquivalence(Pi, Pj))
#                 continue
#             if bool(row["imp_i_to_j"]):
#                 conjs.append(ClassInclusion(Pi, Pj))
#             if bool(row["imp_j_to_i"]):
#                 conjs.append(ClassInclusion(Pj, Pi))

#         return stats, conjs

#     def make_conjectures_from_logic(
#         self,
#         logic,
#         *,
#         use_sorted: bool = True,
#         **kwargs,
#     ) -> Tuple[pd.DataFrame, List[object]]:
#         """
#         Pull hypotheses from ClassLogic:
#           - use_sorted=True  → use logic.sort_by_generality()
#           - use_sorted=False → use logic.nonredundant() equivalent

#         kwargs are forwarded to make_conjectures (thresholds).
#         """
#         hyps = logic.sort_by_generality() if use_sorted else [*logic.nonredundant()]
#         return self.make_conjectures(hyps, **kwargs)


# src/txgraffiti2025/relations/class_relations_miner.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
import pandas as pd

from .core import DataModel, MaskCache
from ..forms.predicates import Predicate
from ..forms.class_relations import ClassInclusion, ClassEquivalence


def _resolve_min_support(min_support: Union[int, float], N: int) -> int:
    """
    Float in (0,1] → ceil(frac * N). Int → clamp to [0,N]. 0 → 0.
    """
    if isinstance(min_support, float):
        if min_support <= 0:
            return 0
        if min_support <= 1:
            return max(0, min(int(np.ceil(min_support * N)), N))
        return max(0, min(int(min_support), N))
    # int
    return max(0, min(int(min_support), N))


@dataclass(slots=True)
class ClassRelationsMiner:
    """
    Mine logical relations between boolean hypotheses (classes):

      • Inclusion:      H_i ⇒ H_j
      • Equivalence:    H_i ⇔ H_j

    Produces a diagnostics DataFrame (pairwise stats) and a list of
    ClassInclusion / ClassEquivalence objects, with optional persistence
    into DataModel.registry["class_relations"].
    """
    model: DataModel
    cache: MaskCache

    # --------------------------- internal utils ---------------------------

    def _mask(self, pred: Predicate) -> np.ndarray:
        """Cached boolean mask as dense ndarray[bool]."""
        s = self.cache.mask(pred)
        return s.to_numpy(dtype=np.bool_, copy=False)

    # --------------------------- public API -------------------------------

    def analyze(
        self,
        hyps: List[Tuple[str, Predicate]],
        *,
        min_support: Union[int, float] = 0.05,   # int (absolute) or float in (0,1]
        max_violations: int = 0,
        min_implication_rate: float = 1.0,       # require n(A∧B)/n(A) ≥ this
        consider_empty: bool = False,
        topk: Optional[int] = None,
        drop_equal_pairs_for_implications: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of pairwise stats with implication/equivalence flags.

        Columns:
          name_i, name_j,
          nA, nB, nI, nU,
          prec_i_to_j, rec_i_to_j, prec_j_to_i, rec_j_to_i, jaccard,
          viol_i_to_j, viol_j_to_i,
          imp_i_to_j, imp_j_to_i, equiv
        """
        if not hyps:
            return pd.DataFrame()

        N = len(self.model.df)
        min_sup_abs = _resolve_min_support(min_support, N)
        rows: List[Dict[str, Any]] = []

        # Precompute masks/sizes
        masks: Dict[str, np.ndarray] = {}
        sizes: Dict[str, int] = {}
        for name, pred in hyps:
            m = self._mask(pred)
            masks[name] = m
            sizes[name] = int(m.sum())

        # Pairwise stats
        for i in range(len(hyps)):
            name_i, _ = hyps[i]
            m_i = masks[name_i]; nA = sizes[name_i]

            if (not consider_empty) and nA == 0:
                continue
            if nA < min_sup_abs:
                continue

            for j in range(i + 1, len(hyps)):
                name_j, _ = hyps[j]
                m_j = masks[name_j]; nB = sizes[name_j]

                if (not consider_empty) and nB == 0:
                    continue
                if nB < min_sup_abs:
                    continue

                inter = (m_i & m_j)
                union = (m_i | m_j)
                nI = int(inter.sum())
                nU = int(union.sum())

                same_mask = np.array_equal(m_i, m_j)
                if drop_equal_pairs_for_implications and same_mask:
                    # We'll still report equivalence below; we just won't score the
                    # two directions of implication redundantly.
                    pass

                viol_i_to_j = int((m_i & ~m_j).sum())
                viol_j_to_i = int((m_j & ~m_i).sum())

                # Rates
                rec_i_to_j = (nI / nA) if nA > 0 else 1.0   # of A, how many also B
                rec_j_to_i = (nI / nB) if nB > 0 else 1.0
                prec_i_to_j = (nI / nB) if nB > 0 else 1.0  # of B, how many are A
                prec_j_to_i = (nI / nA) if nA > 0 else 1.0
                jacc = (nI / nU) if nU > 0 else 1.0

                # Implications are about recall(A→B) and violations(A→B)
                imp_i_to_j = (viol_i_to_j <= int(max_violations)) and (rec_i_to_j >= float(min_implication_rate))
                imp_j_to_i = (viol_j_to_i <= int(max_violations)) and (rec_j_to_i >= float(min_implication_rate))

                # If masks are identical, force equivalence True
                equiv = bool(same_mask or (imp_i_to_j and imp_j_to_i))

                # Optionally suppress implication flags when masks equal,
                # while still reporting equiv=True cleanly.
                if same_mask and drop_equal_pairs_for_implications:
                    imp_i_to_j = False
                    imp_j_to_i = False

                rows.append(dict(
                    name_i=name_i, name_j=name_j,
                    nA=nA, nB=nB, nI=nI, nU=nU,
                    prec_i_to_j=prec_i_to_j, rec_i_to_j=rec_i_to_j,
                    prec_j_to_i=prec_j_to_i, rec_j_to_i=rec_j_to_i,
                    jaccard=jacc,
                    viol_i_to_j=viol_i_to_j, viol_j_to_i=viol_j_to_i,
                    imp_i_to_j=imp_i_to_j, imp_j_to_i=imp_j_to_i,
                    equiv=equiv,
                ))

        cols = [
            "name_i","name_j","nA","nB","nI","nU",
            "prec_i_to_j","rec_i_to_j","prec_j_to_i","rec_j_to_i",
            "jaccard","viol_i_to_j","viol_j_to_i",
            "imp_i_to_j","imp_j_to_i","equiv",
        ]
        out = pd.DataFrame(rows, columns=cols)
        if out.empty:
            return out

        out = out.sort_values(
            ["equiv","imp_i_to_j","imp_j_to_i","jaccard","rec_i_to_j","rec_j_to_i","name_i","name_j"],
            ascending=[False, False, False, False, False, False, True, True],
        ).reset_index(drop=True)

        if topk is not None and topk > 0:
            out = out.head(topk).reset_index(drop=True)
        return out

    def make_conjectures(
        self,
        hyps: List[Tuple[str, Predicate]],
        *,
        min_support: Union[int, float] = 0.05,
        max_violations: int = 0,
        min_implication_rate: float = 1.0,
        consider_empty: bool = False,
        topk: Optional[int] = None,
        drop_equal_pairs_for_implications: bool = False,
        persist_to_registry: bool = True,
    ) -> Tuple[pd.DataFrame, List[object]]:
        """
        Run `analyze` and construct ClassInclusion / ClassEquivalence objects.
        Optionally persist results to model.registry["class_relations"].
        """
        stats = self.analyze(
            hyps,
            min_support=min_support,
            max_violations=max_violations,
            min_implication_rate=min_implication_rate,
            consider_empty=consider_empty,
            topk=topk,
            drop_equal_pairs_for_implications=drop_equal_pairs_for_implications,
        )

        name_to_pred: Dict[str, Predicate] = {name: pred for name, pred in hyps}
        conjs: List[object] = []

        # Prepare registry bucket if present
        if "class_relations" not in self.model.registry:
            self.model.registry["class_relations"] = []

        for _, row in stats.iterrows():
            ni = row["name_i"]; nj = row["name_j"]
            Pi = name_to_pred[ni]; Pj = name_to_pred[nj]

            # Equivalence first
            if bool(row["equiv"]):
                obj = ClassEquivalence(Pi, Pj)
                conjs.append(obj)
                if persist_to_registry:
                    self.model.registry["class_relations"].append({
                        "type": "equiv",
                        "name_i": ni, "name_j": nj,
                        "nA": int(row["nA"]), "nB": int(row["nB"]), "nI": int(row["nI"]), "nU": int(row["nU"]),
                        "jaccard": float(row["jaccard"]),
                        "viol_i_to_j": int(row["viol_i_to_j"]), "viol_j_to_i": int(row["viol_j_to_i"]),
                    })
                # If equivalence, we don’t also log two implications unless you want redundancy.
                continue

            # Otherwise, record any implications that passed
            if bool(row["imp_i_to_j"]):
                obj = ClassInclusion(Pi, Pj)
                conjs.append(obj)
                if persist_to_registry:
                    self.model.registry["class_relations"].append({
                        "type": "incl", "dir": "i_to_j",
                        "name_i": ni, "name_j": nj,
                        "nA": int(row["nA"]), "nB": int(row["nB"]), "nI": int(row["nI"]), "nU": int(row["nU"]),
                        "rec": float(row["rec_i_to_j"]), "prec": float(row["prec_i_to_j"]),
                        "viol": int(row["viol_i_to_j"]),
                    })
            if bool(row["imp_j_to_i"]):
                obj = ClassInclusion(Pj, Pi)
                conjs.append(obj)
                if persist_to_registry:
                    self.model.registry["class_relations"].append({
                        "type": "incl", "dir": "j_to_i",
                        "name_i": nj, "name_j": ni,
                        "nA": int(row["nB"]), "nB": int(row["nA"]), "nI": int(row["nI"]), "nU": int(row["nU"]),
                        "rec": float(row["rec_j_to_i"]), "prec": float(row["prec_j_to_i"]),
                        "viol": int(row["viol_j_to_i"]),
                    })

        return stats, conjs

    # --------------------------- conveniences -----------------------------

    def make_conjectures_from_logic(
        self,
        logic,
        *,
        use_sorted: bool = True,
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[object]]:
        """
        Pull hypotheses from ClassLogic:
          - use_sorted=True  → use logic.sort_by_generality()
          - use_sorted=False → use logic.nonredundant() (auto-normalize if needed)
        """
        if use_sorted:
            if logic.sorted_by_generality() is None:
                logic.sort_by_generality()
            hyps = logic.sorted_by_generality()
        else:
            if logic.nonredundant() is None:
                logic.normalize()
            hyps = logic.nonredundant()

        return self.make_conjectures(hyps or [], **kwargs)

    def hyps_from_registry_classes(self) -> List[Tuple[str, Predicate]]:
        """
        Convert DataModel.registry['classes'] rows into (name, Predicate) pairs.
        Assumes the registry was populated by ClassLogic.to_registry().
        """
        rows = self.model.registry.get("classes", [])
        out: List[Tuple[str, Predicate]] = []
        for r in rows:
            # We saved only the name + (base_parts, extras). Rebuild from model.preds:
            parts = list(r.get("base_parts", [])) + list(r.get("extras", []))
            if not parts:
                # TRUE base as a convenience
                pred = Predicate.from_column(1, truthy_only=True)
                pred.name = "TRUE"
                out.append(("TRUE", pred))
                continue
            pred = None
            for p in parts:
                q = self.model.preds[p]
                pred = q if pred is None else (pred & q)
            out.append((r["name"], pred))
        return out

    def pretty_strings(self, conjs: List[object], *, unicode_ops: bool = True) -> List[str]:
        """
        Human-friendly strings for ClassInclusion / ClassEquivalence objects.
        """
        out: List[str] = []
        for c in conjs:
            out.append(c.pretty(unicode_ops=unicode_ops))
        return out
