import pandas as pd
import numpy as np
from typing import NamedTuple, List
from pandas.api.types import is_bool_dtype

# --- TxGraffiti core imports ---
from txgraffiti.logic import Property, Predicate, Conjecture, Constant
from txgraffiti.generators import linear_programming
from txgraffiti import KnowledgeTable
from txgraffiti.example_data import graph_data

# -------------------------------------------------------------------
# 1) Biconditional‐mining helpers (strictly Boolean preds + equalities)
# -------------------------------------------------------------------
class BicondPred(NamedTuple):
    pred: Predicate
    precision: float
    recall: float

def find_tightness_predicates(
    df: pd.DataFrame,
    H_mask: pd.Series,
    tight_mask: pd.Series,
    bool_preds:     List[Predicate],
    equality_preds: List[Predicate],
    min_precision: float = 0.9,
    min_recall:    float = 0.9
) -> List[BicondPred]:
    N_tight = int(tight_mask.sum())
    if N_tight == 0:
        return []
    out: List[BicondPred] = []
    # Only scan the boolean‐dtype predicates
    for Q in bool_preds + equality_preds:
        mask = Q(df)
        # Skip anything not a Series of bools
        if not (isinstance(mask, pd.Series) and is_bool_dtype(mask.dtype)):
            continue
        P_H     = mask & H_mask
        tp      = int((P_H & tight_mask).sum())
        pred_sz = int(P_H.sum())
        if pred_sz == 0:
            continue
        prec = tp / pred_sz
        rec  = tp / N_tight
        if prec >= min_precision and rec >= min_recall:
            out.append(BicondPred(Q, prec, rec))
    # Sort by F1 score = 2·P·R/(P+R)
    out.sort(key=lambda b: 2*b.precision*b.recall/(b.precision+b.recall), reverse=True)
    return out

def refine_with_biconditional(
    conj: Conjecture,
    df:   pd.DataFrame,
    *,
    bool_preds:     List[Predicate],
    min_precision: float = 0.9,
    min_recall:    float = 0.9
):
    # 1) Masks
    H_mask     = conj.hypothesis(df).astype(bool)
    slack      = conj.conclusion.rhs(df) - conj.conclusion.lhs(df)
    tight_mask = H_mask & (slack == 0)

    # 2) Build equality predicates between numeric columns & small constants
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    equality_preds: List[Predicate] = []
    # feature == feature
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i+1:]:
            name = f"{c1}=={c2}"
            func = lambda tbl, c1=c1, c2=c2: tbl[c1] == tbl[c2]
            equality_preds.append(Predicate(name, func))
    # feature == constant
    for cst in [0,1,2,3]:
        for col in numeric_cols:
            name = f"{col}=={cst}"
            func = lambda tbl, col=col, cst=cst: tbl[col] == cst
            equality_preds.append(Predicate(name, func))

    # 3) Mine only boolean preds + these equalities
    cands = find_tightness_predicates(
        df, H_mask, tight_mask,
        bool_preds=bool_preds,
        equality_preds=equality_preds,
        min_precision=min_precision,
        min_recall=min_recall
    )
    if not cands:
        return conj, None
    return conj, cands[0].pred
