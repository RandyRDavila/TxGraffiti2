# demo_graffiti_base.py  (new-DSL only)
from __future__ import annotations
import pandas as pd

from txgraffiti2025.graffiti_base import GraffitiBase

# New numeric DSL
from txgraffiti2025.graffiti_utils import (
    Expr, ColumnTerm, sqrt, floor, ceil, log, abs_,
)

# New predicates + formulas
from txgraffiti2025.graffiti_predicates import Predicate, TRUE
from txgraffiti2025.graffiti_generic_conjecture import (
    BoolFormula, Conjecture, AllOf, ite, coerce_formula as F
)

def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name":       ["G1", "G2", "G3", "G4"],
            "connected":  [True, True,  False, True,],
            "nontrivial": [1,    1,     0,     1],
            "order":      [4,    5,     1,     9],
            "size":       [3,    7,     0,     9],
        }
    ).set_index("name")

def pick_expr(gb: GraffitiBase, name: str, fallback: str) -> Expr:
    cols = set(gb.get_expr_columns())
    if not cols:
        raise RuntimeError("No numeric columns available.")
    if name in cols:
        return gb.expr(name)
    if fallback in cols:
        return gb.expr(fallback)
    return gb.expr(next(iter(cols)))

def pick_pred(gb: GraffitiBase, name: str, fallback: str) -> Predicate:
    bcols = gb.get_boolean_columns()
    if not bcols:
        raise RuntimeError("No boolean-like columns available.")
    pick = name if name in bcols else (fallback if fallback in bcols else bcols[0])
    bp = gb.get_base_predicates()
    if pick in bp:
        return gb.pred(pick)
    if bp:
        return gb.pred(next(iter(bp.keys())))
    # Edge case: no non-base preds
    return gb.formula_to_predicate(TRUE)

def show_series(title: str, s: pd.Series) -> None:
    print(f"\n{title}:")
    print(s.astype(object))

def main(df: pd.DataFrame | None = None) -> None:
    # 0) Data & base
    if df is None:
        df = _toy_df()
    gb = GraffitiBase(df)
    gb.summary(verbose=True)

    # 1) Numeric expressions (new DSL via GraffitiBase.expr → ColumnTerm)
    order = pick_expr(gb, "order", "size")
    size  = pick_expr(gb, "size", "order")

    # Sanity: these must be new-DSL Expr objects
    assert hasattr(order, "evaluate") and isinstance(order, Expr.__mro__[0]), "order is not a new-DSL Expr"
    assert hasattr(size, "evaluate"), "size is not a new-DSL Expr"

    E_sum   = order + size
    E_diff  = order - size
    E_prod  = order * (size + 1)
    E_div   = order / (size + 1)
    E_fdiv  = order // 2
    E_mod   = order % 3
    E_pow   = (size + 1) ** 2
    E_sqrt  = sqrt(order)
    E_floor = floor(E_div)
    E_ceil  = ceil(E_div)
    E_log2  = log(order + 1, base=2)
    E_abs   = abs_(order - size)
    E_piece = ite(order >= size, order - size, size - order)

    for nm, ex in [
        ("E_sum", E_sum), ("E_diff", E_diff), ("E_prod", E_prod),
        ("E_div", E_div), ("E_fdiv", E_fdiv), ("E_mod", E_mod),
        ("E_pow", E_pow), ("E_sqrt", E_sqrt), ("E_floor", E_floor),
        ("E_ceil", E_ceil), ("E_log2", E_log2), ("E_abs", E_abs),
        ("E_piece(=|order-size|)", E_piece),
    ]:
        show_series(nm, ex.evaluate(gb.df))

    # 2) Predicates and formulas
    connected  = pick_pred(gb, "connected", "nontrivial")
    nontrivial = pick_pred(gb, "nontrivial", "connected")

    # 3) Relations (new DSL): MUST yield BoolFormula
    R_ge = (order >= size + 1)
    R_le = (order <= size + 2)
    R_eq = (order == size)
    R_lt = (order <  size)
    R_gt = (order >  size)

    # Ensure these are new BoolFormula objects, not legacy Compare
    for nm, rel in [("R_ge", R_ge), ("R_le", R_le), ("R_eq", R_eq), ("R_lt", R_lt), ("R_gt", R_gt)]:
        if not isinstance(rel, BoolFormula):
            raise TypeError(f"{nm} is {type(rel).__name__}, expected BoolFormula (new DSL). "
                            "Check that you're importing the new graffiti_utils & generic_conjecture.")

    # 4) Boolean formulas mixing preds & relations

    Pc = F(gb.predicates["connected"])    # base column → OK
    Pn = F(gb.pred("nontrivial"))

    P = Pc & Pn
    Q = Pc | ~Pn
    X = P ^ Q
    # P   = connected & nontrivial
    # Q   = connected | ~nontrivial
    # X   = P ^ Q
    Imp = P >> R_ge
    Bic = (R_ge & R_le).iff(R_eq)

    # Evaluate via new-DSL-aware helper
    for nm, f in [
        ("P = connected ∧ nontrivial", P),
        ("Q = connected ∨ ¬nontrivial", Q),
        ("X = P xor Q", X),
        ("R_ge: order ≥ size+1", R_ge),
        ("R_le: order ≤ size+2", R_le),
        ("R_eq: order = size", R_eq),
        ("R_lt: order < size", R_lt),
        ("R_gt: order > size", R_gt),
        ("Imp: P ⇒ (order ≥ size+1)", Imp),
        ("Bic: (R_ge ∧ R_le) ⇔ R_eq", Bic),
    ]:
        m = gb.mask_of_formula(f)
        print(f"\n{nm}  (true count: {int(m.sum())}/{len(m)})")
        print(pd.Series(m, index=gb.df.index).astype(bool))

    # 5) Conjectures
    C1 = Conjecture(relation=R_le, condition=P, name="Upper bound under P")
    C2 = Conjecture(relation=Bic, name="Characterization")
    for cj in [C1, C2]:
        m = cj.evaluate(gb.df)
        print(f"\nConjecture: {cj.name or '(unnamed)'}")
        print(" pretty:", cj.pretty())
        print(" holds:", int(m.sum()), "/", len(m))
        print(pd.Series(m, index=gb.df.index).astype(bool))

    # 6) compress_conjectures
    cjs = [
        Conjecture(relation=R_ge, condition=P, name="R_ge under P"),
        Conjecture(relation=R_le, condition=P, name="R_le under P"),
        Conjecture(relation=R_ge, condition=P, name="duplicate R_ge under P"),
        Conjecture(relation=R_eq, condition=None, name="eq global"),
    ]
    merged = gb.compress_conjectures(cjs)
    print("\nCompressed conjectures:")
    for cj in merged:
        print(" -", cj.name, " :: ", cj.pretty())

    # 7) Back-compat alias (still new DSL under the hood)
    alias_mask = gb.mask_of_relation(Imp)
    print("\nmask_of_relation(Imp) OK:", list(alias_mask))

if __name__ == "__main__":
    main()
