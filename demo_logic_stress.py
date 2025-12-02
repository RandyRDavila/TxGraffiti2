# demo_logic_stress.py
from __future__ import annotations
import pandas as pd
import numpy as np

from txgraffiti2025.graffiti_base import GraffitiBase
from txgraffiti2025.graffiti_predicates import TRUE
from txgraffiti2025.graffiti_generic_conjecture import (
    coerce_formula as F,
    BoolFormula, Iff,
)

def _df():
    # Mix: booleans, 0/1-coded, NA; some rows force tricky edge cases.
    return pd.DataFrame(
        {
            "name":       ["A","B","C","D","E","F","G","H"],
            # Make 'connected' universally true → becomes base predicate
            "connected":  [ True, True, True, True, True, True, True, True ],
            # 0/1-coded with an NA: still boolean-like; NA should be treated as False
            "nontrivial": [ 1,    1,    0,    1,    0,    np.nan, 1,    0 ],
            # Numbers to test relations (==, <, >, ranges, etc.)
            "order":      [ 4,    5,    1,    9,    3,    7,       2,    5 ],
            "size":       [ 3,    7,    1,    9,    8,    4,       2,    0 ],
        }
    ).set_index("name")

def as_bool_series(x, df: pd.DataFrame) -> pd.Series:
    if isinstance(x, BoolFormula):
        return x.evaluate(df).reindex(df.index).astype(bool, copy=False)
    raise TypeError("Expected BoolFormula")

def assert_series_equal(a: pd.Series, b: pd.Series, msg: str):
    a = a.reindex(b.index).astype(bool, copy=False)
    b = b.astype(bool, copy=False)
    if not np.array_equal(a.values, b.values):
        diffs = (a.astype(int) - b.astype(int)).rename("diff")
        raise AssertionError(f"{msg}\nA:\n{a}\nB:\n{b}\nDiff (A-B):\n{diffs}")

def main():
    df = _df()
    # from txgraffiti.example_data import graph_data as df
    gb = GraffitiBase(df)

    # Predicates (wrap as BoolFormula via coerce_formula)
    Pc = F(gb.predicates["connected"])
    Pn = F(gb.predicates["nontrivial"])

    # Numeric expressions
    order = gb.expr("order")
    size  = gb.expr("size")

    # Relations (same-RHS group)
    R_eq  = (order == size)
    R_lt  = (order <  size)
    R_gt  = (order >  size)
    R_le0 = (order <= size)
    R_ge0 = (order >= size)

    # Shifted-RHS set for extra checks
    R_le2 = (order <= size + 2)
    R_lt2 = (order <  size + 2)
    R_eq2 = (order == size + 2)
    R_ge2 = (order >= size + 2)
    R_gt2 = (order >  size + 2)

    # Helpers
    TRUE_F  = F(TRUE)
    FALSE_F = F(TRUE) & ~F(TRUE)

    # Composite formulas
    P   = Pc & Pn
    Q   = Pc | ~Pn
    X   = P ^ Q
    Imp = P >> Q
    Bic = (R_le0 & R_ge0).iff(R_eq)   # equivalent to (order == size)

    # ───────── Truth-table properties ─────────
    dfP = as_bool_series(P, df)
    dfQ = as_bool_series(Q, df)

    # Commutativity
    assert_series_equal(as_bool_series(P & Q, df), as_bool_series(Q & P, df), "AND commutativity failed")
    assert_series_equal(as_bool_series(P | Q, df), as_bool_series(Q | P, df), "OR commutativity failed")
    assert_series_equal(as_bool_series(P ^ Q, df), as_bool_series(Q ^ P, df), "XOR commutativity failed")

    # Associativity (spot-check)
    assert_series_equal(as_bool_series((P & Q) & Pc, df), as_bool_series(P & (Q & Pc), df), "AND associativity failed")
    assert_series_equal(as_bool_series((P | Q) | Pc, df), as_bool_series(P | (Q | Pc), df), "OR associativity failed")

    # Idempotence
    assert_series_equal(dfP, as_bool_series(P & P, df), "AND idempotence failed")
    assert_series_equal(dfP, as_bool_series(P | P, df), "OR idempotence failed")

    # Identity / Domination
    assert_series_equal(dfP, as_bool_series(P & TRUE_F, df),  "AND identity TRUE failed")
    assert_series_equal(as_bool_series(TRUE_F, df), as_bool_series(P | TRUE_F, df), "OR domination TRUE failed")
    assert_series_equal(as_bool_series(FALSE_F, df), as_bool_series(P & FALSE_F, df), "AND domination FALSE failed")
    assert_series_equal(dfP, as_bool_series(P | FALSE_F, df), "OR identity FALSE failed")

    # Double negation
    assert_series_equal(dfP, as_bool_series(~(~P), df), "Double negation failed")

    # De Morgan
    assert_series_equal(as_bool_series(~(P & Q), df), as_bool_series((~P) | (~Q), df), "De Morgan 1 failed")
    assert_series_equal(as_bool_series(~(P | Q), df), as_bool_series((~P) & (~Q), df), "De Morgan 2 failed")

    # Implication equivalence: P ⇒ Q == (~P) | Q
    assert_series_equal(as_bool_series(Imp, df), as_bool_series((~P) | Q, df), "Implication equivalence failed")

    # IFF equivalence: Iff(P,Q) == (P & Q) | (~P & ~Q)
    assert_series_equal(as_bool_series(Iff(P, Q), df),
                        as_bool_series((P & Q) | ((~P) & (~Q)), df),
                        "Iff equivalence failed")

    # XOR equivalences:
    #  (P ^ Q) == (P | Q) & ~(P & Q)
    assert_series_equal(as_bool_series(X, df),
                        as_bool_series((P | Q) & ~(P & Q), df),
                        "XOR (symmetric difference) equivalence failed")
    #  (P ^ TRUE) == ~P ; (P ^ FALSE) == P
    assert_series_equal(as_bool_series(P ^ TRUE_F, df), as_bool_series(~P, df), "XOR with TRUE failed")
    assert_series_equal(as_bool_series(P ^ FALSE_F, df), dfP,               "XOR with FALSE failed")

    # ───────── Relation sanity checks ─────────
    df_lt = as_bool_series(R_lt, df)
    df_gt = as_bool_series(R_gt, df)
    df_eq = as_bool_series(R_eq, df)

    # Mutually exclusive on each row: (<, >, ==)
    assert_series_equal((df_lt & df_gt), pd.Series(False, index=df.index), "< and > overlap")
    assert_series_equal((df_lt & df_eq), pd.Series(False, index=df.index), "< and == overlap")
    assert_series_equal((df_gt & df_eq), pd.Series(False, index=df.index), "> and == overlap")

    # Same-RHS identities (must use the SAME right-hand side!)
    df_le0 = as_bool_series(R_le0, df)           # order <= size
    df_ge0 = as_bool_series(R_ge0, df)           # order >= size
    assert_series_equal(df_le0, (df_lt | df_eq), "<= != (< or ==) with same RHS")
    assert_series_equal(df_ge0, (df_gt | df_eq), ">= != (> or ==) with same RHS")

    # Shifted-RHS identities
    df_le2 = as_bool_series(R_le2, df)
    df_lt2 = as_bool_series(R_lt2, df)
    df_eq2 = as_bool_series(R_eq2, df)
    df_ge2 = as_bool_series(R_ge2, df)
    df_gt2 = as_bool_series(R_gt2, df)

    assert_series_equal(df_le2, (df_lt2 | df_eq2),
                        "<= != (< or ==) when RHS is size+2")
    # Alternative algebraic shift form: (order - size) <= 2
    R_shift_form = ((order - size) <= 2)
    assert_series_equal(df_le2, as_bool_series(R_shift_form, df),
                        "(order <= size+2) != ((order - size) <= 2)")

    assert_series_equal(df_ge2, (df_gt2 | df_eq2),
                        ">= != (> or ==) when RHS is size+2")

    # ───────── Base integration checks (mask cache & conversions) ─────────
    for label, f in [
        ("P", P), ("Q", Q), ("X", X), ("Imp", Imp), ("Bic", Bic),
        ("R_eq", R_eq), ("R_lt", R_lt), ("R_gt", R_gt),
        ("R_le0", R_le0), ("R_ge0", R_ge0), ("R_le2", R_le2), ("R_ge2", R_ge2),
    ]:
        via_gb = gb.mask_of_formula(f)  # route through GraffitiBase
        direct = as_bool_series(f, df).to_numpy(dtype=bool, copy=False)
        if not np.array_equal(via_gb, direct):
            raise AssertionError(f"mask_of_formula mismatch for {label}")

    print("✅ Logic stress test passed: all boolean operators, relations, and equivalences hold.")

if __name__ == "__main__":
    main()
