#!/usr/bin/env python
# substructure_module_examples.py
# Educational walkthrough for R₅ substructure predicates.

from __future__ import annotations
import pandas as pd

from txgraffiti2025.forms.substructure import (
    SubstructurePredicate,
    exists_sub,
    not_exists_sub,
)
from txgraffiti2025.forms.predicates import Predicate


def banner(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78 + "\n")


def show_mask(title: str, pred: Predicate, df: pd.DataFrame):
    print(f"[{title}] {getattr(pred, 'pretty', pred.__repr__)()}")
    print("repr:", repr(pred))
    print("mask:")
    print(pred.mask(df).astype(int))
    print("")


# ------------------------------------------------------------------------------
# 0) Setup: a tiny “structured object” class we can test
# ------------------------------------------------------------------------------

class Dummy:
    def __init__(self, values):
        self.values = list(values)

    def has_even(self) -> bool:
        return any(v % 2 == 0 for v in self.values)

    def count_gt(self, t: int) -> int:
        return sum(v > t for v in self.values)

    def __repr__(self):
        return f"Dummy({self.values})"


def make_df():
    return pd.DataFrame(
        {
            "object": [
                Dummy([1, 3, 5]),    # no even
                Dummy([1, 2, 3]),    # has even
                None,                 # missing / malformed
                Dummy([2, 4, 6, 8]),  # all even
            ],
            "threshold": [3, 2, 10, 5],
            "flag": [True, False, True, True],
        },
        index=["A", "B", "C", "D"],
    )


def main():
    df = make_df()
    banner("0) DataFrame")
    print(df)

    # ------------------------------------------------------------------------------
    # 1) Basic existence test: fn(obj) -> bool
    # ------------------------------------------------------------------------------

    banner("1) Existence: object has an even element")
    C_even = SubstructurePredicate(
        lambda obj: obj.has_even(),
        name="has_even",
        on_error="false",  # robust to None / malformed objects
    )
    show_mask("C_even", C_even, df)

    banner("1b) Non-existence via logical negation (~)")
    show_mask("~C_even", ~C_even, df)

    # ------------------------------------------------------------------------------
    # 2) Using the convenience helpers exists_sub / not_exists_sub
    # ------------------------------------------------------------------------------

    banner("2) Convenience builders (exists_sub / not_exists_sub)")
    P_exists_even = exists_sub(lambda obj: obj.has_even(), name="has_even", on_error="false")
    P_no_even = not_exists_sub(lambda obj: obj.has_even(), name="no_even", on_error="false")
    show_mask("exists_sub(has_even)", P_exists_even, df)
    show_mask("not_exists_sub(has_even)", P_no_even, df)

    # ------------------------------------------------------------------------------
    # 3) Row-aware test: fn(obj, row) uses another column (threshold)
    # ------------------------------------------------------------------------------

    banner("3) Row-aware existence: count_gt(threshold) ≥ 2")
    # accepts_row → fn(obj, row)
    C_many_large = SubstructurePredicate(
        lambda obj, row: obj.count_gt(int(row["threshold"])) >= 2,
        accepts_row=True,
        on_error="false",
        name="count_gt(threshold) ≥ 2",
    )
    show_mask("C_many_large", C_many_large, df)

    # ------------------------------------------------------------------------------
    # 4) Combine with other predicates (AND / OR / NOT)
    # ------------------------------------------------------------------------------

    banner("4) Composition with other predicates")
    P_flag = Predicate.from_column("flag")   # (flag) column → predicate
    show_mask("flag", P_flag, df)

    # Only consider rows with flag=True and object has_even
    combo = P_flag & C_even
    show_mask("(flag ∧ has_even)", combo, df)

    # Existence OR row-aware existence
    disj = C_even | C_many_large
    show_mask("(has_even ∨ many_large)", disj, df)

    # ------------------------------------------------------------------------------
    # 5) Error handling demo
    # ------------------------------------------------------------------------------

    banner('5) Error handling: on_error="raise" vs "false"')
    # This fn intentionally raises if obj is None
    def will_raise(obj):
        return len(obj.values) > 0  # obj might be None

    # Robust form: errors become False
    robust = SubstructurePredicate(will_raise, name="len(values)>0", on_error="false")
    show_mask('robust (on_error="false")', robust, df)

    # Strict form: will raise on row C
    print('strict (on_error="raise") (expect exception)…')
    try:
        strict = SubstructurePredicate(will_raise, name="len(values)>0", on_error="raise")
        _ = strict.mask(df)
    except Exception as e:
        print("  caught:", type(e).__name__, "-", str(e))

    print("\nDone.\n")


if __name__ == "__main__":
    main()
