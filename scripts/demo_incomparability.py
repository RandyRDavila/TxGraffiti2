# scripts/demo_incomparability.py
# Demo: Incomparability-driven feature mining on your graph dataset.
# - Loads txgraffiti.example_data.graph_data as df
# - Sets df['nontrivial'] = df['connected']
# - Builds DataModel + MaskCache
# - Uses ClassLogic to get the base class (connected ∧ nontrivial)
# - Runs IncomparabilityAnalyzer:
#     * analyze (diagnostics)
#     * register_absdiff_exprs_for_meaningful_pairs
#     * register_minmax_for_often_unequal_pairs
#     * register_minmax_exprs_for_meaningful_pairs
# - Prints compact summaries and registry heads

from __future__ import annotations

import sys
from inspect import signature
import pandas as pd

# ---------- helper: call only with supported kwargs ----------
def _call_supported(func, /, *args, **kwargs):
    """Call func with only those kwargs it supports (portable across versions)."""
    params = signature(func).parameters
    return func(*args, **{k: v for k, v in kwargs.items() if k in params})

# ---------- optional niceties ----------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except Exception:
    Console = None
    Table = None
    Panel = None

from txgraffiti.example_data import graph_data as df
from txgraffiti2025.relations.core import DataModel, MaskCache
from txgraffiti2025.relations.class_logic import ClassLogic
from txgraffiti2025.relations.incomparability import IncomparabilityAnalyzer


def _console():
    return Console() if Console is not None else None


def _print(console, text=""):
    if console is None:
        print(text)
    else:
        console.print(text)


def _df_head(dataframe: pd.DataFrame, n=8, cols=None) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()
    out = dataframe if cols is None else dataframe.loc[:, [c for c in cols if c in dataframe.columns]]
    return out.head(n)


def main():
    console = _console()

    # ── Load your data and set nontrivial = connected ────────────────────
    data = df.copy()
    if "connected" in data.columns:
        data["connected"] = data["connected"].astype(bool, copy=False)
    data["nontrivial"] = data["connected"]

    # ── Build model + cache ──────────────────────────────────────────────
    model = DataModel(data)
    cache = MaskCache(model)

    # Ensure registry buckets exist
    model.registry.setdefault("classes", [])
    model.registry.setdefault("absdiff", [])
    model.registry.setdefault("minmax", [])

    # Summary
    if Panel is not None:
        _print(console, Panel.fit(str(model.summary()), title="DataModel Summary"))
    else:
        _print(console, "=== DataModel ===")
        _print(console, str(model.summary()))
        _print(console)

    # ── Base logic (e.g., connected ∧ nontrivial) ────────────────────────
    logic = ClassLogic(model, cache)
    base_name = logic.base_name()
    base_pred = logic.base_predicate()
    base_parts = tuple(base_name.split(" ∧ ")) if base_name != "TRUE" else tuple()

    _print(console, f"Base name: {base_name}")
    _print(console, f"Base parts: {base_parts}\n")

    # Enumerate + normalize + sort hypothesis classes
    logic.enumerate(max_arity=2, min_support=2, include_base=True)
    logic.normalize()
    classes_sorted = logic.sort_by_generality()

    # Persist sorted nonredundant classes into the model registry
    model.registry["classes"].clear()
    for name, pred in classes_sorted:
        msum = int(cache.mask(pred).sum())
        parts = [] if name == "TRUE" else name.split(" ∧ ")
        extras = parts[len(base_parts):] if base_parts else parts
        model.registry["classes"].append({
            "name": name,
            "arity": len(extras),
            "support": msum,
            "extras": extras,
            "base_parts": list(base_parts),
        })

    # ── Incomparability-driven mining ────────────────────────────────────
    inc = IncomparabilityAnalyzer(model, cache)

    # 1) Diagnostics on base domain (portable call)
    diag = _call_supported(inc.analyze, condition=base_pred, use_base_if_none=True)
    cols_diag = [
        "inv1", "inv2", "n_rows", "n_lt", "n_gt", "n_eq",
        "rate_lt", "rate_gt", "rate_eq", "incomparable", "balance", "support"
    ]
    if Table is not None and not diag.empty:
        t = Table(title="Incomparability (diagnostics head)")
        for col in cols_diag:
            t.add_column(col, overflow="fold")
        for _, r in _df_head(diag, 8, cols_diag).iterrows():
            t.add_row(*(str(r.get(c, "")) for c in cols_diag))
        console.print(t)
    else:
        _print(console, "=== Incomparability (diagnostics head) ===")
        _print(console, _df_head(diag, 8, cols_diag).to_string(index=False))
    _print(console)

    # 2) Register |x−y| for meaningfully incomparable pairs (portable call)
    absdf = _call_supported(
        inc.register_absdiff_exprs_for_meaningful_pairs,
        condition=base_pred,
        use_base_if_none=True,
        require_finite=True,
        min_support=0.10,
        min_side_rate=0.10,
        min_side_count=5,
        min_median_gap=0.5,
        min_mean_gap=0.5,
        name_style="abs_x_minus_y",
        prefix="abs_",
        suffix="",
        overwrite_existing=False,
        top_n_store=20,
    )
    _print(console, "=== Registered abs(x-y) (head) ===")
    cols_abs = [
        "inv1", "inv2", "n_rows", "n_lt", "n_gt", "n_eq",
        "rate_lt", "rate_gt", "rate_eq", "support",
        "median_gap", "mean_gap", "selected", "expr_name"
    ]
    _print(console, _df_head(absdf, 8, cols_abs).to_string(index=False))
    _print(console)

    # Commit selected absdiff to registry
    if isinstance(absdf, pd.DataFrame) and not absdf.empty:
        use = absdf.loc[absdf.get("selected", False) & absdf.get("expr_name").notna()]
        for _, r in use.iterrows():
            model.record_absdiff(
                inv1=r["inv1"],
                inv2=r["inv2"],
                expr_name=r["expr_name"],
                hypothesis=base_name,
                support=float(r.get("support", 0.0)),
                median_gap=float(r.get("median_gap", float("nan"))),
                mean_gap=float(r.get("mean_gap", float("nan"))),
            )

    # 3) Register min/max for pairs that are often unequal (portable call)
    mm_unequal = _call_supported(
        inc.register_minmax_for_often_unequal_pairs,
        condition=base_pred,
        use_base_if_none=True,
        require_finite=True,
        min_support=0.10,
        min_neq_rate=0.50,
        min_neq_count=8,
        must_be_incomparable=False,
        prefix_min="min_",
        prefix_max="max_",
        suffix="",
        overwrite_existing=False,
        top_n_store=50,
    )
    _print(console, "=== Registered min/max (often unequal) (head) ===")
    cols_mm1 = [
        "inv1", "inv2", "n_rows", "n_lt", "n_gt", "n_eq",
        "rate_lt", "rate_gt", "rate_eq", "support",
        "often_unequal", "selected", "expr_min_name", "expr_max_name"
    ]
    _print(console, _df_head(mm_unequal, 8, cols_mm1).to_string(index=False))
    _print(console)

    # Commit selected min/max (often unequal)
    if isinstance(mm_unequal, pd.DataFrame) and not mm_unequal.empty:
        use = mm_unequal.loc[
            mm_unequal.get("selected", False)
            & mm_unequal.get("expr_min_name").notna()
            & mm_unequal.get("expr_max_name").notna()
        ]
        for _, r in use.iterrows():
            model.record_minmax(
                inv1=r["inv1"],
                inv2=r["inv2"],
                key_min=r["expr_min_name"],
                key_max=r["expr_max_name"],
                hypothesis=base_name,
                support=float(r.get("support", 0.0)),
                often_unequal=bool(r.get("often_unequal", False)),
            )

    # 4) Register min/max for meaningfully incomparable pairs (portable call)
    mm_meaning = _call_supported(
        inc.register_minmax_exprs_for_meaningful_pairs,
        condition=base_pred,
        use_base_if_none=True,
        require_finite=True,
        min_support=0.10,
        min_side_rate=0.10,
        min_side_count=5,
        max_eq_rate=0.70,
        key_style="slug",
        overwrite_existing=False,
    )
    _print(console, "=== Registered min/max (meaningful incomparability) (head) ===")
    cols_mm2 = [
        "inv1", "inv2", "n_rows", "n_lt", "n_gt", "n_eq",
        "rate_lt", "rate_gt", "rate_eq", "support",
        "selected", "key_min", "key_max"
    ]
    _print(console, _df_head(mm_meaning, 8, cols_mm2).to_string(index=False))
    _print(console)

    # Commit selected min/max (meaningful incomparability)
    if isinstance(mm_meaning, pd.DataFrame) and not mm_meaning.empty:
        use = mm_meaning.loc[
            mm_meaning.get("selected", False)
            & mm_meaning.get("key_min").notna()
            & mm_meaning.get("key_max").notna()
        ]
        for _, r in use.iterrows():
            model.record_minmax(
                inv1=r["inv1"],
                inv2=r["inv2"],
                key_min=r["key_min"],
                key_max=r["key_max"],
                hypothesis=base_name,
                support=float(r.get("support", 0.0)),
                often_unequal=True,  # by construction here
            )

    # ── Final registry snapshots ─────────────────────────────────────────
    reg = model.summary()["registry_counts"]
    _print(console, "=== Registry counts ===")
    _print(console, str(reg))
    _print(console)

    abs_frame = model.registry_frame("absdiff")
    mm_frame = model.registry_frame("minmax")

    _print(console, "registry['absdiff'] head:")
    _print(console, _df_head(abs_frame, 6).to_string(index=False) if not abs_frame.empty else "(empty)")
    _print(console)

    _print(console, "registry['minmax'] head:")
    _print(console, _df_head(mm_frame, 6).to_string(index=False) if not mm_frame.empty else "(empty)")
    _print(console)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e, file=sys.stderr)
        raise
