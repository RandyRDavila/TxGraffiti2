from __future__ import annotations

import sys, os, io, datetime as _dt
from contextlib import contextmanager
from typing import Sequence, Tuple, Union, Optional, Dict, List, Iterable
import numpy as np
import pandas as pd
import os
import datetime as _dt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List




from txgraffiti2025.graffiti_relations import GraffitiClassRelations
from txgraffiti2025.forms.utils import Expr, ColumnTerm, Const
from txgraffiti2025.forms.predicates import Predicate, Where
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE

# Your imports
from txgraffiti2025.graffiti_relations import GraffitiClassRelations
from txgraffiti2025.graffiti_lp import GraffitiLP, LPFitConfig
from txgraffiti2025.graffiti_lp_lift_integer_aware import lift_integer_aware
from txgraffiti2025.graffiti_intricate_mixed import GraffitiLPIntricate
from txgraffiti2025.graffiti_qualitative import GraffitiQualitative
from txgraffiti2025.graffiti_asymptotic_miner import AsymptoticMiner, AsymptoticSearchConfig
from txgraffiti2025.graffiti_lp_lift_integer_aware import (
    lift_and_process_sorted,
    process_and_sort_conjectures,   # if you want to call it directly
)


class _Tee(io.TextIOBase):
    """Write-through stream that mirrors writes to multiple targets."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)
    def flush(self):
        for st in self._streams:
            st.flush()

def _hr(ch: str = "─", n: int = 80) -> str:
    return ch * n

def _now_stamp() -> str:
    # America/Chicago assumed by your environment; adjust if needed.
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@contextmanager
def tee_report(filepath: str, *, title: str = "TxGraffiti Run Report", include_stderr: bool = True):
    """
    Duplicate all prints to `filepath` (UTF-8). Adds a nice header/footer.
    Usage:
        with tee_report("reports/run.txt", title="My Run"):
            ... your existing code full of print() ...
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    f = open(filepath, "w", encoding="utf-8", newline="\n")
    # header
    header = (
        f"{_hr()}\n"
        f"{title}\n"
        f"{_hr()}\n"
        f"Started: {_now_stamp()}\n"
        f"Working dir: {os.getcwd()}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"{_hr()}\n"
    )
    f.write(header); f.flush()

    old_out = sys.stdout
    old_err = sys.stderr
    try:
        sys.stdout = _Tee(old_out, f)
        if include_stderr:
            sys.stderr = _Tee(old_err, f)
        yield
    finally:
        # footer
        print(_hr())
        print("END OF REPORT")
        print(f"Finished: {_now_stamp()}")
        print(_hr())
        sys.stdout.flush()
        if include_stderr:
            sys.stderr.flush()
        sys.stdout = old_out
        sys.stderr = old_err
        f.close()

def _as_number(v, default=None):
    if v is None:
        return default
    try:
        v = v() if callable(v) else v
        if hasattr(v, "item") and callable(getattr(v, "item")):
            v = v.item()
        return int(v) if isinstance(v, (int, bool)) else float(v)
    except Exception:
        return default

def _mask_for(df: pd.DataFrame, H) -> np.ndarray:
    if H is None or H is TRUE:
        return np.ones(len(df), dtype=bool)
    s = H.mask(df) if hasattr(H, "mask") else H(df)
    return np.asarray(s, dtype=bool)

def _compute_touch_support_batch(df: pd.DataFrame, conjs: Sequence[Conjecture],
                                 rtol: float = 1e-9, atol: float = 1e-9) -> Dict[int, tuple[int,int,float]]:
    """
    Batch compute (touch_count, support_n, touch_rate) for all conjectures.
    Groups by (H, L, R) so we eval each triplet once.
    Returns: {index -> (tc, sup, rate)}
    """
    if not conjs:
        return {}
    # local caches
    def _pred_key(p):
        if p is None or p is TRUE: return "TRUE"
        n = getattr(p, "name", None)
        return f"name:{n}" if n else f"repr:{repr(p)}"
    def _expr_key(e): return repr(e)

    groups: dict[tuple[str, str, str], list[int]] = {}
    for i, c in enumerate(conjs):
        groups.setdefault((_pred_key(c.condition),
                           _expr_key(c.relation.left),
                           _expr_key(c.relation.right)), []).append(i)

    # reusable caches
    mcache = {}
    arrcache = {}
    def _mask(pred):
        k = _pred_key(pred)
        if k in mcache: return mcache[k]
        m = _mask_for(df, pred)
        mcache[k] = m
        return m
    def _arr(expr):
        k = _expr_key(expr)
        a = arrcache.get(k)
        if a is None:
            s = expr.eval(df)
            if hasattr(s, "to_numpy"):
                a = s.to_numpy(dtype=float, copy=False)
            else:
                a = np.asarray(s, dtype=float)
                if a.ndim == 0:
                    a = np.full(len(df), float(a), dtype=float)
            arrcache[k] = a
        return a

    out: Dict[int, tuple[int,int,float]] = {}
    for (Hk, Lk, Rk), idxs in groups.items():
        rep = conjs[idxs[0]]
        Hm = _mask(rep.condition)
        if not np.any(Hm):
            for j in idxs:
                out[j] = (0, 0, 0.0)
            continue
        L = _arr(rep.relation.left)
        R = _arr(rep.relation.right)
        ok = np.isfinite(L) & np.isfinite(R) & Hm
        sup = int(ok.sum())
        if sup == 0:
            tc, rate = 0, 0.0
        else:
            eq = np.isclose(L[ok], R[ok], rtol=rtol, atol=atol)
            tc = int(eq.sum())
            rate = float(tc / sup)
        for j in idxs:
            out[j] = (tc, sup, rate)
    return out

def _dedup_by_string(conjs: Sequence[Conjecture]) -> list[Conjecture]:
    seen, out = set(), []
    for c in conjs:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        out.append(c)
    return out

def _annotate_touch_support(df: pd.DataFrame, conjs: list[Conjecture]) -> None:
    stats = _compute_touch_support_batch(df, conjs)
    for i, c in enumerate(conjs):
        tc, sup, rate = stats.get(i, (0, 0, 0.0))
        setattr(c, "touch_count", int(_as_number(getattr(c, "touch_count", tc), default=tc) or tc))
        setattr(c, "support_n",  int(_as_number(getattr(c, "support_n",  sup), default=sup) or sup))
        setattr(c, "touch_rate", float(_as_number(getattr(c, "touch_rate", rate), default=rate) or rate))

def _sort_by_touch_support(conjs: list[Conjecture]) -> list[Conjecture]:
    def key(c):
        return (int(getattr(c, "touch_count", 0) or 0),
                int(getattr(c, "support_n", 0) or 0))
    return sorted(conjs, key=key, reverse=True)

def _pretty_safe(c) -> str:
    try:
        return c.pretty(show_tol=False)
    except Exception:
        return str(c)

def print_bank(bank: dict[str, list[Conjecture]], k_per_bucket: int = 10, title: str = "FULL CONJECTURE LIST"):
    def section(title: str):
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80 + "\n")
    section(title)
    for name in ("lowers", "uppers", "equals"):
        lst = bank.get(name, [])
        print(f"[{name.upper()}] total={len(lst)}\n")
        for c in lst[:k_per_bucket]:
            print("•", _pretty_safe(c))
            print(f"    touches={getattr(c, 'touch_count', '?')}, support={getattr(c, 'support_n', '?')}")
        print()

def finalize_conjecture_bank(
    df: pd.DataFrame,
    all_lowers: list[Conjecture],
    all_uppers: list[Conjecture],
    all_equals: list[Conjecture],
    *,
    top_k_per_bucket: int = 100,
    apply_morgan: bool = True,
) -> dict[str, list[Conjecture]]:
    """
    1) Dedup per bucket,
    2) annotate touches/support (batch),
    3) sort by (touch_count, support_n),
    4) (optional) Morgan filter per bucket,
    5) take top_k_per_bucket per bucket,
    6) return dict and print summary.
    """
    from txgraffiti2025.processing.post import morgan_filter

    lowers = _dedup_by_string(all_lowers)
    uppers = _dedup_by_string(all_uppers)
    equals = _dedup_by_string(all_equals)

    # annotate
    _annotate_touch_support(df, lowers)
    _annotate_touch_support(df, uppers)
    _annotate_touch_support(df, equals)

    # sort
    lowers = _sort_by_touch_support(lowers)
    uppers = _sort_by_touch_support(uppers)
    equals = _sort_by_touch_support(equals)

    # optional Morgan filter (validity/pruning)
    if apply_morgan:
        lowers = list(morgan_filter(df, lowers).kept)
        uppers = list(morgan_filter(df, uppers).kept)
        equals = list(morgan_filter(df, equals).kept)

        # re-annotate after Morgan (in case of any recomputation)
        _annotate_touch_support(df, lowers)
        _annotate_touch_support(df, uppers)
        _annotate_touch_support(df, equals)

        lowers = _sort_by_touch_support(lowers)
        uppers = _sort_by_touch_support(uppers)
        equals = _sort_by_touch_support(equals)

    # take top-k
    lowers = lowers[:top_k_per_bucket]
    uppers = uppers[:top_k_per_bucket]
    equals = equals[:top_k_per_bucket]

    bank = {"lowers": lowers, "uppers": uppers, "equals": equals}
    return bank

# --- minimal shims you already have in your file ---
def section(title: str):
    print("\n" + "-" * 80)
    print(f"{title.upper()}")
    print("-" * 80 + "\n")

def print_conjs(label: str, conjs, n=6):
    print(f"[{label.upper()}] total={len(conjs)}\n")
    for c in conjs[:n]:
        try:
            s = c.pretty(show_tol=False)
        except Exception:
            s = str(c)
        print("•", s)
        t = getattr(c, "touch_count", "?")
        s_n = getattr(c, "support_n", "?")
        print(f"    touches={t}, support={s_n}\n")

# ────────────────────────────────────────────────────────────────────
# Equality-class helpers
# ────────────────────────────────────────────────────────────────────
def _top_equality_classes(gcr_obj: GraffitiClassRelations,
                          eq_summary: pd.DataFrame,
                          *,
                          k: int = 12,
                          min_rows: int = 25,
                          side: str = "auto") -> List[tuple[str, Any, float, int]]:
    """
    Select top equality classes from an analyze_ratio_bounds_on_base() summary.
    Returns tuples: (name, Predicate, best_rate, n_rows)
    """
    tmp = eq_summary.copy()
    if tmp.empty:
        return []
    tmp["best_rate"] = np.where(tmp["touch_lower_rate"] >= tmp["touch_upper_rate"],
                                tmp["touch_lower_rate"], tmp["touch_upper_rate"])
    tmp["best_side"] = np.where(tmp["touch_lower_rate"] >= tmp["touch_upper_rate"], "lower", "upper")
    tmp["score"] = tmp["best_rate"] * np.log1p(tmp["n_rows"])
    tmp = tmp[(tmp["n_rows"] >= min_rows)]
    if tmp.empty:
        return []
    tmp["_pairkey"] = tmp.apply(lambda r: tuple(sorted([r["inv1"], r["inv2"]])), axis=1)
    tmp = tmp.sort_values(["score"], ascending=False).drop_duplicates("_pairkey", keep="first")
    sel = tmp.head(k)

    picks = []
    for _, row in sel.iterrows():
        which = row["best_side"] if side == "auto" else side
        name, pred = gcr_obj.spawn_equality_classes_from_ratio_row(row, which=which, tol=0.0)
        picks.append((name, pred, float(row["best_rate"]), int(row["n_rows"])))
    return picks


def _build_df_with_eq_booleans(df, gcr, top_eqs):
    """
    NEW: keep ALL original columns and just add the equality-class booleans.
    """
    df_eq = df.copy()  # <- keep everything
    print("Selected equality classes:")
    for name, pred, rate, nrows in top_eqs:
        mask = gcr._mask_cached(pred)
        colname = name
        df_eq[colname] = mask
        print(f"  • {name:40s} (tightness≈{rate:.3f}, n={nrows})  → added as boolean '{colname}'")
    return df_eq



# ────────────────────────────────────────────────────────────────────
# Main reusable pipeline
# ────────────────────────────────────────────────────────────────────
def run_conjecturing_pipeline(
    df: pd.DataFrame,
    *,
    target: str,
    report_title: str = "TxGraffiti • Full Program Run Report",
    enable_eq_bootstrap: bool = True,
    eq_top_k: int = 12,
    eq_min_rows: int = 25,
    ratio_min_support: float = 0.05,
    ratio_pos_denominator: bool = True,
    ratio_touch_atol: float = 0.0,
    ratio_touch_rtol: float = 0.0,
    asym_min_abs_rho: float = 0.45,
    asym_tail_quantile: float = 0.75,
    asym_min_support_n: int = 20,
    k_affine_values: tuple = (1, 2, 3),
    k_affine_hypotheses_limit: int = 20,
    k_affine_min_touch: int = 3,
    k_affine_max_denom: int = 20,
    k_affine_top_m_by_variance: int = 10,
    affine_max_denom: int = 20,
    intricate_weight: float = 0.5,
    intricate_min_touch: int = 10,
    top_k_per_bucket: int = 100,
    tee: Optional[Any] = None,  # pass your tee_report(...) context manager
    finalize_conjecture_bank=None,  # inject the function you already have
    post_prune=None,                # optional hook you wrote earlier
    print_bank=None,                # inject the function you already have
) -> Dict[str, Any]:
    """
    Run the full conjecturing pipeline on df for a given target.
    Returns a dict with final bank and intermediate banks.
    """
    if finalize_conjecture_bank is None or print_bank is None:
        raise ValueError("Please pass finalize_conjecture_bank=... and print_bank=...")

    # --- accumulators ---
    ALL_LOWERS: List = []
    ALL_UPPERS: List = []
    ALL_EQUALS: List = []

    # --- Engines on the original df ---
    gcr = GraffitiClassRelations(df)
    qual = GraffitiQualitative(gcr)

    # Qualitative relations (light preview)
    section("Qualitative Relations (top 12)")
    results = qual.generate_qualitative_relations(
        y_targets=[target],
        method="spearman",
        min_abs_rho=0.4,
        min_n=12,
        top_k_per_hyp=5,
    )
    GraffitiQualitative.print_sample(results, k=12)

    # Asymptotics from qualitative signals
    section("Asymptotic limits (qualitative → definite)")
    miner = AsymptoticMiner(
        gcr,
        cfg=AsymptoticSearchConfig(
            min_abs_rho=asym_min_abs_rho,
            tail_quantile=asym_tail_quantile,
            min_support_n=asym_min_support_n,
        ),
    )
    asym_conjs = miner.generate_asymptotics_for_target(
        target=target,
        hyps=getattr(gcr, "nonredundant_conjunctions_", None),
    )
    print(f"[ASYMPTOTIC] total={len(asym_conjs)}\n")
    for c in asym_conjs[:10]:
        print("•", c.pretty())

    # GCR summary
    section("GraffitiClassRelations summary")
    gcr = GraffitiClassRelations(df)  # ensure fresh init
    print("Boolean columns:", gcr.boolean_cols)
    print("Expr columns:", gcr.expr_cols)
    print("Base hypothesis:", gcr.base_hypothesis_name)

    gcr.enumerate_conjunctions(max_arity=2)
    nonred, _, _ = gcr.find_redundant_conjunctions()
    print("\nNonredundant conjunctions:")
    for n, _ in nonred[:8]:
        print(" ", n)
    print()

    # Atomic examples
    atomic = gcr.build_constant_conjectures(tol=0.0, group_per_hypothesis=False)
    print("Example atomic conjectures:\n")
    for c in atomic[:5]:
        from textwrap import indent
        print(indent(c.pretty(), "  "))
        print()

    gcr.characterize_constant_classes(tol=0.0, group_per_hypothesis=True, limit=50)
    gcr.print_class_characterization_summary()

    # Affine LP fits (original df)
    section("Affine LP fits")
    lp = GraffitiLP(gcr)
    features = [c for c in lp.invariants if c != target]
    cfg = LPFitConfig(
        target=target,
        features=features,
        direction="both",
        max_denom=affine_max_denom,
    )
    lowers, uppers, equals = lp.fit_affine(cfg)
    print_conjs("lower", lowers)
    print_conjs("upper", uppers)
    print_conjs("equal", equals)
    ALL_LOWERS.extend(lowers); ALL_UPPERS.extend(uppers); ALL_EQUALS.extend(equals)

    # K-affine bounds (original df)
    section("K2 affine bounds generation")
    res = lp.generate_k_affine_bounds(
        target=target,
        k_values=k_affine_values,
        hypotheses_limit=k_affine_hypotheses_limit,
        min_touch=k_affine_min_touch,
        max_denom=k_affine_max_denom,
        top_m_by_variance=k_affine_top_m_by_variance,

    )
    print_conjs("lower", res["lowers"])
    print_conjs("upper", res["uppers"])
    print_conjs("equal", res["equals"])
    ALL_LOWERS.extend(res["lowers"]); ALL_UPPERS.extend(res["uppers"]); ALL_EQUALS.extend(res["equals"])

    # Integer-aware lifting
    section("Integer-aware lifting")
    low1 = lift_integer_aware(df=lp.df, gcr=lp.gcr, conjectures=res["lowers"])
    up1  = lift_integer_aware(df=lp.df, gcr=lp.gcr, conjectures=res["uppers"])
    print(f"Before vs After (lowers): {len(res['lowers'])} → {len(low1)}")
    print(f"Before vs After (uppers): {len(res['uppers'])} → {len(up1)}\n")

    for old, new in zip(res["lowers"], low1):
        if old.signature() != new.signature():
            print("Refined:", new.pretty(), f"| touches={getattr(new, 'touch_count', '?')}\n")
    ALL_LOWERS.extend(low1); ALL_UPPERS.extend(up1)

    # Intricate mixed inequalities (original df)
    section("Intricate mixed inequalities")
    lp_intr = GraffitiLPIntricate(df)
    print("Boolean columns:", lp_intr.bool_columns)
    print("Numeric columns:", lp_intr.numeric_columns[:8], "...")
    print("Base hypothesis:", lp_intr.base_hyp, "\n")

    res_intr = lp_intr.run_intricate_mixed_pipeline(
        target_col=target,
        weight=intricate_weight,
        min_touch=intricate_min_touch,
    )
    print_conjs("lower", res_intr["lowers"])
    print_conjs("upper", res_intr["uppers"])
    print_conjs("equal", res_intr["equals"])
    ALL_LOWERS.extend(res_intr["lowers"]); ALL_UPPERS.extend(res_intr["uppers"]); ALL_EQUALS.extend(res_intr["equals"])

    # Optional post-prune before equality bootstrap
    if post_prune is not None:
        ALL_LOWERS, ALL_UPPERS, ALL_EQUALS = post_prune(df, ALL_LOWERS, ALL_UPPERS, ALL_EQUALS)

    # ── Equality-class bootstrap pass (optional) ──────────────────────
    df_eq = df
    if enable_eq_bootstrap:
        section("Equality-class hypotheses bootstrap → rebuild engines")

        summary, _base_conjs = gcr.analyze_ratio_bounds_on_base(
            min_support=ratio_min_support,
            positive_denominator=ratio_pos_denominator,
            touch_atol=ratio_touch_atol,
            touch_rtol=ratio_touch_rtol,
        )
        top_eqs = _top_equality_classes(gcr, summary, k=eq_top_k, min_rows=eq_min_rows, side="auto")
        df_eq = _build_df_with_eq_booleans(df, gcr, top_eqs)

        # Rebuild engines
        gcr_eq = GraffitiClassRelations(df_eq)
        lp_eq = GraffitiLP(gcr_eq)
        lp_intr_eq = GraffitiLPIntricate(df_eq)

        features_eq = [c for c in lp_eq.invariants if c != target]
        print("\nBoolean columns in rebuilt frame (includes equality-class hypotheses):")
        print(gcr_eq.boolean_cols)

        section("Affine LP fits (equality-class hypotheses)")
        cfg_eq = LPFitConfig(
            target=target,
            features=features_eq,
            direction="both",
            max_denom=affine_max_denom,
        )
        lowers_eq, uppers_eq, equals_eq = lp_eq.fit_affine(cfg_eq)
        print_conjs("lower", lowers_eq)
        print_conjs("upper", uppers_eq)
        print_conjs("equal", equals_eq)
        ALL_LOWERS.extend(lowers_eq); ALL_UPPERS.extend(uppers_eq); ALL_EQUALS.extend(equals_eq)

        section("K2 affine bounds (equality-class hypotheses)")
        res_eq = lp_eq.generate_k_affine_bounds(
            target=target,
            k_values=k_affine_values,
            hypotheses_limit=k_affine_hypotheses_limit,
            min_touch=k_affine_min_touch,
            max_denom=k_affine_max_denom,
            top_m_by_variance=k_affine_top_m_by_variance,
        )
        print_conjs("lower", res_eq["lowers"])
        print_conjs("upper", res_eq["uppers"])
        print_conjs("equal", res_eq["equals"])
        ALL_LOWERS.extend(res_eq["lowers"]); ALL_UPPERS.extend(res_eq["uppers"]); ALL_EQUALS.extend(res_eq["equals"])

        section("Intricate mixed inequalities (equality-class hypotheses)")
        res_intr_eq = lp_intr_eq.run_intricate_mixed_pipeline(
            target_col=target,
            weight=intricate_weight,
            min_touch=intricate_min_touch,
        )
        print_conjs("lower", res_intr_eq["lowers"])
        print_conjs("upper", res_intr_eq["uppers"])
        print_conjs("equal", res_intr_eq["equals"])
        ALL_LOWERS.extend(res_intr_eq["lowers"]); ALL_UPPERS.extend(res_intr_eq["uppers"]); ALL_EQUALS.extend(res_intr_eq["equals"])

        # Optional post-prune after bootstrap
        if post_prune is not None:
            ALL_LOWERS, ALL_UPPERS, ALL_EQUALS = post_prune(df_eq, ALL_LOWERS, ALL_UPPERS, ALL_EQUALS)

    df_final = df_eq
    # ── Finalize on df_eq (it contains all needed columns now) ────────
    FINAL = finalize_conjecture_bank(
        df_final,
        ALL_LOWERS, ALL_UPPERS, ALL_EQUALS,
        top_k_per_bucket=top_k_per_bucket,
        apply_morgan=True,
    )

    # Pretty print final bank
    print_bank(FINAL, k_per_bucket=12, title="FULL CONJECTURE LIST (DEDUPED • RANKED • TOP-K)")

    return {
        "final_bank": FINAL,
        "all_lowers": ALL_LOWERS,
        "all_uppers": ALL_UPPERS,
        "all_equals": ALL_EQUALS,
        "df_final": df_eq,
    }


# ────────────────────────────────────────────────────────────────────
# Example __main__: keep tiny and call the function
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Prepare report path + tee
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/txgraffiti_run_{stamp}.txt"

    # Your tee context manager should be in scope:
    # from yourmodule.report_utils import tee_report, finalize_conjecture_bank, print_bank, post_prune

    from txgraffiti.example_data import graph_data as df
    # # df.drop(columns=[''], inplace=True)
    df['nontrivial'] = df['connected']
    TARGET = 'zero_forcing_number'
    # df['nontrivial'] = df['connected']
    df.drop(columns=['cograph', 'vertex_cover_number'], inplace=True)
    # Load your number theory CSV (or build on the fly)
    # df = pd.read_csv("polytope_data_full-2.csv")  # columns like n, q2, r2, ... as you defined

    # df.drop(columns=['|p3 - p6|','|p4 - p6|','|p5 - p6|','|p6 - p7|', 'fullerene', 'Unnamed: 0'], inplace=True)

    # TARGET = 'temperature(p6)'
    with tee_report(report_path, title="TxGraffiti • Full Program Run Report"):
        out = run_conjecturing_pipeline(
            df,
            target=TARGET,
            report_title="TxGraffiti • Full Program Run Report",
            enable_eq_bootstrap=True,
            eq_top_k=22,
            eq_min_rows=25,
            ratio_min_support=0.05,
            ratio_pos_denominator=True,
            ratio_touch_atol=0.0,
            ratio_touch_rtol=0.0,
            asym_min_abs_rho=0.45,
            asym_tail_quantile=0.55,
            asym_min_support_n=20,
            k_affine_values=(1, 2, 3, 4),
            k_affine_hypotheses_limit=25,
            k_affine_min_touch=3,
            k_affine_max_denom=20,
            k_affine_top_m_by_variance=10,
            affine_max_denom=20,
            intricate_weight=0.5,
            intricate_min_touch=3,
            top_k_per_bucket=100,
            tee=tee_report,
            finalize_conjecture_bank=finalize_conjecture_bank,
            # post_prune=post_prune,           # optional; pass None to skip
            print_bank=print_bank,
        )

    print(f"\n📄 Report saved to: {report_path}")
