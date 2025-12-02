# scripts/demo_relations_pipeline.py
from __future__ import annotations

import argparse, math
import pandas as pd

from txgraffiti2025.relations.core import DataModel, MaskCache
from txgraffiti2025.relations.class_logic import ClassLogic
from txgraffiti2025.relations.incomparability import IncomparabilityAnalyzer
from txgraffiti2025.relations.equality import EqualityMiner
from txgraffiti2025.relations.class_relations_miner import ClassRelationsMiner
from txgraffiti2025.relations.boolean_discovery import BooleanDiscoveryMiner
from txgraffiti2025.forms.predicates import Predicate, EQ, LE, GE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _n_rows(cache: MaskCache, pred: Predicate) -> int:
    try:
        return int(cache.mask(pred).sum())
    except Exception:
        return 0

def _short(s: str, k: int = 96) -> str:
    return s if len(s) <= k else s[: k - 1] + "…"

def _literal_count(name: str) -> int:
    # crude: count ∧ separators + 1
    return name.count("∧") + 1


# ─────────────────────────────────────────────────────────────────────────────
# Rule selection to avoid noise
# ─────────────────────────────────────────────────────────────────────────────

def pick_atomic_rules(
    df_pairs: pd.DataFrame,
    *,
    cache: MaskCache,
    model: DataModel,
    k_eq: int = 10,
    k_le: int = 10,
    k_ge: int = 10,
    min_support: float = 0.15,
    min_balance: float = 0.15,
    min_eq_rate: float = 0.985,
    min_rule_rate: float = 0.975,
    require_variance: bool = True,
) -> list[tuple[str, Predicate]]:
    """
    From BooleanDiscoveryMiner pair rules, pick a *small*, useful subset:
      • Equalities:   x == y with high eq_rate and support.
      • Inequalities: x ≤ y or x ≥ y with high rule_rate, support, and *balance*
        (we avoid rules that are basically equalities).
      • Optional variance gate to avoid degenerate ‘constant == constant’.
    Returns list of (name, pred) predicates ready to mine as hypotheses.
    """
    picks: list[tuple[str, Predicate]] = []

    if df_pairs is None or df_pairs.empty:
        return picks

    # Expect columns: kind in {"eq","le","ge"}, inv1, inv2, support, eq_rate, rule_rate,
    #                 rate_lt, rate_gt, rate_eq, pretty_name, pred (optional), var1, var2
    def _ok_var(row) -> bool:
        if not require_variance:
            return True
        v1 = float(row.get("var1", 1.0) or 0.0)
        v2 = float(row.get("var2", 1.0) or 0.0)
        return (v1 > 0.0) or (v2 > 0.0)

    # Equalities
    eq_df = df_pairs[
        (df_pairs["kind"] == "eq")
        & (df_pairs["support"] >= min_support)
        & (df_pairs["eq_rate"] >= min_eq_rate)
    ].copy()
    if require_variance and not eq_df.empty:
        eq_df = eq_df[eq_df.apply(_ok_var, axis=1)]
    eq_df["score"] = 1.0 * eq_df["support"] + 0.5 * (eq_df["eq_rate"] - min_eq_rate)
    eq_df = eq_df.sort_values(["score","support"], ascending=False).head(k_eq)

    for _, r in eq_df.iterrows():
        name = r.get("pretty_name") or f"{r['inv1']} = {r['inv2']}"
        pred = EQ(model.exprs[r["inv1"]], model.exprs[r["inv2"]])
        picks.append((name, pred))

    # Inequalities — require *balance* so it’s not “equality in disguise”
    def _ineq_select(kind: str, k_top: int):
        dfk = df_pairs[
            (df_pairs["kind"] == kind)
            & (df_pairs["support"] >= min_support)
            & (df_pairs["rule_rate"] >= min_rule_rate)
            & (df_pairs["rate_eq"] <= (1.0 - min_balance))  # not mostly equal
        ].copy()
        if dfk.empty:
            return []
        # balance ~ min(rate_lt, rate_gt) ensures both sides occur sometimes
        dfk["balance"] = dfk[["rate_lt","rate_gt"]].min(axis=1)
        dfk = dfk[dfk["balance"] >= min_balance]
        if require_variance:
            dfk = dfk[dfk.apply(_ok_var, axis=1)]
        if dfk.empty:
            return []
        # Rank by (support, rule_rate, balance), then favor shorter pretty name
        dfk["score"] = (1.0 * dfk["support"]
                        + 0.5 * (dfk["rule_rate"] - min_rule_rate)
                        + 0.25 * dfk["balance"])
        dfk["namelen"] = dfk["pretty_name"].str.len()
        dfk = dfk.sort_values(["score","support","balance","namelen"], ascending=[False, False, False, True]).head(k_top)
        out = []
        for _, r in dfk.iterrows():
            name = r.get("pretty_name") or (f"{r['inv1']} ≤ {r['inv2']}" if kind=="le" else f"{r['inv1']} ≥ {r['inv2']}")
            if kind == "le":
                pred = LE(model.exprs[r["inv1"]], model.exprs[r["inv2"]])
            else:
                pred = GE(model.exprs[r["inv1"]], model.exprs[r["inv2"]])
            out.append((name, pred))
        return out

    picks.extend(_ineq_select("le", k_le))
    picks.extend(_ineq_select("ge", k_ge))
    return picks


def select_hypotheses_compact(
    sorted_hyps: list[tuple[str, Predicate]],
    cache: MaskCache,
    *,
    max_hyp: int,
    min_domain: int,
    max_literals: int,
) -> list[tuple[str, Predicate]]:
    """Strict, compact selection of class-logic hypotheses."""
    chosen: list[tuple[str, Predicate]] = []
    for name, pred in sorted_hyps:
        if _n_rows(cache, pred) < min_domain:
            continue
        if _literal_count(name) > max_literals:
            continue
        chosen.append((name, pred))
        if len(chosen) >= max_hyp:
            break
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Noise-controlled relations pipeline demo")
    ap.add_argument("--max-hyp", type=int, default=6)
    ap.add_argument("--min-domain", type=int, default=60)
    ap.add_argument("--max-literals", type=int, default=24)
    ap.add_argument("--k-eq", type=int, default=6, help="max discovered equality rules to mine")
    ap.add_argument("--k-le", type=int, default=6, help="max discovered ≤ rules to mine")
    ap.add_argument("--k-ge", type=int, default=6, help="max discovered ≥ rules to mine")
    ap.add_argument("--mine-mode", choices=["classes","rules","mixed"], default="rules",
                    help="hypotheses source: class logic, discovered rules, or both (prefers rules)")
    ap.add_argument("--min-support", type=float, default=0.15)
    ap.add_argument("--min-balance", type=float, default=0.15)
    ap.add_argument("--min-eq-rate", type=float, default=0.985)
    ap.add_argument("--min-rule-rate", type=float, default=0.975)
    ap.add_argument("--print-every", type=int, default=1)
    args = ap.parse_args()

    # 1) Data
    print("\n══ 1) Load example dataframe ═════════════════════════════════════════════════")
    from txgraffiti.example_data import graph_data as df
    df["nontrivial"] = df["connected"]
    print(f"    - rows: {len(df)}")
    print(f"    - cols: {len(df.columns)}")

    # 2) Model/Cache
    print("══ 2) Initialize DataModel / MaskCache ═══════════════════════════════════════")
    model = DataModel(df)
    cache = MaskCache(model)
    print(f"    - #bool: {len(model.boolean_cols)}")
    print(f"    - #num:  {len(model.numeric_cols)}")

    # 3) ClassLogic (first pass)
    print("══ 3) ClassLogic (first pass) ════════════════════════════════════════════════")
    logic = ClassLogic(model, cache)
    logic.enumerate(max_arity=2)
    nonred, _, _ = logic.normalize()
    sorted_hyps = logic.sort_by_generality()
    print(f"    - #nonredundant: {len(nonred)}")
    print(f"    - #sorted for mining: {len(sorted_hyps)}")

    # 4) Boolean discovery (tight only)
    print("══ 4) BooleanDiscoveryMiner (tight rules only) ═══════════════════════════════")
    bdm = BooleanDiscoveryMiner(model, cache)
    pair_rules = bdm.discover_pairwise_rules(
        condition=None,
        tol=1e-9,
        min_support=args.min_support,
        min_eq_rate=args.min_eq_rate,
        min_rule_rate=args.min_rule_rate,
        max_violations=None,
        name_style="pretty",
    )
    const_rules = bdm.discover_constant_ties(
        condition=None,
        tol=1e-9,
        min_support=args.min_support,
        min_const_rate=args.min_eq_rate,
        rationalize=True,
        max_denom=64,
        name_style="pretty",
    )
    print(f"    - pair rules: {0 if pair_rules is None else len(pair_rules)}")
    print(f"    - const ties: {0 if const_rules is None else len(const_rules)}")

    # 5) Recompute ClassLogic (after discovery) — keeps base booleans + any registered
    print("══ 5) Recompute ClassLogic (after boolean discovery) ═════════════════════════")
    logic = ClassLogic(model, cache)
    logic.enumerate(max_arity=2)
    _, _, _ = logic.normalize()
    sorted_hyps = logic.sort_by_generality()
    print(f"    - #sorted for mining: {len(sorted_hyps)}")

    # 6) Select hypotheses to mine (compact)
    print("══ Select hypotheses to mine ═════════════════════════════════════════════════")
    chosen: list[tuple[str, Predicate]] = []

    if args.mine_mode in ("rules","mixed"):
        # Pick small set of atomic, high-quality rules
        rule_hyps = pick_atomic_rules(
            (pair_rules if pair_rules is not None else pd.DataFrame()),
            cache=cache,
            model=model,
            k_eq=args.k_eq, k_le=args.k_le, k_ge=args.k_ge,
            min_support=args.min_support,
            min_balance=args.min_balance,
            min_eq_rate=args.min_eq_rate,
            min_rule_rate=args.min_rule_rate,
            require_variance=True,
        )
        # keep only those with sufficient domain
        rule_hyps = [(n, p) for (n, p) in rule_hyps if _n_rows(cache, p) >= args.min_domain]
        chosen.extend(rule_hyps[: args.max_hyp])

    if args.mine_mode in ("classes","mixed") and len(chosen) < args.max_hyp:
        extra = select_hypotheses_compact(
            sorted_hyps, cache,
            max_hyp=(args.max_hyp - len(chosen)),
            min_domain=args.min_domain,
            max_literals=args.max_literals,
        )
        chosen.extend(extra)

    print(f"  • Selected {len(chosen)} hypotheses to mine.")
    for name, pred in chosen[:10]:
        print(f"    · n={_n_rows(cache, pred):4d} | {_short(name)}")
    if not chosen:
        print("    · (none) — try lowering --min-domain or --min-support")
        return

    # 7) Optional class relations among *chosen* hyps (tiny)
    print("══ 6) ClassRelationsMiner (optional, concise) ════════════════════════════════")
    crm = ClassRelationsMiner(model, cache)
    rel_stats, rel_conjs = crm.make_conjectures(
        chosen,
        min_support=args.min_support,
        max_violations=0,
        min_implication_rate=1.0,
    )
    print(f"    - relation stats: {0 if rel_stats is None else len(rel_stats)}")
    print(f"    - relation conjectures: {0 if rel_conjs is None else len(rel_conjs)}")

    # 8) Numeric miners with aggressive pruning under *chosen* hyps
    print("══ 7) Numeric miners with pruning (under selected hyps) ══════════════════════")
    inc = IncomparabilityAnalyzer(model, cache)
    eqm = EqualityMiner(model, cache)

    kept_abs, kept_mm, kept_const, kept_pairs, eq_conjs = [], [], [], [], []

    for idx, (hyp_name, hyp_pred) in enumerate(chosen, 1):
        if args.print_every and (idx % args.print_every == 0 or idx == 1):
            print(f"── Mining under: {_short(hyp_name, 160)} (n={_n_rows(cache, hyp_pred)}) ──")

        # incomparability (not printed; only used internally for thresholds if you want)
        # inc_df = inc.analyze(condition=hyp_pred, include_all_pairs=True)

        # |x-y| on meaningfully incomparable pairs (kept if selected)
        abs_df = inc.register_absdiff_exprs_for_meaningful_pairs(
            condition=hyp_pred,
            require_finite=True,
            min_support=max(0.10, args.min_support),
            min_side_rate=0.20,
            min_side_count=3,
            min_median_gap=0.0,
            min_mean_gap=0.0,
            overwrite_existing=False,
            top_n_store=8,
        )
        if not abs_df.empty:
            sel = abs_df[abs_df["selected"]].copy()
            if not sel.empty:
                kept_abs.append(sel.head(8))

        # min/max for often-unequal (avoid equality-like)
        mm_df = inc.register_minmax_for_often_unequal_pairs(
            condition=hyp_pred,
            require_finite=True,
            min_support=max(0.10, args.min_support),
            min_neq_rate=0.60,
            min_neq_count=5,
            must_be_incomparable=True,
            prefix_min="min_",
            prefix_max="max_",
            overwrite_existing=False,
            top_n_store=8,
        )
        if not mm_df.empty:
            sel = mm_df[mm_df["selected"]].copy()
            if not sel.empty:
                kept_mm.append(sel.head(8))

        # constants / equalities (very strict)
        const_df = eqm.analyze_constants(
            condition=hyp_pred,
            tol=1e-9,
            require_finite=True,
            rationalize=True, max_denom=32,
            min_support=max(0.15, args.min_support),
        )
        if not const_df.empty:
            sel = const_df[const_df["selected"]].copy()
            if not sel.empty:
                kept_const.append(sel.head(8))

        pair_df = eqm.analyze_pair_equalities(
            condition=hyp_pred,
            tol=1e-9,
            require_finite=True,
            min_support=max(0.15, args.min_support),
            min_eq_rate=max(0.985, args.min_eq_rate),
        )
        if not pair_df.empty:
            sel = pair_df[pair_df["selected"]].copy()
            if not sel.empty:
                kept_pairs.append(sel.head(8))

        # optional: Eq(...) objects
        conjs = eqm.make_eq_conjectures(constants=const_df, pairs=pair_df, condition=hyp_pred)
        if conjs:
            eq_conjs.extend(conjs[:8])

    # 9) Summaries
    print("══ 8) Summaries (kept results only) ══════════════════════════════════════════")
    def _concat(xs):
        return pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    kept_abs_all   = _concat(kept_abs)
    kept_mm_all    = _concat(kept_mm)
    kept_const_all = _concat(kept_const)
    kept_pairs_all = _concat(kept_pairs)

    print(f"    - class relation conjectures: {len(rel_conjs or [])}")
    print(f"    - kept |x−y| rows: {len(kept_abs_all)}")
    print(f"    - kept min/max rows: {len(kept_mm_all)}")
    print(f"    - kept near-constants: {len(kept_const_all)}")
    print(f"    - kept pair-equalities: {len(kept_pairs_all)}")
    print(f"    - Eq(...) conjectures: {len(eq_conjs)}")

    # Expr registry tail (sanity)
    tail = list(model.exprs.keys())[-12:]
    print("\n=== Expr registry keys (tail) ===")
    print(tail)


if __name__ == "__main__":
    main()
