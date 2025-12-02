# scripts/demo_class_relations_miner.py
from __future__ import annotations

import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from txgraffiti2025.relations.core import DataModel, MaskCache
from txgraffiti2025.relations.class_logic import ClassLogic
NAME_DELIM = " ∧ "
from txgraffiti2025.relations.class_relations_miner import ClassRelationsMiner

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _split_parts(name: str) -> tuple[str, ...]:
    if name == "TRUE" or not name:
        return tuple()
    return tuple(p.strip() for p in name.split(NAME_DELIM) if p.strip())

def _extras(name: str, base_parts: tuple[str, ...]) -> tuple[str, ...]:
    parts = set(_split_parts(name))
    base = set(base_parts)
    return tuple(sorted(parts - base))

def _filter_trivial_base_inclusions(
    stats: pd.DataFrame,
    base_name: str,
) -> pd.DataFrame:
    """Drop A ⊆ base and base ⊆ B inclusions; keep everything else."""
    if stats.empty:
        return stats
    mask_keep = np.ones(len(stats), dtype=bool)

    # Rows with accepted directional implications:
    #  - if imp_i_to_j and name_j == base_name -> drop
    #  - if imp_j_to_i and name_i == base_name -> drop
    if "imp_i_to_j" in stats.columns and "imp_j_to_i" in stats.columns:
        drop_i = stats["imp_i_to_j"] & (stats["name_j"] == base_name)
        drop_j = stats["imp_j_to_i"] & (stats["name_i"] == base_name)
        mask_keep &= ~(drop_i | drop_j)

    # Keep all equivalences (including those not involving base)
    return stats.loc[mask_keep].reset_index(drop=True)

def _rows_to_registry(
    stats: pd.DataFrame,
    base_parts: tuple[str, ...],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, r in stats.iterrows():
        ni = str(r["name_i"]); nj = str(r["name_j"])
        rec_base = {
            "name_i": ni,
            "name_j": nj,
            "nA": int(r["nA"]),
            "nB": int(r["nB"]),
            "nI": int(r["nI"]),
            "nU": int(r["nU"]),
            "jaccard": float(r["jaccard"]),
        }
        if bool(r["equiv"]):
            rows.append({
                "type": "equiv",
                **rec_base,
                "dir": None,
                "rec": None,
                "prec": None,
                "viol": None,
                "extras_i": list(_extras(ni, base_parts)),
                "extras_j": list(_extras(nj, base_parts)),
            })
        else:
            # Write an entry for each accepted implication direction
            if bool(r["imp_i_to_j"]):
                rows.append({
                    "type": "incl",
                    **rec_base,
                    "dir": "i_to_j",
                    "rec": float(r["rec_i_to_j"]),
                    "prec": float(r["prec_i_to_j"]),
                    "viol": int(r["viol_i_to_j"]),
                    "extras_i": list(_extras(ni, base_parts)),
                    "extras_j": list(_extras(nj, base_parts)),
                })
            if bool(r["imp_j_to_i"]):
                rows.append({
                    "type": "incl",
                    **rec_base,
                    "dir": "j_to_i",
                    "rec": float(r["rec_j_to_i"]),
                    "prec": float(r["prec_j_to_i"]),
                    "viol": int(r["viol_j_to_i"]),
                    "extras_i": list(_extras(nj, base_parts)),  # note swap
                    "extras_j": list(_extras(ni, base_parts)),
                })
    return rows

def _print_conjectures_compact(stats: pd.DataFrame) -> None:
    if stats.empty:
        print("Conjectures (pretty): <none>")
        return
    print("Conjectures (pretty):")
    for _, r in stats.iterrows():
        ni = r["name_i"]; nj = r["name_j"]
        if bool(r["equiv"]):
            print(f" • {ni} ≡ {nj}")
        else:
            if bool(r["imp_i_to_j"]):
                print(f" • {ni} ⊆ {nj}")
            if bool(r["imp_j_to_i"]):
                print(f" • {nj} ⊆ {ni}")

# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

def build_demo_df() -> pd.DataFrame:
    """Tiny baked-in demo if user doesn't point to a csv."""
    data = dict(
        order=[4,5,6,7,8,9,10,11],
        size=[17,13,11,14,16,19,21,22],
        connected=[True]*8,
        nontrivial=[True]*8,
        planar=[1,1,0,0,1,0,1,0],
        triangle_free=[1,0,1,0,1,0,1,0],
        bipartite=[1,0,1,0,1,0,1,0],
    )
    return pd.DataFrame(data)

def main():
    p = argparse.ArgumentParser(description="Demo: Class relations miner with cleaner output.")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV path for your dataset.")
    p.add_argument("--max_arity", type=int, default=2, help="Max extra predicates per conjunction.")
    p.add_argument("--min_support", type=float, default=0.10, help="Support threshold (abs if >=1 else fraction).")
    p.add_argument("--max_violations", type=int, default=0, help="Max allowed violations for an implication.")
    p.add_argument("--min_implication_rate", type=float, default=1.0, help="n(A∧B)/n(A) must be ≥ this.")
    p.add_argument("--use_sorted", action="store_true", help="Use logic.sort_by_generality() instead of nonredundant().")
    p.add_argument("--include_base", action="store_true", help="Include the pure base in enumeration.")
    p.add_argument("--skip_trivial_base", action="store_true", help="Skip inclusions of the form base∧X ⊆ base.")
    p.add_argument("--topk", type=int, default=20, help="Print at most this many stats rows.")
    args = p.parse_args()

    # --- Load data ---
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = build_demo_df()

    model = DataModel(df)
    cache = MaskCache(model)
    logic = ClassLogic(model, cache)

    print("=== DataModel ===")
    print(model.summary())
    print()

    # --- enumerate + normalize + optionally sorted ---
    logic.enumerate(max_arity=args.max_arity, min_support=0, include_base=args.include_base)
    logic.normalize()
    hyps = logic.sort_by_generality() if args.use_sorted else logic.nonredundant()

    base_name = logic.base_name()
    base_parts = _split_parts(base_name)
    print(f"Base name: {base_name}")
    print(f"Base parts: {base_parts}")
    print()

    # --- mine relations ---
    miner = ClassRelationsMiner(model, cache)
    stats, conjs = miner.make_conjectures(
        hyps,
        min_support=args.min_support,
        max_violations=args.max_violations,
        min_implication_rate=args.min_implication_rate,
        consider_empty=False,
        topk=None,  # we’ll truncate only for printing
        # return_only_strict=False,
    )

    if args.skip_trivial_base and not stats.empty:
        stats = _filter_trivial_base_inclusions(stats, base_name)

    # --- show a head for readability ---
    print("Top relations (stats head):")
    if stats.empty:
        print("<none>")
    else:
        print(stats.head(args.topk).to_string(index=False))
    print()

    # --- pretty print conjectures using compact names ---
    _print_conjectures_compact(stats)
    print()

    # --- persist to registry with jaccard kept for incl rows ---
    payload_rows = _rows_to_registry(stats, base_parts)
    model.registry.setdefault("class_relations", []).extend(payload_rows)

    # --- show a small registry sample ---
    reg = pd.DataFrame(model.registry.get("class_relations", []))
    print("\nRegistry 'class_relations' sample:")
    if reg.empty:
        print("<empty>")
    else:
        cols = ["type","name_i","name_j","nA","nB","nI","nU","jaccard","dir","rec","prec","viol"]
        present = [c for c in cols if c in reg.columns]
        print(reg[present].head(12).to_string(index=False))

if __name__ == "__main__":
    main()
