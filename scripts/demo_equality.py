# scripts/demo_equality.py
from __future__ import annotations

import argparse
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Optional pretty printing
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except Exception:
    Console = None
    Table = None
    Panel = None
    box = None

# NetworkX (version-agnostic helpers)
import networkx as nx


# ──────────────────────────────────────────────────────────────────────────────
# NetworkX helpers (version-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

def _nx_random_tree(n: int, *, seed: Optional[int] = None) -> nx.Graph:
    """Uniform random labeled tree via Prüfer sequence (works on any NetworkX)."""
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if n == 1:
        return G
    prufer = [rng.randrange(n) for _ in range(n - 2)]
    degree = [1] * n
    for v in prufer:
        degree[v] += 1
    # connect leaves to prufer elements
    leaves = [i for i, d in enumerate(degree) if d == 1]
    leaves.sort()
    for v in prufer:
        u = leaves.pop(0)
        G.add_edge(u, v)
        degree[u] -= 1
        degree[v] -= 1
        if degree[v] == 1:
            # insert v into leaves keeping it sorted
            import bisect
            bisect.insort(leaves, v)
    u, v = [i for i, d in enumerate(degree) if d == 1]
    G.add_edge(u, v)
    return G


def _safe_diameter(G: nx.Graph) -> float:
    if not nx.is_connected(G):
        return float("nan")
    try:
        return nx.diameter(G)
    except Exception:
        # fallback via eccentricity
        ecc = nx.eccentricity(G)
        return float(max(ecc.values()))


def _safe_radius(G: nx.Graph) -> float:
    if not nx.is_connected(G):
        return float("nan")
    try:
        return nx.radius(G)
    except Exception:
        ecc = nx.eccentricity(G)
        return float(min(ecc.values()))


def _is_planar(G: nx.Graph) -> bool:
    try:
        planar, _ = nx.algorithms.planarity.check_planarity(G)
        return bool(planar)
    except Exception:
        # Older NX sometimes errors on multigraph checks—be conservative
        return False


def _is_chordal(G: nx.Graph) -> bool:
    try:
        return bool(nx.is_chordal(G))
    except Exception:
        return False


def _is_triangle_free(G: nx.Graph) -> bool:
    try:
        tri = nx.triangles(G)
        return sum(tri.values()) == 0
    except Exception:
        # fallback: brute-force small cycles length 3
        for u in G:
            N = set(G[u])
            for v in N:
                if v == u:
                    continue
                if len(N & set(G[v])) > 0:
                    return False
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Corpus + features
# ──────────────────────────────────────────────────────────────────────────────

def _named_graphs(seed: int) -> List[Tuple[str, nx.Graph]]:
    rng = random.Random(seed)
    out: List[Tuple[str, nx.Graph]] = []

    # Classics
    out += [
        ("Petersen", nx.petersen_graph()),
        ("Complete_6", nx.complete_graph(6)),
        ("Cycle_8", nx.cycle_graph(8)),
        ("Path_10", nx.path_graph(10)),
        ("Star_12", nx.star_graph(11)),  # order=12
        ("Balanced_Bipartite_6_6", nx.complete_bipartite_graph(6, 6)),
        ("Grid_4x4", nx.grid_2d_graph(4, 4)),
    ]

    # ER graphs
    for n, p in [(9, 0.13), (12, 0.16), (15, 0.12)]:
        G = nx.gnp_random_graph(n, p, seed=rng.randrange(10_000))
        out.append((f"Gnp_{n}_{p:.2f}_{rng.randrange(100)}", G))

    # Random trees
    for k in [8, 12, 16]:
        out.append((f"Tree_{k}", _nx_random_tree(k, seed=rng.randrange(10_000))))

    return out


def _random_corpus(n: int, seed: int) -> List[Tuple[str, nx.Graph]]:
    rng = random.Random(seed)
    base = _named_graphs(seed)
    out = list(base)
    # Add random mixture to hit target n
    while len(out) < n:
        typ = rng.choice(["cycle", "path", "gnp", "k", "kbip", "tree"])
        if typ == "cycle":
            m = rng.randint(5, 20)
            out.append((f"Cycle_{m}_{rng.randrange(100)}", nx.cycle_graph(m)))
        elif typ == "path":
            m = rng.randint(5, 20)
            out.append((f"Path_{m}_{rng.randrange(100)}", nx.path_graph(m)))
        elif typ == "gnp":
            m = rng.randint(6, 18)
            p = rng.uniform(0.08, 0.22)
            out.append((f"Gnp_{m}_{p:.2f}_{rng.randrange(100)}", nx.gnp_random_graph(m, p, seed=rng.randrange(10_000))))
        elif typ == "k":
            m = rng.randint(4, 10)
            out.append((f"Complete_{m}_{rng.randrange(100)}", nx.complete_graph(m)))
        elif typ == "kbip":
            a = rng.randint(3, 9); b = rng.randint(3, 9)
            out.append((f"KB_{a}_{b}_{rng.randrange(100)}", nx.complete_bipartite_graph(a, b)))
        else:  # tree
            m = rng.randint(6, 18)
            out.append((f"Tree_{m}_{rng.randrange(100)}", _nx_random_tree(m, seed=rng.randrange(10_000))))
    return out[:n]


def _feature_row(name: str, G: nx.Graph) -> Dict[str, object]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    min_deg = min(degs) if degs else 0
    max_deg = max(degs) if degs else 0
    connected = nx.is_connected(G)
    tree = nx.is_tree(G)
    bip = nx.is_bipartite(G)
    planar = _is_planar(G)
    chordal = _is_chordal(G)
    tri_free = _is_triangle_free(G)
    diam = _safe_diameter(G)
    rad = _safe_radius(G)
    try:
        avg_c = nx.average_clustering(G)
    except Exception:
        avg_c = float("nan")
    return dict(
        name=name,
        order=float(n),
        size=float(m),
        minimum_degree=float(min_deg),
        maximum_degree=float(max_deg),
        diameter=float(diam),
        radius=float(rad),
        avg_clustering=float(avg_c),
        connected=bool(connected),
        tree=bool(tree),
        bipartite=bool(bip),
        planar=bool(planar),
        chordal=bool(chordal),
        triangle_free=bool(tri_free),
        dataset_nonempty=True,
    )


def build_dataframe(graphs: Sequence[Tuple[str, nx.Graph]]) -> pd.DataFrame:
    rows = [_feature_row(name, G) for name, G in graphs]
    df = pd.DataFrame(rows)
    # Ensure stable column order
    cols_bool = ["connected", "tree", "bipartite", "planar", "chordal", "triangle_free", "dataset_nonempty"]
    cols_num = ["name", "order", "size", "minimum_degree", "maximum_degree", "diameter", "radius", "avg_clustering"]
    return df[cols_bool + cols_num]


# ──────────────────────────────────────────────────────────────────────────────
# Graffiti/TxGraffiti plumbing
# ──────────────────────────────────────────────────────────────────────────────

from txgraffiti2025.relations.core import DataModel, MaskCache
from txgraffiti2025.relations.incomparability import IncomparabilityAnalyzer
from txgraffiti2025.relations.equality import EqualityMiner
from txgraffiti2025.forms.generic_conjecture import TRUE  # only for explicit base if desired


def _make_model(df: pd.DataFrame, bool_like: List[str], numeric: List[str]) -> DataModel:
    """Construct DataModel without mutating frozen instances.

    We try several constructor signatures. If none match, we assume the
    DataModel can infer columns from df dtypes and proceed with just `df`.
    """
    # Prefer explicit signatures if supported
    try:
        return DataModel(df=df, bool_cols=bool_like, numeric_cols=numeric)
    except TypeError:
        pass
    try:
        return DataModel(df=df, boolean_cols=bool_like, numeric_cols=numeric)
    except TypeError:
        pass
    try:
        return DataModel(df, bool_cols=bool_like, numeric_cols=numeric)
    except TypeError:
        pass
    try:
        return DataModel(df, boolean_cols=bool_like, numeric_cols=numeric)
    except TypeError:
        pass

    # Final fallback: rely on DataModel's internal inference
    # (no attribute setting to avoid FrozenInstanceError)
    return DataModel(df)



# ──────────────────────────────────────────────────────────────────────────────
# Pretty print helpers
# ──────────────────────────────────────────────────────────────────────────────

def _console(enabled: bool):
    if not enabled or Console is None:
        return None
    return Console()


def _print(console, text: str = "", style: Optional[str] = None):
    if console is None:
        print(text)
    else:
        console.print(text, style=style)


def _df_table(console, title: str, df: pd.DataFrame, max_rows: int = 12):
    if console is None or Table is None:
        print(f"\n{title}\n")
        print(df.head(max_rows).to_string(index=False))
        return
    t = Table(title=title, box=box.SIMPLE_HEAVY)
    for c in df.columns:
        t.add_column(str(c))
    for _, row in df.head(max_rows).iterrows():
        t.add_row(*[str(x) for x in row.tolist()])
    console.print(t)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    *,
    n_graphs: int,
    seed: int,
    max_arity: int,
    console=None,
):
    _print(console, "[bold cyan]DataFrame snapshot[/bold cyan]")
    _print(console, "Building a NetworkX graph corpus…")

    graphs = _random_corpus(n_graphs, seed)
    df = build_dataframe(graphs)

    # Partition columns
    bool_like = ["connected", "tree", "bipartite", "planar", "chordal", "triangle_free", "dataset_nonempty"]
    numeric = ["name", "order", "size", "minimum_degree", "maximum_degree", "diameter", "radius", "avg_clustering"]

    # Show partition
    if console is not None and Table is not None:
        t = Table(title="Heuristic column partition (pre-DataModel)", box=box.SIMPLE_HEAVY)
        t.add_column("Type")
        t.add_column("Columns")
        t.add_row("Boolean-like", ", ".join(bool_like))
        t.add_row("Numeric", ", ".join(numeric))
        console.print(t)

    # Build model/cache (robust to constructor changes)
    model = _make_model(df=df, bool_like=bool_like, numeric=numeric)
    cache = MaskCache(model)

    # (Optional) choose a base domain; here we’ll just use TRUE, but you can
    # set `model.base_pred` externally if your DataModel supports it.
    base_condition = None  # or use a Predicate if you want: connected ∧ dataset_nonempty

    # 1) Incomparability → register |x−y|, min(x,y), max(x,y) to enrich expr space
    _print(console, "\n[bold cyan]Incomparability → |x−y|, min(x,y), max(x,y)[/bold cyan]")
    inc = IncomparabilityAnalyzer(model, cache)

    inc_table = inc.analyze(condition=base_condition, include_all_pairs=False)
    _df_table(console, "Incomparability under base domain (top 10)", inc_table.head(10))

    abs_table = inc.register_absdiff_exprs_for_meaningful_pairs(
        condition=base_condition,
        min_support=0.10,
        min_side_rate=0.10,
        min_side_count=4,
        min_median_gap=0.5,
        min_mean_gap=0.5,
        top_n_store=20,
    )
    _df_table(console, "Registered |x−y| (ranked by mean_gap)", abs_table.head(12))

    minmax_bal = inc.register_minmax_exprs_for_meaningful_pairs(
        condition=base_condition,
        min_support=0.10,
        min_side_rate=0.10,
        min_side_count=4,
        max_eq_rate=0.70,
        key_style="slug",
    )
    _df_table(console, "Registered min/max for balanced incomparable pairs", minmax_bal.head(12))

    # 2) Equality mining (constants and near-equalities)
    _print(console, "\n[bold cyan]Equality mining: constants and pairwise equalities[/bold cyan]")
    eqm = EqualityMiner(model, cache)

    consts = eqm.analyze_constants(
        condition=base_condition,
        tol=1e-9,
        require_finite=True,
        rationalize=True,
        max_denom=64,
        min_support=0.10,
    )
    _df_table(console, "Detected constant-valued invariants on domain", consts.head(12))

    pairs = eqm.analyze_pair_equalities(
        condition=base_condition,
        tol=1e-9,
        require_finite=True,
        min_support=0.10,
        min_eq_rate=0.95,
    )
    _df_table(console, "Detected near-equalities x ≈ y on domain", pairs.head(12))

    # 3) Optional: build Eq(...) conjectures
    conjs = eqm.make_eq_conjectures(constants=consts, pairs=pairs, condition=base_condition, only_selected=True)
    _print(console, f"\n[bold green]Constructed {len(conjs)} Eq-conjecture(s).[/bold green]")

    # Small preview of conjecture reprs if available
    preview = []
    for c in conjs[:8]:
        try:
            preview.append(repr(c))
        except Exception:
            preview.append("<repr unavailable>")
    if preview:
        if console and Table:
            tt = Table(title="Conjectures (preview)", box=box.SIMPLE_HEAVY)
            tt.add_column("Repr")
            for r in preview:
                tt.add_row(r)
            console.print(tt)
        else:
            print("\nConjectures (preview):")
            for r in preview:
                print(" •", r)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Demo: Equality mining over NetworkX corpus")
    ap.add_argument("--n", type=int, default=50, dest="n", help="number of graphs")
    ap.add_argument("--seed", type=int, default=7, help="PRNG seed")
    ap.add_argument("--max-arity", type=int, default=2, help="kept for symmetry with other demos")
    ap.add_argument("--no-rich", action="store_true", help="disable rich output")
    args = ap.parse_args()

    console = _console(not args.no_rich)
    run_pipeline(
        n_graphs=args.n,
        seed=args.seed,
        max_arity=args.max_arity,
        console=console,
    )


if __name__ == "__main__":
    main()
