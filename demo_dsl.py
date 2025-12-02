# scripts/demo_dsl.py
from __future__ import annotations

import argparse, math, random, sys, os, shutil, subprocess
from contextlib import contextmanager
from textwrap import dedent
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# ───────────────────────── Optional niceties (no hard deps) ───────────────────────── #
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.theme import Theme
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
except Exception:
    Console = None
    Table = None
    Panel = None
    Theme = None
    box = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None


def _console(enabled: bool, wrap: bool = True):
    if not enabled or Console is None:
        return None
    theme = Theme(
        {
            "ok": "bold green",
            "warn": "bold yellow",
            "err": "bold red",
            "title": "bold cyan",
            "dim": "dim",
            "b": "bold",
            "muted": "grey50",
        }
    )
    return Console(theme=theme, soft_wrap=wrap, record=False)


def _print(console, text: str = "", style: Optional[str] = None, **kwargs):
    if console is None:
        print(text)
    else:
        console.print(text, style=style, **kwargs)


def _hr(console, char: str = "─", n: int = 70):
    _print(console, char * n, style="dim")


def _banner(console):
    msg = "TxGraffiti • Demo DSL Runner"
    if console is None or Panel is None:
        print("=" * len(msg))
        print(msg)
        print("=" * len(msg))
    else:
        _print(console, Panel.fit(msg, style="title", padding=(0, 2)))


def _short(s: str, n: int = 40) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _auto_rows(auto_rows: bool, explicit_rows: Optional[int]) -> Optional[int]:
    if not auto_rows:
        return explicit_rows
    try:
        height = shutil.get_terminal_size((100, 30)).lines
        # leave room for headers/pager prompt
        return max(8, height - 12)
    except Exception:
        return explicit_rows


def _as_table(console, title: str, df: pd.DataFrame, max_rows: Optional[int], max_cols: Optional[int]):
    if console is None or Table is None:
        print(f"\n{title}")
        show = df if max_rows in (None, 0) else df.head(max_rows)
        pd.set_option("display.max_columns", None if max_cols in (None, 0) else max_cols)
        print(show.to_string(max_cols=None if max_cols in (None, 0) else max_cols))
        extra_rows = len(df) - (len(show) if max_rows not in (None, 0) else len(df))
        if max_rows not in (None, 0) and extra_rows > 0:
            print(f"... ({extra_rows} more rows)")
        return

    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("(index)", style="dim")

    cols = list(df.columns)
    if max_cols not in (None, 0) and len(cols) > max_cols:
        cols = cols[:max_cols] + ["…"]
    for c in cols:
        table.add_column(str(c))

    show_df = df if max_rows in (None, 0) else df.head(max_rows)
    limit = len(df.columns) if max_cols in (None, 0) else min(max_cols, len(df.columns))
    for idx, row in show_df.iterrows():
        vals = [str(idx)]
        for c in df.columns[:limit]:
            vals.append(_short(str(row[c])))
        if max_cols not in (None, 0) and len(df.columns) > max_cols:
            vals.append("…")
        table.add_row(*vals)
    console.print(table)


@contextmanager
def _pager(console, enabled: bool, force_less_flags: Optional[str] = None):
    """
    Prefer Rich's pager (uses $PAGER or less -R). If Rich missing, try $PAGER, else pydoc, else no-op.
    """
    if not enabled:
        yield
        return

    # Rich pager (best UX)
    if console is not None and hasattr(console, "pager"):
        # rich honors $PAGER; you can set PAGER="less -SR"
        with console.pager():
            yield
        return

    # External pager via $PAGER (second best)
    pager_cmd = os.environ.get("PAGER")
    if pager_cmd:
        # We can't stream as a context, so capture then feed
        class _Collector:
            def __init__(self): self.buf = []
            def write(self, s): self.buf.append(s)
            def get(self): return "".join(self.buf)
        c = _Collector()
        old = sys.stdout
        sys.stdout = c
        try:
            yield
        finally:
            sys.stdout = old
            try:
                p = subprocess.Popen(pager_cmd, shell=True, stdin=subprocess.PIPE)
                p.communicate(c.get().encode())
            except Exception:
                print(c.get())
        return

    # pydoc pager fallback
    try:
        import pydoc
        class _Collector:
            def __init__(self): self.buf = []
            def write(self, s): self.buf.append(s)
            def get(self): return "".join(self.buf)
        c = _Collector()
        old = sys.stdout
        sys.stdout = c
        try:
            yield
        finally:
            sys.stdout = old
            pydoc.pager(c.get())
        return
    except Exception:
        # last resort
        yield


def _pager_if(console, text: str, use_pager: bool):
    if use_pager:
        with _pager(console, True):
            _print(console, text)
    else:
        _print(console, text)


# ───────────────────────── Try optional math deps ───────────────────────── #
try:
    import networkx as nx
except Exception:
    nx = None

try:
    import sympy as sp
    from sympy import Poly
except Exception:
    sp = None
    Poly = None


# ───────────────────────── Try txgraffiti imports with nice errors ───────────────────────── #
def _try_txgraffiti(console, quiet: bool = False):
    try:
        from txgraffiti2025.graffiti_utils import (
            to_expr, sqrt, floor, ceil, log, min_, max_, absdiff,
        )
        from txgraffiti2025.graffiti_predicates import Predicate, Where, TRUE
        from txgraffiti2025.graffiti_generic_conjecture import (
            Ge, Le, Eq, Lt, Gt, AndF, OrF, XorF, NotF, Implies, Iff, AllOf, ite, Conjecture,
        )
        from txgraffiti2025.graffiti_base import GraffitiBase
        from txgraffiti2025.graffiti_class_logic import GraffitiClassLogic
        return dict(
            to_expr=to_expr, sqrt=sqrt, floor=floor, ceil=ceil, log=log,
            min_=min_, max_=max_, absdiff=absdiff,
            Predicate=Predicate, Where=Where, TRUE=TRUE,
            Ge=Ge, Le=Le, Eq=Eq, Lt=Lt, Gt=Gt,
            AndF=AndF, OrF=OrF, XorF=XorF, NotF=NotF, Implies=Implies, Iff=Iff, AllOf=AllOf,
            ite=ite, Conjecture=Conjecture,
            GraffitiBase=GraffitiBase, GraffitiClassLogic=GraffitiClassLogic,
        )
    except Exception as e:
        if not quiet:
            _print(console, f"[err]Could not import txgraffiti2025 modules: {e}[/err]")
            _print(console, "[warn]Ensure package is installed and PYTHONPATH is set.[/warn]")
        return None


# ───────────────────────── Random seeds ───────────────────────── #
def _seed_everything(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


# ───────────────────────── Helpers ───────────────────────── #
def _first_n_numeric(df: pd.DataFrame, k: int = 5) -> list[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        if len(cols) >= k:
            break
    return cols


def _bool_cols(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if df[c].dtype == bool or str(df[c].dtype).lower().startswith("boolean"):
            out.append(c)
    return out


def _pick_two_distinct_numeric(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    nums = _first_n_numeric(df, k=6)
    if len(nums) >= 2:
        return nums[0], nums[1]
    return (None, None)


# ───────────────────────── 1) Graphs ───────────────────────── #
def _safe_radius_diameter(G) -> tuple[float, float]:
    if nx is None:
        return (np.nan, np.nan)
    if nx.is_connected(G):
        try:
            return float(nx.radius(G)), float(nx.diameter(G))
        except Exception:
            return (np.nan, np.nan)
    return (np.nan, np.nan)


def _is_planar(G) -> bool:
    if nx is None:
        return False
    try:
        is_planar, _ = nx.check_planarity(G, counterexample=False)
        return bool(is_planar)
    except Exception:
        return False


def build_graph_df() -> Optional[pd.DataFrame]:
    if nx is None:
        return None
    items = [
        ("K3", nx.complete_graph(3)),
        ("K4", nx.complete_graph(4)),
        ("K5", nx.complete_graph(5)),
        ("C4", nx.cycle_graph(4)),
        ("C5", nx.cycle_graph(5)),
        ("P4", nx.path_graph(4)),
        ("P5", nx.path_graph(5)),
        ("Q3", nx.hypercube_graph(3)),
        ("Petersen", nx.petersen_graph()),
    ]
    rows = []
    for name, G in items:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        connected = nx.is_connected(G)
        tree = nx.is_tree(G) if connected else False
        planar = _is_planar(G)
        rad, diam = _safe_radius_diameter(G)
        rows.append(
            {
                "name": name,
                "order": float(n),
                "size": float(m),
                "radius": float(rad) if math.isfinite(rad) else np.nan,
                "diameter": float(diam) if math.isfinite(diam) else np.nan,
                "connected": bool(connected),
                "planar": bool(planar),
                "tree": bool(tree),
            }
        )
    df = pd.DataFrame(rows).set_index("name").sort_index()
    df["avg_deg"] = (2.0 * df["size"]) / df["order"]
    return df


# ───────────────────────── 2) Polynomials ───────────────────────── #
def _cheap_discriminant(P: Poly) -> float:
    deg = int(P.degree())
    if deg <= 5:
        try:
            return float(sp.discriminant(P))
        except Exception:
            return float("nan")
    return float("nan")


def _fast_real_root_count(P: Poly) -> float:
    try:
        coeffs = np.array([float(c) for c in P.all_coeffs()], dtype=float)
        if coeffs.size <= 1:
            return 0.0
        roots = np.roots(coeffs)
        return float(np.sum(np.abs(roots.imag) < 1e-8))
    except Exception:
        return float("nan")


def _cheap_irreducible(P: Poly) -> bool:
    try:
        deg = int(P.degree())
        if deg <= 4:
            fac = sp.factor(P.as_expr())
            return bool(sp.degree(fac) == deg and fac == P.as_expr())
    except Exception:
        pass
    return False


def _is_palindromic_coeffs(coeffs: list[int]) -> bool:
    return coeffs == list(reversed(coeffs))


def _is_reciprocal_poly(P: Poly, x) -> bool:
    deg = int(P.degree())
    if deg <= 0:
        return True
    c = [int(cc) for cc in P.all_coeffs()]
    return c[0] != 0 and c[-1] != 0 and abs(c[0]) == abs(c[-1])


def build_polynomial_df(n_random: int = 40) -> Optional[pd.DataFrame]:
    if sp is None:
        return None
    x = sp.symbols("x")
    polys: list[tuple[str, sp.Poly]] = []
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
        polys.append((f"cyclo_{n}", sp.Poly(sp.cyclotomic_poly(n, x), x, domain=sp.QQ)))
    for n in [1, 2, 3, 4, 5, 6]:
        polys.append((f"chebyshev_T{n}", sp.Poly(sp.chebyshevt(n, x).expand(), x, domain=sp.QQ)))
        polys.append((f"legendre_P{n}", sp.Poly(sp.legendre(n, x).expand(), x, domain=sp.QQ)))
    pal_vecs = [[1, 0, 1], [1, -1, 2, -1, 1], [2, 0, -3, 0, 2]]
    for i, c in enumerate(pal_vecs):
        p = sum(c[j] * x ** (len(c) - 1 - j) for j in range(len(c)))
        polys.append((f"pal_{i}", sp.Poly(p, x, domain=sp.QQ)))
    rr = [(x - 2) ** 2, (x + 1) ** 3, (x ** 2 + 1) ** 2, (x - 1) ** 2 * (x + 2)]
    for i, p in enumerate(rr):
        polys.append((f"repeated_{i}", sp.Poly(sp.expand(p), x, domain=sp.QQ)))
    rng = np.random.default_rng(7)
    for i in range(n_random):
        deg = int(rng.integers(2, 7))
        if rng.random() < 0.5:
            coeffs = np.zeros(deg + 1, dtype=int)
            k = max(2, int(0.3 * (deg + 1)))
            idx = rng.choice(np.arange(deg + 1), size=k, replace=False)
            coeffs[idx] = rng.integers(-4, 5, size=k)
            if coeffs[0] == 0:
                coeffs[0] = int(rng.integers(1, 4))
        else:
            coeffs = rng.integers(-4, 5, size=deg + 1)
            if coeffs[0] == 0:
                coeffs[0] = int(rng.integers(1, 4))
        poly = sum(int(coeffs[j]) * x ** (deg - j) for j in range(deg + 1))
        polys.append((f"rand_{i}_deg{deg}", sp.Poly(sp.expand(poly), x, domain=sp.QQ)))
    rows = []
    for name, P in polys:
        coeffs = [int(c) for c in P.all_coeffs()]
        deg = int(P.degree())
        terms = int(sum(c != 0 for c in coeffs))
        height = float(max(abs(c) for c in coeffs)) if coeffs else 0.0
        l1 = float(sum(abs(c) for c in coeffs))
        leading = int(coeffs[0]) if coeffs else 0
        const = int(coeffs[-1]) if coeffs else 0
        disc = _cheap_discriminant(P)
        real_root_count = _fast_real_root_count(P)
        monic = bool(leading == 1)
        squarefree = bool((disc != 0.0) and math.isfinite(disc))
        irreducible = _cheap_irreducible(P)
        palindromic = _is_palindromic_coeffs(coeffs)
        reciprocal = _is_reciprocal_poly(P, x)
        even_poly = bool(P.as_expr().subs(x, -x) - P.as_expr() == 0)
        odd_poly = bool(P.as_expr().subs(x, -x) + P.as_expr() == 0)
        is_cyclo = name.startswith("cyclo_")
        is_cheb = name.startswith("chebyshev_T")
        is_leg = name.startswith("legendre_P")
        sparse = bool(terms <= max(2, (deg + 1) // 2))
        rows.append(
            {
                "name": name,
                "degree": float(deg),
                "terms": float(terms),
                "height": float(height),
                "l1_coeff": float(l1),
                "leading": float(leading),
                "constant": float(const),
                "discriminant": float(disc) if math.isfinite(disc) else np.nan,
                "real_root_count": float(real_root_count) if math.isfinite(real_root_count) else np.nan,
                "root_density": float(real_root_count / deg) if (deg > 0 and math.isfinite(real_root_count)) else np.nan,
                "monic": monic,
                "squarefree": squarefree,
                "irreducible": irreducible,
                "palindromic": palindromic,
                "reciprocal": reciprocal,
                "even_poly": even_poly,
                "odd_poly": odd_poly,
                "cyclotomic": is_cyclo,
                "chebyshev": is_cheb,
                "legendre": is_leg,
                "sparse": sparse,
            }
        )
    df = pd.DataFrame(rows).set_index("name").sort_index()
    df["gap_deg_terms"] = df["degree"] - df["terms"]
    df["height_over_deg"] = df["height"] / df["degree"].replace(0, np.nan)
    return df


# ───────────────────────── 3) Matrices ───────────────────────── #
def _circulant(vec: np.ndarray) -> np.ndarray:
    n = len(vec)
    return np.vstack([np.roll(vec, i) for i in range(n)])


def _fro_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))


def build_matrix_df(n_random: int = 24, sizes=(2, 3, 4)) -> Optional[pd.DataFrame]:
    mats: list[tuple[str, np.ndarray]] = []
    for n in sizes:
        mats.append((f"I{n}", np.eye(n)))
        mats.append((f"Z{n}", np.zeros((n, n))))
        mats.append((f"J{n}", np.ones((n, n))))
        mats.append((f"DiagStep_{n}", np.diag(np.arange(1, n + 1))))
        mats.append((f"DiagAlt_{n}", np.diag([1 if i % 2 == 0 else -1 for i in range(n)])))
        J = np.zeros((n, n))
        for i in range(n - 1):
            J[i, i + 1] = 1.0
        mats.append((f"JordanNil_{n}", J))
        R = np.random.default_rng(7).integers(-2, 3, size=(n, n)).astype(float)
        PSD = R.T @ R
        mats.append((f"PSD_{n}", PSD))
        R2 = np.random.default_rng(11).integers(-2, 3, size=(n, n)).astype(float)
        Sk = R2 - R2.T
        mats.append((f"Skew_{n}", Sk))
        base = np.arange(1, n + 1, dtype=float)
        mats.append((f"Circulant_{n}", _circulant(base)))
    for theta in [0.0, math.pi / 6, math.pi / 4, math.pi / 3]:
        c, s = math.cos(theta), math.sin(theta)
        mats.append((f"Rot2_{theta:.2f}", np.array([[c, -s], [s, c]], dtype=float)))
    for n in sizes:
        u = np.random.default_rng(3).normal(size=(n, 1))
        u /= np.linalg.norm(u)
        H = np.eye(n) - 2 * (u @ u.T)
        mats.append((f"House_{n}", H))
    for n in sizes:
        P = np.zeros((n, n))
        P[0, 0] = 1.0
        mats.append((f"Proj_e1_{n}", P))
    rng = np.random.default_rng(23)
    for i in range(n_random):
        n = int(rng.choice(sizes))
        A = rng.integers(-3, 4, size=(n, n)).astype(float)
        mats.append((f"Rand_{i}_n{n}", A))
    rows = []
    for name, A in mats:
        n = A.shape[0]
        rank = float(np.linalg.matrix_rank(A))
        trace = float(np.trace(A))
        try:
            det = float(np.linalg.det(A))
        except Exception:
            det = float("nan")
        abs_det = float(abs(det)) if math.isfinite(det) else np.nan
        fro = _fro_norm(A)
        try:
            evals = np.linalg.eigvals(A)
            spectral_radius = float(np.max(np.abs(evals)))
            num_pos = float(np.sum((evals.real > 1e-9) & (np.abs(evals.imag) < 1e-8)))
            min_eig = float(np.min(evals.real))
        except Exception:
            spectral_radius = np.nan
            num_pos = np.nan
            min_eig = np.nan
        symmetric = bool(np.allclose(A, A.T, atol=1e-9))
        skew_symm = bool(np.allclose(A, -A.T, atol=1e-9))
        orthogonal = bool(np.allclose(A.T @ A, np.eye(n), atol=1e-8))
        invertible = bool(math.isfinite(det) and abs(det) > 1e-10)
        full_rank = bool(rank >= n - 1e-9)
        psd = bool(symmetric and math.isfinite(min_eig) and min_eig >= -1e-8)
        nsd = bool(symmetric and math.isfinite(min_eig) and spectral_radius <= 1e-8)
        idempotent = bool(np.allclose(A @ A, A, atol=1e-8))
        involutory = bool(np.allclose(A @ A, np.eye(n), atol=1e-8))
        nilpotent2 = bool(np.allclose(A @ A, np.zeros_like(A), atol=1e-8))
        nilpotent3 = bool(np.allclose(A @ A @ A, np.zeros_like(A), atol=1e-8))
        rows.append(
            {
                "name": name,
                "order": float(n),
                "rank": float(rank),
                "trace": float(trace) if math.isfinite(trace) else np.nan,
                "det": float(det) if math.isfinite(det) else np.nan,
                "abs_det": float(abs_det) if math.isfinite(abs_det) else np.nan,
                "fro_norm": float(fro),
                "spectral_radius": float(spectral_radius) if math.isfinite(spectral_radius) else np.nan,
                "num_pos_eigs": float(num_pos) if math.isfinite(num_pos) else np.nan,
                "symmetric": symmetric,
                "skew_symmetric": skew_symm,
                "orthogonal": orthogonal,
                "invertible": invertible,
                "full_rank": full_rank,
                "psd": psd,
                "nsd": nsd,
                "idempotent": idempotent,
                "involutory": involutory,
                "nilpotent2": nilpotent2,
                "nilpotent3": nilpotent3,
            }
        )
    df = pd.DataFrame(rows).set_index("name").sort_index()
    df["rank_gap"] = df["order"] - df["rank"]
    df["cond_like"] = df["fro_norm"] / (df["abs_det"].replace(0, np.nan))
    return df


# ───────────────────────── 4) Integers ───────────────────────── #
def build_integer_df(n_min: int = 2, n_max: int = 200) -> Optional[pd.DataFrame]:
    if sp is None:
        return None
    out = []
    for n in range(n_min, n_max + 1):
        fac = sp.ntheory.factorint(n)
        primes = sorted(fac)
        exps = list(fac.values())
        tau = sp.ntheory.divisor_count(n)
        sigma = sp.ntheory.divisor_sigma(n)
        phi = sp.ntheory.totient(n)
        mu = sp.ntheory.mobius(n)
        omega = len(primes)
        big_omega = sum(exps)
        rad = 1
        for p in primes:
            rad *= p
        lpf = max(primes) if primes else np.nan
        spf = min(primes) if primes else np.nan
        is_prime = bool(sp.ntheory.isprime(n))
        squarefree = bool(mu != 0)
        prime_power = bool(len(primes) == 1)
        powerful = bool(all(e >= 2 for e in exps))
        perfect = bool(sigma == 2 * n)
        abundant = bool(sigma > 2 * n)
        deficient = bool(sigma < 2 * n)
        smooth_7 = bool((not np.isnan(lpf)) and lpf <= 7)
        smooth_11 = bool((not np.isnan(lpf)) and lpf <= 11)
        try:
            carmichael = bool(sp.ntheory.residue_ntheory.is_carmichael(n))
        except Exception:
            carmichael = False
        out.append(
            {
                "name": f"n={n}",
                "n": float(n),
                "tau": float(tau),
                "sigma": float(sigma),
                "phi": float(phi),
                "mu": float(mu),
                "omega": float(omega),
                "Omega": float(big_omega),
                "rad": float(rad),
                "lpf": float(lpf) if not np.isnan(lpf) else np.nan,
                "spf": float(spf) if not np.isnan(spf) else np.nan,
                "sigma_over_n": float(sigma / n),
                "phi_over_n": float(phi / n),
                "tau_over_logn": float(tau / np.log(n)),
                "is_prime": is_prime,
                "squarefree": squarefree,
                "prime_power": prime_power,
                "powerful": powerful,
                "perfect": perfect,
                "abundant": abundant,
                "deficient": deficient,
                "smooth_7": smooth_7,
                "smooth_11": smooth_11,
                "carmichael": carmichael,
            }
        )
    df = pd.DataFrame(out).set_index("name").sort_index()
    df["abundance"] = df["sigma"] - 2.0 * df["n"]
    df["omega_vs_log"] = df["omega"] / np.log(df["n"])
    return df


# ───────────────────────── Common per-domain runner ───────────────────────── #
def _mk_predicates_from_bool_cols(df: pd.DataFrame, Predicate):
    return {c: Predicate.from_column(c) for c in _bool_cols(df)}


def _pick_dense_predicate(df: pd.DataFrame, Where):
    num_cols = _first_n_numeric(df, 8)
    if "avg_deg" in df.columns:
        return Where(lambda d: d["avg_deg"] > (d["avg_deg"].median()), name="avg_deg>median")
    elif "sigma_over_n" in df.columns:
        return Where(lambda d: d["sigma_over_n"] > (d["sigma_over_n"].median()), name="sigma_over_n>median")
    elif len(num_cols) >= 1:
        col0 = num_cols[0]
        return Where(lambda d, c=col0: d[c] > (d[c].median()), name=f"{col0}>median")
    return None


def _render_relations_block(df: pd.DataFrame, txmod: Dict, limit_lists: bool = True) -> str:
    a_name, b_name = _pick_two_distinct_numeric(df)
    if a_name is None:
        return "(not enough numeric columns for relations demo)"
    to_expr, sqrt, ceil = txmod["to_expr"], txmod["sqrt"], txmod["ceil"]
    min_, max_, absdiff = txmod["min_"], txmod["max_"], txmod["absdiff"]

    a, b = to_expr(a_name), to_expr(b_name)
    mn, mx, adiff = min_(a, b), max_(a, b), absdiff(a, b)
    rt_b, ceil_rt_b = sqrt(b), ceil(sqrt(b))

    lines = []
    lines.append(f"a={a_name}: {a.pretty()}   b={b_name}: {b.pretty()}   √b: {rt_b.pretty()}   ⌈√b⌉: {ceil_rt_b.pretty()}")
    lines.append(f"min(a,b): {mn.pretty()}   max(a,b): {mx.pretty()}   |a−b|: {adiff.pretty()}")

    rels = [
        (f"{a_name} ≥ ⌈√{b_name}⌉", (a >= ceil_rt_b)),
        (f"{a_name} ≤ {b_name}", (a <= b)),
        (f"{a_name} = {a_name}", (a == a)),
        (f"{a_name} < {b_name}", (a < b)),
        (f"{a_name} > {b_name}", (a > b)),
        (f"{a_name} ≥ min({a_name},{b_name})", (a >= mn)),
        (f"max({a_name},{b_name}) ≥ {a_name}", (mx >= a)),
        (f"|{a_name}−{b_name}| ≥ 0", (adiff >= 0)),
        (f"min({a_name},{b_name}) ≤ ⌈√{b_name}⌉", (mn <= ceil_rt_b)),
    ]

    for lab, rel in rels:
        vals = rel(df).to_list()
        if limit_lists and len(vals) > 32:
            vals = vals[:32] + ["…"]
        lines.append(f"{lab}: {vals}")
    return "\n".join(lines)


def run_domain(
    console,
    df: pd.DataFrame,
    domain_name: str,
    *,
    max_arity: int,
    skip_gcl: bool,
    txmod: dict,
    sections: set,
    pager: bool,
    page_all: bool,
    table_rows: Optional[int],
    table_cols: Optional[int],
):
    with _pager(console, page_all):
        _hr(console)
        _print(console, f"[title]{domain_name}[/title]")
        _hr(console)

        # SECTION: data
        if "data" in sections:
            _as_table(console, f"{domain_name} • DataFrame (head)", df, max_rows=table_rows, max_cols=table_cols)

        # Predicates + relations require at least one boolean or numeric column
        preds = {}
        Dense = None
        if "preds" in sections or "relations" in sections:
            preds = _mk_predicates_from_bool_cols(df, txmod["Predicate"])
            Dense = _pick_dense_predicate(df, txmod["Where"])

        # SECTION: preds
        if "preds" in sections:
            _print(console, "\n[b]Predicates (sample)[/b]")
            for name, P in list(preds.items())[:5]:
                _print(console, f"{name:12s}: {P(df).to_list()}")
            if Dense is not None:
                _print(console, f"{Dense.name:12s}: {Dense(df).to_list()}")

        # SECTION: relations
        if "relations" in sections:
            block = _render_relations_block(df, txmod)
            _pager_if(console, "\n[b]Relations demo[/b]\n" + block, pager and not page_all)

            # Mixed combos only if we had a & b
            a_name, b_name = _pick_two_distinct_numeric(df)
            if a_name is not None:
                to_expr, sqrt, ceil = txmod["to_expr"], txmod["sqrt"], txmod["ceil"]
                min_, max_, absdiff = txmod["min_"], txmod["max_"], txmod["absdiff"]
                a, b = to_expr(a_name), to_expr(b_name)
                rel_ge, rel_le, rel_lt, rel_gt = (a >= ceil(sqrt(b))), (a <= b), (a < b), (a > b)

                TRUE = txmod["TRUE"]
                P1 = next(iter(preds.values())) if preds else TRUE
                mix_and = (rel_ge & P1)
                mix_or = (rel_lt | (Dense if Dense is not None else TRUE))
                mix_xor = (rel_le ^ (Dense if Dense is not None else TRUE))
                mix_not = ~(rel_gt)

                _print(console, "\n[b]Mixed BoolFormula combos[/b]")
                _print(console, f"({a_name} ≥ ⌈√{b_name}⌉) ∧ {getattr(P1,'name','TRUE')}: {mix_and(df).to_list()}")
                _print(console, f"({a_name} < {b_name}) ∨ {getattr(Dense,'name','TRUE')}: {mix_or(df).to_list()}")
                _print(console, f"({a_name} ≤ {b_name}) ⊕ {getattr(Dense,'name','TRUE')}: {mix_xor(df).to_list()}")
                _print(console, f"¬({a_name} > {b_name}): {mix_not(df).to_list()}")

                ite = txmod["ite"]
                pw = ite(P1, a, b)
                _print(console, f"\nPiecewise ite({getattr(P1,'name','TRUE')}, {a_name}, {b_name}) →")
                _print(console, str(pw.evaluate(df).to_list()))

                Conjecture = txmod["Conjecture"]
                C1 = Conjecture(relation=rel_ge, name=f"C1: {a_name} ≥ ⌈√{b_name}⌉")
                _print(console, f"\n[ok]{C1.name} holds on {int(C1.evaluate(df).sum())} / {len(df)}[/ok]")
                _print(console, f"C1 pretty: {C1.pretty()}")

        # Throttle cols for polynomial domain (keeps later sections readable)
        if domain_name.startswith("Polynomials"):
            keep_bools = ["monic", "squarefree", "palindromic", "sparse", "chebyshev", "cyclotomic"]
            keep_nums = ["degree", "terms", "height", "root_density", "gap_deg_terms", "height_over_deg"]
            keep = [c for c in df.columns if c in (keep_bools + keep_nums)]
            if keep:
                df = df[keep].copy()

        # SECTION: base
        if "base" in sections:
            _print(console, "\n[b]GraffitiBase[/b]")
            base = txmod["GraffitiBase"](df)
            with _pager(console, pager and not page_all):
                base.summary(verbose=True)
        else:
            base = txmod["GraffitiBase"](df)

        # SECTION: gcl
        if "gcl" in sections and not skip_gcl:
            _print(console, "\n[b]GraffitiClassLogic (pipeline)[/b]")
            gcl = txmod["GraffitiClassLogic"](base, run_pipeline=True, max_arity=max_arity, tol=0.0)
            nonred = [n for (n, _) in getattr(gcl, "nonredundant_conjunctions_", [])]
            _print(console, f"Nonredundant conjunctions: {nonred}")
            _print(console, "Constant exprs (by conjunction name):")
            for k, v in getattr(gcl, "constant_exprs_", {}).items():
                if v:
                    _print(console, f"  {k} → {v}")
            with _pager(console, pager and not page_all):
                gcl.summary_conjectures(verbose=True)
        elif "gcl" in sections and skip_gcl:
            _print(console, "\n[warn](Skipping GraffitiClassLogic)[/warn]")


# ───────────────────────── CLI / main ───────────────────────── #
DOMAINS = ("graphs", "polynomials", "matrices", "integers")


def build_domain(name: str, poly_n: int, mat_n: int, int_max: int) -> tuple[str, Optional[pd.DataFrame]]:
    name = name.lower()
    if name == "graphs":
        return ("Graphs (NetworkX curated)", build_graph_df())
    if name == "polynomials":
        return ("Polynomials (SymPy)", build_polynomial_df(n_random=poly_n))
    if name == "matrices":
        return ("Integer Matrices", build_matrix_df(n_random=mat_n, sizes=(2, 3, 4)))
    if name == "integers":
        return ("Integers (SymPy number theory)", build_integer_df(2, int_max))
    raise ValueError(f"Unknown domain: {name}")


def _checkbox_menu_questionary(options: list[str]) -> Optional[list[str]]:
    try:
        import questionary
        return questionary.checkbox("Select domains:", choices=options).ask()
    except Exception:
        return None


def _fzf_multi(options: list[str]) -> Optional[list[str]]:
    fzf = shutil.which("fzf")
    if not fzf:
        return None
    try:
        p = subprocess.run([fzf, "-m"], input="\n".join(options).encode(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if p.returncode != 0:
            return None
        sel = p.stdout.decode().strip().splitlines()
        return [s for s in sel if s in options]
    except Exception:
        return None


def interactive_pick(console) -> list[str]:
    picked = _checkbox_menu_questionary(list(DOMAINS))
    if picked:
        return picked
    picked = _fzf_multi(list(DOMAINS))
    if picked:
        return picked

    _print(console, "\n[b]Select domains (comma-separated numbers):[/b]")
    for i, d in enumerate(DOMAINS, 1):
        _print(console, f"  {i}) {d}")
    sel = input("> ").strip()
    idxs = []
    for tok in sel.split(","):
        tok = tok.strip()
        if tok.isdigit():
            k = int(tok)
            if 1 <= k <= len(DOMAINS):
                idxs.append(k - 1)
    return [DOMAINS[i] for i in idxs]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the TxGraffiti DSL & ClassLogic demo on multiple math domains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-d", "--domain", action="append", choices=DOMAINS, help="Domain to run (can repeat). If omitted, runs all unless --menu is used.")
    p.add_argument("--menu", action="store_true", help="Interactive picker to choose domains.")

    # Sections control
    p.add_argument(
        "--sections",
        type=str,
        default="data,preds,relations,base,gcl",
        help="Comma list of sections to show: data,preds,relations,base,gcl",
    )

    p.add_argument("--max-arity", type=int, default=2, help="Max arity for GraffitiClassLogic.")
    p.add_argument("--skip-gcl", action="store_true", help="Skip GraffitiClassLogic (only show DataFrame + Base).")

    # dataset sizes
    p.add_argument("--poly-n", type=int, default=40, help="Number of random polynomials.")
    p.add_argument("--mat-n", type=int, default=24, help="Number of random matrices.")
    p.add_argument("--int-max", type=int, default=200, help="Max integer n for number theory table.")

    # UX
    p.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    p.add_argument("--no-color", action="store_true", default=False, help="Disable rich colors if available.")
    p.add_argument("--no-wrap", action="store_true", help="Disable soft wrapping (Rich).")

    # Paging & layout
    default_pager = sys.stdout.isatty()
    p.add_argument("--pager", action="store_true", default=default_pager, help="Page long sections (relations, summaries).")
    p.add_argument("--no-pager", action="store_true", help="Force-disable paging even on a TTY.")
    p.add_argument("--page-all", action="store_true", help="Pipe the *entire* domain output through a pager.")
    p.add_argument("--rows", type=int, default=20, help="Rows to show in tables (0 shows all).")
    p.add_argument("--auto-rows", action="store_true", help="Auto-fit rows to terminal height.")
    p.add_argument("--max-cols", type=int, default=12, help="Max columns to show in tables (0 shows all).")

    p.add_argument("--export", type=str, default=None, help="Export each built dataframe as CSV to this folder.")
    p.add_argument("--save-md", type=str, default=None, help="Also save a Markdown report per domain to this folder.")
    p.add_argument("--quiet", action="store_true", help="Less verbose error messages.")
    return p.parse_args(argv)


def _export_df(path_dir: str, key: str, df: pd.DataFrame, console):
    try:
        os.makedirs(path_dir, exist_ok=True)
        out = os.path.join(path_dir, f"{key}.csv")
        df.to_csv(out)
        _print(console, f"[ok]Exported {key} → {out}[/ok]")
    except Exception as e:
        _print(console, f"[err]Export failed for {key}: {e}[/err]")


def _render_markdown_report(domain_name: str, df: pd.DataFrame, txmod: dict, sections: set, max_rows: Optional[int]) -> str:
    parts = [f"# {domain_name}\n"]
    if "data" in sections:
        head = df if max_rows in (None, 0) else df.head(max_rows)
        parts.append("## DataFrame (head)\n")
        parts.append(head.to_markdown())
        parts.append("")
    if "preds" in sections:
        preds = _mk_predicates_from_bool_cols(df, txmod["Predicate"])
        parts.append("## Predicates (sample)\n")
        for name, P in list(preds.items())[:5]:
            parts.append(f"- **{name}**: {P(df).to_list()}")
        parts.append("")
    if "relations" in sections:
        parts.append("## Relations demo\n")
        parts.append("```\n" + _render_relations_block(df, txmod, limit_lists=False) + "\n```")
    # base/gcl summaries are printed by txgraffiti classes to stdout; keep MD minimal.
    return "\n".join(parts)


def main(argv=None):
    args = parse_args(argv)
    _seed_everything(args.seed)

    # normalize pager flags
    pager = args.pager and not args.no_pager

    console = _console(not args.no_color, wrap=not args.no_wrap)
    _banner(console)

    txmod = _try_txgraffiti(console, quiet=args.quiet)
    if txmod is None:
        sys.exit(1)

    chosen = args.domain or []
    if args.menu:
        chosen = interactive_pick(console)
    if not chosen:
        chosen = list(DOMAINS)

    # parse sections
    sec = [s.strip().lower() for s in args.sections.split(",") if s.strip()]
    valid = {"data", "preds", "relations", "base", "gcl"}
    sections = set([s for s in sec if s in valid])
    if not sections:
        sections = {"data"}

    rows_to_show = _auto_rows(args.auto_rows, None if args.rows == 0 else args.rows)

    # optional progress
    use_progress = console is not None and Progress is not None and sys.stdout.isatty()
    if use_progress:
        progress = Progress(
            SpinnerColumn(style="title"),
            TextColumn("[title]{task.description}"),
            transient=True,
            console=console,
        )
        progress.start()
    else:
        progress = None

    try:
        for key in chosen:
            if progress:
                progress.add_task(description=f"Building {key}…", total=None)

            dom_name, df = build_domain(key, args.poly_n, args.mat_n, args.int_max)
            if df is None:
                _print(console, f"[warn]{dom_name.split()[0]} deps unavailable; skipping.[/warn]")
                continue

            if args.export:
                _export_df(args.export, key, df, console)
            if args.save_md:
                os.makedirs(args.save_md, exist_ok=True)
                md = _render_markdown_report(dom_name, df, txmod, sections, rows_to_show)
                out = os.path.join(args.save_md, f"{key}.md")
                with open(out, "w", encoding="utf-8") as f:
                    f.write(md)
                _print(console, f"[ok]Saved Markdown → {out}[/ok]")

            run_domain(
                console,
                df,
                dom_name,
                max_arity=args.max_arity,
                skip_gcl=args.skip_gcl,
                txmod=txmod,
                sections=sections,
                pager=pager,
                page_all=args.page_all,
                table_rows=rows_to_show,
                table_cols=None if args.max_cols == 0 else args.max_cols,
            )
    finally:
        if progress:
            progress.stop()


if __name__ == "__main__":
    main()
