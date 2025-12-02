# scripts/demo_dsl.py
from __future__ import annotations
import math, random, argparse, sys
import numpy as np
import pandas as pd

# --- Try optional deps ---
try:
    import networkx as nx
except Exception:
    nx = None

try:
    import sympy as sp
    from sympy import Matrix, symbols, Poly
except Exception:
    sp = None
    Matrix = None
    Poly = None

# Adjust these imports if your package structure differs:
from txgraffiti2025.graffiti_utils import (
    to_expr, sqrt, floor, ceil, log,
    min_, max_, absdiff,
)
from txgraffiti2025.graffiti_predicates import Predicate, Where, TRUE
from txgraffiti2025.graffiti_generic_conjecture import (
    Ge, Le, Eq, Lt, Gt,
    AndF, OrF, XorF, NotF, Implies, Iff, AllOf,
    ite, Conjecture,
)
from txgraffiti2025.graffiti_base import GraffitiBase
from txgraffiti2025.graffiti_class_logic import GraffitiClassLogic

random.seed(7)
np.random.seed(7)

# ───────────────────────────── helpers ───────────────────────────── #

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

def _pick_two_distinct_numeric(df: pd.DataFrame) -> tuple[str, str] | tuple[None, None]:
    nums = _first_n_numeric(df, k=6)
    if len(nums) >= 2:
        return nums[0], nums[1]
    return (None, None)

def _mk_predicates_from_bool_cols(df: pd.DataFrame) -> dict[str, Predicate]:
    preds = {}
    for c in _bool_cols(df):
        preds[c] = Predicate.from_column(c)
    return preds

def _print_section(title: str):
    bar = "─" * 58
    print(f"\n{bar}\n{title}\n{bar}")

def _rational_or_float(val) -> float:
    try:
        vv = float(val)
        if math.isfinite(vv):
            return float(vv)
    except Exception:
        pass
    return float("nan")

# ───────────────────────────── 1) GRAPHS ───────────────────────────── #

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

def build_graph_df() -> pd.DataFrame | None:
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
        ("Petersen", nx.petersen_graph())
    ]
    rows = []
    for name, G in items:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        connected = nx.is_connected(G)
        tree = nx.is_tree(G) if connected else False
        planar = _is_planar(G)
        rad, diam = _safe_radius_diameter(G)
        rows.append({
            "name": name,
            "order": float(n),
            "size": float(m),
            "radius": float(rad) if math.isfinite(rad) else np.nan,
            "diameter": float(diam) if math.isfinite(diam) else np.nan,
            "connected": bool(connected),
            "planar": bool(planar),
            "tree": bool(tree),
        })
    df = pd.DataFrame(rows).set_index("name").sort_index()
    df["avg_deg"] = (2.0 * df["size"]) / df["order"]
    return df

# ───────────────────── 2) SYMPY POLYNOMIALS (fast) ───────────────────── #

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

def build_polynomial_df(n_random: int = 40) -> pd.DataFrame | None:
    if sp is None:
        return None
    x = sp.symbols('x')

    polys: list[tuple[str, sp.Poly]] = []
    for n in [1,2,3,4,5,6,7,8,9,10,12]:
        polys.append((f"cyclo_{n}", sp.Poly(sp.cyclotomic_poly(n, x), x, domain=sp.QQ)))
    for n in [1,2,3,4,5,6]:
        polys.append((f"chebyshev_T{n}", sp.Poly(sp.chebyshevt(n, x).expand(), x, domain=sp.QQ)))
        polys.append((f"legendre_P{n}", sp.Poly(sp.legendre(n, x).expand(), x, domain=sp.QQ)))

    pal_vecs = [[1,0,1],[1,-1,2,-1,1],[2,0,-3,0,2]]
    for i, c in enumerate(pal_vecs):
        p = sum(c[j]*x**(len(c)-1-j) for j in range(len(c)))
        polys.append((f"pal_{i}", sp.Poly(p, x, domain=sp.QQ)))

    rr = [(x-2)**2,(x+1)**3,(x**2+1)**2,(x-1)**2*(x+2)]
    for i, p in enumerate(rr):
        polys.append((f"repeated_{i}", sp.Poly(sp.expand(p), x, domain=sp.QQ)))

    rng = np.random.default_rng(7)
    for i in range(n_random):
        deg = int(rng.integers(2, 7))
        if rng.random() < 0.5:
            coeffs = np.zeros(deg+1, dtype=int)
            k = max(2, int(0.3*(deg+1)))
            idx = rng.choice(np.arange(deg+1), size=k, replace=False)
            coeffs[idx] = rng.integers(-4, 5, size=k)
            if coeffs[0] == 0: coeffs[0] = int(rng.integers(1, 4))
        else:
            coeffs = rng.integers(-4, 5, size=deg+1)
            if coeffs[0] == 0: coeffs[0] = int(rng.integers(1, 4))
        poly = sum(int(coeffs[j]) * x**(deg-j) for j in range(deg+1))
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
        odd_poly  = bool(P.as_expr().subs(x, -x) + P.as_expr() == 0)
        is_cyclo  = name.startswith("cyclo_")
        is_cheb   = name.startswith("chebyshev_T")
        is_leg    = name.startswith("legendre_P")
        sparse    = bool(terms <= max(2, (deg+1)//2))

        rows.append({
            "name": name,
            "degree": float(deg),
            "terms": float(terms),
            "height": float(height),
            "l1_coeff": float(l1),
            "leading": float(leading),
            "constant": float(const),
            "discriminant": float(disc) if math.isfinite(disc) else np.nan,
            "real_root_count": float(real_root_count) if math.isfinite(real_root_count) else np.nan,
            "root_density": float(real_root_count/deg) if (deg>0 and math.isfinite(real_root_count)) else np.nan,
            "monic": monic, "squarefree": squarefree, "irreducible": irreducible,
            "palindromic": palindromic, "reciprocal": reciprocal,
            "even_poly": even_poly, "odd_poly": odd_poly,
            "cyclotomic": is_cyclo, "chebyshev": is_cheb, "legendre": is_leg,
            "sparse": sparse,
        })

    df = pd.DataFrame(rows).set_index("name").sort_index()
    df["gap_deg_terms"] = df["degree"] - df["terms"]
    df["height_over_deg"] = df["height"] / df["degree"].replace(0, np.nan)
    return df

# ─────────────── 3) INTEGER MATRICES (richer) ─────────────── #

def _circulant(vec: np.ndarray) -> np.ndarray:
    n = len(vec)
    return np.vstack([np.roll(vec, i) for i in range(n)])

def _fro_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord='fro'))

def build_matrix_df(n_random: int = 24, sizes=(2, 3, 4, 5)) -> pd.DataFrame | None:
    if sp is None:
        return None

    mats: list[tuple[str, np.ndarray]] = []
    for n in sizes:
        mats.append((f"I{n}", np.eye(n)))
        mats.append((f"Z{n}", np.zeros((n, n))))
        mats.append((f"J{n}", np.ones((n, n))))
        mats.append((f"DiagStep_{n}", np.diag(np.arange(1, n+1))))
        mats.append((f"DiagAlt_{n}", np.diag([1 if i%2==0 else -1 for i in range(n)])))
        J = np.zeros((n, n))
        for i in range(n-1):
            J[i, i+1] = 1.0
        mats.append((f"JordanNil_{n}", J))
        R = np.random.default_rng(7).integers(-2, 3, size=(n, n)).astype(float)
        PSD = R.T @ R
        mats.append((f"PSD_{n}", PSD))
        R2 = np.random.default_rng(11).integers(-2, 3, size=(n, n)).astype(float)
        Sk = R2 - R2.T
        mats.append((f"Skew_{n}", Sk))
        base = np.arange(1, n+1, dtype=float)
        mats.append((f"Circulant_{n}", _circulant(base)))

    for theta in [0.0, math.pi/6, math.pi/4, math.pi/3]:
        c, s = math.cos(theta), math.sin(theta)
        mats.append((f"Rot2_{theta:.2f}", np.array([[c,-s],[s,c]], dtype=float)))
    for n in sizes:
        u = np.random.default_rng(3).normal(size=(n,1))
        u /= np.linalg.norm(u)
        H = np.eye(n) - 2*(u@u.T)
        mats.append((f"House_{n}", H))
    for n in sizes:
        P = np.zeros((n, n)); P[0,0] = 1.0
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
            spectral_radius = np.nan; num_pos = np.nan; min_eig = np.nan

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

        rows.append({
            "name": name, "order": float(n), "rank": float(rank),
            "trace": float(trace) if math.isfinite(trace) else np.nan,
            "det": float(det) if math.isfinite(det) else np.nan,
            "abs_det": float(abs_det) if math.isfinite(abs_det) else np.nan,
            "fro_norm": float(fro),
            "spectral_radius": float(spectral_radius) if math.isfinite(spectral_radius) else np.nan,
            "num_pos_eigs": float(num_pos) if math.isfinite(num_pos) else np.nan,
            "symmetric": symmetric, "skew_symmetric": skew_symm, "orthogonal": orthogonal,
            "invertible": invertible, "full_rank": full_rank, "psd": psd, "nsd": nsd,
            "idempotent": idempotent, "involutory": involutory,
            "nilpotent2": nilpotent2, "nilpotent3": nilpotent3,
        })

    df = pd.DataFrame(rows).set_index("name").sort_index()
    df["rank_gap"] = df["order"] - df["rank"]
    df["cond_like"] = df["fro_norm"] / (df["abs_det"].replace(0, np.nan))
    return df

# ─────────────── 4) NUMBER THEORY INTEGERS (richer) ─────────────── #

def build_integer_df(n_min: int = 2, n_max: int = 200) -> pd.DataFrame | None:
    if sp is None:
        return None
    out = []
    for n in range(n_min, n_max + 1):
        fac = sp.ntheory.factorint(n)
        primes = sorted(fac); exps = list(fac.values())
        tau = sp.ntheory.divisor_count(n)
        sigma = sp.ntheory.divisor_sigma(n)
        phi = sp.ntheory.totient(n)
        mu = sp.ntheory.mobius(n)
        omega = len(primes); big_omega = sum(exps)
        rad = 1
        for p in primes: rad *= p
        lpf = max(primes) if primes else np.nan
        spf = min(primes) if primes else np.nan
        is_prime = bool(sp.ntheory.isprime(n))
        squarefree = bool(mu != 0)
        prime_power = bool(len(primes) == 1)
        powerful = bool(all(e >= 2 for e in exps))
        perfect = bool(sigma == 2*n)
        abundant = bool(sigma > 2*n)
        deficient = bool(sigma < 2*n)
        smooth_7 = bool((not np.isnan(lpf)) and lpf <= 7)
        smooth_11 = bool((not np.isnan(lpf)) and lpf <= 11)
        try:
            carmichael = bool(sp.ntheory.residue_ntheory.is_carmichael(n))
        except Exception:
            carmichael = False
        out.append({
            "name": f"n={n}", "n": float(n), "tau": float(tau), "sigma": float(sigma),
            "phi": float(phi), "mu": float(mu), "omega": float(omega), "Omega": float(big_omega),
            "rad": float(rad), "lpf": float(lpf) if not np.isnan(lpf) else np.nan,
            "spf": float(spf) if not np.isnan(spf) else np.nan,
            "sigma_over_n": float(sigma/n), "phi_over_n": float(phi/n),
            "tau_over_logn": float(tau/np.log(n)),
            "is_prime": is_prime, "squarefree": squarefree, "prime_power": prime_power,
            "powerful": powerful, "perfect": perfect, "abundant": abundant,
            "deficient": deficient, "smooth_7": smooth_7, "smooth_11": smooth_11,
            "carmichael": carmichael,
        })
    df = pd.DataFrame(out).set_index("name").sort_index()
    df["abundance"] = df["sigma"] - 2.0*df["n"]
    df["omega_vs_log"] = df["omega"] / np.log(df["n"])
    return df

# ───────────────────── common per-domain runner ───────────────────── #

def run_domain(df: pd.DataFrame, domain_name: str, *, max_arity: int, skip_gcl: bool):
    _print_section(f"{domain_name} • DataFrame")
    print(df); print()

    preds = _mk_predicates_from_bool_cols(df)

    num_cols = _first_n_numeric(df, 8)
    Dense = None
    if "avg_deg" in df.columns:
        Dense = Where(lambda d: d["avg_deg"] > (d["avg_deg"].median()), name="avg_deg>median")
    elif "sigma_over_n" in df.columns:
        Dense = Where(lambda d: d["sigma_over_n"] > (d["sigma_over_n"].median()), name="sigma_over_n>median")
    elif len(num_cols) >= 1:
        col0 = num_cols[0]
        Dense = Where(lambda d, c=col0: d[c] > (d[c].median()), name=f"{col0}>median")

    _print_section(f"{domain_name} • Predicates (sample)")
    for name, P in list(preds.items())[:5]:
        print(f"{name:12s}:", P(df).to_list())
    if Dense is not None:
        print(f"{Dense.name:12s}:", Dense(df).to_list())
    print()

    a_name, b_name = _pick_two_distinct_numeric(df)
    if a_name is not None:
        a, b = to_expr(a_name), to_expr(b_name)
        mn, mx, adiff = min_(a, b), max_(a, b), absdiff(a, b)
        rt_b, ceil_rt_b = sqrt(b), ceil(sqrt(b))

        _print_section(f"{domain_name} • Expr pretty samples")
        print(f"a={a_name}:", a.pretty(), " b={b_name}:", b.pretty(),
              " √b:", rt_b.pretty(), " ⌈√b⌉:", ceil_rt_b.pretty())
        print("min(a,b):", mn.pretty(), " max(a,b):", mx.pretty(), " |a−b|:", adiff.pretty()); print()

        rel_ge, rel_le, rel_eq = (a >= ceil_rt_b), (a <= b), (a == a)
        rel_lt, rel_gt = (a < b), (a > b)
        rel_min1, rel_max1, rel_ad0 = (a >= mn), (mx >= a), (adiff >= 0)
        rel_min2 = (mn <= ceil_rt_b)

        _print_section(f"{domain_name} • Relations demo")
        for lab, rel in [
            (f"{a_name} ≥ ⌈√{b_name}⌉", rel_ge),
            (f"{a_name} ≤ {b_name}", rel_le),
            (f"{a_name} = {a_name}", rel_eq),
            (f"{a_name} < {b_name}", rel_lt),
            (f"{a_name} > {b_name}", rel_gt),
            (f"{a_name} ≥ min({a_name},{b_name})", rel_min1),
            (f"max({a_name},{b_name}) ≥ {a_name}", rel_max1),
            (f"|{a_name}−{b_name}| ≥ 0", rel_ad0),
            (f"min({a_name},{b_name}) ≤ ⌈√{b_name}⌉", rel_min2),
        ]:
            print(f"{lab:32s}:", rel(df).to_list())
        print()

        if preds or Dense is not None:
            P1 = next(iter(preds.values())) if preds else TRUE
            mix_and = (rel_ge & P1)
            mix_or  = (rel_lt | (Dense if Dense is not None else TRUE))
            mix_xor = (rel_le ^ (Dense if Dense is not None else TRUE))
            mix_not = ~(rel_gt)

            _print_section(f"{domain_name} • Mixed BoolFormula combos")
            print(f"({a_name} ≥ ⌈√{b_name}⌉) ∧ {getattr(P1,'name','TRUE')}:",
                  mix_and(df).to_list())
            print(f"({a_name} < {b_name}) ∨ {getattr(Dense,'name','TRUE')}:",
                  mix_or(df).to_list())
            print(f"({a_name} ≤ {b_name}) ⊕ {getattr(Dense,'name','TRUE')}:",
                  mix_xor(df).to_list())
            print(f"¬({a_name} > {b_name}):", mix_not(df).to_list()); print()

            pw = ite(P1, a, b)
            print(f"Piecewise ite({getattr(P1,'name','TRUE')}, {a_name}, {b_name}) →")
            print(pw.evaluate(df).to_list()); print()

            C1 = Conjecture(relation=rel_ge, name=f"C1: {a_name} ≥ ⌈√{b_name}⌉")
            print(C1.name, "holds on", int(C1.evaluate(df).sum()), "/", len(df))
            print("C1 pretty:", C1.pretty()); print()

    # Throttle columns for polynomials to speed GCL
    if domain_name.startswith("Polynomials"):
        keep_bools = ["monic","squarefree","palindromic","sparse","chebyshev","cyclotomic"]
        keep_nums  = ["degree","terms","height","root_density","gap_deg_terms","height_over_deg"]
        keep = [c for c in df.columns if c in (keep_bools + keep_nums)]
        if keep:
            df = df[keep].copy()

    _print_section(f"{domain_name} • GraffitiBase")
    base = GraffitiBase(df)
    base.summary(verbose=True)

    if skip_gcl:
        _print_section(f"{domain_name} • (skipping GraffitiClassLogic)")
        return

    _print_section(f"{domain_name} • GraffitiClassLogic (pipeline)")
    gcl = GraffitiClassLogic(base, run_pipeline=True, max_arity=max_arity, tol=0.0)
    print("Nonredundant conjunctions:",
          [n for (n, _) in getattr(gcl, "nonredundant_conjunctions_", [])])
    print("Constant exprs (by conjunction name):")
    for k, v in getattr(gcl, "constant_exprs_", {}).items():
        if v: print(" ", k, "→", v)
    print()
    gcl.summary_conjectures(verbose=True)

# ───────────────────────────── CLI / main ───────────────────────────── #

DOMAINS = ("graphs", "polynomials", "matrices", "integers")

from txgraffiti.example_data import graph_data as df

def build_domain(name: str, poly_n: int, mat_n: int, int_max: int) -> tuple[str, pd.DataFrame | None]:
    name = name.lower()
    if name == "graphs":
        return ("Graphs (NetworkX curated)", df)
    if name == "polynomials":
        return ("Polynomials (SymPy)", build_polynomial_df(n_random=poly_n))
    if name == "matrices":
        return ("Integer Matrices (SymPy)", build_matrix_df(n_random=mat_n, sizes=(2,3,4)))
    if name == "integers":
        return ("Integers (SymPy number theory)", build_integer_df(2, int_max))
    raise ValueError(f"Unknown domain: {name}")

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run DSL/Graffiti pipelines on selectable domains.")
    p.add_argument("-d","--domain", action="append", choices=DOMAINS,
                   help="Domain to run (can repeat). If omitted, runs all unless --menu is used.")
    p.add_argument("--menu", action="store_true",
                   help="Interactive picker to choose domains.")
    p.add_argument("--max-arity", type=int, default=2,
                   help="Max arity for GraffitiClassLogic (default: 2).")
    p.add_argument("--skip-gcl", action="store_true",
                   help="Skip GraffitiClassLogic (only show DataFrame + Base).")
    # dataset sizes
    p.add_argument("--poly-n", type=int, default=40,
                   help="Number of random polynomials to generate (default: 40).")
    p.add_argument("--mat-n", type=int, default=24,
                   help="Number of random matrices to generate (default: 24).")
    p.add_argument("--int-max", type=int, default=200,
                   help="Max integer n for number theory table (default: 200).")
    return p.parse_args(argv)

def interactive_pick() -> list[str]:
    print("\nSelect domains (comma-separated numbers):")
    for i, d in enumerate(DOMAINS, 1):
        print(f"  {i}) {d}")
    sel = input("> ").strip()
    idxs = []
    for tok in sel.split(","):
        tok = tok.strip()
        if tok.isdigit():
            k = int(tok)
            if 1 <= k <= len(DOMAINS):
                idxs.append(k-1)
    return [DOMAINS[i] for i in idxs]

def main(argv=None):
    args = parse_args(argv)

    chosen = args.domain or []
    if args.menu:
        chosen = interactive_pick()
    if not chosen:
        # default = all
        chosen = list(DOMAINS)

    # Build and run each chosen domain
    for key in chosen:
        dom_name, df = build_domain(key, args.poly_n, args.mat_n, args.int_max)
        if df is None:
            print(f"{dom_name.split()[0]} deps unavailable; skipping.")
            continue
        run_domain(df, dom_name, max_arity=args.max_arity, skip_gcl=args.skip_gcl)

if __name__ == "__main__":
    main()
