# ============================================================
# Colab-ready: Polars DSL + Generators (ratio/mix/product)
# Adapts the ideas from your pandas/Numpy DSL to Polars
# ============================================================

# If starting fresh:
# !pip -q install polars pandas pyarrow txgraffiti

import polars as pl
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Iterable
from itertools import combinations, product
from fractions import Fraction
import math

# ------------------------------
# Core: tiny expression & conjecture layer on Polars Expr
# ------------------------------
def col(name: str) -> pl.Expr:
    return pl.col(name)

@dataclass
class Relation:
    label: str
    bool_expr: pl.Expr
    rhs: Optional[pl.Expr] = None           # <-- new: keep RHS for scoring/inspection
    sharp_expr: Optional[pl.Expr] = None    # rows achieving equality for ≤/≥/==


@dataclass
class Conjecture:
    name: str
    premises: List[pl.Expr] = field(default_factory=list)  # ANDed together
    relation: Relation = None


def le(lhs: pl.Expr, rhs: pl.Expr, label: str) -> Relation:
    return Relation(label, lhs <= rhs, rhs, lhs == rhs)

def ge(lhs: pl.Expr, rhs: pl.Expr, label: str) -> Relation:
    return Relation(label, lhs >= rhs, rhs, lhs == rhs)

def lt(lhs: pl.Expr, rhs: pl.Expr, label: str) -> Relation:
    return Relation(label, lhs < rhs, rhs, None)

def gt(lhs: pl.Expr, rhs: pl.Expr, label: str) -> Relation:
    return Relation(label, lhs > rhs, rhs, None)

def eq(lhs: pl.Expr, rhs: pl.Expr, label: str) -> Relation:
    return Relation(label, lhs == rhs, rhs, lhs == rhs)


# Simple “algebra” helpers
def to_const(x: float, max_denom: int = 30) -> float:
    # keep as float but quantize with a bounded-denominator rational for readability/stability
    return float(Fraction(x).limit_denominator(max_denom))

def with_floor(e: pl.Expr) -> pl.Expr:
    return e.floor()

def with_ceil(e: pl.Expr) -> pl.Expr:
    return e.ceil()

# ------------------------------
# Namespace: df.conj.eval(...)  (evaluate conjectures at once)
# ------------------------------
@pl.api.register_dataframe_namespace("conj")
class ConjNamespace:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def eval(self, conjectures: List[Conjecture], *, streaming: bool = False) -> pl.DataFrame:
        if not conjectures:
            return pl.DataFrame([])

        exprs, jobs = [], []
        for jid, conj in enumerate(conjectures):
            C = pl.lit(True)
            for p in conj.premises:
                C = C & p
            R = conj.relation.bool_expr
            S = conj.relation.sharp_expr

            exprs.extend([
                C.sum().alias(f"supp__{jid}"),
                (C & R).sum().alias(f"ok__{jid}"),
                (C & ~R).sum().alias(f"bad__{jid}"),
            ])
            if S is not None:
                exprs.append((C & S).sum().alias(f"touch__{jid}"))
            jobs.append((jid, conj.name, conj.relation.label, S is not None))

        row = (
            self._df.lazy().select(exprs).collect(streaming=streaming)
            if streaming else self._df.select(exprs)
        ).row(0, named=True)

        N = self._df.height
        out = []
        for jid, name, label, has_sharp in jobs:
            supp  = int(row.get(f"supp__{jid}", 0) or 0)
            ok    = int(row.get(f"ok__{jid}", 0) or 0)
            bad   = int(row.get(f"bad__{jid}", 0) or 0)
            touch = int(row.get(f"touch__{jid}", 0) or 0) if has_sharp else None
            holds = (supp > 0 and bad == 0)
            out.append({
                "name": name,
                "relation": label,
                "holds": holds,
                "support": supp,
                "violations": bad,
                "touch": touch,
                "touch%": None if (touch is None or supp == 0) else round(100*touch/supp, 1),
            })

        df_out = pl.DataFrame(out)
        # Sort: (not holding first), then by sharpness (touch desc), then name
        return df_out.sort(
            by=[pl.col("holds").cast(pl.Int8), pl.col("touch").fill_null(-1), "name"],
            descending=[False, True, False],
        )

# ------------------------------
# Namespace: df.smart (helpers we’ll reuse, esp. ratio bounds)
# ------------------------------
@pl.api.register_dataframe_namespace("smart")
class SmartNamespace:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def ratio_bounds(
        self,
        *,
        target: Optional[str] = None,
        denominators: Optional[List[str]] = None,
        hypotheses: Optional[List[str]] = None,  # boolean cols
        include_global: bool = True,
        require_positive: bool = True,
        skip_self: bool = True,
        round_digits: int = 6,
        streaming: bool = False,
    ) -> pl.DataFrame:
        df = self._df

        num_cols = [c for c, dt in df.schema.items() if dt in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        )]
        bool_cols = [c for c, dt in df.schema.items() if dt == pl.Boolean]
        if not num_cols:
            return pl.DataFrame([])

        H_list: List[str] = []
        if hypotheses:
            for h in hypotheses:
                if h not in bool_cols:
                    raise ValueError(f"Not a boolean column: {h}")
                H_list.append(h)
        if include_global:
            H_list = ["__ALL__"] + H_list
        if not H_list:
            H_list = ["__ALL__"]

        num_list = [target] if (target and target in num_cols) else num_cols
        den_list = [d for d in (denominators or num_cols) if d in num_cols]

        exprs, jobs = [], []
        # precompute supports per H
        for H in H_list:
            Hmask = pl.lit(True) if H == "__ALL__" else pl.col(H)
            exprs.append(pl.when(Hmask).then(1).otherwise(0).sum().alias(f"supp__{H}"))

        jid = 0
        for H in H_list:
            Hmask = pl.lit(True) if H == "__ALL__" else pl.col(H)
            for num, den in product(num_list, den_list):
                if skip_self and num == den:
                    continue
                pos = (pl.col(den) > 0) if require_positive else (pl.col(den) != 0)
                valid = Hmask & pos
                ratio = pl.when(valid).then(pl.col(num) / pl.col(den)).otherwise(None)
                exprs.extend([
                    ratio.min().alias(f"mn__{jid}"),
                    ratio.max().alias(f"mx__{jid}"),
                    valid.sum().alias(f"used__{jid}"),
                ])
                jobs.append((jid, H, num, den))
                jid += 1

        row = df.lazy().select(exprs).collect(streaming=streaming).row(0, named=True) if streaming else df.select(exprs).row(0, named=True)

        out = []
        for jid, H, num, den in jobs:
            cmin = row[f"mn__{jid}"]; cmax = row[f"mx__{jid}"]
            used = int(row[f"used__{jid}"] or 0)
            supp = int(row[f"supp__{H}"] or 0) if H != "__ALL__" else df.height
            if cmin is None or cmax is None or used == 0:
                continue
            out.append({
                "hypothesis": "ALL" if H == "__ALL__" else H,
                "num": num, "den": den,
                "cmin": round(float(cmin), round_digits),
                "cmax": round(float(cmax), round_digits),
                "support": supp, "used_rows": used,
            })
        return pl.DataFrame(out).sort(["hypothesis", "num", "den"])

# ------------------------------
# Generators that adapt your old pipeline’s ideas
# ------------------------------

def list_boolean_and_numeric_cols(df: pl.DataFrame) -> Tuple[List[str], List[str]]:
    bool_cols = [c for c, dt in df.schema.items() if dt == pl.Boolean]
    num_cols = [c for c, dt in df.schema.items() if dt in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ) and c not in bool_cols]
    return bool_cols, num_cols

def _premises_from_name(df: pl.DataFrame, Hname: str) -> List[pl.Expr]:
    return [] if Hname == "ALL" else [pl.col(Hname)]

def _strength_ge(df: pl.DataFrame, rhs: pl.Expr, prem: List[pl.Expr]) -> float:
    mask = pl.lit(True)
    for p in prem: mask = mask & p
    vals = df.select(pl.when(mask).then(rhs).otherwise(None).mean()).item()
    return float(vals) if vals is not None else float("-inf")

def _strength_le(df: pl.DataFrame, rhs: pl.Expr, prem: List[pl.Expr]) -> float:
    m = _strength_ge(df, rhs, prem)
    return -m if math.isfinite(m) else float("-inf")

def gen_linear_bounds(
    df: pl.DataFrame,
    *,
    target: str,
    hypotheses: Optional[List[str]] = None,
    include_global: bool = True,
    floor_ceil: bool = True,
    denom_whitelist: Optional[List[str]] = None,
) -> List[Conjecture]:
    """
    Ratio bounds → linear bounds y ≥ cmin*x, y ≤ cmax*x.
    Adds ceil/floor variants ONLY if they remain true on the slice.
    """
    bounds = df.smart.ratio_bounds(
        target=target,
        denominators=denom_whitelist,
        hypotheses=hypotheses,
        include_global=include_global,
        require_positive=True,
        skip_self=True,
        round_digits=6,
    )

    conjs: List[Conjecture] = []
    for r in bounds.iter_rows(named=True):
        H, num, den = r["hypothesis"], r["num"], r["den"]
        if num != target:  # only target/den
            continue
        cmin, cmax = to_const(r["cmin"]), to_const(r["cmax"])
        prem = _premises_from_name(df, H)
        x = col(den); y = col(target)

        # Lower: y ≥ cmin*x  (+ optional ceil if still holds)
        base_ge = ge(y, pl.lit(cmin) * x, f"{target} ≥ {cmin}·{den}")
        cand = [("base_ge", base_ge)]
        if floor_ceil:
            cand.append(("ceil_ge", ge(y, with_ceil(pl.lit(cmin) * x), f"{target} ≥ ceil({cmin}·{den})")))
        # choose strongest true (highest mean RHS)
        picks = []
        for tag, rel in cand:
            cj = Conjecture(f"[{H}] lower {target}|{den} ({tag})", premises=prem, relation=rel)
            picks.append((tag, cj, _strength_ge(df, rel.rhs, prem)))   # <-- here
        picks = sorted(picks, key=lambda t: t[2], reverse=True)
        conjs.append(picks[0][1])




        # Upper: y ≤ cmax*x  (+ optional floor if still holds)
        base_le = le(y, pl.lit(cmax) * x, f"{target} ≤ {cmax}·{den}")
        cand = [("base_le", base_le)]
        if floor_ceil:
            cand.append(("floor_le", le(y, with_floor(pl.lit(cmax) * x), f"{target} ≤ floor({cmax}·{den})")))
        picks = []
        for tag, rel in cand:
            cj = Conjecture(f"[{H}] upper {target}|{den} ({tag})", premises=prem, relation=rel)
            picks.append((tag, cj, _strength_le(df, rel.rhs, prem)))   # <-- and here
        picks = sorted(picks, key=lambda t: t[2], reverse=True)
        conjs.append(picks[0][1])

    return conjs

def gen_mix_bounds(
    df: pl.DataFrame,
    *,
    target: str,
    hypotheses: Optional[List[str]] = None,
    include_global: bool = True,
    secondaries: Optional[List[str]] = None,
    primary_whitelist: Optional[List[str]] = None,
) -> List[Conjecture]:
    """
    2-feature “½-mix” bounds:
      Lower:  y ≥ ½(cmin*x + s_cmin*sqrt(z))   (and ceil variants)
      Upper:  y ≤ ½(cmax*x + s_cmax*sqrt(z))   (and floor variants)
    Also a squared mix with z^2. Pick the strongest variant that is true on the slice (by mean RHS).
    """
    bool_cols, num_cols = list_boolean_and_numeric_cols(df)
    primaries = [p for p in (primary_whitelist or num_cols) if p != target]
    secondaries = [s for s in (secondaries or num_cols) if s != target]

    # choose hypotheses
    H_list = (hypotheses or [])[:]
    if include_global:
        H_list = ["ALL"] + H_list

    y = col(target)
    conjs: List[Conjecture] = []

    # For each hypothesis slice:
    for H in H_list:
        prem = _premises_from_name(df, H)
        Hmask = pl.lit(True) if H == "ALL" else pl.col(H)

        # Compute cmin/cmax for target/x for all primaries in one pass
        rb_x = df.smart.ratio_bounds(
            target=target,
            denominators=primaries,
            hypotheses=None if H == "ALL" else [H],
            include_global=(H == "ALL"),
            require_positive=True,
            skip_self=True,
        ).filter(pl.col("hypothesis") == H)

        # Precompute sqrt(z) and z^2 min/max ratios in one pass
        exprs: List[pl.Expr] = []
        jobs: List[Tuple[int, str]] = []
        jid = 0
        for z in secondaries:
            pos_sqrt = (col(z) > 0)      # sqrt requires z > 0
            pos_sq   = (col(z) != 0)     # z^2 > 0 unless z == 0

            valid_sqrt = Hmask & pos_sqrt
            valid_sq   = Hmask & pos_sq

            ratio_sqrt = pl.when(valid_sqrt).then(y / col(z).sqrt()).otherwise(None)
            ratio_sq   = pl.when(valid_sq).then(y / (col(z) * col(z))).otherwise(None)

            exprs += [
                ratio_sqrt.min().alias(f"mn_sqrt__{jid}"),
                ratio_sqrt.max().alias(f"mx_sqrt__{jid}"),
                valid_sqrt.sum().alias(f"used_sqrt__{jid}"),
                ratio_sq.min().alias(f"mn_sq__{jid}"),
                ratio_sq.max().alias(f"mx_sq__{jid}"),
                valid_sq.sum().alias(f"used_sq__{jid}"),
            ]
            jobs.append((jid, z))
            jid += 1

        row = df.select(exprs).row(0, named=True) if exprs else {}

        # Iterate over (x, z), build mixes and choose strongest true variant
        for r in rb_x.iter_rows(named=True):
            den = r["den"]
            cmin_x = to_const(r["cmin"])
            cmax_x = to_const(r["cmax"])
            x = col(den)

            for jid, z in jobs:
                # sqrt mix
                used_sqrt = int(row.get(f"used_sqrt__{jid}", 0) or 0)
                if used_sqrt > 0:
                    s_cmin = to_const(row[f"mn_sqrt__{jid}"])
                    s_cmax = to_const(row[f"mx_sqrt__{jid}"])

                    base_lo = 0.5 * (pl.lit(cmin_x) * x + pl.lit(s_cmin) * col(z).sqrt())
                    base_hi = 0.5 * (pl.lit(cmax_x) * x + pl.lit(s_cmax) * col(z).sqrt())

                    # Lower ≥ : base, ceil_whole, ceil_split-1
                    lo_variants = [
                        ("base",
                         ge(y, base_lo, f"{target} ≥ ½({cmin_x}·{den}+{s_cmin}·√{z})")),
                        ("ceil_w",
                         ge(y, with_ceil(base_lo), f"{target} ≥ ceil(½({cmin_x}·{den}+{s_cmin}·√{z}))")),
                        ("ceil_s",
                         ge(
                             y,
                             with_ceil(0.5 * pl.lit(cmin_x) * x) +
                             with_ceil(0.5 * pl.lit(s_cmin) * col(z).sqrt()) - 1,
                             f"{target} ≥ ceil(½{cmin_x}·{den})+ceil(½{s_cmin}·√{z})-1",
                         )),
                    ]
                    scored = [
                        (
                            t,
                            Conjecture(f"[{H}] mix√ lower {target}|{den},{z} ({t})", premises=prem, relation=rel),
                            _strength_ge(df, rel.rhs, prem),
                        )
                        for t, rel in lo_variants
                    ]
                    scored.sort(key=lambda t: t[2], reverse=True)
                    conjs.append(scored[0][1])

                    # Upper ≤ : base, floor_whole, floor_split
                    hi_variants = [
                        ("base",
                         le(y, base_hi, f"{target} ≤ ½({cmax_x}·{den}+{s_cmax}·√{z})")),
                        ("floor_w",
                         le(y, with_floor(base_hi), f"{target} ≤ floor(½({cmax_x}·{den}+{s_cmax}·√{z}))")),
                        ("floor_s",
                         le(
                             y,
                             with_floor(0.5 * pl.lit(cmax_x) * x) +
                             with_floor(0.5 * pl.lit(s_cmax) * col(z).sqrt()),
                             f"{target} ≤ floor(½{cmax_x}·{den})+floor(½{s_cmax}·√{z})",
                         )),
                    ]
                    scored = [
                        (
                            t,
                            Conjecture(f"[{H}] mix√ upper {target}|{den},{z} ({t})", premises=prem, relation=rel),
                            _strength_le(df, rel.rhs, prem),
                        )
                        for t, rel in hi_variants
                    ]
                    scored.sort(key=lambda t: t[2], reverse=True)
                    conjs.append(scored[0][1])

                # square mix
                used_sq = int(row.get(f"used_sq__{jid}", 0) or 0)
                if used_sq > 0:
                    q_cmin = to_const(row[f"mn_sq__{jid}"])
                    q_cmax = to_const(row[f"mx_sq__{jid}"])
                    z2 = col(z) * col(z)

                    base_lo = 0.5 * (pl.lit(cmin_x) * x + pl.lit(q_cmin) * z2)
                    base_hi = 0.5 * (pl.lit(cmax_x) * x + pl.lit(q_cmax) * z2)

                    lo_variants = [
                        ("base",
                         ge(y, base_lo, f"{target} ≥ ½({cmin_x}·{den}+{q_cmin}·{z}^2)")),
                        ("ceil",
                         ge(y, with_ceil(base_lo), f"{target} ≥ ceil(½({cmin_x}·{den}+{q_cmin}·{z}^2))")),
                    ]
                    scored = [
                        (
                            t,
                            Conjecture(f"[{H}] mix² lower {target}|{den},{z} ({t})", premises=prem, relation=rel),
                            _strength_ge(df, rel.rhs, prem),
                        )
                        for t, rel in lo_variants
                    ]
                    scored.sort(key=lambda t: t[2], reverse=True)
                    conjs.append(scored[0][1])

                    hi_variants = [
                        ("base",
                         le(y, base_hi, f"{target} ≤ ½({cmax_x}·{den}+{q_cmax}·{z}^2)")),
                        ("floor",
                         le(y, with_floor(base_hi), f"{target} ≤ floor(½({cmax_x}·{den}+{q_cmax}·{z}^2))")),
                    ]
                    scored = [
                        (
                            t,
                            Conjecture(f"[{H}] mix² upper {target}|{den},{z} ({t})", premises=prem, relation=rel),
                            _strength_le(df, rel.rhs, prem),
                        )
                        for t, rel in hi_variants
                    ]
                    scored.sort(key=lambda t: t[2], reverse=True)
                    conjs.append(scored[0][1])

    return conjs


def gen_product_bounds_caro(
    df: pl.DataFrame,
    *,
    hypotheses: Optional[List[str]] = None,
    include_global: bool = True,
    nontrivial_only: bool = True,
) -> List[Conjecture]:
    """
    Search product inequalities W*x ≤ y*z under each hypothesis, with
    Caro-style non-triviality pruning:
      skip if (W≤y & x≤z) OR (W≤z & x≤y) holds universally on the slice.
    Single batched pass per H.
    """
    bool_cols, num_cols = list_boolean_and_numeric_cols(df)
    H_list = (hypotheses or [])[:]
    if include_global: H_list = ["ALL"] + H_list

    conjs: List[Conjecture] = []
    for H in H_list:
        prem = _premises_from_name(df, H)
        Hmask = pl.lit(True) if H == "ALL" else pl.col(H)

        # Build a batched plan that, for each quadruple, checks:
        # - universal L<=R (for Wx≤yz)
        # - triviality via (W≤y & x≤z) OR (W≤z & x≤y)
        exprs, jobs = [], []
        jid = 0
        for W, x in combinations(num_cols, 2):
            for y, z in combinations(num_cols, 2):
                if {W, x} == {y, z}:   # identical multiset → tautology; skip
                    continue
                L = col(W)*col(x); R = col(y)*col(z)
                # universal L<=R on slice: d = R-L ≥ 0  and equality rows
                d = R - L
                valid = Hmask & pl.all_horizontal([col(W).is_finite(), col(x).is_finite(), col(y).is_finite(), col(z).is_finite(), L.is_finite(), R.is_finite()])
                exprs += [
                    (pl.when(valid).then(d >= 0).otherwise(True)).all().alias(f"ok__{jid}"),
                    (pl.when(valid).then(d == 0).otherwise(False)).sum().alias(f"touch__{jid}"),
                ]
                if nontrivial_only:
                    # check (W≤y & x≤z) OR (W≤z & x≤y) universally on the same slice
                    c1 = (col(W) <= col(y)) & (col(x) <= col(z))
                    c2 = (col(W) <= col(z)) & (col(x) <= col(y))
                    exprs.append((pl.when(valid).then(c1 | c2).otherwise(True)).all().alias(f"triv__{jid}"))
                jobs.append((jid, W, x, y, z))
                jid += 1

        if not jobs:
            continue

        row = df.select(exprs).row(0, named=True)
        for jid, W, x, y, z in jobs:
            ok = bool(row[f"ok__{jid}"])
            if not ok:
                continue
            if nontrivial_only and bool(row[f"triv__{jid}"]):
                continue
            touch = int(row[f"touch__{jid}"] or 0)
            rel = le(col(W)*col(x), col(y)*col(z), f"{W}·{x} ≤ {y}·{z}")
            conjs.append(Conjecture(f"[{H}] Caro {W}*{x}≤{y}*{z} (touch={touch})", premises=prem, relation=rel))

    return conjs

# ------------------------------
# Demo: run on the txgraffiti example dataset
# ------------------------------
from txgraffiti.example_data import graph_data as pdf  # pandas DF
dfp = pl.from_pandas(pdf)

# Choose hypotheses (boolean columns present in data)
bool_cols, num_cols = list_boolean_and_numeric_cols(dfp)
print("Booleans:", bool_cols)
print("Numerics:", num_cols)

TARGET = "independence_number"

# 1) Linear bounds from ratios (+ optional floor/ceil variants)
lin_conjs = gen_linear_bounds(
    dfp,
    target=TARGET,
    hypotheses=bool_cols,   # per-boolean slices
    include_global=True,
    floor_ceil=True,
)

# 2) 2-feature mixes (sqrt & square) using the 1/2 distribution trick
mix_conjs = gen_mix_bounds(
    dfp,
    target=TARGET,
    hypotheses=bool_cols,
    include_global=True,
)

# 3) Caro-style product inequalities with non-triviality pruning
caro_conjs = gen_product_bounds_caro(
    dfp,
    hypotheses=bool_cols,
    include_global=True,
    nontrivial_only=True,
)

# Evaluate all generated conjectures in one fused pass
all_conjs = lin_conjs + mix_conjs + caro_conjs
res = dfp.conj.eval(all_conjs)

# Rank like your old _touch_count: sort by holding then by touch desc
res_top = res.filter(pl.col("holds") == True).sort(pl.col("touch").fill_null(0), descending=True)

print("\n===== Top holding conjectures (by touch) =====")
print(res_top.head(40))

# If you want to see a few non-holding (near-misses) for debug:
print("\n===== A few non-holding (for inspection) =====")
print(res.filter(pl.col("holds") == False).head(10))
