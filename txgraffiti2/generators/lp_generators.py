import pulp
import numpy as np
import pandas as pd
from typing import List
from fractions import Fraction
import shutil
import itertools

from txgraffiti2.logic.conjecture_logic import (
    Property,
    Predicate,
    Inequality,
    Conjecture,
)

__all__ = [
    "generate_lp_conjecture",
]

def get_available_solver():
    """
    Automatically selects a compatible LP solver.
    Prefers system-installed CBC (via COIN_CMD), falls back to GLPK.
    """
    # Try system-installed CBC
    cbc_path = shutil.which("cbc")
    if cbc_path:
        return pulp.COIN_CMD(path=cbc_path, msg=False)

    # Try GLPK as fallback
    glpk_path = shutil.which("glpsol")
    if glpk_path:
        pulp.LpSolverDefault.msg = 0
        return pulp.GLPK_CMD(path=glpk_path, msg=False)

    raise RuntimeError(
        "No compatible LP solver found.\n"
        "Install CBC (`brew install cbc` or `apt-get install coinor-cbc`) "
        "or GLPK (`brew install glpk`)."
    )

def generate_lp_conjecture(
    df: pd.DataFrame,
    features: List[Property],
    target:   Property,
    hypothesis: Predicate,
    tol: float = 1e-8
) -> Conjecture:
    """
    Solve an LP to minimize the worst‐case residual of
      a·features + b <= target
    over all rows satisfying hypothesis(df), but then
    display only nonzero terms in the name.
    """
    # 1) restrict
    mask  = hypothesis(df)
    subdf = df[mask]
    if subdf.empty:
        raise ValueError(f"No rows satisfy {hypothesis.name!r}")

    # 2) build X, y
    X = np.column_stack([p(subdf).values for p in features])
    y = subdf[target.name].values
    n, k = X.shape

    # 3) LP setup
    prob = pulp.LpProblem("chebyshev_bound", pulp.LpMinimize)
    a_vars = [pulp.LpVariable(f"a_{i}", lowBound=None) for i in range(k)]
    b_var  = pulp.LpVariable("b", lowBound=None)
    M_var  = pulp.LpVariable("M", lowBound=0)
    prob += M_var, "Minimize_max_residual"

    # constraints
    for i in range(n):
        xi, yi = X[i], y[i]
        lhs = pulp.lpSum(a_vars[j] * xi[j] for j in range(k)) + b_var
        prob += lhs <= yi,         f"ub_row_{i}"
        prob += (yi - lhs) <= M_var, f"cheby_{i}"

    # 4) solve
    solver = get_available_solver()
    prob.solve(solver)

    # 5) extract
    a_sol = [v.value() for v in a_vars]
    b_sol = b_var.value()

    # 6) build cleaned name
    terms = []
    for coeff, prop in zip(a_sol, features):
        if abs(coeff) < tol:
            continue
        frac = Fraction(str(coeff)).limit_denominator()
        if frac == 1:
            terms.append(f"{prop.name}")
        else:
            terms.append(f"{frac}*{prop.name}")

    # constant term
    if abs(b_sol) >= tol:
        b_frac = Fraction(str(b_sol)).limit_denominator()
        terms.append(str(b_frac))

    # if everything cancelled, show 0
    if not terms:
        rhs_name = "(0)"
    else:
        rhs_name = "(" + " + ".join(terms) + ")"

    # 7) build rhs Property
    def rhs_func(df, coeffs=a_sol, intercept=b_sol, feats=features):
        s = intercept
        for c, p in zip(coeffs, feats):
            s = s + c * p(df)
        return s

    rhs_prop = Property(rhs_name, rhs_func)

    # 8) wrap as Conjecture
    ineq = Inequality(rhs_prop, "<=", target)
    return Conjecture(hypothesis, ineq)

def generate_lp_conjectures(
    df: pd.DataFrame,
    base_props: List[Property],
    target:     Property,
    hypothesis: Predicate,
    max_features: int = None
) -> List[Conjecture]:
    """
    For r = 1..max_features (or len(base_props) if None),
    try generate_lp_conjecture(df, combo, target, hypothesis)
    for every combination of base_props of size r.
    Return all unique Conjectures sorted by touch_count(df) descending.
    """
    if max_features is None:
        max_features = len(base_props)

    all_conjs = []
    for r in range(1, max_features + 1):
        for combo in itertools.combinations(base_props, r):
            try:
                conj = generate_lp_conjecture(
                    df=df,
                    features=list(combo),
                    target=target,
                    hypothesis=hypothesis
                )
            except Exception:
                # e.g. no rows satisfy hyp, or LP infeasible
                continue

            # guard: only keep if it truly holds everywhere
            if conj.is_true(df):
                all_conjs.append(conj)

    # dedupe
    unique = set(all_conjs)

    # sort by descending touch‐count on the full df
    sorted_conjs = sorted(
        unique,
        key=lambda c: c.conclusion.touch_count(df),
        reverse=True
    )
    return sorted_conjs
