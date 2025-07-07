import numpy as np
import pandas as pd
import pulp
import shutil

from fractions import Fraction
from typing import List, Tuple
from txgraffiti.generators.registry import register_gen
from txgraffiti.logic import Constant, Property, Predicate, Conjecture, Inequality


__all__ = [
    'linear_programming',
]

def get_available_solver():
    cbc = shutil.which("cbc")
    if cbc:
        return pulp.COIN_CMD(path=cbc, msg=False)
    glpk = shutil.which("glpsol")
    if glpk:
        pulp.LpSolverDefault.msg = 0
        return pulp.GLPK_CMD(path=glpk, msg=False)
    raise RuntimeError("No LP solver found (install CBC or GLPK)")

def _solve_sum_slack_lp(
    X: np.ndarray,
    y: np.ndarray,
    sense: str = "upper"
) -> Tuple[np.ndarray, float]:
    """
    Solve LP:
      minimize sum_i s_i
      s.t.  for all i:
         if sense=="upper":   a·x_i + b - y_i   = s_i ≥ 0
         if sense=="lower":   y_i - (a·x_i + b)   = s_i ≥ 0

      Returns (a_sol, b_sol).
    """
    n, k = X.shape
    prob = pulp.LpProblem("sum_slack", pulp.LpMinimize)

    # decision vars
    a_vars = [pulp.LpVariable(f"a_{j}", lowBound=None) for j in range(k)]
    b_var  = pulp.LpVariable("b", lowBound=None)
    s_vars = [pulp.LpVariable(f"s_{i}", lowBound=0) for i in range(n)]

    # objective: minimize total slack
    prob += pulp.lpSum(s_vars)

    # constraints
    for i in range(n):
        xi = X[i]
        yi = y[i]
        lhs = pulp.lpSum(a_vars[j]*xi[j] for j in range(k)) + b_var
        if sense == "upper":
            # slack = (a·x + b) - y
            prob += lhs - yi == s_vars[i]
        else:
            # sense=="lower": slack = y - (a·x + b)
            prob += yi - lhs == s_vars[i]

    # solve
    solver = get_available_solver()
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP did not solve optimally: {pulp.LpStatus[status]}")

    a_sol = np.array([v.value() for v in a_vars], dtype=float)
    b_sol = float(b_var.value())
    return a_sol, b_sol

@register_gen
def linear_programming(
    df: pd.DataFrame,
    *,
    features:   List[Property],
    target:     Property,
    hypothesis: Predicate,
    tol:        float = 1e-8
) -> Conjecture:
    """
    Solve a sum‐of‐slacks LP to find coefficients (a,b) such that
      hypothesis → (y ≤ a^T x + b)
      hypothesis → (y ≥ a^T x + b)

    Returns a generator of Conjectures.
    """
    # 1) restrict to hypothesis‐true subset
    mask = hypothesis(df)
    sub = df[mask]
    if sub.empty:
        raise ValueError(f"No rows satisfy {hypothesis.name!r}")

    # 2) build numeric arrays
    X = np.column_stack([p(sub).values for p in features])
    y = sub[target.name].values

    for sense in ["upper", "lower"]:
    # 3) solve for (a_sol, b_sol)
        a_sol, b_sol = _solve_sum_slack_lp(X, y, sense=sense)

        # 4) reconstruct rhs Property:  a^T x + b
        rhs: Property = Constant(Fraction(b_sol).limit_denominator())
        for coeff, prop in zip(a_sol, features):
            if abs(coeff) < tol:
                continue
            rhs = rhs + (prop * Fraction(float(coeff)).limit_denominator())

        # 5) form the correct inequality
        if sense == "upper":
            ineq = Inequality(target, "<=", rhs)   # rhs ≥ y  ↔  y ≤ rhs
        else:
            ineq = Inequality(target, ">=", rhs)   # rhs ≤ y  ↔  y ≥ rhs
        yield Conjecture(hypothesis, ineq)
