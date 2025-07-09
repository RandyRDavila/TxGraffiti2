"""
Linear programming-based conjecture generator.

This module defines generators that formulate inequalities between numeric
graph invariants using sum-of-slacks linear programs. Given a hypothesis
and a list of feature properties, the LP computes the best linear upper and
lower bounds on a target invariant, yielding inequalities suitable for
conjecture generation.

Requires an LP solver (CBC or GLPK) available on the system path.
"""

import numpy as np
import pandas as pd
import pulp
import shutil


from fractions import Fraction
from typing import List, Tuple, Iterator
from txgraffiti.generators.registry import register_gen
from txgraffiti.logic import *


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
    Solve a sum-of-slacks linear program to fit a hyperplane bounding y.

    The objective is to minimize the total slack in approximating `y` using
    a linear combination of `X` plus a bias term, either as an upper or
    lower bound:

    - If `sense == "upper"`: minimize ∑ (a·x_i + b - y_i), subject to nonnegative slack
    - If `sense == "lower"`: minimize ∑ (y_i - a·x_i - b), subject to nonnegative slack

    Parameters
    ----------
    X : np.ndarray
        A 2D array of shape (n_samples, n_features) representing the feature values.

    y : np.ndarray
        A 1D array of target values, same length as the number of rows in `X`.

    sense : {'upper', 'lower'}, optional
        Whether to fit an upper or lower bounding hyperplane. Default is `"upper"`.

    Returns
    -------
    Tuple[np.ndarray, float]
        Tuple `(a, b)` where `a` is the weight vector and `b` is the intercept term.

    Raises
    ------
    RuntimeError
        If the LP does not solve optimally or no solver is found.
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
) -> Iterator[Conjecture]:
    """
    Generate linear inequality conjectures via sum-of-slacks linear programming.

    For a set of objects (rows) satisfying the `hypothesis`, this function fits
    two bounding hyperplanes — one upper and one lower — that linearly relate
    the `target` invariant to a weighted combination of `features`. It does so by
    minimizing the sum of slack variables in a linear program, producing:

    - `hypothesis → target ≤ a₁·f₁ + ... + a_k·f_k + b`
    - `hypothesis → target ≥ a₁·f₁ + ... + a_k·f_k + b`

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing invariant values of mathematical objects.

    features : List[Property]
        A list of properties to use as explanatory variables (features).

    target : Property
        The property to predict or bound, appearing on the LHS.

    hypothesis : Predicate
        Logical condition defining the subpopulation over which bounds are computed.

    tol : float, optional
        Tolerance for treating small coefficients as zero (default is `1e-8`).

    Yields
    ------
    Conjecture
        Conjectures expressing linear upper and lower bounds on the target
        under the given hypothesis.

    Raises
    ------
    ValueError
        If no rows satisfy the hypothesis.
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
