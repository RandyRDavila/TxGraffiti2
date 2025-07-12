Linear Programming Derived Inequalities
=======================================

.. currentmodule:: txgraffiti.generators.optimization.linear_programming

Overview
--------

The **`linear_programming`** generator uses a sum-of-slacks linear program to fit
two bounding hyperplanes (upper and lower) relating a numeric **target**
Property to a linear combination of one or more **feature** Properties,
restricted to the subpopulation defined by a **hypothesis** Predicate.

Requirements
------------

- An LP solver installed on your PATH, either **CBC** (`cbc`) or **GLPK**
  (`glpsol`).
- Python packages: **pulp**, **numpy**, **pandas**.

Algorithm
---------

1. **Restriction**
   Select only the rows where `hypothesis(df) == True`.

2. **Matrix assembly**
   Build feature matrix `X` (shape `(n_samples, n_features)`) and target
   vector `y`.

3. **Sum-of-slacks LP**
   Solve two linear programs (one for an upper bound, one for a lower bound)
   that minimize the total slack:

   - *Upper bound*:

    ..math
       \min \sum_{i=1}^n (a\cdot x_i + b - y_i)
       \quad\text{s.t.}\;a x_i + b - y_i \ge 0.


   - *Lower bound*:

    ..math
        \min \sum_{i=1}^n (y_i - (a\cdot x_i + b))
        \quad\text{s.t.}\;y_i - (a x_i + b) \ge 0.


4. **Reconstruct RHS**
   Convert the optimal coefficients \((a, b)\) into a `Property` expression
   \(\,b + \sum_j a_j \, f_j\), where each \(f_j\) is the \(j\)th feature.

5. **Emit conjectures**
   Yield two `Conjecture` objects per run:
   - `hypothesis → target ≤ (a·features + b)`
   - `hypothesis → target ≥ (a·features + b)`


Example
-------

Here’s a minimal example on a toy DataFrame:

.. code-block:: python

    import pandas as pd
    from txgraffiti.logic import Property, Predicate
    from txgraffiti.generators.optimization import linear_programming

    # Sample data
    df = pd.DataFrame({
        'alpha':     [1, 2, 3, 4],
        'beta':      [3, 1, 1, 2],
        'connected': [True, True, True, True],
    })

    # Lift into TxGraffiti objects
    A = Property('alpha', lambda df: df['alpha'])
    B = Property('beta',  lambda df: df['beta'])
    H = Predicate('connected', lambda df: df['connected'])

    # Generate linear bounds on alpha in terms of beta under H
    for conj in linear_programming(
            df,
            features=[B],
            target=A,
            hypothesis=H
    ):
        print(conj)

See Also
--------

- :ref:`key_features/generators/ratios` — simple ratio‐based bounds
