.. _key_features/generators/linear_programming:

Linear Programming Derived Inequalities
=======================================

.. currentmodule:: txgraffiti.generators.optimization.linear_programming

Overview
--------

The **`linear_programming`** generator uses a **sum-of-slacks linear program** to fit
two bounding hyperplanes—**upper** and **lower**—relating a numeric **target**
`Property` to a linear combination of one or more **feature** `Property` objects,
restricted to a subpopulation defined by a `Predicate` hypothesis.

Requirements
------------

- An LP solver installed on your system PATH:
  - **CBC** (`cbc`)
  - or **GLPK** (`glpsol`)
- Python packages:
  - `pulp`
  - `numpy`
  - `pandas`

Algorithm
---------

1. **Restriction**

   Select rows where the hypothesis holds:

   .. math::

      D_H = \{x \mid H(x) = \mathrm{True}\}

2. **Matrix Assembly**

   Construct the feature matrix \(X \in \mathbb{R}^{n \times k}\) and target vector
   \(y \in \mathbb{R}^n\).

3. **Sum-of-Slacks LPs**

   Solve two linear programs (LPs) minimizing the total slack:

   - **Upper Bound**:

     .. math::

        \min \sum_{i=1}^n (a \cdot x_i + b - y_i)
        \quad \text{subject to} \quad a \cdot x_i + b \ge y_i

   - **Lower Bound**:

     .. math::

        \min \sum_{i=1}^n (y_i - a \cdot x_i - b)
        \quad \text{subject to} \quad y_i \ge a \cdot x_i + b

4. **Reconstruct RHS**

   Express the optimal linear inequality as:

   .. math::

      \text{target} \ge b + \sum_{j=1}^k a_j f_j
      \quad \text{or} \quad
      \text{target} \le b + \sum_{j=1}^k a_j f_j

   where each \(f_j\) is a feature `Property`.

5. **Emit Conjectures**

   Yield two `Conjecture` objects:

   - \( H(x) \Rightarrow T(x) \ge \text{RHS}(x) \)
   - \( H(x) \Rightarrow T(x) \le \text{RHS}(x) \)

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

Expected Output
---------------

.. code-block:: text

   (connected) → (alpha >= ((-1/2 * beta) + 5/2))
   (connected) → (alpha <= ((-1 * beta) + 4))

See Also
--------

- :ref:`key_features/generators/ratios` — simple ratio-based bounds
- :ref:`key_features/generators/convex_hull` — geometry-derived linear bounds
