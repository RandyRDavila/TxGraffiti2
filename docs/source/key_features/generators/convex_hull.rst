.. _key_features/generators/convex_hull:

Convex Hull Derived Inequalities
================================

The **`convex_hull`** generator creates linear inequality conjectures by computing the **convex hull** of feature-target vectors, restricted to rows satisfying a given hypothesis. Each **facet** of the convex hull becomes a symbolic inequality.

.. math::

   \text{If } H(x) \text{ then } T(x) \;\le\; a_1 F_1(x) + \cdots + a_k F_k(x) + b
   \quad \text{or} \quad
   T(x) \;\ge\; \cdots

where:

- \(H(x)\) is a `Predicate` (logical condition),
- \(T(x)\) is the **target** `Property`,
- \(F_1, \dots, F_k\) are the **features** (`Property`),
- The coefficients \(a_i\) and intercept \(b\) are derived from the facet normal.

Motivation
----------

The convex hull captures the **tightest linear outer bounds** on a set of vectors. By interpreting its facets, we extract strong candidate inequalities of the form:

.. math::

   \text{target} \;\le\; \text{linear combination of features}
   \quad\text{or}\quad
   \text{target} \;\ge\; \text{linear combination of features}

This allows discovery of subtle linear relationships involving multiple features at once.

How It Works
------------

1. **Restrict** the dataset to rows where the hypothesis holds:

   .. math::

      D_H = \{x : H(x) = \text{True}\}

2. **Embed** each row as a vector in \(\mathbb{R}^{k+1}\):

   .. math::

      v_i = [F_1(x_i), \dots, F_k(x_i), T(x_i)]

3. **Compute the convex hull** of these vectors using SciPy’s `ConvexHull`.

4. **Translate each facet** of the hull (an affine hyperplane):

   .. math::

      a_1 x_1 + \cdots + a_k x_k + a_y y + b_0 = 0

   into an inequality of the form:

   .. math::

      y \le \sum (-a_i/a_y) x_i - b_0/a_y

   or

   .. math::

      y \ge \cdots

5. **Emit a conjecture** for each such facet, using symbolic expressions.

Usage Example
-------------

.. code-block:: python

   import pandas as pd
   from txgraffiti.logic.conjecture_logic import Property, Predicate
   from txgraffiti.generators import convex_hull

   # Sample dataset
   df = pd.DataFrame({
       'alpha':    [1, 2, 3],
       'beta':     [3, 1, 1],
       'connected':[True, True, True],
       'tree':     [False, False, True],
   })

   # Lift target and features
   alpha  = Property('alpha', lambda df: df['alpha'])
   beta   = Property('beta',  lambda df: df['beta'])
   conn   = Predicate('connected', lambda df: df['connected'])

   # Run convex hull generator
   conjs = list(convex_hull(
       df,
       features   = [beta],
       target     = alpha,
       hypothesis = conn
   ))

   for c in conjs:
       print(c)

Expected Output
---------------

.. code-block:: text

   (connected) → (alpha >= ((-1/2 * beta) + 5/2))
   (connected) → (alpha <= ((-1 * beta) + 4))

Explanation
-----------

- Convex hull facets yield tight outer bounds for `alpha` in terms of `beta`.
- Rational coefficients are extracted from the hull geometry and rounded via `Fraction.limit_denominator()`.

These bounds reflect extremal structure of the data under the given hypothesis.

Advanced Options
----------------

- **drop_side_facets=True** (default): omit inequalities where the `target` does not appear meaningfully.
- **tol=1e-8**: threshold for ignoring near-zero coefficients.
- **Multiple features**: pass a list of `Property` objects for multivariate bounds.

Integration with Playground
----------------------------

You can use `convex_hull` inside `ConjecturePlayground` just like any other generator:

.. code-block:: python

   from txgraffiti.playground import ConjecturePlayground

   pg = ConjecturePlayground(df, object_symbol='G')
   pg.discover(
       methods   = [convex_hull],
       features  = ['beta', 'gamma'],
       target    = 'alpha',
       hypothesis= 'connected',
   )
