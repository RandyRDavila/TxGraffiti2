.. _key_features/generators/ratios:

Ratio-Based Inequalities
=========================

The **`ratios`** generator produces simple linear bounds of the form:

.. math::

   \text{If } H(x) \text{ then} \quad
   T(x) \ge c \cdot F(x)
   \quad \text{and} \quad
   T(x) \le C \cdot F(x)

where:

- \(H(x)\) is a `Predicate` (hypothesis),
- \(T(x)\) is the **target** `Property`,
- \(F(x)\) is a **feature** `Property`,
- \(c\) is a **lower bound constant**:

  .. math::

     c = \min_{x \in H^{-1}(\text{True})} \frac{T(x)}{F(x)}

- \(C\) is an **upper bound constant**:

  .. math::

     C = \max_{x \in H^{-1}(\text{True})} \frac{T(x)}{F(x)}

The generator emits exactly **two `Conjecture` objects per feature**: one lower-bound and one upper-bound inequality.

Workflow
--------

1. **Restrict** the dataset to rows where the hypothesis holds:

   .. math::

      D_H = \{x : H(x) = \text{True}\}

2. **Compute ratios** for each object in \(D_H\):

   .. math::

      r_i = \frac{T(x_i)}{F(x_i)}

3. **Extract constants**:

   .. math::

      c = \min_i r_i,
      \quad
      C = \max_i r_i

4. **Emit conjectures**:

   .. math::

      H(x) \Rightarrow T(x) \ge c \cdot F(x),
      \quad
      H(x) \Rightarrow T(x) \le C \cdot F(x)

Usage Example
-------------

.. code-block:: python

   import pandas as pd
   from txgraffiti.logic.conjecture_logic import Property, Predicate
   from txgraffiti.generators import ratios

   # Sample dataset
   df = pd.DataFrame({
       'alpha':    [1, 2, 3],
       'beta':     [3, 1, 1],
       'connected':[True, True, True],
       'tree':     [False, False, True],
   })

   # Define target and features
   target   = Property('alpha', lambda df: df['alpha'])
   features = [Property('beta', lambda df: df['beta'])]

   # Define hypothesis
   hyp = Predicate('connected', lambda df: df['connected'])

   # Generate conjectures
   conjs = list(ratios(df, features=features, target=target, hypothesis=hyp))

   for c in conjs:
       print(c)

Expected Output
---------------

.. code-block:: text

   (connected) → (alpha >= (1/3 * beta))
   (connected) → (alpha <= (3 * beta))

Explanation
-----------

For each row with `connected == True`, the `ratios` generator computes:

.. math::

   \left[\frac{1}{3}, \frac{2}{1}, \frac{3}{1}\right] = [0.333…, 2.0, 3.0]

- **Minimum** ratio is \(c = \tfrac{1}{3}\), yielding: \(\alpha \ge \tfrac{1}{3} \beta\)
- **Maximum** ratio is \(C = 3\), yielding: \(\alpha \le 3 \beta\)

These two inequalities form conjectures about how `alpha` is bounded by `beta` for connected graphs.

Integration with Playground
----------------------------

Use `ratios` inside the `ConjecturePlayground` to automate discovery:

.. code-block:: python

   from txgraffiti.playground import ConjecturePlayground
   from txgraffiti.heuristics import morgan_accept, dalmatian_accept
   from txgraffiti.processing import remove_duplicates, sort_by_touch_count

   pg = ConjecturePlayground(df, object_symbol='G')
   pg.discover(
       methods         = [ratios],
       features        = ['beta', 'gamma'],
       target          = 'alpha',
       hypothesis      = 'connected',
       heuristics      = [morgan_accept, dalmatian_accept],
       post_processors = [remove_duplicates, sort_by_touch_count],
   )

The `ratios` generator contributes clean, data-driven bounds to your symbolic conjecture workflow.
