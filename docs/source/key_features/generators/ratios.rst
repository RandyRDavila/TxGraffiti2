.. _key_features/generators/ratios:

Ratio Derived Inequalities
==========================

The **`ratios`** generator produces simple linear bounds of the form

.. math::

   \text{if }H(x)\text{ then }
   \begin{cases}
     T(x) \;\ge\; c\,F(x),\\
     T(x) \;\le\; C\,F(x),
   \end{cases}

where:

- **H** is a `Predicate` (the hypothesis)
- **T** is the **target** `Property`
- **F** is a single **feature** `Property`
- **c** is the lower bound constant:

.. math::
    c = \min_{H(x)=\mathrm{True}}\frac{T(x)}{F(x)}

- **C** is the lower bound constant:

.. math::
    C = \max_{H(x)=\mathrm{True}}\frac{T(x)}{F(x)}

It yields exactly two `Conjecture` objects per feature: one lower‐bound and one upper‐bound.

Key Steps
---------

1. **Restrict** to the rows where the hypothesis holds:
   \(\{x \mid H(x)=\mathrm{True}\}\).

2. **Compute ratios** are:

.. math::
    r_i = \frac{T(x_i)}{F(x_i)}

 .. math::
    c = \min_i r_i

.. math::
    C = \max_i r_i.

3. **Emit** two conjectures per feature:

.. math::

    H(x)\;\Longrightarrow\;T(x) \;\ge\; c\,F(x)

.. math::

    H(x)\;\Longrightarrow\;T(x) \;\le\; C\,F(x)

Usage Example
-------------

.. code-block:: python

   import pandas as pd
   from txgraffiti.logic.conjecture_logic import Property, Predicate
   from txgraffiti.generators import ratios

   # Sample DataFrame
   df = pd.DataFrame({
       'alpha':    [1, 2, 3],
       'beta':     [3, 1, 1],
       'connected':[True, True, True],
       'tree':     [False, False, True],
   })

   # 1) Lift columns to Property
   target   = Property('alpha',   lambda df: df['alpha'])
   features = [ Property('beta',  lambda df: df['beta']) ]

   # 2) Define hypothesis
   hypothesis = Predicate('connected', lambda df: df['connected'])

   # 3) Run the ratios generator
   conjs = list(ratios(
       df,
       features   = features,
       target     = target,
       hypothesis = hypothesis
   ))

   print(conjs)

Expected Output
---------------

.. code-block:: text

   [
     <Conj (connected) → (alpha >= (1/3 * beta))>,
     <Conj (connected) → (alpha <= (3 * beta))>
   ]

Explanation
-----------

- For each row with `connected == True`, compute `alpha/beta`:
  \([1/3, 2/1, 3/1] = [0.333…, 2.0, 3.0]\).

- The **minimum** ratio \(c = 1/3\) yields
  \((\alpha \ge \tfrac13\,\beta)\).

- The **maximum** ratio \(C = 3\) yields
  \((\alpha \le 3\,\beta)\).

These two `Conjecture` objects assert that, among connected graphs, the `alpha`
value always lies between \(0.333\,\beta\) and \(3\,\beta\).

Integration
-----------

You can combine `ratios` with:

- **Multiple features**: supply a list of `Property` objects
- **Heuristics**: filter by tightness, accuracy, or custom logic
- **Post-processing**: dedupe, sort by touch-count, or strengthen to equalities

For example, within a `ConjecturePlayground`:

.. code-block:: python

   pg = ConjecturePlayground(df, object_symbol='G')
   pg.discover(
       methods   = [ratios],
       features  = ['beta', 'gamma', ...],
       target    = 'alpha',
       hypothesis= 'connected',
       heuristics= [morgan, dalmatian],
       post_processors = [remove_duplicates, sort_by_touch_count],
   )

The `ratios` generator will contribute linear bounds to your automated‐discovery workflow.
