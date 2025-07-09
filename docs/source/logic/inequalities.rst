Inequalities
============

An **Inequality** is a specialized `Predicate` representing a comparison between two
`Property` expressions.  It allows you to symbolically express and then evaluate
statements of the form

.. math::

   L(x) \;\le\; R(x)
   \quad\text{or}\quad
   L(x) \;\ge\; R(x)

where both \(L\) and \(R\) are built from your DataFrame columns (via `Property`).

Key Features
------------

- **Operator‐based construction**
  Use the standard comparison operators between `Property` objects (or lifted scalars):

  .. code-block:: python

     from txgraffiti.logic.conjecture_logic import Property

     P = Property('x', lambda df: df['x'])
     Q = Property('y', lambda df: df['y'])

     # builds an Inequality for y ≤ 2·x + 3
     ineq1 = Q <= (2 * P + 3)

     # builds an Inequality for x ≥ y − 5
     ineq2 = P >= (Q - 5)

- **Slack**
  For each row, the **slack** is how far the inequality is from being tight (zero):

  .. math::

     \mathrm{slack}(i) =
     \begin{cases}
       R(x_i) - L(x_i), & \text{if }L \le R,\\
       L(x_i) - R(x_i), & \text{if }L \ge R.
     \end{cases}

  .. code-block:: python

     slack = ineq1.slack(df)
     # if ineq1 is Q ≤ 2P+3, slack[i] = (2P[i]+3) - Q[i]

- **Touch count**
  The number of rows on which the inequality holds **with equality** (\(\mathrm{slack}=0\)):

  .. code-block:: python

     count = ineq1.touch_count(df)
     print(f"Tight on {count} rows")

- **Seamless integration with `Predicate` and `Conjecture`**
  You can `&` or `|` inequalities, and pass them as conclusions in `Conjecture`:

  .. code-block:: python

     from txgraffiti.logic.conjecture_logic import Conjecture

     hyp = (P >= 1)
     conj = Conjecture(hyp, ineq1)
     print("Holds universally?", conj.is_true(df))
     print("Accuracy:", conj.accuracy(df))

Motivation
----------

When exploring data you frequently want to find or test linear bounds such as

.. math::

   y \;\le\; a\,x \;+\; b
   \quad\text{or}\quad
   y \;\ge\; c\,x \;+\; d

for which you need:

1. **Symbolic names** for pretty‐printing and export,
2. **Numeric evaluation** on each row,
3. **Metrics** for tightness and support (how many points lie exactly on the bound).

`Inequality` provides all three in one object.

Illustrative Example
--------------------

Here’s a complete example showing math and code:

.. math::

   \text{Suppose }x=[1,2,3,4,5],\;y=[2,5,8,11,14].
   \text{We hypothesize }y \le 3x + 1.

.. code-block:: python

   import pandas as pd
   from txgraffiti.logic.conjecture_logic import (
       Property, Inequality, Conjecture
   )

   # 1) Build the DataFrame
   df = pd.DataFrame({
       'x': [1,2,3,4,5],
       'y': [2,5,8,11,14]
   })

   # 2) Lift columns to Property
   X = Property('x', lambda df: df['x'])
   Y = Property('y', lambda df: df['y'])

   # 3) Construct the inequality: y ≤ 3·x + 1
   bound = Inequality(lhs=X, op="<=", rhs=(3 * X + 1))
   print("Inequality name:", bound.name)
   # → "x <= (3 * x + 1)"   (names mirror construction)

   # 4) Compute slack and touch count
   slack = bound.slack(df)
   touched = bound.touch_count(df)
   print("Slack per row:", slack.tolist())
   print(f"Tight on {touched}/{len(df)} rows")

   # 5) Wrap in a conjecture: if x ≥ 2 then y ≤ 3x+1
   conjecture = Conjecture(hypothesis=(X >= 2), conclusion=bound)
   print("Conjecture holds?", conjecture.is_true(df))
   print("Accuracy:", conjecture.accuracy(df))

Output:

.. code-block:: text

   Inequality name: x <= (3 * x + 1)
   Slack per row: [2, 0, 0, 0, 0]
   Tight on 4/5 rows
   Conjecture holds? True
   Accuracy: 1.00

Now you have a symbolic, evaluable bound that you can feed into any of the TxGraffiti
generators, heuristics, or export pipelines.
