Conjecture Objects
==================

A **Conjecture** is the central object in TxGraffiti’s DSL for expressing and
evaluating logical implications over tabular data.  It represents

.. math::

   H(x)\;\Longrightarrow\;C(x)

where:

- **H** (the *hypothesis*) and **C** (the *conclusion*) are both `Predicate` objects
- Evaluation is deferred until you apply the conjecture to a `pandas.DataFrame`

Each `Conjecture` carries not only its symbolic form but also methods to test,
measure, and inspect it on your dataset.

Key Features
------------

- **Construction**
  Create a conjecture by passing two `Predicate` instances, or use the right‐shift
  operator `>>` for syntactic sugar:

  .. code-block:: python

     from txgraffiti.logic.conjecture_logic import Conjecture

     # explicit constructor
     conj1 = Conjecture(hyp_pred, conc_pred)

     # using >> operator
     conj2 = hyp_pred >> conc_pred

- **Row-wise mask**
  Calling the `Conjecture` on a DataFrame yields a Boolean `Series`:

  .. code-block:: python

     mask = conj(df)
     # equivalent to (~hyp_pred(df)) | conc_pred(df)

- **Universal truth check**
  Test if the implication holds on *every* row:

  .. code-block:: python

     holds_all = conj.is_true(df)
     print("Always true?", holds_all)

- **Accuracy**
  Among the rows where **H** is true, compute the fraction where **C** also holds:

  .. code-block:: python

     acc = conj.accuracy(df)
     print(f"Accuracy under hypothesis: {acc:.2%}")

- **Counterexamples**
  Retrieve the subset of rows that violate the implication:

  .. code-block:: python

     bad_rows = conj.counterexamples(df)
     print("Counterexamples:\n", bad_rows)

Motivation
----------

In exploratory data analysis and automated conjecturing, you frequently need to:

1. **Define** a row-wise condition **H** on your data
2. **Define** a conclusion **C** you think follows from **H**
3. **Test** the implication across your entire dataset
4. **Quantify** how often it holds (accuracy) and inspect failures (counterexamples)

A standalone boolean mask or pair of masks is cumbersome: you lose the logical
structure, naming, and integrated metrics.  `Conjecture` packages all of these
concerns into a single, composable object that plays nicely with generators,
heuristics, and export tools.

Illustrative Example
--------------------

Suppose we have two numeric columns \(x\) and \(y\), and we wish to test:

.. math::

   \text{if }x \ge 2\text{ then }y \le 1.2\,x.

1) **Prepare the DataFrame**:

   .. code-block:: python

      import pandas as pd

      df = pd.DataFrame({
          'x': [1, 2, 3, 4, 5],
          'y': [1.1, 1.9, 3.0, 4.2, 4.8],
      })

2) **Lift to `Property`**:

   .. code-block:: python

      from txgraffiti.logic.conjecture_logic import Property

      X = Property('x', lambda df: df['x'])
      Y = Property('y', lambda df: df['y'])

3) **Define hypothesis** \(\,H(x)= (x \ge 2)\) and **conclusion**
   \(C(x)= (y \le 1.2\,x)\):

   .. code-block:: python

      hyp = (X >= 2)
      conc = (Y <= 1.2 * X)

4) **Construct the `Conjecture`**:

   .. code-block:: python

      from txgraffiti.logic.conjecture_logic import Conjecture

      conj = Conjecture(hyp, conc)

5) **Evaluate and summarize**:

   .. code-block:: python

      mask    = conj(df)
      holds   = conj.is_true(df)
      accuracy = conj.accuracy(df)
      cex     = conj.counterexamples(df)

      print("Mask:", mask.tolist())
      print("Holds on all rows?", holds)
      print(f"Accuracy under H: {accuracy:.2%}")
      print("Counterexamples:\n", cex)

Output:

.. code-block:: text

   Mask: [True, True, True, True, True]
   Holds on all rows? True
   Accuracy under H: 100.00%
   Counterexamples:
   Empty DataFrame

After this you can pass your `Conjecture` into discovery pipelines, apply
heuristics (e.g. Morgan, Dalmatian), post‐process (sorting, deduplication),
and even export to formal theorem stubs for Lean or Coq.
