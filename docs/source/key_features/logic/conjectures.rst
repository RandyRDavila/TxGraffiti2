Conjecture
==========

A **Conjecture** packages a row‐wise logical implication

.. math::

   H(x)\;\Longrightarrow\;C(x)

where **H** (the hypothesis) and **C** (the conclusion) are both `Predicate` objects.
It provides:

- **Deferred evaluation**: calling `conj(df)` returns a Boolean mask
- **Universal check**: `conj.is_true(df)` tells you if it holds on every row
- **Accuracy, counterexamples**, and integration into discovery pipelines

Creation Methods
----------------

There are three equivalent ways to build a `Conjecture(H, C)`:

1. **Direct constructor**
2. **`.implies(..., as_conjecture=True)`**
3. **Shift operator `>>`**

Below we illustrate all three.

.. code-block:: python

   import pandas as pd
   from txgraffiti.logic.conjecture_logic import (
       Property, Predicate, Conjecture, TRUE
   )

   # Sample DataFrame
   df = pd.DataFrame({
       'alpha':    [1, 2, 3],
       'beta':     [3, 1, 1],
       'connected':[True, True, True],
       'tree':     [False, False, True],
   })

   # Lift numeric columns to Property
   alpha = Property('alpha',   lambda df: df['alpha'])
   beta  = Property('beta',    lambda df: df['beta'])

   # Hypothesis and conclusion
   hypothesis = Predicate('connected', lambda df: df['connected'])
   conclusion = alpha <= (2 * beta + 3)

   # 1) Direct constructor
   conj1 = Conjecture(hypothesis, conclusion)
   print(f"conj1 = {conj1}")
   print(conj1(df))
   print(f"conj1.is_true(df) = {conj1.is_true(df)}\n")

   # 2) Using .implies(..., as_conjecture=True)
   conj2 = hypothesis.implies(conclusion, as_conjecture=True)
   print(f"conj2 = {conj2}")
   print(conj2(df))
   print(f"conj2.is_true(df) = {conj2.is_true(df)}\n")

   # 3) Using >> operator
   conj3 = hypothesis >> conclusion
   print(f"conj3 = {conj3}")
   print(conj3(df))
   print(f"conj3.is_true(df) = {conj3.is_true(df)}\n")

# Expected output:

.. code-block:: text

   conj1 = <Conj (connected) → (alpha <= ((2 * beta) + 3))>
   0    True
   1    True
   2    True
   dtype: bool
   conj1.is_true(df) = True

   conj2 = <Conj (connected) → (alpha <= ((2 * beta) + 3))>
   0    True
   1    True
   2    True
   dtype: bool
   conj2.is_true(df) = True

   conj3 = <Conj (connected) → (alpha <= ((2 * beta) + 3))>
   0    True
   1    True
   2    True
   dtype: bool
   conj3.is_true(df) = True

Key Methods
-----------

- **`conj(df)`**
  Returns a `pandas.Series[bool]` mask equal to \((¬H(df)) ∨ C(df)\).

- **`conj.is_true(df)`**
  Returns **True** iff every row satisfies the implication.

- **`conj.accuracy(df)`**
  Fraction of rows where `H` is true that also satisfy `C`.

- **`conj.counterexamples(df)`**
  Returns a `DataFrame` of rows violating the implication.

Integration
-----------

`Conjecture` objects can be:

- Filtered by **heuristics** (e.g. Morgan, Dalmatian)
- Sorted or deduplicated in **post‐processors**
- Exported to formal‐proof stubs (Lean 4) via `export_to_lean`

Use `Conjecture` as the core unit in your automated‐discovery workflows.
