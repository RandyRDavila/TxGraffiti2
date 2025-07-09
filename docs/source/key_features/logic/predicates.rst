Predicates
==========

A **Predicate** is a row-wise Boolean test on a DataFrame. Under the hood it wraps
a function `df → pd.Series[bool]` but provides:

- **Operator overloading**: use `&`, `|`, `^`, and `~` to build complex logic
- **Automatic simplifications**:
  - `P & P` ⇒ `P`
  - `P & True` ⇒ `P`
  - `P | False` ⇒ `P`
  - `P & ~P` ⇒ `False`
  - `P | ~P` ⇒ `True`
- **Deferred evaluation**: you only get a Boolean Series when you call it on a DataFrame

Key Features
------------

- **Basic comparisons**
  Compare two `Property` objects (or a `Property` and a scalar) to form an `Inequality`
  (a subclass of `Predicate`):

  .. code-block:: python

     from txgraffiti.logic.conjecture_logic import Property

     P = Property('value', lambda df: df['value'])
     is_large = (P > 100)           # Inequality returned, e.g. "<value > 100>"

- **Logical connectives**
  Combine predicates with standard Boolean operators:

  .. code-block:: python

     # assume pred1 and pred2 are Predicates
     both   = pred1 & pred2    # AND
     either = pred1 | pred2    # OR
     xor    = pred1 ^ pred2    # XOR
     not_p  = ~pred1           # NOT

- **Readable names**
  The `.name` attribute records the symbolic form:

  .. code-block:: python

     print((pred1 & ~pred2).name)
     # → "(pred1) ∧ ¬(pred2)"

- **Evaluation on DataFrame**
  Call the predicate to filter rows:

  .. code-block:: python

     mask = both(df)   # pandas.Series of True/False
     filtered_df = df[mask]

Motivation
----------

When constructing conjectures, you need to express hypotheses and conclusions as
composable, inspectable Boolean formulas. A raw function or mask lacks the ability
to be combined algebraically or simplified. `Predicate` gives you:

1. **Algebraic composition** of tests
2. **Symbolic naming** for pretty-printing and export
3. **Built-in logic laws** to simplify expressions

Example
-------

Here’s a typical workflow using `Predicate`:

.. code-block:: python

   import pandas as pd
   from txgraffiti.logic.conjecture_logic import Property, Predicate

   # Sample data
   df = pd.DataFrame({
       'value': [5, 12, 27, 30],
       'category': ['A', 'B', 'A', 'B'],
   })

   # Lift a numeric column
   V = Property('value', lambda df: df['value'])

   # Define simple predicates
   is_even     = Predicate('is_even',  lambda df: df['value'] % 2 == 0)
   is_large    = (V >= 20)                   # Inequality → Predicate
   is_category = Predicate(
       'is_A', lambda df: df['category'] == 'A'
   )

   # Combine them
   combo = is_even & is_large & is_category

   # Inspect the formula
   print(combo.name)
   # → "(is_even) ∧ (value >= 20) ∧ (is_A)"

   # Filter the DataFrame
   result = df[combo(df)]
   print(result)
   #    value category
   # 2     27        A

Once you have a `Predicate`, you can pass it as a hypothesis or conclusion into a
`Conjecture`, or use it directly with `.forall(pred)` / `.exists(pred)` in your session.
