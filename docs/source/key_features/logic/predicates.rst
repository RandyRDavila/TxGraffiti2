Predicates
==========

A **Predicate** is a first‐class, composable Boolean test on each row of a `pandas.DataFrame`.
Under the hood it wraps a function `df → pd.Series[bool]` but gives you:

- A **symbolic name** for pretty‐printing and export
- **Operator overloads** (`&`, `|`, `^`, `~`) with built-in simplifications
- Deferred, row-wise evaluation only when you apply it to a DataFrame

Motivation
----------

When building conjectures you often need to express hypotheses like “\(G\) is a tree”
or “degree ≥ 3” as reusable, named Boolean tests.  A raw mask (`df['tree']`) has no
symbolic identity, can’t be combined algebraically, and doesn’t prettify.  `Predicate`
fills that gap.

Defining a Predicate
--------------------

You can lift any boolean‐valued column or expression into a `Predicate`.  For example:

.. code-block:: python

   import pandas as pd
   from txgraffiti.logic.conjecture_logic import Predicate

   df = pd.DataFrame({
       'alpha':    [1, 2, 3],
       'beta':     [3, 1, 1],
       'connected':[True, True, True],
       'tree':     [False, False, True],
   })

   # 1) Lift an existing Boolean column:
   tree = Predicate('tree', lambda df: df['tree'])

   # 2) Inspect its symbolic form:
   print(tree)
   # → <Predicate tree>

Evaluating a Predicate
----------------------

Applying a `Predicate` to your DataFrame produces a `pandas.Series[bool]` mask:

.. code-block:: python

   mask = tree(df)
   print(mask)

   # Output:
   # 0    False
   # 1    False
   # 2     True
   # Name: tree, dtype: bool

You can use this mask to filter rows:

.. code-block:: python

   trees_only = df[tree(df)]
   print(trees_only)

   #    alpha  beta  connected   tree
   # 2      3     1       True   True

Logical Operators
-----------------

Predicates support standard Boolean connectives with automatic simplifications:

- **AND**: `p & q`
- **OR**:  `p | q`
- **XOR**: `p ^ q`
- **NOT**: `~p`

Example:

.. code-block:: python

   # Build two simple Predicates from columns:
   is_high_alpha = Predicate('high_alpha',  lambda df: df['alpha'] >= 3)
   is_low_beta  = Predicate('low_beta',    lambda df: df['beta']  <= 1)

   # Combine them:
   combo = is_high_alpha & is_low_beta

   print(combo.name)
   # → "(high_alpha) ∧ (low_beta)"

   # Evaluate:
   print(combo(df))
   # 0    False
   # 1    False
   # 2     True
   # dtype: bool

Automatic Simplifications
-------------------------

- **Idempotence**: `P & P` ⇒ `P`;  `P | P` ⇒ `P`
- **Domination**: `P & False` ⇒ `False`;  `P | True` ⇒ `True`
- **Complement**: `P & ~P` ⇒ `False`;  `P | ~P` ⇒ `True`
- **Absorption**: `P & (P | Q)` ⇒ `P`;  `P | (P & Q)` ⇒ `P`

Comparison & Implication
------------------------

You can compare numeric `Property` objects (or literals) to obtain an `Inequality` (a subclass of `Predicate`):

.. code-block:: python

   from txgraffiti.logic.conjecture_logic import Property

   A = Property('alpha', lambda df: df['alpha'])
   C = Property('cost',  lambda df: df['beta'] * 2)

   test = (A + 1 >= C)
   print(test.name)
   # → "(alpha + 1) ≥ (beta * 2)"

   # Combine numeric tests with Boolean predicates:
   conj = tree & test
   print(conj.name)
   # → "(tree) ∧ (alpha + 1 ≥ beta * 2)"

Quantifiers
-----------

Within a `ConjecturePlayground`, you can wrap predicates in `∀` or `∃` notation:

.. code-block:: python

   from txgraffiti.playground import ConjecturePlayground

   pg = ConjecturePlayground(df, object_symbol='G')
   print(pg.forall(tree))   # prints "∀ G: tree"
   print(pg.exists(tree))   # prints "∃ G: tree"

Summary
-------

`Predicate` is your building block for hypotheses and conclusions:

1. **Lift** any Boolean test into a named object
2. **Combine** via `&`, `|`, `^`, `~` with algebraic laws
3. **Evaluate** only when you call it on a DataFrame
4. **Inspect** its `.name` for symbolic output

Use them to drive automated discovery, logic‐based filtering, and exporting to proof assistants.
