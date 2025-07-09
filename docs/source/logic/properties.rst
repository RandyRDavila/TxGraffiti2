Properties
==========

The ``Property`` class is the core abstraction for **numeric** data in TxGraffiti.
It wraps a DataFrame column (or any Pandas‐Series–producing function) and lets you
develop algebraic expressions entirely within the DSL, deferring actual DataFrame
access until evaluation time.

Key Features
------------

- **Auto-lifting of scalars**
  You can mix ``Property`` objects and plain numbers seamlessly:

  .. code-block:: python

     from txgraffiti.logic.conjecture_logic import Property, Constant
     P = Property('price', lambda df: df['price'])
     expr = P * 1.2 + 5          # automatically lifts 5 → Constant(5)

- **Standard arithmetic operators**
  Supports ``+``, ``-``, ``*``, ``/``, and ``**``, with identity simplifications:

  .. code-block:: python

     zero = Constant(0)
     assert (P + zero).name == P.name   # P + 0 simplifies to P

- **Human-readable names**
  Every composed expression records its textual form in ``.name``:

  .. code-block:: python

     expr = (P * 0.9) - Constant(2)
     print(expr.name)
     # → "(price * 0.9) - 2"

- **Deferred evaluation**
  The underlying lambda is not executed until you call the ``Property`` on a DataFrame:

  .. code-block:: python

     series = expr(df)   # returns a pandas.Series of computed values

Motivation
----------

When building conjectures, you often need to:

  1.  **Compose** linear or nonlinear expressions over columns,
  2.  **Inspect** the symbolic form for printing or exporting,
  3.  **Evaluate** those expressions on your DataFrame rows,
  4.  **Re-use** them in predicate logic or inequality generation.

A raw lambda alone can’t carry its own “name” or simplify repeated constants.
``Property`` unifies these concerns by pairing a Python function with a
symbolic name and operator‐overloads that keep everything readable.

Example
-------

Below is a simple end-to-end example of defining and using a ``Property``.

.. code-block:: python

   import pandas as pd
   from txgraffiti.logic.conjecture_logic import Property, Constant, Inequality

   # 1) Sample DataFrame
   df = pd.DataFrame({
       'price': [10.0, 20.0, 15.0],
       'cost':  [ 8.0, 18.0, 14.0],
   })

   # 2) Lift columns to Property
   P = Property('price', lambda df: df['price'])
   C = Property('cost',  lambda df: df['cost'])
   K = Constant(0.9)   # a fixed discount factor

   # 3) Build a new Property for discounted margin
   margin = (P * K) - C
   print("Expression name:", margin.name)
   # → "(price * 0.9) - cost"

   # 4) Evaluate on the DataFrame
   print(margin(df))
   # 0    1.0   # 10*0.9 - 8.0
   # 1    0.0   # 20*0.9 - 18.0
   # 2   -0.5   # 15*0.9 - 14.0

   # 5) Use in an Inequality
   ineq = Inequality(margin, '>=', Constant(0))
   print(ineq.name)
   # → "(price * 0.9) - cost >= 0"

You can now wrap ``ineq`` in a ``Conjecture`` or feed ``margin`` into any generator
to discover bounds or test hypotheses over your data.
