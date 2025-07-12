Properties & Constants
======================

The **Property** class lifts either a DataFrame column or a scalar constant into a symbolic
object that supports arithmetic, comparisons, and lazy evaluation.  A **Constant** is simply
a `Property` that always returns the same number for every row.

Example DataFrame
-----------------

.. code-block:: python

   import pandas as pd

   df = pd.DataFrame({
       'alpha':     [1, 2, 3],
       'beta':      [3, 1, 1],
       'connected': [True, True, True],
       'tree':      [False, False, True],
   })

Constants
---------

Create constant‐valued properties and combine them:

.. code-block:: python

   from txgraffiti import Constant

   c2 = Constant(2)
   c3 = Constant(3)

   print(c2)           # <Constant 2>
   print(c2(df))       # Series: [2, 2, 2]

   print(c2 + c3)      # <Constant (2 + 3)>
   print((c2 + c3)(df))
   # [5, 5, 5]

   expr = 2*c2 - c3
   print(expr)         # <Constant ((2 * 2) - 3)>
   print(expr(df))     # [1, 1, 1]

Properties
----------

Lift DataFrame columns into symbolic properties and build expressions:

.. code-block:: python

   from txgraffiti import Property

   alpha = Property('alpha', lambda df: df['alpha'])
   beta  = Property('beta',  lambda df: df['beta'])

   print(alpha)        # <Property alpha>
   print(alpha(df))    # [1, 2, 3]

   expr1 = 2 * alpha
   print(expr1)        # <Property (2 * alpha)>
   print(expr1(df))    # [2, 4, 6]

   combo = alpha*alpha + 5
   print(combo)        # <Property ((alpha * alpha) + 5)>
   print(combo(df))    # [6, 9, 14]

Mixed Example
-------------

Combine constants and columns seamlessly:

.. code-block:: python

   # c2 and c3 from above, alpha from above
   bound = c2 * alpha + c3
   print(bound)        # <Property ((2 * alpha) + 3)>
   print(bound(df))    # [5, 7, 9]

   # Using multiple properties and constants:
   expr2 = (alpha + beta) * c3 - c2
   print(expr2)        # <Property (((alpha + beta) * 3) - 2)>
   print(expr2(df))    # [10, 4, 4]
