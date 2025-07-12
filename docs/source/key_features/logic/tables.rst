KnowledgeTable
==============

.. currentmodule:: txgraffiti.logic.tables

The ``KnowledgeTable`` class is a core data structure in **TxGraffiti**. It is a lightweight subclass of `pandas.DataFrame` designed to work seamlessly with symbolic logic components like `Property`, `Predicate`, and `Conjecture`.

It automatically wraps columns as symbolic `Property` objects and supports rich algebraic expressions and logical composition directly on the table's named fields.

Use ``KnowledgeTable`` when you want:

- Symbolic access to data columns as `Property` objects.
- Tight integration with the `txgraffiti.logic` layer.
- A robust, expressive environment for reasoning over tabular data.

----

Quick Example
-------------

The example below illustrates how to use `KnowledgeTable` to create and manipulate symbolic expressions.

.. code-block:: python

    from txgraffiti.logic.tables import KnowledgeTable

    kt = KnowledgeTable({
        'alpha': [1, 2, 3],
        'beta':  [3, 1, 1],
        'tree':  [False, False, True],
    })

    alpha = kt.alpha    # ← Property('alpha')
    beta  = kt.beta     # ← Property('beta')

    # Properties support arithmetic
    expr1 = alpha + beta
    print(expr1(kt))  # [4, 3, 4]

    # Logical operations using Predicates
    pred = kt.tree & (alpha == 3)
    print(pred(kt))  # [False, False, True]

----

API Highlights
--------------

- ``kt.<column_name>`` gives you a symbolic `Property` object.
- `Property` objects support `+, -, *, /, **, <, >, ==, !=`, etc.
- `Predicate` objects support `&` (AND), `|` (OR), and `~` (NOT).
- Compatible with `Conjecture(hypothesis ⇒ conclusion)`.
