.. _heuristics/morgan:

Morgan Acceptance Heuristic
===========================

.. currentmodule:: txgraffiti.heuristics.davila

Overview
--------

The **Morgan heuristic**, proposed by Davila, filters conjectures based on the **generality of the hypothesis**. It ensures that only the *most general* form of a conjecture is retained for each logical conclusion.

This heuristic rejects redundant conjectures whose hypotheses are strict subsets of already accepted ones with the **same inequality conclusion**.

Acceptance Criteria
-------------------

A candidate conjecture :math:`C = H \Rightarrow (LHS \leq RHS)` is accepted if:

- There does **not** exist any previously accepted conjecture with:
  - The **same** logical conclusion (possibly in flipped form), and
  - A **strictly more general** hypothesis (i.e., a hypothesis that covers more rows in the dataset than `H`).

If such a more general conjecture exists, the new one is rejected.

Function Signature
------------------

.. autofunction:: morgan_accept
    :no-index:

.. Example
.. -------

.. .. code-block:: python

..     import pandas as pd
..     from txgraffiti.logic import Predicate, Property, Conjecture, Inequality
..     from txgraffiti.heuristics.davila import morgan_accept

..     df = pd.DataFrame({
..         'alpha':     [1, 2, 3],
..         'beta':      [3, 2, 1],
..         'connected': [True, True, True],
..         'tree':      [False, True, False],
..     })

..     A = Property('alpha', lambda df: df['alpha'])
..     B = Property('beta',  lambda df: df['beta'])

..     # Two hypotheses: one general, one specific
..     P_general = Predicate('connected', lambda df: df['connected'])
..     P_subset  = Predicate('tree',      lambda df: df['tree'])

..     # Define conjectures with the same conclusion but different hypotheses
..     c_general = Conjecture(P_general, Inequality(A, '<=', B))
..     c_subset  = Conjecture(P_subset,  Inequality(A, '<=', B))

..     # The more general one is accepted
..     assert morgan_accept(c_general, [c_subset], df) is True

..     # The more specific one is rejected if the general one already exists
..     assert morgan_accept(c_subset, [c_general], df) is False

Notes
-----

- This heuristic relies on the helper function :func:`same_conclusion` to compare bounds up to logical equivalence.
- Hypotheses are compared using boolean masks on the dataset to determine strict containment.

See Also
--------

- :ref:`heuristics/dalmatian` — Dalmatian filter: global truth + significance test
