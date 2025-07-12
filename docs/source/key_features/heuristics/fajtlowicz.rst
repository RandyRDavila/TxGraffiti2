.. _heuristics/dalmatian:

Dalmatian Acceptance Heuristic
==============================

.. currentmodule:: txgraffiti.heuristics.fajtlowicz

Overview
--------

The **Dalmatian heuristic**, introduced by Fajtlowicz, provides a principled
filter for deciding whether a newly proposed upper-bound conjecture should be
accepted, based on **validity** and **strict improvement** over existing bounds.

It is named for the “spotted” pattern of truth: accepted conjectures must
cover all points (no false spots) while improving at least one (showing a
distinct spot of significance).

Acceptance Criteria
-------------------

A candidate conjecture :math:`C = H \Rightarrow (LHS \leq RHS)` is accepted
if and only if:

1. (**Truth Test**) The inequality holds for all objects satisfying the hypothesis `H`.

2. (**Significance Test**) There exists at least one object for which the new RHS is
   **strictly tighter** (i.e., strictly less) than every previously accepted RHS bound
   with the same hypothesis and left-hand side.

If no matching prior bounds exist, the candidate is accepted by default.

Function Signature
------------------

.. autofunction:: dalmatian_accept
    :no-index:

.. Example
.. -------

.. .. code-block:: python

..     import pandas as pd
..     from txgraffiti.logic import Property, Predicate, Inequality, Conjecture
..     from txgraffiti.heuristics.fajtlowicz import dalmatian_accept

..     df = pd.DataFrame({
..         'alpha':     [1, 1, 1],
..         'beta':      [1, 1, 1],
..         'connected': [True, True, True],
..     })

..     P = Predicate('connected', lambda df: df['connected'])
..     A = Property('alpha', lambda df: df['alpha'])
..     B = Property('beta',  lambda df: df['beta'])

..     # Define several conjectures of increasing strength
..     weak   = Conjecture(P, Inequality(A, "<=", B + 2))
..     strong = Conjecture(P, Inequality(A, "<=", B + 1))
..     best   = Conjecture(P, Inequality(A, "<=", B))

..     # Case 1: no existing bounds ⇒ accept
..     assert dalmatian_accept(best, [], df) is True

..     # Case 2: weaker than existing ⇒ reject
..     assert dalmatian_accept(weak, [strong], df) is False

..     # Case 3: strictly tighter than existing ⇒ accept
..     assert dalmatian_accept(best, [strong], df) is True

.. See Also
.. --------

.. - :ref:`heuristics/morgan` — Morgan filter: simple uniqueness & truth
.. - :ref:`key_features/generators/linear_programming` — LP-based inequality generation
