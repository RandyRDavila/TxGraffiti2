.. _heuristics/sophie:

Sophie Acceptance Heuristic
===========================

.. currentmodule:: txgraffiti.heuristics.delavina

Overview
--------

The **Sophie heuristic**, introduced by Ermelinda DeLaViña, filters conjectures based on the **novelty of their hypothesis coverage**. It encourages exploration by ensuring that each accepted conjecture contributes something *new* to the covered portion of the dataset.

This heuristic evaluates a candidate conjecture by checking whether it covers any rows not already covered by previously accepted ones.

Acceptance Criteria
-------------------

Let :math:`H` be the hypothesis of a new conjecture. Define:

- :math:`\text{cover}(H)` — the set of rows where `H(df)` evaluates to `True`.

Then the **Sophie heuristic** accepts a new conjecture if:

.. math::

    \text{cover}(H_{\text{new}}) \not\subseteq \bigcup_i \text{cover}(H_i),

where :math:`H_i` are the hypotheses of all previously accepted conjectures.

Function Signature
------------------

.. autofunction:: sophie_accept
    :no-index:

Notes
-----

- This heuristic works well in conjunction with significance-based heuristics like :ref:`heuristics/dalmatian` or generality-based heuristics like :ref:`heuristics/morgan`.
- It can be used to encourage diversity in the discovered conjectures, by demanding nonredundant hypothesis domains.

See Also
--------

- :ref:`heuristics/dalmatian` — Dalmatian: sharp upper bounds over shared hypothesis
- :ref:`heuristics/morgan` — Morgan: filter out overly specific hypotheses
