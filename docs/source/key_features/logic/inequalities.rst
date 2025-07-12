Inequalities
============

An **Inequality** is a symbolic statement comparing two numeric expressions built from your data.

These statements take the form:

.. math::

   L(G) \;\le\; R(G)
   \quad\text{or}\quad
   L(G) \;\ge\; R(G)

where \(L\) and \(R\) are expressions over graph invariants (or other structured data features).
They serve as the building blocks for numerical conjectures in `txgraffiti`.

Concrete Example
----------------

Here’s a worked example using the built-in `graph_data` dataset:

.. code-block:: python

   from txgraffiti.playground    import ConjecturePlayground
   from txgraffiti.generators    import convex_hull, linear_programming, ratios
   from txgraffiti.heuristics    import dalmatian_accept, morgan_accept
   from txgraffiti.processing    import remove_duplicates, sort_by_touch_count
   from txgraffiti.example_data  import graph_data
   from txgraffiti               import Predicate

   ai = ConjecturePlayground(
       graph_data,
       object_symbol='G',
       base='connected',
   )

   # Define custom predicate: G is cubic
   regular = Predicate('regular', lambda df: df['min_degree'] == df['max_degree'])
   cubic   = regular & (ai.max_degree == 3)

   # Run discovery using three methods
   ai.discover(
       methods         = [ratios, convex_hull, linear_programming],
       features        = [ai.independence_number],
       target          = ai.zero_forcing_number,
       heuristics      = [dalmatian_accept, morgan_accept],
       hypothesis      = [cubic],
       post_processors = [remove_duplicates, sort_by_touch_count],
   )

   # Print resulting inequalities
   for idx, conj in enumerate(ai.conjectures[:3], start=1):
       print(f"Conjecture {idx}: {ai.forall(conj)}")

Output:

.. code-block:: text

   Conjecture 1: ∀ G: ((connected) ∧ (regular) ∧ (max_degree == 3)) → (zero_forcing_number >= 3)

   Conjecture 2: ∀ G: ((connected) ∧ (regular) ∧ (max_degree == 3)) → (zero_forcing_number >= (-2 + independence_number))

   Conjecture 3: ∀ G: ((connected) ∧ (regular) ∧ (max_degree == 3)) → (zero_forcing_number >= ((2 * independence_number) + -8))

Each of these is an `Inequality` object automatically constructed from symbolic arithmetic and evaluated row-by-row on the data.

Touch and Slack
---------------

For any `Inequality`, you can compute:

- **Slack**: the numeric distance between the LHS and RHS.
- **Touch count**: how many rows satisfy the inequality exactly (i.e., with zero slack).

.. code-block:: python

   ineq = ai.conjectures[1].conclusion
   print("Touch count:", ineq.touch_count(graph_data))
   print("Slack values:", ineq.slack(graph_data).tolist())

Integration
-----------

Inequalities can be used directly as:

- `Predicate` objects for filtering or logic composition
- Conclusions inside a `Conjecture` statement
- Targets for filtering, ranking, and exporting

.. seealso::

   - :doc:`properties`
   - :doc:`predicates`
