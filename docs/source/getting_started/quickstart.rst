Quickstart
==========

Get started in just a few lines: load a dataset, define hypotheses, and generate mathematical conjectures using symbolic expressions and heuristics.

Step 1: Load your data
----------------------

You can use any `pandas.DataFrame`—TxGraffiti works on tabular data with numeric or boolean columns.

Try the built-in graph dataset:

.. code-block:: python

   from txgraffiti.example_data import graph_data
   df = graph_data  # structured invariants for connected graphs

Step 2: Initialize a playground
-------------------------------

Wrap your dataset in a `ConjecturePlayground` session. This gives you symbolic access to all features.

.. code-block:: python

   from txgraffiti.playground import ConjecturePlayground

   ai = ConjecturePlayground(
       df,
       object_symbol="G",      # used in printed formulas (∀ G: …)
       base="connected",       # optional: global assumption
   )

Step 3: Define hypotheses
-------------------------

You can build logical predicates using symbolic expressions:

.. code-block:: python

   regular = ai.max_degree == ai.min_degree
   cubic   = regular & (ai.max_degree == 3)
   small   = ai.max_degree <= 3
   not_complete = ~ai.Kn

Step 4: Generate conjectures
----------------------------

Choose your generator methods, features, target invariant, and heuristics:

.. code-block:: python

   from txgraffiti.generators import ratios, convex_hull, linear_programming
   from txgraffiti.heuristics import dalmatian_accept, morgan_accept
   from txgraffiti.processing import remove_duplicates, sort_by_touch_count

   ai.discover(
       methods         = [ratios, convex_hull, linear_programming],
       features        = [ai.independence_number],
       target          = ai.zero_forcing_number,
       hypothesis      = [cubic, small & not_complete],
       heuristics      = [dalmatian_accept, morgan_accept],
       post_processors = [remove_duplicates, sort_by_touch_count],
   )

Step 5: Print your top conjectures
----------------------------------

.. code-block:: python

   for i, conj in enumerate(ai.conjectures[:5], start=1):
       print(f"Conjecture {i}.", ai.forall(conj))
       print("    Accuracy:", f"{conj.accuracy(df):.0%}\n")

Example output:

.. code-block:: text

   Conjecture 1. ∀ G: ((connected) ∧ (regular) ∧ (max_degree == 3)) → (zero_forcing_number ≥ 3)
       Accuracy: 100%

   Conjecture 2. ∀ G: ((connected) ∧ (max_degree ≤ 3) ∧ (¬(Kn))) → (zero_forcing_number ≤ (1 + independence_number))
       Accuracy: 100%

   Conjecture 3. ∀ G: ((connected) ∧ (regular) ∧ (max_degree == 3)) → (zero_forcing_number ≥ ((2 · independence_number) − 8))
       Accuracy: 100%

   Conjecture 4. ∀ G: ((connected) ∧ (max_degree ≤ 3) ∧ (¬(Kn))) → (zero_forcing_number ≥ ((6/5 · independence_number) − 37/5))
       Accuracy: 100%

   Conjecture 5. ∀ G: ((connected) ∧ (max_degree ≤ 3) ∧ (¬(Kn))) → (zero_forcing_number ≤ ((−1/5 · independence_number) + 47/5))
       Accuracy: 100%

----

What's next?
------------

- Try other generators like `in_reverie` or `make_upper_linear_conjecture`.
- Swap in your own dataset of tabular numerical data.
- Explore how to export formulas to Lean 4 or generate new predicates recursively.

.. seealso::

   - :doc:`installation <installation>`
   - :doc:`logic/index <../key_features/logic/index>`
   - :doc:`playground/index <../key_features/playground/index>`

