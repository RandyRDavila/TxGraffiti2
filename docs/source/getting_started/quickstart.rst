Quickstart
==========

Follow these steps to generate your first conjectures in under ten lines of code.

1.  **Import and load data**
    You can use your own `pandas.DataFrame`, or try the built-in `graph_data` toy dataset:

    .. code-block:: python

       import pandas as pd
       from txgraffiti.example_data import graph_data

       # Option A: use the bundled graph dataset
       df = graph_data

       # Option B: load your own CSV
       # df = pd.read_csv("my_graph_invariants.csv")

2.  **Create a ConjecturePlayground**
    Wrap the DataFrame in a session.  Choose a symbol (e.g. `G`) to represent each row:

    .. code-block:: python

       from txgraffiti.playground import ConjecturePlayground

       pg = ConjecturePlayground(
           df,
           object_symbol="G",    # used in ∀/∃ output
           base="connected"      # optional: assume every graph is connected
       )

3.  **Define hypotheses**
    Lift Boolean columns or build new ones using `Predicate` expressions:

    .. code-block:: python

       # Existing Boolean columns become Predicates automatically:
       regular = (pg.max_degree == pg.min_degree)

       # You can also combine or create new Predicates:
       cubic   = regular & (pg.max_degree == 3)
       bipartite = pg.bipartite  # if 'bipartite' column exists

4.  **Run discovery**
    Choose one or more **generator** functions, supply numeric **features** and a **target**,
    and optionally pass **heuristics** and **post-processors** to filter & sort results:

    .. code-block:: python

       from txgraffiti.generators import convex_hull, linear_programming, ratios
       from txgraffiti.heuristics import morgan, dalmatian
       from txgraffiti.processing import remove_duplicates, sort_by_touch_count

       pg.discover(
           methods         = [convex_hull, linear_programming, ratios],
           features        = ['order', 'matching_number', 'min_degree'],
           target          = 'independence_number',
           hypothesis      = [cubic, regular],
           heuristics      = [morgan, dalmatian],
           post_processors = [remove_duplicates, sort_by_touch_count],
           # You can also pass generator-specific kwargs:
           # drop_side_facets = True,
           # round_decimals   = 3,
           # min_fraction     = 0.10,
       )

5.  **Inspect your top conjectures**
    The `.conjectures` list is sorted by your post-processors (e.g. touch-count):

    .. code-block:: python

       for i, conj in enumerate(pg.conjectures[:5], start=1):
           print(f"Conjecture {i}.", pg.forall(conj))
           print("  → accuracy:", f"{conj.accuracy(df):.2%}\n")

    Example output:

    .. code-block:: text

       Conjecture 1. ∀ G: (cubic) ∧ (order ≤ 2 · matching_number)
         → accuracy: 100.00%

       Conjecture 2. ∀ G: (connected) ∧ (min_degree ≥ 2)
         → (independence_number ≤ order − 2)
         accuracy:  85.71%

That’s it!  You’ve generated and evaluated your first conjectures—mix and match
different generators, hypotheses, and processing steps to discover new patterns
in your data.
