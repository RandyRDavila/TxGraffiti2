GraPHedron and Geometric Conjecturing
=====================================

The **GraPHedron** system introduced a geometric approach to automated conjecturing:
each mathematical object is mapped to a point in Euclidean space based on its invariant
values, and the **convex hull** of these points is computed. The facets of this hull
correspond to **tight inequalities** that define the boundary of the invariant cloud.

TxGraffiti replicates and extends this idea using the `convex_hull` generator.

Convex Hull Generator
---------------------

.. autofunction:: txgraffiti.generators.hull_generators.convex_hull

**Input**:
- A list of numeric invariants (as `Property` objects)
- A single `target` invariant to bound
- A `Predicate` hypothesis to restrict which rows to include

**Output**:
- Linear inequality conjectures of the form:

  .. math::

      \text{hypothesis} \Rightarrow \text{target} \leq \text{RHS} \quad \text{or} \quad \text{target} \geq \text{RHS}

Where RHS is a linear combination of the input features.

Example
-------

Suppose we have the following data table:

.. code-block:: python

    import networkx as nx
    import pandas as pd
    from txgraffiti.logic import Property, TRUE
    from txgraffiti.generators.geometric import convex_hull
    from itertools import combinations
    from graphcalc import diameter, size  # replace with your actual graph invariant functions

    # Generate all connected 4-vertex graphs
    paw = nx.Graph()
    paw.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 0), (0, 4)])

    graphs = [
            nx.star_graph(3),                      # (2, 3)
            nx.path_graph(4),                      # (3, 3)
            paw,         # (2, 3) + isolated node
            nx.cycle_graph(4),                     # (2, 4)
            nx.Graph([(0,1), (1,2), (2,3), (0,2), (1,3)]),  # (2, 5)
            nx.complete_graph(4)                   # (1, 6)
        ]


    # Build dataframe of invariants
    df = pd.DataFrame({
        "name": [f"G{i}" for i in range(len(graphs))],
        "D": [diameter(G) for G in graphs],
        "m": [size(G) for G in graphs],
        'connected_graph': [True for G in graphs],
    })

    D = Property("D", lambda df: df["D"])
    m = Property("m", lambda df: df["m"])
    connected = Predicate("connected graph", lambda df: df["connected_graph"])

    for conj in convex_hull(df, features=[D], target=m, hypothesis=connected):
        print(conj)



**Output**

.. code-block:: text

    <Conj (connected graph) → (m >= ((-3 * D) + 9))>
    <Conj (connected graph) → (m >= 3)>
    <Conj (connected graph) → (m <= ((-1/2 * D) + 13/2))>

This means that the values of `t` always lie between `a + b` and `2a + 3b` on this dataset.
TxGraffiti automatically infers these bounding relationships using the convex hull geometry
of the 3D point cloud (a, b, t).

Underlying Geometry
-------------------

- Each row in the dataset is treated as a point in ℝⁿ.
- The convex hull is computed using `scipy.spatial.ConvexHull`.
- Each face of the hull corresponds to a linear inequality.
- Small coefficients (or side facets) can be filtered using `drop_side_facets=True`.

Use Cases
---------

- Reproducing the geometric insights of GraPHedron.
- Exploring the boundary structure of invariant spaces.
- Producing tight **upper and lower bounds** on a target invariant.

Learn more about GraPHedron’s original motivation in the `timeline.rst` and `symbolic_heuristics.rst` files.

