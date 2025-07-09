Geometric Approaches
=====================

Some systems frame conjecture discovery geometrically, treating inequalities as **faces** or **facets** of a convex hull over invariant vectors.

**Key Systems**:
- **GraPHedron**: Plots graph invariants in high-dimensional space and extracts conjectures from the boundaries.
- **PHOEG**: A modern framework that integrates polyhedral and visual exploration of graph invariant spaces.

**Geometric Insight**:
- Each graph corresponds to a point in ℝⁿ (via its invariants).
- Inequalities define halfspaces.
- Conjectures correspond to supporting hyperplanes.

**TxGraffiti Parallel**:
TxGraffiti doesn’t compute full convex hulls, but:
- Its linear conjectures naturally define supporting halfspaces.
- It can identify when two bounds coincide (indicating equality).
- Its filtering heuristics implicitly prune interior inequalities.

Future versions may include convex hull support for full geometric inference.
