Optimization-Based Conjecturing
===============================

The idea of using **optimization** to generate conjectures reframes the problem: instead of asking *which inequalities are true*, we ask *what are the sharpest possible bounds?*

**Notable Example**:
- **AutoGraphiX (2000s)**: Used Variable Neighborhood Search (VNS) to find extremal graphs for target inequalities.

**Key Shift**:
- Search over graph space (or object space) instead of formula space.
- Treat conjecture testing as a **numerical optimization** problem.

**TxGraffiti’s Role**:
TxGraffiti implements this paradigm through linear and mixed-integer programming:
- Finds optimal coefficients for linear bounds.
- Supports both upper and lower bounds over tabular datasets.
- Allows optimization conditioned on Boolean properties (e.g., only connected graphs).

This structure allows TxGraffiti to discover conjectures like:
\[
\alpha(G) \leq \mu(G) \quad \text{for all regular graphs}.
\]

Such bounds are not just valid—they're empirically tight.
