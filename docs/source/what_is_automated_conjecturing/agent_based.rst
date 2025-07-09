Agent-Based Conjecturing
=========================

Conjecture generation is not a one-shot task—it's an **interactive loop**. Agent-based frameworks explicitly model this duality through **conjecture and refutation**.

**The Optimist–Pessimist Model**:
- **Optimist**: Proposes conjectures using heuristics or optimization.
- **Pessimist**: Tries to refute them via counterexamples (often using reinforcement learning).

**Feedback Loop**:
1. Optimist proposes.
2. Pessimist finds a counterexample.
3. Optimist retrains or adjusts bounds.
4. Repeat.

**TxGraffiti Integration**:
TxGraffiti plays the Optimist role:
- Proposes inequalities from data.
- Filters and ranks them for strength.
- Can update its conjecture space in response to new counterexamples.

In future versions, this loop can be automated—turning TxGraffiti into a full participant in agent-based discovery.
