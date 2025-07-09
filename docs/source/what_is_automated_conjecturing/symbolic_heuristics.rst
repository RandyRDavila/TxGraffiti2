Symbolic and Heuristic Approaches
==================================

The earliest systems for automated conjecturing were built using symbolic logic and rule-based heuristics. These systems searched for regularities in formal expressions or simple datasets, guided by hard-coded rules or expert knowledge.

**Notable Examples**:

- **Wang’s Program II (1959)**: Generated thousands of logical formulas, illustrating the combinatorial explosion of naive symbolic enumeration.
- **Lenat’s AM (1976–1977)**: Introduced the idea of "interestingness" as a heuristic for guiding exploration in elementary number theory.
- **Epstein’s Graph Theorist (1980s)**: Used symbolic representations and minimal examples to infer and prove conjectures in graph theory.

**Key Characteristics**:
- Operated over propositional or algebraic formulas.
- Depended on handcrafted rules.
- Lacked robust filtering mechanisms, leading to many trivial conjectures.

**Relation to TxGraffiti**:
TxGraffiti avoids raw symbolic enumeration, but its internal architecture is influenced by these early efforts—especially in how it lifts expressions into `Property` and `Predicate` objects and supports user-defined symbolic logic.

