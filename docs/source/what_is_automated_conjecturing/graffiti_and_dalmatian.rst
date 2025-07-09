Graffiti and the Dalmatian Heuristic
====================================

In the 1980s, Siemion Fajtlowicz introduced **Graffiti**, a revolutionary system that empirically evaluated inequalities on graph datasets. Its success came from pairing brute-force generation with powerful **heuristic filters**.

**Core Ideas from Graffiti**:
- Generate inequalities involving graph invariants.
- Retain only those that are true for all known graphs.
- Filter trivial or implied conjectures using IRIN and CNCL heuristics.

**The Dalmatian Heuristic**:
- **Truth Test**: The inequality must hold for all known graphs.
- **Significance Test**: It must give a strictly better bound for at least one graph than previously known inequalities.
- The name “Dalmatian” comes from the idea that each conjecture should "touch" a different spot in the data, like spots on a Dalmatian.

**TxGraffiti Connection**:
TxGraffiti implements a modernized and modular version of this pipeline:
- Computes conjectures that pass both truth and significance tests.
- Tracks touch numbers.
- Applies transitive pruning to reduce redundancy.

This is the philosophical core of TxGraffiti's "Hazel", "Morgan", and "Dalmatian" filters.
