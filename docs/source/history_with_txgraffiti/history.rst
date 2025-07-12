.. _history:

History
=======

*“Hi-de-hi, ho-de-ho — discovery begins with a call, and an echo.”*
— *Inspired by Cab Calloway, Minnie the Moocher (1931)*

The dream of machines that *conjecture like mathematicians* has a long and fascinating history.
**TxGraffiti** continues this tradition by building upon generations of symbolic, heuristic, and data-driven systems that have shaped the evolving field of **automated mathematical discovery**.

Origins: Symbolic and Heuristic Beginnings
------------------------------------------

- **Wang’s Program II (1959)**
  One of the earliest documented attempts to generate mathematical statements automatically. It produced thousands of logical formulas but lacked mechanisms to identify meaningful ones.

- **Lenat’s AM (1976–77)**
  Simulated mathematical creativity through hundreds of hand-coded heuristics. Rediscovered fundamental notions like primes, divisibility, and proposed versions of famous conjectures.

- **Epstein’s Graph Theorist (1980s)**
  Focused on symbolic graph reasoning and the automation of proofs, using algebraic definitions and transformations to uncover property relationships.

The Graffiti Era
----------------

- **Fajtlowicz’s Graffiti (1980s–2000s)**
  A foundational system that generated conjectures by evaluating inequalities on graph invariants. Introduced **IRIN**, **CNCL**, and the now-famous **Dalmatian heuristic** to refine results. Its conjectures led to dozens of published theorems.

- **DeLaViña’s Graffiti.pc (1990s–2010s)**
  Extended Graffiti into a graphical environment for research and education. Introduced the **Sophie heuristic** for structural inference and integrated conjecture discovery into undergraduate learning.

Optimization and Geometry
-------------------------

- **AutoGraphiX (2000s)**
  Recast conjecturing as an optimization problem, searching graph space using Variable Neighborhood Search to minimize or maximize target expressions.

- **GraPHedron & PHOEG**
  Applied geometric and polyhedral analysis to identify conjecture boundaries, using convex hulls of graph invariant vectors to discover inequality facets.

The Modern Era: TxGraffiti and Beyond
-------------------------------------

- **TxGraffiti (2017–present)**
  A hybrid system that merges linear optimization with heuristic filters. It generates inequalities over tabular data, ranks them by **touch number**, and applies Dalmatian-style pruning.
  TxGraffiti has independently rediscovered known theorems and produced novel open problems in graph theory, and now powers both a public Python package and an interactive web application.

Data-Driven and Neural Approaches
---------------------------------

- **The Ramanujan Machine (2019–present)**
  Uses symbolic expression search and numerical precision to conjecture continued fraction identities for mathematical constants like π and ζ(3).

- **Learning Algebraic Varieties (2018–present)**
  Infers defining polynomials and geometric structure from sample points using tools from algebraic geometry and topological data analysis.

- **DeepMind’s Neural Mathematician (2021–present)**
  Trains neural networks on mathematical datasets to predict invariants and discover new theorems in areas such as knot theory and representation theory.

A New Paradigm: Agent-Based Discovery
-------------------------------------

- **The Optimist–Pessimist Model (2024–present)**
  Formalizes the interaction between conjecture generation (Optimist) and counterexample search (Pessimist).
  The **Optimist** agent (powered by TxGraffiti) generates inequalities with supporting heuristics.
  The **Pessimist** agent uses reinforcement learning to explore graph space and find violations, forming a feedback loop of empirical refinement.

Why It Matters
--------------

From Wang’s early logic engine to TxGraffiti’s optimization pipelines and DeepMind’s learning-guided insights, the landscape of automated conjecturing has evolved into a rich, multi-agent, multi-modal field.

Today’s systems no longer simply enumerate formulas — they *dream*, *doubt*, *refine*, and *learn*.
**TxGraffiti** carries this legacy forward with modern tools for automated reasoning, bridging symbolic mathematics with interactive AI.

