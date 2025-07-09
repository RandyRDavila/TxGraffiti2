Welcome to TxGraffiti
=====================

*A machine that dreams in mathematics, scrawling conjectures on the walls of discovery.*

**TxGraffiti** is a Python toolkit for **automated mathematical conjecturing**.
It can generate novel inequalities, rediscover classical theorems, and explore
structural patterns in objects such as graphs, groups, and numerical sequences.

Originally prototyped in **2016**, TxGraffiti has since powered dozens of research
projects—spanning domination, zero forcing, independence numbers, and beyond.
Inspired by Siemion Fajtlowicz’s original *Graffiti* program, this modern Python
implementation (co‐developed by Jillian Eddy and Randy Davila) adds:

- Symbolic **Property** and **Predicate** objects for clear DSL-style expressions
- Geometry- and LP-driven **generators** for systematic bound discovery
- **Heuristics** (Morgan, Dalmatian, …) to filter and rank the most intriguing conjectures
- Built-in support for exporting to **Lean 4** stubs for formal proof development

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart

Key Features
------------

- **Automated discovery** of numeric **inequalities** and **equalities**
- Rich **boolean hypotheses** (`Predicate`) on graph‐ or tabular data
- Multiple **conjecture generators**: convex-hull, ratio bounds, Chebyshev LP, …
- Pluggable **heuristics** and **post-processors** to refine and sort results
- Seamless **extension**: register your own generators, predicates, heuristics, or processors
- **Export** to JSON, CSV, or **Lean 4** theorem stubs

Core Concepts
-------------

.. toctree::
   :maxdepth: 1
   :caption: Logic

   logic/properties
   logic/predicates
   logic/inequalities
   logic/conjectures

.. toctree::
   :maxdepth: 1
   :caption: Discovery

   playground/conjecture_playground

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   txgraffiti.generators
   txgraffiti.heuristics
   txgraffiti.logic
   txgraffiti.playground
   txgraffiti.processing
   txgraffiti.export

Learn More
----------

- Interactive website: https://txgraffiti.streamlit.app
- Papers & publications: search “txgraffiti” on arXiv

---

*Let your machine dream. Let it write mathematics.*
Happy conjecturing!
