Welcome to TxGraffiti
=====================

*A machine that dreams in mathematics, scrawling conjectures on the walls of discovery.*

**TxGraffiti** is a Python toolkit for **automated mathematical conjecturing**.

It empowers machines to explore mathematical structure—uncovering patterns, generating inequalities, and proposing candidate theorems backed by data. TxGraffiti blends symbolic logic, optimization, and heuristics into a unified system capable of operating across graph theory, number theory, combinatorics, and more.

Originally prototyped in **2016**, TxGraffiti descends from a rich lineage of systems—from Fajtlowicz’s *Graffiti* and DeLaViña’s *Graffiti.pc*, to Melót's *GraPHedron*—and has since powered dozens of research projects, produced open conjectures, and contributed to peer-reviewed theorems.

This modern version of *TxGraffiti* was designed and developed by **Randy Davila**, with contributions from **Jillian Eddy**, and introduces new capabilities for discovery and interaction:

- Symbolic logic for properties, predicates, and conjectures
- Linear and geometric inequality generation via LP and convex hulls
- Heuristic filtering and aesthetic ranking
- Interactive exploration, export to Lean 4, and educational demos

Let your machine dream. Let it write mathematics.


------------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Key Features

   key_features/logic/index
   key_features/generators/index
   key_features/heuristics/index
   key_features/playground/index

.. toctree::
   :maxdepth: 1
   :caption: Automated Conjecturing

   history_with_txgraffiti/history
   history_with_txgraffiti/wangs_program_ii/index
   history_with_txgraffiti/graffiti/index
   history_with_txgraffiti/graffiti_pc/index
   history_with_txgraffiti/graphedron/index

----------------

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/txgraffiti.generators
   api/txgraffiti.heuristics
   api/txgraffiti.logic
   api/txgraffiti.playground
   api/txgraffiti.processing
   api/txgraffiti.export_utils
   api/txgraffiti.example_data

Learn More
-------------

- 🌐 `Interactive Web App <https://txgraffiti.streamlit.app>`_
- 📄 `Papers on arXiv <https://arxiv.org/search/?query=txgraffiti&searchtype=all>`_

---

*Happy conjecturing.*
