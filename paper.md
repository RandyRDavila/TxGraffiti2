---
title: 'TxGraffiti: A Python Package for Symbolic Conjecture Generation and Automated Mathematical Discovery'
tags:
  - Python
  - automated conjecturing
  - symbolic logic
  - optimization
  - mathematical discovery
authors:
  - name: Randy Davila
    orcid: 0000-0002-9908-3760
    affiliation: "1, 2"
  - name: Jillian Eddy
    orcid: 0000-0003-3645-928X
    affiliation: "3"
affiliations:
 - name: Department of Computational Applied Mathematics & Operations Research, Rice University, United States
   index: 1
 - name: RelationalAI, United States
   index: 2
 - name: University of California, Davis, United States
   index: 3
date: 7 May 2025
bibliography: paper.bib
---


# Summary

**TxGraffiti** is a Python package for symbolic conjecture generation and automated discovery in mathematics. The system produces logical implications—typically inequalities—relating numerical and Boolean invariants of structured objects such as graphs, integers, or datasets. Built on foundations laid by earlier systems like *Graffiti*, *Graffiti.pc*, *CONJECTURING*, and *GraPHedron*, TxGraffiti reimagines the conjecturing process with a fully symbolic architecture, recursive logic construction, and modern optimization methods.

At its core, TxGraffiti represents mathematical properties as first-class symbolic expressions and uses logical inference, linear programming, and heuristic evaluation to generate conjectures with minimal assumptions. It supports filtering by correctness, sharpness, generality, and structural significance—prioritizing conjectures with mathematical relevance. Generated conjectures can be exported to formal languages such as Lean 4 for verification.

TxGraffiti is general-purpose and extensible. Though often applied to graph theory, it supports arbitrary domains expressible via tabular data (e.g., Pandas DataFrames), enabling usage across combinatorics, number theory, and symbolic machine learning. It is actively used in mathematical research and AI pipelines for conjecture generation, hypothesis mining, and symbolic pattern discovery.

# Statement of Need

Researchers in mathematics and symbolic reasoning frequently seek meaningful relationships among structural properties—such as inequalities between graph invariants, or predicates defining new object classes. While previous systems have shown that automated tools can assist in conjecture generation, they typically lack symbolic abstraction, logic composability, extensible data interfaces, formal output formats, and modern usability.

**TxGraffiti** addresses these limitations by providing:

- A symbolic, logic-based representation of mathematical properties and conjectures,
- Integration with Pandas for rapid experimentation on tabular datasets,
- Mixed-integer programming backends for bounding inequalities over datasets,
- Heuristic filters for prioritizing general, sharp, and novel results,
- Export capabilities to Lean 4 and other formal systems,
- A modular, extensible design with full unit testing and documentation.

TxGraffiti is designed for researchers working at the intersection of mathematics, symbolic AI, and programmatic discovery. It lowers the barrier to experimentation with conjecture generation and supports integration into both research workflows and educational tools.

# Design and Implementation

TxGraffiti is built around a symbolic framework for expressing and evaluating mathematical conjectures. The system supports programmatic reasoning over numeric and Boolean invariants and generates conjectures in the form of logical implications that can be optimized, filtered, and formally verified.

### Symbolic Expression Layer

At the foundation of TxGraffiti is the `Property` class, which lifts columns of tabular data (e.g., a graph invariant such as independence number or average degree) into symbolic numeric expressions. These `Property` objects support arithmetic operations, symbolic comparison, and string representations suitable for display or formal translation.

Boolean-valued expressions are encapsulated as `Predicate` objects, which can be logically combined using operators like `∧`, `∨`, and `¬`. Predicates may represent simple class membership (e.g., “connected graph”) or complex logical conditions (e.g., “connected ∧ ¬tree”).

### Conjecture Representation

Conjectures in TxGraffiti are modeled as symbolic logical implications of the form:

```bash
(hypothesis) → (predicate)
```

Typically, the predicate is an inequality between two symbolic expressions. The `Conjecture` class encodes these implications and tracks rich metadata, including:

- A hypothesis `Predicate` (or `True` for universally quantified conjectures),
- A symbolic inequality between `Property` objects,
- Metadata such as equality cases, sharpness, significance, and touched objects.

Conjectures can be printed, ranked, exported to formal systems, or evaluated on datasets to check correctness and collect evidence.

### Discovery Pipeline

Conjecture discovery is organized into a modular pipeline:

- **Inequality Generators** (`txgraffiti.generators`) propose candidate inequalities between symbolic expressions. These include:
  - `ratios`: compares ratios of expressions,
  - `convex_hull`: mines bounding inequalities from convex geometry,
  - `linear`: uses mixed-integer programming to optimize bounds.

- **Acceptance Heuristics** (`txgraffiti.heuristics`) evaluate and filter generated conjectures. Notable heuristics include:
  - `morgan_accept`: favors conjectures with generality,
  - `dalmatian_accept` and `calloway`: perform correctness checks, coverage analysis, and local sharpness scoring.

- **Playground Interface** (`txgraffiti.playground.ConjecturePlayground`) provides a high-level interface for experimentation. It enables users to apply generators and heuristics to datasets with minimal boilerplate and retrieve conjectures ranked by strength or novelty.

### Integration and Extensibility

TxGraffiti is designed to be general-purpose and modular:

- Operates directly on Pandas DataFrames,
- Supports any domain expressible via tabular invariants (e.g., graphs, integers, sports statistics),
- Conjectures can be exported to **Lean 4** for formal verification using `txgraffiti.export_utils`,
- Users can define custom properties, predicates, generators, or heuristics by subclassing existing components.

All major components are fully documented and tested. The package includes continuous integration workflows for PyPI publication and ReadTheDocs deployment, ensuring stable releases and accessible documentation.

# Example Usage

TxGraffiti is designed to make automated conjecture generation accessible and expressive for mathematical researchers and symbolic AI systems. The core discovery interface is the `ConjecturePlayground`, which allows users to pose conjecturing tasks over any dataset of structured objects and their associated invariants.

Below is a minimal working example that discovers conjectures about the independence number of graphs from a bundled dataset of precomputed invariants.

```python
from txgraffiti.playground    import ConjecturePlayground
from txgraffiti.generators    import convex_hull, ratios
from txgraffiti.heuristics    import morgan_accept, dalmatian_accept
from txgraffiti.processing    import remove_duplicates, sort_by_touch_count
from txgraffiti.example_data  import graph_data

# Step 1. Initialize the conjecture engine on a dataset of graphs
ai = ConjecturePlayground(
    graph_data,
    object_symbol='G'  # used for printing ∀ G: ...
)

# Step 2. Define optional structural predicates
regular = (ai.max_degree == ai.min_degree)
cubic   = regular & (ai.max_degree == 3)

# Step 3. Run discovery using convex geometry and ratio-based generators
ai.discover(
    methods         = [convex_hull, ratios],
    features        = ['order', 'matching_number', 'min_degree'],
    target          = 'independence_number',
    hypothesis      = [ai.connected & ai.bipartite, ai.connected & regular],
    heuristics      = [morgan_accept, dalmatian_accept],
    post_processors = [remove_duplicates, sort_by_touch_count],
)

# Step 4. Output the top conjectures
for idx, conj in enumerate(ai.conjectures[:2], start=1):
    print(f"Conjecture {idx}. {ai.forall(conj)}\n")
```

The output of this code should look like:

```bash
Conjecture 1. ∀ G: ((connected) ∧ (bipartite)) → (independence_number == ((-1 * matching_number) + order))

Conjecture 2. ∀ G: ((connected) ∧ (max_degree == min_degree) ∧ (bipartite)) → (independence_number == matching_number)
```

Additional examples—including applications to number theory, real-world datasets, and recursive conjecture pipelines—are available in the [TxGraffiti README](https://github.com/randydavila/txgraffiti) and [online documentation](https://txgraffiti2.readthedocs.io). These examples illustrate how the symbolic core generalizes beyond graph theory to support rich, domain-agnostic mathematical discovery.

# Validation and Testing

TxGraffiti includes a comprehensive suite of unit and integration tests covering core symbolic logic components (`Property`, `Predicate`, `Conjecture`), discovery modules (`generators`, `heuristics`), and data interfaces. The package is continuously tested using GitHub Actions against multiple Python versions to ensure correctness and stability.

Tests are written using `pytest` and cover:

- Logical correctness of symbolic expression evaluation,
- Generator output validity across varied datasets,
- Heuristic acceptance criteria and performance,
- End-to-end discovery pipelines via the `ConjecturePlayground`.

In addition, all conjectures generated during testing are validated against real datasets to ensure logical soundness. Documentation builds and distribution workflows are also validated via continuous integration.
