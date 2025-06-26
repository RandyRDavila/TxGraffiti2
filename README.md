# TxGraffiti2: Automated Conjecture Generation Library

![Python CI](https://github.com/RandyRDavila/txgraffiti2/actions/workflows/python-ci.yml/badge.svg)

**TxGraffiti2** is a Python package for building, evaluating, and refining mathematical conjectures over tabular data (e.g., graph invariants). It provides a clean, composable API for:

* **Numeric expressions** (`Property`): lift columns or constants into first‑class objects supporting `+, -, *, /, **`.
* **Boolean predicates** (`Predicate`): define row‑wise tests and combine them with `∧`, `∨`, and `¬`.
* **Inequalities** (`Inequality`): compare `Property` objects to generate rich, named predicates, with helper methods for slack and touch counts.
* **Implications** (`Conjecture`): express and verify if a hypothesis implies a conclusion, with methods for accuracy, counterexample extraction, and more.

This repo will evolve into a full framework for:

1. **Conjecture generation** via linear programming and heuristic filtering.
2. **Conjecture ranking** using sharpness, significance, and geometric scores.
3. **Counterexample search** integrated into a feedback loop (Optimist–Pessimist agents).
4. **Dataset management** for known mathematical objects (graphs, polytopes, integers, etc.).
5. **Notebook examples** showcasing end‑to‑end workflows from data to publishable conjectures.

---

## Features

* * Lift pandas columns and constants to symbolic `Property` objects
  * Auto‑simplify identities (e.g. `p + 0 → p`, `p*1 → p`)
* * Combine row‑wise tests with logical operators in `Predicate`
  * Build named, composable boolean expressions
* * Form inequalities between properties with `<=, <, >=, >, ==, !=`
  * Compute `slack`, `touch_count`, and extract counterexamples
* * Package hypotheses and conclusions into `Conjecture` objects
  * Evaluate truth, accuracy, and failing rows automatically

## Installation

```bash
pip install txgraffiti2  # coming soon
# or
git clone https://github.com/RandyRDavila/txgraffiti2.git
cd txgraffiti2
pip install -e .
```

## Quickstart

```python
import pandas as pd
from txgraffiti2.conjecture_logic import Property, Predicate, Conjecture

# sample data of graph invariants
G = pd.DataFrame({
    'alpha': [1,2,3],
    'gamma': [2,3,1],
    'connected': [True,True,False]
})

alpha = Property('alpha', lambda df: df['alpha'])
gamma = Property('gamma', lambda df: df['gamma'])
conn  = Predicate('connected', lambda df: df['connected'])

# conjecture: if connected then alpha <= gamma + 1
eq = Inequality(alpha, '<=', gamma + 1)
conj = Conjecture(conn, alpha <= gamma + 1)
print("True?", conj.is_true(G))
print("Accuracy:", conj.accuracy(G))
print("Counterexamples:\n", conj.counterexamples(G))
```

## Roadmap

* [ ] Conjecture generation module (LP + heuristics)
* [ ] Ranking and filtering heuristics (Dalmatian → Calloway)
* [ ] Integrated counterexample search (Pessimist agent)
* [ ] Curated datasets & example notebooks
* [ ] CLI and Jupyter widgets for interactive exploration

## Testing

Run the existing pytest suite:

```bash
pytest -q
```

Contributions, issues, and suggestions are very welcome! See [CONTRIBUTING.md](/CONTRIBUTING.md) for guidelines.

---

© 2025 Randy Davila and collaborators. Licensed under MIT.
