# TxGraffiti: Automated Conjecture Generation Library for Python

**TxGraffiti** is a Python package for building, evaluating, and refining mathematical conjectures over tabular data (e.g., graph invariants). It provides a clean, composable API for:

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

## Installation

```bash
pip install txgraffiti  # coming soon
# or
git clone https://github.com/RandyRDavila/txgraffiti2.git
cd txgraffiti2
pip install -e .
```

---

## Quickstart

Below is a minimal example of using `txgraffiti` on a built in dataset of precomputed values on simple, connected, and nontrivial graphs.

```python
from txgraffiti.playground    import ConjecturePlayground # the main class for finding conjectures
from txgraffiti.generators    import convex_hull, linear_programming, ratios # methods for producing inequalities
from txgraffiti.heuristics    import morgan, dalmatian # heuristics to reduce number of statements accepted.
from txgraffiti.processing    import remove_duplicates, sort_by_touch_count # post processing for removal and sorting of conjectures.
from txgraffiti.example_data  import graph_data   # bundled toy dataset

# 2) Instantiate your playground
#    object_symbol will be used when you pretty-print "∀ G.connected: …"
ai = ConjecturePlayground(
    graph_data,
    object_symbol='G'
)

# 3) (Optional) define any custom predicates
regular = (ai.max_degree == ai.min_degree)
cubic   = regular & (ai.max_degree == 3)

# 4) Run discovery
ai.discover(
    methods         = [convex_hull, linear_programming, ratios],
    features        = ['order', 'matching_number', 'min_degree'],
    target          = 'independence_number',
    hypothesis      = [ai.connected & ai.bipartite,
                       ai.connected & regular],
    heuristics      = [morgan, dalmatian],
    post_processors = [remove_duplicates, sort_by_touch_count],
)

# 5) Print your top conjectures
for idx, conj in enumerate(ai.conjectures[:10], start=1):
    # wrap in ∀-notation for readability and conversion to Lean4
    formula = ai.forall(conj)
    print(f"Conjecture {idx}. {formula}\n")
```

The output of the above code should look something like the following:

```bash
Conjecture 1. ∀ G: ((connected) ∧ (bipartite)) → (independence_number == ((-1 * matching_number) + order))

Conjecture 2. ∀ G: ((connected) ∧ (max_degree == min_degree) ∧ (bipartite)) → (independence_number == matching_number)
```

## Testing

Run the existing pytest suite:

```bash
pytest -q
```

Contributions, issues, and suggestions are very welcome! See [CONTRIBUTING.md](/CONTRIBUTING.md) for guidelines.

---

© 2025 Randy Davila and Jillian Eddy. Licensed under MIT.
