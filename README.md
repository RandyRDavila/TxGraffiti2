# TxGraffiti: Automated Conjecture Generation in Python

[![PyPI version](https://img.shields.io/pypi/v/txgraffiti.svg)](https://pypi.org/project/txgraffiti/)
[![Documentation Status](https://readthedocs.org/projects/txgraffiti2/badge/?version=latest)](https://txgraffiti2.readthedocs.io/en/latest/)
[![Build Status](https://github.com/RandyRDavila/TxGraffiti2/actions/workflows/ci.yml/badge.svg)](https://github.com/RandyRDavila/TxGraffiti2/actions)
[![License](https://img.shields.io/github/license/RandyRDavila/TxGraffiti2)](LICENSE)
[![codecov](https://codecov.io/gh/RandyRDavila/txgraffiti2/branch/main/graph/badge.svg)](https://codecov.io/gh/RandyRDavila/txgraffiti2)

---

**TxGraffiti** is a Python package for **automated mathematical conjecture generation**.

It uncovers patterns, equalities, and inequalities in structured datasets by forming symbolic expressions and proposing data-backed conjectures. While originally developed to explore graph-theoretic invariants, TxGraffiti is domain-agnostic and can be applied to any tabular data where mathematical relationships may be discovered.

Built on principles from the Graffiti family of programs, **TxGraffiti** blends logic, optimization, and heuristics to create meaningful, testable mathematical statements. It is designed for:

- 📐 Mathematicians exploring new bounds and relationships
- 📊 Data scientists modeling symbolic structure in tabular data
- 🤖 AI researchers studying machine-driven discovery
- 📚 Educators demonstrating the intersection of math and computation

The system combines symbolic logic, heuristic filtering, and optimization techniques to produce clear, interpretable conjectures—making it a powerful tool for researchers, educators, and AI-assisted discovery.

---

## Features

- Work with **properties** (numeric features), **predicates** (boolean tests), and **inequalities**
- Automatically **generate conjectures** using convex hull, LP, and ratio methods
- Apply **heuristics** to reduce noise and prioritize meaningful conjectures
- Compose logical hypotheses and filter conjectures by truth and significance
- Use built-in datasets on graphs and integers, or plug in your own
- Export results to Lean4, search for counterexamples, and iterate

---

## 📦 Installation

Install the latest release from PyPI:

```bash
pip install txgraffiti
```

To install the development version from source:

```bash
git clone https://github.com/RandyRDavila/TxGraffiti2.git
cd TxGraffiti2

# Optional: create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows, use: .venv\Scripts\activate

# Install the package in editable mode with development dependencies
pip install -e .[dev]
```

TxGraffiti requires Python 3.8 or later.

---

## Example: Graph Theory Conjectures

Below is a minimal example of using `txgraffiti` on a built in dataset of precomputed values on simple, connected, and nontrivial graphs.

```python
from txgraffiti.playground    import ConjecturePlayground  # main interface for discovery
from txgraffiti.generators    import convex_hull, ratios
from txgraffiti.heuristics    import morgan_accept, dalmatian_accept
from txgraffiti.processing    import remove_duplicates, sort_by_touch_count
from txgraffiti.example_data  import graph_data            # bundled toy dataset

# 1. Instantiate your playground
ai = ConjecturePlayground(
    graph_data,
    object_symbol='G'  # used in pretty-printing: ∀ G: ...
)

# 2. (Optional) Define custom predicates
regular = (ai.max_degree == ai.min_degree)
cubic   = regular & (ai.max_degree == 3)

# 3. Run conjecture discovery
ai.discover(
    methods         = [convex_hull, ratios],
    features        = ['order', 'matching_number', 'min_degree'],
    target          = 'independence_number',
    hypothesis      = [ai.connected & ai.bipartite,
                       ai.connected & regular],
    heuristics      = [morgan_accept, dalmatian_accept],
    post_processors = [remove_duplicates, sort_by_touch_count],
)

# 4. Print your top conjectures
for idx, conj in enumerate(ai.conjectures[:10], start=1):
    print(f"Conjecture {idx}. {ai.forall(conj)}\n")

```

The output of the above code should look something like the following:

```bash
Conjecture 1. ∀ G: ((connected) ∧ (bipartite)) → (independence_number == ((-1 * matching_number) + order))

Conjecture 2. ∀ G: ((connected) ∧ (max_degree == min_degree) ∧ (bipartite)) → (independence_number == matching_number)
```

## Example: Integer Dataset

Next, we conjecture on the built in integer dataset.

```python
from txgraffiti.playground    import ConjecturePlayground
from txgraffiti.generators    import convex_hull, ratios
from txgraffiti.heuristics    import morgan_accept, dalmatian_accept
from txgraffiti.processing    import remove_duplicates, sort_by_touch_count
from txgraffiti.example_data  import integer_data   # bundled toy dataset

# 2) Instantiate your playground
#    object_symbol will be used when you pretty-print "∀ G.connected: …"
ai = ConjecturePlayground(
    integer_data,
    object_symbol='n.PositiveInteger'
)

ai.discover(
    methods         = [convex_hull, ratios],
    features        = ['sum_divisors', 'divisor_count', 'totient', 'prime_factor_count'],
    target          = 'collatz_steps',
    hypothesis      = [ai.is_square, ai.is_fibonacci, ai.is_power_of_two],
    heuristics      = [morgan_accept, dalmatian_accept],
    post_processors = [remove_duplicates, sort_by_touch_count],
)

# 5) Print your top conjectures
for idx, conj in enumerate(ai.conjectures[:10], start=1):
    # wrap in ∀-notation for readability
    formula = ai.forall(conj)
    print(f"Conjecture {idx}. {formula}\n")
```

The output of the above code should look something like the following:

```bash
Conjecture 1. ∀ n.PositiveInteger: ((is_power_of_two) ∧ (is_fibonacci)) → (collatz_steps == prime_factor_count)

Conjecture 2. ∀ n.PositiveInteger: (is_square) → (collatz_steps >= (((17/8 * divisor_count) + -17/8) + (-9/8 * prime_factor_count)))

Conjecture 3. ∀ n.PositiveInteger: (is_square) → (collatz_steps <= (((((-17/10 * sum_divisors) + -391/8) + (1887/40 * divisor_count)) + (34/5 * totient)) + (-1847/40 * prime_factor_count)))

Conjecture 4. ∀ n.PositiveInteger: (is_power_of_two) → (collatz_steps <= prime_factor_count)

Conjecture 5. ∀ n.PositiveInteger: (is_square) → (collatz_steps >= prime_factor_count)

Conjecture 6. ∀ n.PositiveInteger: (is_fibonacci) → (collatz_steps >= prime_factor_count)
```

## Testing

Run the existing pytest suite:

```bash
pytest
```

## Contributing

Contributions, ideas, and suggestions are welcome!
To get involved:

1. Fork the repository
2. Create a new branch
3. Submit a pull request

See [CONTRIBUTING.md](/CONTRIBUTING.md) for details.

---

## License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.

---

## Authors

- Randy Davila, PhD – Lead developer

- Jillian Eddy – Co-developer, logic design
