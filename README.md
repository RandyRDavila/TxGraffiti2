# TxGraffiti: Automated Conjecture Generation in Python

[![PyPI version](https://img.shields.io/pypi/v/txgraffiti.svg)](https://pypi.org/project/txgraffiti/)
[![Documentation Status](https://readthedocs.org/projects/txgraffiti2/badge/?version=latest)](https://txgraffiti2.readthedocs.io/en/latest/)
[![Build Status](https://github.com/RandyRDavila/TxGraffiti2/actions/workflows/python-ci.yml/badge.svg)](https://github.com/RandyRDavila/TxGraffiti2/actions)
[![License](https://img.shields.io/github/license/RandyRDavila/TxGraffiti2)](LICENSE)

---

**TxGraffiti** is a Python library for building, evaluating, and discovering mathematical conjectures from structured data—particularly graph invariants and number-theoretic quantities.

Inspired by the original *Graffiti* program of Siemion Fajtlowicz, this package automates the creative mathematical process using a combination of symbolic logic, optimization, heuristics, and postprocessing.

---

## Features

- Work with **properties** (numeric features), **predicates** (boolean tests), and **inequalities**
- Automatically **generate conjectures** using convex hull, LP, and ratio methods
- Apply **heuristics** to reduce noise and prioritize meaningful conjectures
- Compose logical hypotheses and filter conjectures by truth and significance
- Use built-in datasets on graphs and integers, or plug in your own
- Export results to Lean4, search for counterexamples, and iterate

---

## Installation

Install from PyPI:

```bash
pip install txgraffiti
```

To install from source:

```bash
git clone https://github.com/RandyRDavila/TxGraffiti2.git
cd TxGraffiti2
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## Example: Graph Theory Conjectures

Below is a minimal example of using `txgraffiti` on a built in dataset of precomputed values on simple, connected, and nontrivial graphs.

```python
from txgraffiti.playground    import ConjecturePlayground # the main class for finding conjectures
from txgraffiti.generators    import convex_hull, linear_programming, ratios # methods for producing inequalities
from txgraffiti.heuristics    import morgan_acceptance, dalmatian_acceptance # heuristics to reduce number of statements accepted.
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
    heuristics      = [morgan_acceptance, dalmatian_acceptance],
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

## Example: Integer Dataset

Next, we conjecture on the built in integer dataset.

```python
from txgraffiti.playground    import ConjecturePlayground
from txgraffiti.generators    import convex_hull, linear_programming, ratios
from txgraffiti.heuristics    import morgan_acceptance, dalmatian_acceptance
from txgraffiti.processing    import remove_duplicates, sort_by_touch_count
from txgraffiti.example_data  import integer_data   # bundled toy dataset

# 2) Instantiate your playground
#    object_symbol will be used when you pretty-print "∀ G.connected: …"
ai = ConjecturePlayground(
    integer_data,
    object_symbol='n.PositiveInteger'
)

ai.discover(
    methods         = [convex_hull, linear_programming, ratios],
    features        = ['sum_divisors', 'divisor_count', 'totient', 'prime_factor_count'],
    target          = 'collatz_steps',
    hypothesis      = [ai.is_square, ai.is_fibonacci, ai.is_power_of_two],
    heuristics      = [morgan_acceptance, dalmatian_acceptance],
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
