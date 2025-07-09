Wang's Program II (1959)
========================

This page recreates the logic-based conjecturing process of Wang's early AI system using `txgraffiti`'s symbolic logic layer.

Wang’s system generated thousands of logical formulas in propositional logic and checked whether they were tautologies. We simulate that below using `Predicate` and `Conjecture`.

.. code-block:: python

    import pandas as pd
    from txgraffiti.logic.conjecture_logic import Predicate, Conjecture, TRUE, FALSE

    # Step 1: Simulate a truth table over 3 variables: A, B, C
    from itertools import product

    rows = list(product([False, True], repeat=3))
    df = pd.DataFrame(rows, columns=["A", "B", "C"])

    # Step 2: Create symbolic predicates
    A = Predicate("A", lambda df: df["A"])
    B = Predicate("B", lambda df: df["B"])
    C = Predicate("C", lambda df: df["C"])

    # Step 3: Create and test some formulas
    formulas = [
        A >> A,
        A >> B,
        A >> (A | B),
        (A & B) >> C,
        (A & ~A),
        (A | ~A),
        (A & B) >> (A | B),
        ((A >> B) & (B >> C)) >> (A >> C),  # hypothetical syllogism
    ]

    # Step 4: Evaluate and print which are tautologies
    for f in formulas:
        print(f"{f}  ::  Tautology?  {f.is_true(df)}")
