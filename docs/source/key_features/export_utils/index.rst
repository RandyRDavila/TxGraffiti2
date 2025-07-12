.. _export_utils:

Exporting
===================

Overview
--------

The **`txgraffiti.export_utils.lean4`** module provides tools to translate
TxGraffiti `Conjecture` objects into Lean 4 theorem stubs or propositions.
It includes:

- **`LEAN_SYMBOLS`**
  A mapping from common mathematical and Python operators to their Lean 4
  equivalents (e.g. `">=" → "≥"`, `"==" → "="`).

- **`auto_var_map(df, *, skip=('name',))`**
  Build a variable-to-binder map for Lean.  By default it returns
  `{col: "<col> G"}` for every column in `df` except those in `skip`.

- **`conjecture_to_lean4(conj, name, object_symbol='G', object_decl='SimpleGraph V')`**
  Convert a single `Conjecture` into a Lean 4 `theorem` stub with explicit
  hypotheses and a `sorry` proof.

Examples
--------

Low-Level Translation
+++++++++++++++++++++

.. code-block:: python

    import pandas as pd
    from txgraffiti.logic import Property, Predicate
    from txgraffiti.export_utils.lean4 import auto_var_map, conjecture_to_lean4

    # 1) Prepare a small DataFrame
    df = pd.DataFrame({
        'alpha':     [1, 2, 3],
        'beta':      [3, 1, 1],
        'connected': [True, True, True],
    })

    # 2) Lift into TxGraffiti objects
    A = Property('alpha',     lambda df: df['alpha'])
    B = Property('beta',      lambda df: df['beta'])
    H = Predicate('connected', lambda df: df['connected'])

    # 3) Build a simple conjecture: connected → (alpha ≤ beta)
    conj = H >> (A <= B)

    # 4) Auto‐generate the Lean variable map
    vm = auto_var_map(df)
    print(vm)
    # {'alpha': 'alpha G', 'beta': 'beta G', 'connected': 'connected G'}

    # 5) Translate to a Lean 4 theorem stub
    lean_code = conjecture_to_lean4(
        conj,
        name="alpha_leq_beta",
        object_symbol="G",
        object_decl="SimpleGraph V"
    )
    print(lean_code)

    # Output:
    # theorem alpha_leq_beta (G : SimpleGraph V)
    #     (h1 : connected G) : alpha G ≤ beta G :=
    # sorry


Using ConjecturePlayground.export_to_lean
+++++++++++++++++++++++++++++++++++++++++

You can also export *all* discovered conjectures from a playground session
directly to a `.lean` file:

.. code-block:: python

    from txgraffiti.playground import ConjecturePlayground
    from txgraffiti.generators.ratios import ratios
    from txgraffiti.heuristics.fajtlowicz import dalmatian_accept

    # Wrap your DataFrame (or KnowledgeTable) in a playground
    pg = ConjecturePlayground(df, object_symbol="G", base=H)

    # Discover conjectures via the 'ratios' generator and the Dalmatian heuristic
    pg.discover(
        methods    =[ratios],
        features   =[B],
        target     =A,
        hypothesis =H,
        heuristics =[dalmatian_accept],
    )

    # Export all cached conjectures into 'conjs.lean' with names 'txg_1', 'txg_2', ...
    pg.export_to_lean(
        path="conjs.lean",
        name_prefix="txg",
        object_symbol="G"
    )
    # Writes N theorem stubs to conjs.lean
