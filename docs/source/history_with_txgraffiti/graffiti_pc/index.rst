.. _sophie:

DeLaViña's Sophie Heuristic
=====================================

Overview
--------

The **Sophie heuristic** was developed by Ermilinda DeLaViña and Bill Waller
and first appeared in DeLaViña’s *Graffiti.pc*. It governs which new
conjectures to keep in a discovery pipeline by examining the **cover set**
of each hypothesis: a conjecture is accepted only if it “covers” at least
one data point not already covered by any previously accepted conjecture.

How It Works
------------

1. **Cover set**
   For a conjecture ``c`` with hypothesis predicate ``H``, its cover set on
   a pandas DataFrame ``df`` is the Boolean mask:

   .. code-block:: python

       cover_c = c.hypothesis(df)

2. **Union of existing covers**
   Given a list ``accepted`` of conjectures already kept, form their union:

   .. code-block:: python

       if accepted:
           old_union = pd.concat(
               [c.hypothesis(df) for c in accepted],
               axis=1
           ).any(axis=1)
       else:
           old_union = pd.Series(False, index=df.index)

3. **Acceptance rule**
   Accept the new conjecture ``c_new`` if and only if there exists at least
   one row where ``c_new`` covers a point not in ``old_union``:

   .. code-block:: python

       new_cover = c_new.hypothesis(df)
       delta     = new_cover & ~old_union
       accept    = bool(delta.any())


Example
-------

Here is a complete example demonstrating Sophie in action on a small dataset:

.. code-block:: python

    import pandas as pd
    from txgraffiti.logic import Property, Predicate
    from txgraffiti.heuristics import sophie_accept

    # 1) Sample data
    df = pd.DataFrame({
        'alpha': [1, 2, 3, 4],
        'p':     [True, True, False, False],
        'q':     [False, True, True,  False],
    })

    # 2) Lift into TxGraffiti objects
    A = Property('alpha', lambda df: df['alpha'])
    P = Predicate('p',     lambda df: df['p'])
    Q = Predicate('q',     lambda df: df['q'])

    # 3) Build three conjectures
    #    c1 covers rows 0 and 1
    c1 = P >> (A <= 10)
    #    c2 covers rows 1 and 2
    c2 = Q >> (A <= 10)
    #    c3 covers rows 0,1,2 but adds no new row beyond c1,c2
    P_or_Q = P | Q
    c3 = P_or_Q >> (A <= 10)

    # 4) Apply Sophie
    accepted = []

    # c1 is first → covers new rows ⇒ accept
    assert sophie_accept(c1, accepted, df) is True
    accepted.append(c1)

    # c2 adds row 2 beyond c1’s cover ⇒ accept
    assert sophie_accept(c2, accepted, df) is True
    accepted.append(c2)

    # c3 adds no new rows beyond {0,1,2} ⇒ reject
    assert sophie_accept(c3, accepted, df) is False
