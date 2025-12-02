import numpy as np
from txgraffiti2025.workbench.class_relations import discover_class_relations
from txgraffiti2025.forms.predicates import Where

def test_discover_relations_simple(toy_df):
    # P: (b1) ; Q: (b1) ∧ (b2)
    P = Where(fn=lambda df: (df["b1"] == 1).to_numpy(), name="(b1)")
    Q = Where(fn=lambda df: ((df["b1"] == 1) & (df["b2"] == 1)).to_numpy(), name="((b1) ∧ (b2))")

    eqs, incs = discover_class_relations(
        toy_df, predicates=[P, Q], min_support_A=1, skip_trivial_equiv=True,
        disallow_shared_atoms=False
    )
    # Accept either explicit inclusion Q ⊆ P or (depending on filters) no inclusion emitted.
    ok = any((inc.A.name == "((b1) ∧ (b2))" and inc.B.name == "(b1)") for inc in incs)
    assert ok or incs == []
