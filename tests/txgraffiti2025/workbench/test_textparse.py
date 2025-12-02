import numpy as np
from txgraffiti2025.workbench.textparse import split_top_level_conj, canon_atom, atoms_for_pred, predicate_to_conjunction, predicate_to_if_then
from txgraffiti2025.forms.predicates import Where

def test_split_top_level_conj_handles_parens():
    s = "((A) ∧ (B ∧ (C)))"
    parts = split_top_level_conj(s)
    assert parts == ["(A)", "(B ∧ (C))"]

def test_atoms_for_pred_with_derived_where(toy_df):
    # Build a Where predicate whose 'name' encodes derived form: "[ H ] :: φ"
    H_str  = "(b1) ∧ (b2)"
    phi_str = "(connected)"
    name = f"[ {H_str} ] :: {phi_str}"
    p = Where(fn=lambda df: np.ones(len(df), dtype=bool), name=name)
    atoms = atoms_for_pred(p)
    assert canon_atom("(b1)") in atoms
    assert canon_atom("(b2)") in atoms
    assert canon_atom(phi_str) in atoms

def test_renderers(toy_df):
    name = "[ (A ∧ B) ] :: (C)"
    p = Where(fn=lambda df: np.ones(len(df), dtype=bool), name=name)
    conj = predicate_to_conjunction(p)
    ifthen = predicate_to_if_then(p)
    assert "∧" in conj and "(" in conj and ")" in conj
    assert ifthen.startswith("If ") and "⇒" in ifthen
