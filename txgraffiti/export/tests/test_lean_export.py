import pandas as pd
import pytest

from txgraffiti.export import conjecture_to_lean

# Dummy Conjecture with .name only, to avoid full dependency in the test
class DummyConj:
    def __init__(self, name): self.name = name

@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "degree":   [3, 4],
            "order":    [10, 12],
            "connected": [True, False],
            "zero_forcing_number": [4, 5],
            # 'name' column is intentionally included and must be skipped
            "name": ["G1", "G2"],
        }
    )

def test_basic_implication(df):
    conj = DummyConj("degree ≥ 3 → zero_forcing_number ≤ order / 2")
    lean = conjecture_to_lean(conj, df)
    assert lean == (
        "∀ G : SimpleGraph V, degree G ≥ 3 → zero_forcing_number G ≤ order G / 2"
    )

def test_and_or_not(df):
    conj = DummyConj("(connected) ∧ (degree ≥ 3) → ¬(zero_forcing_number > order)")
    lean = conjecture_to_lean(conj, df)
    assert lean == (
        "∀ G : SimpleGraph V, (connected G) ∧ (degree G ≥ 3) → ¬(zero_forcing_number G > order G)"
    )

def test_theorem_wrapper(df):
    conj = DummyConj("degree ≥ 2 → connected")
    lean = conjecture_to_lean(conj, df, theorem_name="deg_two_connected")
    assert lean.startswith("theorem deg_two_connected :")
    assert "degree G ≥ 2 → connected G" in lean
