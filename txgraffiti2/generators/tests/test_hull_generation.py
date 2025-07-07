import pandas as pd
import pytest

from txgraffiti2.logic.conjecture_logic import Property, Predicate, Inequality, Conjecture
from txgraffiti2.generators.hull_generators import generate_hull_conjectures

@pytest.fixture
def df():
    # Example DataFrame (3D cube vertices)
    return pd.DataFrame({
    'is a point in the cube': [True, True, True, True, True, True, True, True],
    'feature1': [0,0,0,0,1,1,1,1],
    'feature2': [0,0,1,1,0,1,0,1],
    'target': [0,1,0,1,0,0,1,1],
    })

# def test_generate_hull_conjectures(df):
#     # Test the hull conjecture generation

#     features = [Property('feature1', lambda df: df['feature1']),
#             Property('feature2', lambda df: df['feature2']),
#             ]
#     target = Property('target', lambda df: df['target'])
#     hypothesis = Property('is a point in the cube', lambda df: df['is a point in the cube'])

#     conjectures = generate_hull_conjectures(df, features, target, hypothesis)
#     assert len(conjectures) > 0, "Expected at least one conjecture to be generated"

#     # Check the conjectures are of the expected form
#     good_conjectures = ['<Conjecture (is a point in the cube) → (target ≥ (0))', '(is a point in the cube) → (target ≤ (1)) >']

#     for conj in conjectures:
#         assert repr(conj) in good_conjectures, f"Unexpected conjecture: {repr(conj)}"
