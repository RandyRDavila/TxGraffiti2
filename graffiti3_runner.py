#!/usr/bin/env python
"""
graffiti4_runner.py

Minimal runner for the Graffiti4 workspace on the example graph dataset.

Usage (from project root):

    PYTHONPATH=src python graffiti4_runner.py
"""

from __future__ import annotations

import pandas as pd

# at the top of graffiti4.py
from txgraffiti.graffiti3.heuristics.morgan import morgan_filter#, dalmatian_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter
from txgraffiti.graffiti3.graffiti3 import Graffiti3, print_g3_result
from txgraffiti.example_data import graph_data as df

df["nontrivial"] = df["connected"]
df.drop(
    columns=[
        "vertex_cover_number",
        "cograph",
        "cubic",
        "chordal",
        "tree",
        "size",
        "triameter",
    ],
    inplace=True,
)

# df = pd.read_csv('polytope_data.csv')
# df.drop(columns=['Unnamed: 0'], inplace=True)

g3 = Graffiti3(
    df,
    max_boolean_arity=2,
    morgan_filter=morgan_filter,
    dalmatian_filter=dalmatian_filter,
    sophie_cfg=dict(
        eq_tol=1e-4,
        min_target_support=5,
        min_h_support=3,
        max_violations=0,
        min_new_coverage=1,
    ),
)

result = g3.conjecture(
    target="independence_number",
    complexity=1,
    include_invariant_products=True,
    include_abs=True,
    include_min_max=True,
)

print_g3_result(result)
