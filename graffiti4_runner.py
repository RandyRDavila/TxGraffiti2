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
from txgraffiti2025.graffiti4_heuristics import morgan_filter#, dalmatian_filter
from txgraffiti2025.graffiti4_dalmatian import dalmatian_filter
from txgraffiti2025.graffiti4 import Graffiti4, print_g4_result
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

g4 = Graffiti4(
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

result = g4.conjecture(
    target="independence_number",
    complexity=3,
    include_invariant_products=False,
    include_abs=False,
    include_min_max=False,
)

print_g4_result(result)
