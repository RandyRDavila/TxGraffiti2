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
from txgraffiti.graffiti3.graffiti3 import Graffiti3, print_g3_result, Stage
from txgraffiti.example_data import graph_data as df

df["nontrivial"] = df["connected"]
# df.drop(
#     columns=[
#         "vertex_cover_number",
#         "cograph",
#         "cubic",
#         "chordal",
#         "tree",
#         "size",
#         "triameter",
#         "K_4_free",
#         "triangle_free",
#         "claw_free",
#         "subcubic",
#         "regular",
#         "planar",
#     ],
#     inplace=True,
# )

# df["cycle"] = df["regular"] & (df["maximum_degree"] == 2)
# df["path"] = (df["minimum_degree"] == 1) & (df["maximum_degree"] == 2)
# df["complete_graph"] = df["independence_number"] == 1
# df["order > 3"] = df["order"] > 3

df.drop(
    columns=[
        "vertex_cover_number",
        "cograph",
        "cubic",
        "chordal",
        # "tree",
        "size",
        # "triameter",
        # "K_4_free",
        # "triangle_free",
        # "claw_free",
        # "subcubic",
        # "regular",
        # "planar",
        # "bipartite",
        "eulerian",
    ],
    inplace=True,
)


# df = df.iloc[:101]

# df = pd.read_csv('polytope_data_full.csv')
# df.drop(columns=['Unnamed: 0', 'p6_odd', 'p3_odd', 'p4_odd', 'p5_odd'], inplace=True)

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


STAGES = [
    # Stage.CONSTANT,
    Stage.RATIO,
    # Stage.LP1,
    # Stage.LP,
    # Stage.LP3,
    # Stage.POLY_SINGLE,
    Stage.MIXED,
]

TARGETS = [
        # "zero_forcing_number",
        # "spectral_radius",
        "independence_number",
        # "domination_number",
        # "total_domination_number"
    ]

result = g3.list_conjecture(
    targets=TARGETS,
    stages=STAGES,
    include_invariant_products=True,
    include_abs=False,
    include_min_max=False,
    include_log=False,
    enable_sophie=True,
    sophie_stages=[Stage.CONSTANT, Stage.RATIO],
    quick=True,
)

print_g3_result(result, k_conjectures=10, k_sophie=10)
