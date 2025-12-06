# In the terminal run the command: PYTHONPATH=src python demo/polytope_demo.py
from __future__ import annotations

import pandas as pd

# at the top of graffiti4.py
from txgraffiti.graffiti3.heuristics.morgan import morgan_filter#, dalmatian_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter
from txgraffiti.graffiti3.graffiti3 import Graffiti3, print_g3_result, Stage
from txgraffiti.example_data import polytope_data as df




df.drop(columns=['temperature(p6)', 'p4_odd', 'p5_odd', 'p3_odd', ], inplace=True)


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
    Stage.CONSTANT,
    Stage.RATIO,
    Stage.LP1,
    Stage.LP2,
    Stage.LP3,
    Stage.LP4,
    Stage.POLY_SINGLE,
    Stage.MIXED,
    Stage.SQRT,
    Stage.LOG,
    Stage.SQRT_LOG,
    Stage.GEOM_MEAN,
    Stage.LOG_SUM,
    Stage.SQRT_PAIR,
    Stage.SQRT_SUM,
    Stage.EXP_EXPONENT,

]

# Target invariants to conjecture on: p5 and p6.
TARGETS = [
        "p5",
        "p6",
    ]

# Conjecture on the target invariants using the stages defined above.
result = g3.conjecture(
    targets=TARGETS,
    stages=STAGES,
    include_invariant_products=False,
    include_abs=False,
    include_min_max=False,
    include_log=False,
    enable_sophie=True,
    sophie_stages=STAGES,
    quick=True,
    show=True,
)
