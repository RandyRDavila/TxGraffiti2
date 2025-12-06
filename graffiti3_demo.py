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
#         "bipartite",
#         "eulerian",
#     ],
#     inplace=True,
# )


# df = df.iloc[:101]

# df = pd.read_csv('polytope_data.csv')
# df.drop(columns=['Unnamed: 0', 'temperature(p6)', 'p4_odd', 'p5_odd', 'p3_odd', ], inplace=True)

df = pd.read_csv('qbits-2.csv')
df.drop(columns=['Unnamed: 0',], inplace=True)
bad_cols = [
    "ansatz_hardware_efficient",
    "ansatz_strongly_entangling",
    "ansatz_convolutional",
    "depth",
    "goal_ce",
    "input_type",
    "sample_idx",
    "high_depth",
    "product_like",
    # "highly_entangled",
    # "weakly_entangled",
    "ent_gap",
    "ent_gap_abs",
    "nontrivial",
    # "n_qubits",
    # "sq_entropy_range",
    "half_entropy",
    "half_entropy_norm",
    "half_purity",
    "sq_entropy_min",
    # "sq_entropy_max",
    "sq_entropy_mean",
    # "sq_entropy_range",
    "mz_mean",
    # "mz_min",
    # "mz_max",
    "mz_abs_mean",
    # "mz_total",
    "mz_abs_total",
]
df.drop(columns=bad_cols, inplace=True)
# One-hot encode input_type as booleans
# input_dummies = pd.get_dummies(df["input_type"], prefix="input", dtype=bool)

# Drop the original string column
# df = pd.concat([df.drop(columns=["input_type"]), input_dummies], axis=1)



DATA_DOC = {
    "data description": (
        "Tabular dataset built from the NTangled_Datasets repository "
        "(GitHub: LSchatzki/NTangled_Datasets). Each row corresponds to a single "
        "n-qubit pure state |ψ⟩ produced by a trained variational ansatz acting on a "
        "random product-state input. The original files provide trained weights for "
        "different ansatz families and target 'concentratable entanglement' (goal_ce). "
        "For each resulting state, we compute derived quantum invariants (entropies, "
        "magnetization summaries) and boolean properties describing entanglement regime, "
        "size/depth regimes, magnetization balance, and ansatz type."
    ),

    # ───────── ansatz type (booleans) ───────── #

    "ansatz_hardware_efficient": (
        "Boolean flag indicating whether the underlying circuit ansatz for this row is "
        "of the Hardware Efficient family (True) or not (False)."
    ),
    "ansatz_strongly_entangling": (
        "Boolean flag indicating whether the circuit ansatz belongs to the Strongly "
        "Entangling family (True) or not (False)."
    ),
    "ansatz_convolutional": (
        "Boolean flag indicating whether the circuit ansatz belongs to the Convolutional "
        "family (True) or not (False)."
    ),

    # ───────── structural / meta columns ───────── #

    "n_qubits": (
        "Number of qubits in the circuit and in the resulting pure state |ψ⟩ for this row."
    ),
    "depth": (
        "Effective circuit depth parsed from the NTangled filename. Roughly corresponds "
        "to the number of repeated entangling layers in the ansatz."
    ),
    "goal_ce": (
        "Target entanglement level (concentratable entanglement) associated with the "
        "trained ansatz, parsed from the filename and rescaled from the integer tag "
        "(e.g., '050' → 0.50)."
    ),
    "input_type": (
        "Categorical string describing how input product states were sampled in the "
        "original NTangled setup (e.g. 'ps', 'uniform', etc.), carried over directly "
        "from the filename."
    ),
    "sample_idx": (
        "Index of the random product-state input used for this particular state under a "
        "fixed ansatz + weight file. Distinguishes multiple samples generated from the "
        "same trained weights."
    ),

    # ───────── entropic invariants ───────── #

    "sq_entropy_mean": (
        "Mean single-qubit von Neumann entropy over all qubits. For each qubit i, "
        "we form the one-qubit reduced density matrix ρ_i and compute S(ρ_i); this "
        "column stores the average of these entropies over i."
    ),
    "sq_entropy_min": (
        "Minimum single-qubit von Neumann entropy among all qubits for this state. "
        "Captures the least entangled site locally."
    ),
    "sq_entropy_max": (
        "Maximum single-qubit von Neumann entropy among all qubits for this state. "
        "Captures the most entangled site locally."
    ),
    "half_entropy": (
        "Von Neumann entropy S(ρ_A) of the reduced state on the first half of the "
        "qubits A, where |A| = floor(n_qubits/2). The complement is traced out. "
        "For n_qubits < 2 this is set to 0."
    ),
    "half_purity": (
        "Purity of the reduced half-system state ρ_A, defined as Tr(ρ_A^2). "
        "Equal to 1 for product states across the A|B bipartition and decreases "
        "as entanglement across the cut increases."
    ),

    # ───────── magnetization / Pauli-Z invariants ───────── #

    "mz_mean": (
        "Mean Pauli-Z expectation value over all qubits: average_i ⟨Z_i⟩ for the state |ψ⟩."
    ),
    "mz_min": (
        "Minimum single-qubit Pauli-Z expectation ⟨Z_i⟩ over all qubits."
    ),
    "mz_max": (
        "Maximum single-qubit Pauli-Z expectation ⟨Z_i⟩ over all qubits."
    ),
    "mz_abs_mean": (
        "Mean absolute Pauli-Z expectation: average_i |⟨Z_i⟩|. Measures the typical "
        "magnitude of local polarization per qubit."
    ),
    "mz_total": (
        "Total Pauli-Z magnetization: sum_i ⟨Z_i⟩ over all qubits for the state |ψ⟩."
    ),

    # ───────── basic boolean regimes ───────── #

    "even_n_qubits": (
        "Boolean flag indicating whether n_qubits is even (True) or odd (False)."
    ),
    "high_depth": (
        "Boolean flag marking 'deep' circuits: True if the circuit depth is at or above "
        "a chosen threshold (e.g., depth ≥ 3), False otherwise."
    ),
    "product_like": (
        "Boolean flag indicating whether the state behaves like a near-product state, "
        "as diagnosed by single-qubit entropies (e.g., sq_entropy_max < 1e-3)."
    ),
    "highly_entangled": (
        "Boolean flag indicating whether the state is highly entangled across the "
        "half-system cut, typically defined by half_entropy exceeding some fraction "
        "of the maximum possible value (e.g., half_entropy > 0.8 * (n_qubits // 2))."
    ),
    "weakly_entangled": (
        "Boolean flag indicating whether the state is only weakly entangled across the "
        "half-system cut, typically defined by half_entropy being small relative to "
        "its maximum (e.g., half_entropy < 0.2 * (n_qubits // 2))."
    ),
    "magnetization_balanced": (
        "Boolean flag indicating whether the total Pauli-Z magnetization is close to "
        "zero, e.g., |mz_total| < 0.1 * n_qubits. Intended to mark states with "
        "approximately balanced computational-basis populations."
    ),

    # ───────── normalized entanglement + gaps ───────── #

    "half_entropy_norm": (
        "Half-system entropy normalized by the maximum possible value across the cut: "
        "half_entropy divided by (n_qubits // 2) when n_qubits ≥ 2. In ideal cases "
        "takes values in [0, 1]."
    ),
    "ent_gap": (
        "Signed entanglement gap: half_entropy − goal_ce. Positive values mean the "
        "realized half-system entropy exceeds the target; negative values indicate "
        "the state under-shoots the target entanglement."
    ),
    "ent_gap_abs": (
        "Absolute entanglement gap: |half_entropy − goal_ce|. Measures how far the "
        "realized entanglement is from the target level, ignoring direction."
    ),

    # ───────── derived spread / magnitude measures ───────── #

    "sq_entropy_range": (
        "Spread of single-qubit entropies: sq_entropy_max − sq_entropy_min. "
        "Quantifies how non-uniform local entanglement is across different qubits."
    ),
    "mz_abs_total": (
        "Absolute value of the total Pauli-Z magnetization: |mz_total|. Used as a "
        "scalar measure of overall polarization magnitude."
    ),

    # ───────── nontriviality flag ───────── #

    "nontrivial": (
        "Boolean flag indicating that the instance is considered nontrivial for "
        "entanglement analysis, typically taken as n_qubits ≥ 2."
    ),
}


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
    Stage.LP1,
    Stage.LP,
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

import random
TARGETS = [
        # "zero_forcing_number",
        # "spectral_radius",
        # "p5",
        # "p4",
        # "p6",
        # "independence_number",
        # "sum(pk_for_k>7)",
        # "half_entropy_norm",
        # "ent_gap_abs",
        # "n_qubits",
        # "sq_entropy_range",
        # # "half_entropy",
        # # "half_entropy_norm",
        # # "half_purity",
        # # "sq_entropy_min",
        "sq_entropy_max",
        # # "sq_entropy_mean",
        # "sq_entropy_range",
        # # "mz_mean",
        # "mz_min",
        # "mz_max",
        # # "mz_abs_mean",
        # "mz_total",
        # "mz_abs_total",
    ]

result = g3.list_conjecture(
    targets=TARGETS,
    stages=STAGES,
    include_invariant_products=True,
    include_abs=False,
    include_min_max=False,
    include_log=False,
    enable_sophie=True,
    sophie_stages=STAGES,
    quick=True,
)

print_g3_result(
    result,
    k_conjectures=10,
    k_sophie=10,
    data_doc=DATA_DOC,
)
