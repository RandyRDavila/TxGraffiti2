import pytest
import pandas as pd

from txgraffiti.example_data import graph_data as df_full

from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.generic_conjecture import Ge
from txgraffiti2025.forms.predicates import Predicate, LEQ, AndPred

from txgraffiti2025.processing.pipeline.full_pipeline import (
    run_full_pipeline,
    PipelineConfig,
    GlobalPruneConfig,
)
from txgraffiti2025.processing.pipeline.mini_batch import MiniBatchConfig


@pytest.mark.integration
def test_pipeline_lifts_subcubic_domination_lower_bound_to_connected_symbolic_reciprocal():
    """
    Reproduce the classic lift via the full pipeline:

        Base (connected ∧ subcubic):      γ ≥ (1/4)·n
        Lift to (connected):               γ ≥ (1/(Δ+1))·n

    We restrict the DataFrame to the essential columns and rely on:
      - ratios generator (lower bounds only)
      - reciprocal lifting enabled in the pipeline
      - global consolidation (Dalmatian → Morgan → Hazel)
    """
    # Restrict to core columns used in this theorem
    needed_cols = {"connected", "maximum_degree", "order", "domination_number"}
    assert needed_cols <= set(df_full.columns)
    df = df_full[list(needed_cols)].copy()

    # Hypotheses:
    #   H_base = connected ∧ (Δ ≤ 3)
    H_conn = Predicate.from_column("connected")
    H_subcubic = LEQ("maximum_degree", 3.0)
    H_base = AndPred(H_conn, H_subcubic)

    # Pipeline config:
    #   - Use only ratios (no external solvers needed)
    #   - Lower bounds only (Ge)
    #   - Enable reciprocal lifting (single wave)
    #   - Keep pruning on (local & global) but allow enough beam to keep the lift
    pipe_cfg = PipelineConfig(
        generators={"ratios"},
        context_mode="one",
        direction="lower",
        drop_frac_local=0.50,
        hazel_atol_local=1e-9,
        beam_k_per_H=50,
        max_seeds_per_generator_per_pair=10,
        global_prune=GlobalPruneConfig(
            hazel_drop_frac=0.25,
            hazel_atol=1e-9,
            beam_k_global=200,
        ),
    )
    # Turn on reciprocal lifting; leave other lifts off
    pipe_cfg.lifting.enable_coefficient_from_reciprocals = True
    pipe_cfg.lifting.enable_coefficient_from_constants = False
    pipe_cfg.lifting.enable_intercept_lifts = False
    pipe_cfg.lifting.waves = 1
    pipe_cfg.lifting.max_lifts_per_seed = 10
    pipe_cfg.lifting.max_new_per_wave = 500
    pipe_cfg.lifting.require_superset_for_lifts = True  # connected should be a strict superset of base

    # Only "order" is the context feature we want for γ ≥ c·n
    numeric_contexts = ["order"]

    # Run the pipeline *on the two hypotheses we care about*:
    #   First H_base (to get γ ≥ 1/4·n), then H_conn so it’s available as a candidate superset.
    res = run_full_pipeline(
        df,
        target="domination_number",
        numeric_columns=numeric_contexts,
        hypotheses=[H_base, H_conn],
        config=pipe_cfg,
        constants_cache=None,  # not needed for reciprocal lifting
    )

    # Sanity: pipeline must produce some final conjectures
    assert isinstance(res.final, list)
    assert len(res.final) > 0

    # Build the expected symbolic relation:
    #   γ ≥ (1/(Δ+1)) · n   under the (connected) hypothesis
    coeff_symbolic = Const(1) / (to_expr("maximum_degree") + Const(1))
    expected_rel_repr = repr(Ge("domination_number", coeff_symbolic * to_expr("order")))
    expected_conn_mask = H_conn.mask(df).reindex(df.index, fill_value=False).astype(bool)

    # Search in final results
    found = False
    for c in res.final:
        try:
            cond_mask = c.condition.mask(df).reindex(df.index, fill_value=False).astype(bool) if c.condition is not None else pd.Series(True, index=df.index)
            if cond_mask.equals(expected_conn_mask) and repr(c.relation) == expected_rel_repr:
                assert c.is_true(df), "Proposed conjecture must hold on the data"
                found = True
                break
        except Exception:
            continue

    assert found, "Pipeline did not recover (connected) ⇒ γ ≥ (1/(Δ+1))·n"
