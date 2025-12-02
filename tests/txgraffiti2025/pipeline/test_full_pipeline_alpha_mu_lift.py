import pytest
import pandas as pd

from txgraffiti.example_data import graph_data as df_full

from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.generic_conjecture import Ge, Le
from txgraffiti2025.forms.predicates import Where
from txgraffiti2025.processing.pre.constants_cache import precompute_constant_ratios
from txgraffiti2025.processing.pipeline.full_pipeline import (
    run_full_pipeline,
    PipelineConfig,
    GlobalPruneConfig,
)

# Optional debugging helpers (safe import; tests don't depend on them)
try:
    from txgraffiti2025.processing.inspect import print_conjectures_summary, conjectures_dataframe
except Exception:  # pragma: no cover
    def print_conjectures_summary(*args, **kwargs):  # fallback no-op
        pass
    def conjectures_dataframe(*args, **kwargs):  # fallback no-op
        return pd.DataFrame()


@pytest.mark.integration
def test_pipeline_lifts_regular_alpha_le_mu_to_connected_with_Delta_over_delta():
    """
    Pipeline should recover:

        (connected ∧ regular) ⇒ α ≤ 1·μ
        and lift to
        (connected) ⇒ α ≤ (Δ/δ)·μ

    via coefficient-from-constants generalization using Δ/δ discovered on the
    regular slice (where Δ == δ so Δ/δ == 1).
    """

    # Restrict to the needed columns only
    keep = [
        "connected", "regular",
        "independence_number", "matching_number",
        "maximum_degree", "minimum_degree",
    ]
    assert set(keep) <= set(df_full.columns)
    df = df_full[keep].copy()

    # Hypotheses: H1 = connected ∧ regular (base), H2 = connected (superset)
    H1 = Where(lambda d: d["connected"] & d["regular"])
    H2 = Where(lambda d: d["connected"])

    # Build a constants cache on H1 that includes Δ/δ == 1 (with shifts=(0,))
    cache = precompute_constant_ratios(
        df,
        hypotheses=[H1],
        numeric_cols=["maximum_degree", "minimum_degree"],
        shifts=(0,),          # exact Δ/δ
        min_support=3,
        max_denominator=200,
    )

    # Configure the pipeline:
    #  - ratios only (solver-free)
    #  - UPPER bounds (α ≤ c·μ)
    #  - gentle pruning (Hazel drop=0) to avoid losing the lifted conjecture
    #  - enable coefficient-from-constants lifting; others off
    #  - keep_one_symbolic_per_bucket=True to retain one symbolic ratio if present
    pipe_cfg = PipelineConfig(
        generators={"ratios"},
        context_mode="one",
        direction="upper",
        drop_frac_local=0.0,
        hazel_atol_local=1e-9,
        beam_k_per_H=50,
        max_seeds_per_generator_per_pair=10,
        global_prune=GlobalPruneConfig(
            hazel_drop_frac=0.0,
            hazel_atol=1e-9,
            beam_k_global=200,
        ),
        keep_one_symbolic_per_bucket=True,  # <-- key for Δ/δ retention
    )
    pipe_cfg.lifting.enable_coefficient_from_constants = True
    pipe_cfg.lifting.enable_coefficient_from_reciprocals = False
    pipe_cfg.lifting.enable_intercept_lifts = False
    pipe_cfg.lifting.waves = 1
    pipe_cfg.lifting.max_lifts_per_seed = 10
    pipe_cfg.lifting.max_new_per_wave = 500
    pipe_cfg.lifting.require_superset_for_lifts = True

    # Only "matching_number" is the desired context feature for α ≤ c·μ
    numeric_contexts = ["matching_number"]

    # Run with the two hypotheses in order so H1 is treated as the base for cache relevance
    res = run_full_pipeline(
        df,
        target="independence_number",
        numeric_columns=numeric_contexts,
        hypotheses=[H1, H2],
        config=pipe_cfg,
        constants_cache=cache,  # inject the precomputed Δ/δ cache on H1
    )

    # Sanity
    assert isinstance(res.final, list) and len(res.final) > 0

    # (Debug print, visible with `pytest -s`)
    print("\n=== FINAL CONJECTURES (top 50) ===")
    print_conjectures_summary(df, res.final, top=50)

    # Search for the lifted α ≤ (Δ/δ)·μ under (connected)
    conn_mask = H2.mask(df).reindex(df.index, fill_value=False).astype(bool)

    found = False
    for c in res.final:
        try:
            if c.condition is None:
                continue
            cm = c.condition.mask(df).reindex(df.index, fill_value=False).astype(bool)
            if not cm.equals(conn_mask):
                continue
            rtxt = repr(c.relation)
            # structural symbol check: must contain Δ, δ, and μ
            if ("maximum_degree" in rtxt) and ("minimum_degree" in rtxt) and ("matching_number" in rtxt):
                assert c.is_true(df)
                found = True
                break
        except Exception:
            # Be robust to odd relations; just skip them
            continue

    if not found:
        # Helpful diagnostic if the exact form wasn’t found:
        table = conjectures_dataframe(df, res.final)
        connected_rows = table[table["condition_pretty"].str.contains("connected", na=False)]
        preview_cols = ["relation_pretty", "kind", "touch", "holds_n", "applicable_n"]
        preview = connected_rows[preview_cols].head(15).to_string(index=False)
        raise AssertionError(
            "Expected (connected) ⇒ α ≤ (maximum_degree/minimum_degree)·matching_number\n"
            "Top ‘connected’ finals were:\n"
            f"{preview}\n"
            "If a stronger-but-different coefficient appears (e.g., α ≤ μ), "
            "you may broaden acceptance or record both forms."
        )

    assert found
