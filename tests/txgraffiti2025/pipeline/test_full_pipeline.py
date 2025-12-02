import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms.predicates import Where
from txgraffiti2025.processing.pipeline.full_pipeline import (
    run_full_pipeline,
    PipelineConfig,
    GlobalPruneConfig,
)

# ------------------------------------------------------------
# Test data
# ------------------------------------------------------------

def df_basic():
    # y = 2x exactly; w is weakly related to y; boolean hypotheses provided
    return pd.DataFrame({
        "H_all":  [True]*12,
        "H_first_half": [True]*6 + [False]*6,
        "x":      list(range(1, 13)),                # 1..12
        "w":      [1,1,2,3,5,8,13,21,34,55,89,144], # noisy/poor predictor for y
        "y":      [2*k for k in range(1, 13)],       # exactly 2x
    })


def H_all():
    return Where(lambda d: d["H_all"].astype(bool))

def H_first_half():
    return Where(lambda d: d["H_first_half"].astype(bool))


# ------------------------------------------------------------
# Core behaviors (ratios-only so tests are solver-free)
# ------------------------------------------------------------

def test_full_pipeline_ratios_only_nonempty_and_sorted():
    """
    With a simple df where y = 2x, ratios-only generation should produce at least
    one nontrivial conjecture; Hazel returns results sorted by touch (non-increasing).
    """
    df = df_basic()
    cfg = PipelineConfig(
        generators={"ratios"},            # avoid LP/convex to keep tests solver-free
        context_mode="one",
        direction="both",
        drop_frac_local=0.50,             # aggressive local pruning
        beam_k_per_H=10,
        global_prune=GlobalPruneConfig(
            hazel_drop_frac=0.25,
            hazel_atol=1e-9,
            beam_k_global=50,
        ),
    )
    res = run_full_pipeline(
        df,
        target="y",
        numeric_columns=["x","w"],        # explicit contexts (exclude target)
        hypotheses=[H_all(), H_first_half()],
        config=cfg,
        constants_cache=None,
    )

    # Nonempty frontier/final
    assert isinstance(res.final, list)
    assert len(res.final) >= 1

    # Scoreboard sanity
    assert isinstance(res.scoreboard, pd.DataFrame)
    assert "touch" in res.scoreboard.columns

    # Hazel ordering: non-increasing touch
    touches = res.scoreboard.sort_values("touch", ascending=False)["touch"].to_list()
    assert touches == sorted(touches, reverse=True)


def test_global_beam_cap_respected():
    """
    Global beam cap should limit the final number of conjectures even if
    many survive global pruning.
    """
    df = df_basic()
    cap = 3
    cfg = PipelineConfig(
        generators={"ratios"},
        context_mode="one",
        direction="both",
        drop_frac_local=0.50,
        beam_k_per_H=50,
        global_prune=GlobalPruneConfig(
            hazel_drop_frac=0.25,
            hazel_atol=1e-9,
            beam_k_global=cap,
        ),
    )
    res = run_full_pipeline(
        df,
        target="y",
        numeric_columns=["x","w"],
        hypotheses=[H_all(), H_first_half()],
        config=cfg,
    )
    assert len(res.final) <= cap


def test_seeds_grouped_by_hypothesis_have_entries():
    """
    Seeds should be grouped per hypothesis key; each provided hypothesis should
    appear in the dictionary (even if its seed list ends up empty after pruning).
    """
    df = df_basic()
    Hs = [H_all(), H_first_half()]
    cfg = PipelineConfig(
        generators={"ratios"},
        context_mode="one",
        direction="both",
        drop_frac_local=0.50,
        beam_k_per_H=5,
        global_prune=GlobalPruneConfig(
            hazel_drop_frac=0.25,
            hazel_atol=1e-9,
            beam_k_global=25,
        ),
    )
    res = run_full_pipeline(
        df,
        target="y",
        numeric_columns=["x","w"],
        hypotheses=Hs,
        config=cfg,
    )
    # Keys present for each hypothesis' repr
    keys = set(res.seeds_by_hypothesis.keys())
    for H in Hs:
        assert repr(H) in keys


def test_provenance_contains_mini_batch_stage_records():
    """
    Provenance should include mini_batch stage entries for seeds that made it past local pruning.
    """
    df = df_basic()
    cfg = PipelineConfig(
        generators={"ratios"},
        context_mode="one",
        direction="both",
        drop_frac_local=0.50,
        beam_k_per_H=10,
        global_prune=GlobalPruneConfig(
            hazel_drop_frac=0.25,
            hazel_atol=1e-9,
            beam_k_global=50,
        ),
    )
    res = run_full_pipeline(
        df,
        target="y",
        numeric_columns=["x","w"],
        hypotheses=[H_all(), H_first_half()],
        config=cfg,
    )
    assert isinstance(res.provenance, list)
    # at least one provenance record and it has expected fields
    assert any(p.get("stage") == "mini_batch" for p in res.provenance)
    sample = res.provenance[0]
    assert "name" in sample and "relation_text" in sample and "condition_text" in sample


def test_final_equals_frontier_when_lifting_disabled():
    """
    With lifting disabled (default), final results should match the frontier
    returned by the global consolidation step.
    """
    df = df_basic()
    cfg = PipelineConfig(
        generators={"ratios"},
        context_mode="one",
        direction="both",
        drop_frac_local=0.50,
        beam_k_per_H=10,
        global_prune=GlobalPruneConfig(
            hazel_drop_frac=0.25,
            hazel_atol=1e-9,
            beam_k_global=50,
        ),
        # All lifting toggles are False by default in LiftingConfig
    )
    res = run_full_pipeline(
        df,
        target="y",
        numeric_columns=["x","w"],
        hypotheses=[H_all(), H_first_half()],
        config=cfg,
    )
    # Compare by structural identity
    fin_keys = {(repr(c.condition), repr(c.relation)) for c in res.final}
    fr_keys  = {(repr(c.condition), repr(c.relation)) for c in res.frontier}
    assert fin_keys == fr_keys


# ------------------------------------------------------------
# Optional: hypothesis autodetection path (smoke test)
# ------------------------------------------------------------

def test_autodetect_hypotheses_smoke():
    """
    If hypotheses=None, the pipeline detects base and enumerates boolean columns.
    This is a smoke test to ensure no crashes and non-empty output on simple data.
    """
    df = df_basic()
    cfg = PipelineConfig(
        generators={"ratios"},
        context_mode="one",
        direction="both",
        drop_frac_local=0.50,
        beam_k_per_H=10,
        global_prune=GlobalPruneConfig(
            hazel_drop_frac=0.25,
            hazel_atol=1e-9,
            beam_k_global=50,
        ),
    )
    res = run_full_pipeline(
        df,
        target="y",
        numeric_columns=["x","w"],
        hypotheses=None,  # allow autodetection
        config=cfg,
    )
    assert isinstance(res.final, list)
    assert len(res.final) >= 1
