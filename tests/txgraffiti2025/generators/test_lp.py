import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.generators.lp import lp_bounds, LPConfig
from txgraffiti2025.forms import Predicate


@pytest.fixture
def df_simple():
    # y = 2*x + 1 exactly
    return pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [1.0, 3.0, 5.0, 7.0],
        "H": [True, True, True, True],
    })


@pytest.fixture
def hyp_true():
    # Predicate selecting H==True
    return Predicate.from_column("H")


def _mock_solver_factory(a_vec, b_val):
    """Return a function to monkeypatch _solve_sum_slack_lp with fixed solution."""
    def _mock_solve_sum_slack_lp(X, y, *, sense: str):
        # Sanity: the shapes should match coefficient length
        k = X.shape[1]
        assert k == len(a_vec)
        return np.array(a_vec, dtype=float), float(b_val)
    return _mock_solve_sum_slack_lp


def test_lp_bounds_both_directions_with_mock_solver(df_simple, hyp_true, monkeypatch):
    # Mock LP to return the exact fit: a=2, b=1
    monkeypatch.setattr(
        "txgraffiti2025.generators.lp._solve_sum_slack_lp",
        _mock_solver_factory([2.0], 1.0)
    )

    cfg = LPConfig(features=["x"], target="y", direction="both", max_denominator=20)
    conjs = list(lp_bounds(df_simple, hypothesis=hyp_true, config=cfg))

    # Expect two conjectures: lower (Ge) and upper (Le)
    assert len(conjs) == 2
    names = {c.name for c in conjs}
    assert any("lp_lower" in n for n in names)
    assert any("lp_upper" in n for n in names)

    # Both should hold exactly on all applicable rows
    for c in conjs:
        applicable, holds, failures = c.check(df_simple, auto_base=False)
        assert applicable.equals(hyp_true.mask(df_simple))
        assert holds[applicable].all()
        assert failures.empty


def test_lp_bounds_upper_only_and_lower_only(df_simple, hyp_true, monkeypatch):
    monkeypatch.setattr(
        "txgraffiti2025.generators.lp._solve_sum_slack_lp",
        _mock_solver_factory([2.0], 1.0)
    )

    cfg_up = LPConfig(features=["x"], target="y", direction="upper")
    cfg_lo = LPConfig(features=["x"], target="y", direction="lower")

    conjs_up = list(lp_bounds(df_simple, hypothesis=hyp_true, config=cfg_up))
    conjs_lo = list(lp_bounds(df_simple, hypothesis=hyp_true, config=cfg_lo))

    assert len(conjs_up) == 1 and "lp_upper" in conjs_up[0].name
    assert len(conjs_lo) == 1 and "lp_lower" in conjs_lo[0].name


def test_lp_bounds_min_support_filters_out(df_simple, hyp_true, monkeypatch):
    # Any mock is fine; we should not reach the solver if min_support not met.
    monkeypatch.setattr(
        "txgraffiti2025.generators.lp._solve_sum_slack_lp",
        _mock_solver_factory([1.0], 0.0)
    )

    # Require more support than available to suppress output
    cfg = LPConfig(features=["x"], target="y", direction="both", min_support=10)
    conjs = list(lp_bounds(df_simple, hypothesis=hyp_true, config=cfg))
    assert conjs == []


def test_lp_bounds_respects_hypothesis_mask(monkeypatch):
    # y = 3*x - 2, but only last two rows are applicable (H==True)
    df = pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [-2.0, 1.0, 4.0, 7.0],
        "H": [False, False, True, True],
    })
    H = Predicate.from_column("H")

    # Mock exact fit a=3, b=-2
    monkeypatch.setattr(
        "txgraffiti2025.generators.lp._solve_sum_slack_lp",
        _mock_solver_factory([3.0], -2.0)
    )

    # Need min_support=2 since only two rows are inside the hypothesis
    cfg = LPConfig(features=["x"], target="y", direction="both", min_support=2)
    conjs = list(lp_bounds(df, hypothesis=H, config=cfg))
    assert len(conjs) == 2

    for c in conjs:
        applicable, holds, failures = c.check(df, auto_base=False)
        # Only the last two are applicable
        assert applicable.tolist() == [False, False, True, True]
        assert holds[applicable].all()
        assert failures.empty



def test_lp_bounds_fraction_pretty_repr(df_simple, hyp_true, monkeypatch):
    # Coeffs chosen to be rational-friendly: a = 2/3, b = 1/2
    monkeypatch.setattr(
        "txgraffiti2025.generators.lp._solve_sum_slack_lp",
        _mock_solver_factory([2.0/3.0], 0.5)
    )

    cfg = LPConfig(features=["x"], target="y", direction="upper", max_denominator=12)
    conjs = list(lp_bounds(df_simple, hypothesis=hyp_true, config=cfg))
    assert len(conjs) == 1
    r = repr(conjs[0].relation)
    # Expect fraction-like rendering somewhere in the right-hand Expr
    # (Exact formatting comes from Const.__repr__ and BinOp.__repr__.)
    assert ("1/2" in r) or ("(1/2)" in r)
    assert ("2/3" in r) or ("(2/3)" in r)


def test_lp_bounds_no_hypothesis_none(monkeypatch):
    # When hypothesis=None, applicable should be all rows (global)
    df = pd.DataFrame({
        "x": [0.0, 1.0, 2.0],
        "y": [1.0, 3.0, 5.0],
    })

    monkeypatch.setattr(
        "txgraffiti2025.generators.lp._solve_sum_slack_lp",
        _mock_solver_factory([2.0], 1.0)
    )

    cfg = LPConfig(features=["x"], target="y", direction="lower")
    conjs = list(lp_bounds(df, hypothesis=None, config=cfg))
    assert len(conjs) == 1

    applicable, holds, failures = conjs[0].check(df, auto_base=False)
    assert applicable.tolist() == [True, True, True]
    assert holds.all()
    assert failures.empty
