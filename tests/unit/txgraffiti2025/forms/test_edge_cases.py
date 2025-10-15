import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms import (
    # utils
    to_expr, Const, ColumnTerm, LinearForm, floor, ceil, abs_, log, exp, sqrt,
    # predicates
    EQ, NE, LT, GT, LE, GE, BETWEEN, IN, IsInteger, Where, RowWhere,
    # relations/conjecture
    Eq, Le, Ge, AllOf, AnyOf, Conjecture,
    # implication & class relations
    Implication, Equivalence,
    ClassInclusion, ClassEquivalence,
    # substructure
    SubstructurePredicate,
)

# ---------- utils & expressions ----------

def test_to_expr_type_error():
    class X: pass
    with pytest.raises(TypeError):
        to_expr(X())

def test_linearform_missing_column_raises(df_basic):
    lf = LinearForm(0.0, [(1.0, "missing")])
    with pytest.raises(KeyError):
        lf.eval(df_basic)

def test_unary_floor_ceil_negatives_and_abs():
    df = pd.DataFrame({"x": [-2.7, -0.1, 0, 0.1, 2.7]})
    f = floor("x").eval(df)
    c = ceil("x").eval(df)
    a = abs_("x").eval(df)
    assert f.tolist() == [-3.0, -1.0, 0.0, 0.0, 2.0]
    assert c.tolist() == [-2.0, 0.0, 0.0, 1.0, 3.0]
    assert a.tolist() == [2.7, 0.1, 0.0, 0.1, 2.7]

def test_modulo_and_negation_alignment(df_basic):
    expr = (-(to_expr("a")) % 2)
    out = expr.eval(df_basic)
    assert len(out) == len(df_basic) and (out.index == df_basic.index).all()

# ---------- predicates ----------

def test_between_exclusive_boundaries(df_basic):
    m = BETWEEN("a", 2.0, 3.0, inc_lo=False, inc_hi=False).mask(df_basic)
    # a = [1,2,3,4] -> only (2,3) exclusive -> none True
    assert m.sum() == 0

def test_where_non_bool_raises(df_basic):
    with pytest.raises(ValueError):
        Where(lambda df: df["a"] + 1).mask(df_basic)

def test_rowwhere_alignment(df_basic):
    m = RowWhere(lambda row: row["a"] % 2 == 0).mask(df_basic)
    assert m.index.equals(df_basic.index)

def test_isinteger_tolerance_boundary():
    df = pd.DataFrame({"v": [1.0, 2.0000000004, 2.00000001]})
    m1 = IsInteger("v", tol=1e-9).mask(df)
    m2 = IsInteger("v", tol=1e-8).mask(df)
    assert m1.tolist() == [True, True, False]
    assert m2.tolist() == [True, True, True]

# ---------- relations & conjecture ----------

def test_allof_anyof_empty_parts(df_basic):
    # AllOf([]) -> vacuously true; AnyOf([]) -> vacuously false
    assert AllOf([]).evaluate(df_basic).all()
    assert (~AnyOf([]).evaluate(df_basic)).all()

def test_conjecture_no_condition_equivalent_to_relation(df_basic):
    R = Le("a", "c")
    conj = Conjecture(R, None)
    app, holds, fails = conj.check(df_basic)
    assert app.all()
    assert holds.equals(R.evaluate(df_basic))
    assert fails.empty

def test_conjecture_alignment_with_shuffled_index(df_basic):
    # Shuffle df index and verify alignment is preserved
    df2 = df_basic.sample(frac=1.0, random_state=0)
    R = Ge("c", "a")
    app, holds, fails = Conjecture(R, None).check(df2)
    assert holds.index.equals(df2.index)
    assert app.index.equals(df2.index)
    assert fails.index.isin(df2.index).all()

# ---------- implication & equivalence ----------

def test_implication_failure_and_slack_sign(df_basic):
    # premise true where a>=3, conclusion false (b>=3) -> failures there with negative slack
    imp = Implication(Ge("a", 3.0), Ge("b", 3.0))
    app, holds, fails = imp.check(df_basic)
    # With no explicit condition, `applicable` is all True by design
    assert app.all()
    # Failures happen exactly on rows where premise holds (a>=3) but conclusion fails (b<3): 2 rows
    assert len(fails) == 2
    # Conclusion slack = left-right for Ge; left=b, right=3 -> negative
    assert (fails["__slack__"] <= 0).all()

def test_equivalence_under_condition(df_basic):
    C = GE("a", 2.0) & LE("a", 3.0)
    eqv = Equivalence(Le("a","c"), Ge("c","a"), condition=C)
    app, holds, fails = eqv.check(df_basic)
    assert app.sum() == 2
    assert holds.sum() == len(df_basic)  # outside C vacuously true

# ---------- class relations ----------

def test_class_relations_misaligned_indices(df_basic):
    # Build a mask on a filtered df; ensure reindex doesn't break
    A = GE("a", 2.0)
    B = GE("c", 3.0)
    incl = ClassInclusion(A, B)
    sub = df_basic.iloc[[1,3]]  # misaligned subset
    mask = incl.mask(sub)
    assert mask.index.equals(sub.index)

def test_class_equivalence_difference_slice(df_basic):
    A = GE("a", 2.0)
    B = GE("a", 3.0)
    eqv = ClassEquivalence(A, B)
    bad = eqv.violations(df_basic)
    assert not bad.empty

# ---------- substructure ----------

def test_substructure_on_error_raise(df_objects):
    pred = SubstructurePredicate(lambda obj: 1/0, object_col="object", on_error="raise")
    with pytest.raises(ZeroDivisionError):
        pred.mask(df_objects)

def test_substructure_returns_series_aligned(df_objects):
    pred = SubstructurePredicate(lambda obj: obj is not None, object_col="object", on_error="false")
    m = pred.mask(df_objects)
    assert m.index.equals(df_objects.index)

# ---------- qualitative ----------

def test_monotone_constant_series_handled():
    df = pd.DataFrame({"x":[1,1,1,1], "y":[2,3,4,5]})
    from txgraffiti2025.forms import MonotoneRelation
    mono = MonotoneRelation("x","y","increasing", method="spearman", min_abs_rho=0.0)
    res = mono.evaluate_global(df)
    assert res["rho"] == 0.0 and res["n"] == 4

def test_monotone_requires_min_samples(df_basic):
    from txgraffiti2025.forms import MonotoneRelation
    mono = MonotoneRelation("a","c", min_abs_rho=0.0)
    res = mono.evaluate_global(df_basic.iloc[:1])  # n=1 -> not enough
    assert res["n"] == 1 and res["ok"] is False
