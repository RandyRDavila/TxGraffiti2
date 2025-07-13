import pandas as pd
import pytest

from txgraffiti.playground.conjecture import (
    ConjecturePlayground,
    ForAll,
    Exists,
    find_strengthened_equalities,
)
from txgraffiti.logic import Property, Predicate, Conjecture, Inequality, Constant, TRUE

# ————— Fixtures —————

@pytest.fixture
def df():
    # three rows, one tree
    return pd.DataFrame({
        'alpha':     [1, 2, 3],
        'beta':      [3, 2, 1],
        'connected': [True, True, True],
        'tree':      [False, True, False],
    })

@pytest.fixture
def pg(df):
    # playground with default base=TRUE
    return ConjecturePlayground(df, object_symbol="G")


# ————— ForAll / Exists —————

def test_forall_and_counterexamples(df, pg):
    P = pg.connected  # holds everywhere
    F = pg.tree       # holds at one row only

    fa = ForAll(P, df, object_symbol="G")
    assert isinstance(fa, ForAll)
    assert bool(fa.is_true()) is True

    ex = Exists(F, df, object_symbol="G")
    assert isinstance(ex, Exists)
    assert bool(ex.is_true()) is True
    # witness should include exactly the row where tree == True
    w = ex.witness()
    assert len(w) == 1
    assert bool(w.iloc[0]['tree']) is True

    # repr
    assert repr(fa) == "∀ G: connected"
    assert repr(ex) == "∃ G: tree"


# ————— .prop() and __getattr__ lifting —————

def test_prop_and_getattr(df, pg):
    # .prop
    p_alpha = pg.prop('alpha')
    assert isinstance(p_alpha, Property)
    assert (p_alpha(df) == df['alpha']).all()

    # getattr on numeric col → Property
    p_beta = pg.beta
    assert isinstance(p_beta, Property)
    assert (p_beta(df) == df['beta']).all()

    # getattr on bool col → Predicate
    pred = pg.connected
    assert isinstance(pred, Predicate)
    assert (pred(df) == df['connected']).all()


# ————— forall()/exists() wrappers —————

def test_pg_forall_exists_methods(df, pg):
    # pg.forall returns a ForAll
    fa = pg.forall(pg.tree)
    assert isinstance(fa, ForAll)
    assert fa.is_true() is False  # tree isn't true on all rows

    # pg.exists returns an Exists
    ex = pg.exists(pg.tree)
    assert isinstance(ex, Exists)
    assert ex.is_true() is True


# ————— generate() with a dummy generator —————

def dummy_gen(df, *, features, target, hypothesis, **kwargs):
    # always yield a single conjecture: hypothesis → (target <= first feature)
    return iter([ hypothesis >> (target <= features[0]) ])

def test_generate_with_dummy(df, pg):
    # use strings for features/target/hypothesis
    gens = list(pg.generate(
        methods=[dummy_gen],
        features=['alpha'],
        target='beta',
        hypothesis='connected'
    ))
    # one conj
    assert len(gens) == 1
    conj = gens[0]
    # should be connected → (beta <= alpha)
    assert isinstance(conj, Conjecture)
    lhs = conj.hypothesis(df)
    assert lhs.all()
    # check conclusion
    ineq = conj.conclusion
    assert isinstance(ineq, Inequality)


# # ————— find_strengthened_equalities —————

def test_find_strengthened_equalities(df, pg):
    P = pg.connected
    alpha = pg.alpha
    beta  = pg.beta

    # build ≤ and ≥
    le = Conjecture(P, Inequality(alpha, '<=', beta))
    ge = Conjecture(P, Inequality(alpha, '>=', beta))
    eqs = find_strengthened_equalities([le, ge])
    # should produce one equality conjecture: connected → (alpha == beta)
    assert len(eqs) == 1
    eqc = eqs[0]
    assert isinstance(eqc, Conjecture)
    assert eqc.hypothesis is P
    assert eqc.conclusion.op == '=='
    # must hold exactly where alpha == beta
    mask = eqc.conclusion(df)
    assert (mask == (df['alpha']==df['beta'])).all()


# # ————— append_row() and reset() —————

def test_append_row_and_reset(df, pg):
    # generate one dummy conjecture so cache is nonempty
    pg.generate(methods=[dummy_gen], features=['alpha'], target='beta', hypothesis='connected')
    assert pg.conjectures == []  # nothing in discover, but generate didn't fill cache

    # append a row
    pg.conjectures = ["stub"]
    pg.append_row({'alpha':10,'beta':10,'connected':True,'tree':False})
    # conjectures got cleared
    assert pg.conjectures == []

    # reset with new_df
    pg.conjectures = ["stub"]
    new_df = df.head(2)
    pg.reset(new_df=new_df)
    assert pg.df.equals(new_df)
    assert pg.conjectures == []

    # reset without new_df
    pg.conjectures = ["stub"]
    pg.reset()
    assert pg.conjectures == []
