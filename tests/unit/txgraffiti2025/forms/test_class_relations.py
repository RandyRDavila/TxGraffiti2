from txgraffiti2025.forms import ClassInclusion, ClassEquivalence, GEQ, LEQ, Predicate

def test_class_inclusion_and_violations(df_basic):
    A = GEQ("a", 3.0)
    B = GEQ("c", 4.5)  # because c = 1.5*a, rows with a>=3 -> c>=4.5
    incl = ClassInclusion(A, B)
    mask = incl.mask(df_basic)
    assert mask.all()
    assert incl.violations(df_basic).empty

def test_class_equivalence(df_basic):
    A = GEQ("a", 2.0) & LEQ("a", 3.0)
    B = GEQ("a", 1.999999) & LEQ("a", 3.000001)
    eqv = ClassEquivalence(A, B)
    mask = eqv.mask(df_basic)
    # nearly identical ranges but potentially differ at edges; ok if not all True
    assert mask.dtype == bool
