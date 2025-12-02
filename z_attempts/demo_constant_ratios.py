# In your demo:
from txgraffiti2025.processing.pre.hypotheses import enumerate_boolean_hypotheses, detect_base_hypothesis
from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses
from txgraffiti2025.processing.post.constant_ratios import find_constant_ratios_for_conjecture
from txgraffiti.example_data import graph_data as df

base = detect_base_hypothesis(df)
hyps_all = enumerate_boolean_hypotheses(
    df, treat_binary_ints=True, include_base=True, include_pairs=True, skip_always_false=True
)
kept_hyps, _ = simplify_and_dedup_hypotheses(df, hyps_all, min_support=30, treat_binary_ints=True)

# Build a ratio-style probe conjecture (optional, for coeff matching):
from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le
conj = Conjecture(relation=Le(to_expr("independence_number"), Const(1.0) * to_expr("order")), condition=None)

hits = find_constant_ratios_for_conjecture(
    df,
    conj,
    hypotheses=kept_hyps,         # subclasses to search within
    shifts=tuple(range(-1, 1)),   # broaden a bit
    constancy="cv",               # robust mode
    cv_tol=0.08,                  # ~8% variability
    min_support= max(12, int(0.1*len(df))),
    max_denominator=30,
)

for h in hits[:12]:
    hyp_txt = repr(h.hypothesis) if h.hypothesis is not None else "TRUE"
    print(f"[{hyp_txt}]  ({h.numerator}+{h.shift_num})/({h.denominator}+{h.shift_den}) "
          f"≈ {h.value_display} | support={h.support} | cv={h.cv:.3f}"
          f"{' (≈ coeff)' if h.matches_conj_coeff else ''}")
