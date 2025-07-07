import pandas as pd
from typing import List, Iterator
from fractions import Fraction

from txgraffiti.logic import *
from txgraffiti.generators.registry import register_gen

__all__ = [
    'ratios',
]

@register_gen
def ratios(
    df: pd.DataFrame,
    *,
    features:   List[Property],
    target:     Property,
    hypothesis: Predicate,
) -> Iterator[Conjecture]:
    """
    For each feature f in `features`, compute the min and max of t/f
    over rows where `hypothesis` holds, then emit:
      hypothesis → (t >= c_min * f)
      hypothesis → (t <= c_max * f)
    """
    mask = hypothesis(df)
    # if hypothesis never true, nothing to yield
    if not mask.any():
        return

    t_vals = target(df)[mask]
    for f in features:
        f_vals = f(df)[mask]

        # avoid division by zero
        nonzero = f_vals != 0
        if not nonzero.any():
            continue

        ratios = t_vals[nonzero] / f_vals[nonzero]
        cmin, cmax = Fraction(float(ratios.min())).limit_denominator(), Fraction(float(ratios.max())).limit_denominator()

        # build RHS Property-expressions
        low_rhs  = Constant(cmin) * f
        high_rhs = Constant(cmax) * f

        # yield t >= cmin * f
        yield Conjecture(
            hypothesis,
            Inequality(target, ">=", low_rhs)
        )

        # yield t <= cmax * f
        yield Conjecture(
            hypothesis,
            Inequality(target, "<=", high_rhs)
        )
