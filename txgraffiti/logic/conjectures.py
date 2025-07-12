# txgraffiti.logic.conjectures.py

import pandas as pd
from txgraffiti.logic.predicates import Predicate

# ───────── Conjecture ─────────

__all__ = [
    'Conjecture',
]

class Conjecture(Predicate):
    """
    A logical implication between two predicates.

    Represents a rule of the form: (hypothesis) → (conclusion).

    Parameters
    ----------
    hypothesis : Predicate
        The antecedent of the implication.
    conclusion : Predicate
        The consequent of the implication.

    Examples
    --------
    >>> conj = P >> Q
    >>> conj.is_true(df)
    True
    """
    def __init__(self, hypothesis: Predicate, conclusion: Predicate):
        name = f"({hypothesis.name}) → ({conclusion.name})"
        func = lambda df: (~hypothesis(df)) | conclusion(df)
        super().__init__(name, func)
        object.__setattr__(self, 'hypothesis',  hypothesis)
        object.__setattr__(self, 'conclusion',  conclusion)

    def is_true(self, df: pd.DataFrame) -> bool:
        """
        Check if the conjecture holds on all rows of the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The data to evaluate.

        Returns
        -------
        bool
            True if all rows satisfy the implication.
        """
        return bool(self(df).all())

    def accuracy(self, df: pd.DataFrame) -> float:
        """
        Compute the conditional accuracy of the conjecture.

        This is defined as the fraction of rows satisfying the conclusion
        among those satisfying the hypothesis.

        Parameters
        ----------
        df : pd.DataFrame
            The data to evaluate.

        Returns
        -------
        float
            The accuracy of the conjecture.
        """
        hyp = self.hypothesis(df)
        if not hyp.any():
            return 0.0
        return float(self(df)[hyp].mean())

    def counterexamples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the rows that violate the conjecture.

        Parameters
        ----------
        df : pd.DataFrame
            The data to search.

        Returns
        -------
        pd.DataFrame
            Subset of rows where the implication fails.
        """
        return df[~self(df)]

    def __repr__(self):
        return f"<Conj {self.name}>"
