import pandas as pd
from txgraffiti.logic.properties import Property
from txgraffiti.logic.predicates import Predicate, TRUE


__all__ = [
    'KnowledgeTable',
]

import pandas as pd
from txgraffiti.logic.properties import Property
from txgraffiti.logic.predicates import Predicate, TRUE

class KnowledgeTable(pd.DataFrame):
    """
    A pandas DataFrame subclass that “lifts” its columns into TxGraffiti
    Properties and Predicates, preserving the subclass through most
    DataFrame operations.

    Parameters
    ----------
    *args, **kwargs
        Passed through to `pandas.DataFrame`. See `pandas.DataFrame`
        for supported parameters.

    Attributes
    ----------
    <column_name> : Property or Predicate
        If a column is boolean, `kt.<column_name>` returns a
        `txgraffiti.logic.predicates.Predicate` that masks the DataFrame
        on that column. Otherwise it returns a
        `txgraffiti.logic.properties.Property` on that column.

    Methods
    -------
    to_session(object_symbol="G", base=TRUE)
        Create a `ConjecturePlayground` for this table.

    Notes
    -----
    - We implement the pandas `_constructor` hook so that methods like
      `df.loc[...]`, `.assign()`, etc., return a `KnowledgeTable` again.
    - Only existing column names are intercepted in `__getattr__`; all other
      attributes fall back to the normal DataFrame API.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti.logic.tables import KnowledgeTable
    >>> df = pd.DataFrame({
    ...     'alpha':     [1, 2, 3],
    ...     'connected': [True, False, True],
    ... })
    >>> kt = KnowledgeTable(df)
    >>> # Numeric column lifts to Property:
    >>> prop = kt.alpha
    >>> print(prop(df))
    0    1
    1    2
    2    3
    Name: alpha, dtype: int64

    >>> # Boolean column lifts to Predicate:
    >>> pred = kt.connected
    >>> print(pred(df))
    0     True
    1    False
    2     True
    Name: connected, dtype: bool

    >>> # Start a conjecturing session:
    >>> session = kt.to_session(object_symbol="G", base=kt.connected)
    >>> session  # doctest: +ELLIPSIS
    <txgraffiti.playground.conjecture.ConjecturePlayground object at ...>
    """
    # Tell pandas to preserve our subclass in most operations
    _metadata = []

    @property
    def _constructor(self):
        return KnowledgeTable

    def __getattr__(self, name):
        """
        Lift a column into a Property or Predicate on attribute access.

        Parameters
        ----------
        name : str
            The attribute name being accessed.

        Returns
        -------
        Predicate or Property
            - If ``name`` matches a boolean column, returns
              ``Predicate(name, lambda df: df[name])``.
            - If ``name`` matches a non-boolean column, returns
              ``Property(name, lambda df: df[name])``.
        """
        if name in self.columns:
            series = self[name]
            if pd.api.types.is_bool_dtype(series):
                return Predicate(name, lambda df, c=name: df[c])
            return Property(name, lambda df, c=name: df[c])
        return super().__getattr__(name)

    def to_session(self, object_symbol="G", base: Predicate = TRUE):
        """
        Create a ConjecturePlayground session from this table.

        Parameters
        ----------
        object_symbol : str, optional
            The symbol to use for the “object” in generated conjectures
            (default is ``"G"``).
        base : Predicate, optional
            A base hypothesis to conjoin with any user‐provided
            hypothesis (default is ``TRUE``, i.e. no restriction).

        Returns
        -------
        ConjecturePlayground
            An instance of :class:`txgraffiti.playground.conjecture.ConjecturePlayground`
            initialized with this table.
        """
        from txgraffiti.playground.conjecture import ConjecturePlayground
        return ConjecturePlayground(self,
                                    object_symbol=object_symbol,
                                    base=base)
