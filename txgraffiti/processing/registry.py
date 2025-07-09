from typing import Callable, List
import pandas as pd

from txgraffiti.logic import Conjecture

__all__ = [
    'register_post',
    'list_posts',
    'get_post',
]

_POST_FUNCS: dict[str, Callable[[List[Conjecture], pd.DataFrame], List[Conjecture]]] = {}

def register_post(name: str):
    """
    Decorator to register a post-processing function under a given name.

    Parameters
    ----------
    name : str
        The name under which to register the function.

    Returns
    -------
    Callable
        A decorator that registers the function.
    """
    def deco(fn: Callable[[List[Conjecture], pd.DataFrame], List[Conjecture]]):
        _POST_FUNCS[name] = fn
        return fn
    return deco

def list_posts() -> list[str]:
    """
    Return all registered post-processor names.

    Returns
    -------
    list of str
        Names of all registered post-processing functions.
    """
    return list(_POST_FUNCS.keys())

def get_post(name: str) -> Callable[[List[Conjecture], pd.DataFrame], List[Conjecture]]:
    """
    Retrieve a registered post-processing function by name.

    Parameters
    ----------
    name : str
        The name of the post-processing function to retrieve.

    Returns
    -------
    Callable[[List[Conjecture], pandas.DataFrame], List[Conjecture]]
        The registered post-processing function.

    Raises
    ------
    ValueError
        If no post-processing function is registered under the given name.
    """
    try:
        return _POST_FUNCS[name]
    except KeyError:
        raise ValueError(f"No such post‐processor: {name!r}")

