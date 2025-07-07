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
    Decorator to register a post‐processor under `name`.
    A post‐processor is a function
      fn(conjs: List[Conjecture], df: pd.DataFrame) -> List[Conjecture]
    """
    def deco(fn: Callable[[List[Conjecture], pd.DataFrame], List[Conjecture]]):
        _POST_FUNCS[name] = fn
        return fn
    return deco

def list_posts() -> list[str]:
    """Return all registered post‐processor names."""
    return list(_POST_FUNCS.keys())

def get_post(name: str) -> Callable[[List[Conjecture], pd.DataFrame], List[Conjecture]]:
    try:
        return _POST_FUNCS[name]
    except KeyError:
        raise ValueError(f"No such post‐processor: {name!r}")

