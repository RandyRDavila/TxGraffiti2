from typing import List, Iterator, Callable
from txgraffiti.logic import Conjecture

__all__ = [
    'register_gen',
    'list_gens',
]

_GEN_FUNCS: list[Callable[..., Iterator[Conjecture]]] = []

def register_gen(fn: Callable[..., Iterator[Conjecture]]) -> Callable[..., Iterator[Conjecture]]:
    """
    Use as a plain decorator:

        @register_gen
        def convex_hull(...): ...
    """
    _GEN_FUNCS.append(fn)
    return fn

def list_gens() -> list[Callable[..., Iterator[Conjecture]]]:
    """Return all registered generator functions."""
    return list(_GEN_FUNCS)
