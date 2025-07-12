# txgraffiti/utils/safe_generator.py
from typing import Callable, Generator, Any, Iterable
from txgraffiti.utils.logging_config import setup_logger

logger = setup_logger("txgraffiti.generators")

def safe_generator(func: Callable[..., Iterable]) -> Callable[..., Generator]:
    """
    Wraps a generator function to yield only successful results,
    logging errors instead of halting iteration.

    Parameters
    ----------
    func : Callable[..., Iterable]
        The generator function to wrap.

    Returns
    -------
    Callable[..., Generator]
        A wrapped generator that logs and skips failures.
    """
    def wrapper(*args, **kwargs):
        try:
            for result in func(*args, **kwargs):
                yield result
        except Exception as e:
            logger.exception(f"Generator {func.__name__} failed with args={args}, kwargs={kwargs}")
            return  # Stop iteration gracefully

    return wrapper
