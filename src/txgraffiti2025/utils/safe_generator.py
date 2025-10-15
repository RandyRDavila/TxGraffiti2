"""
Decorator for safely wrapping generator functions.

Prevents generator crashes from halting larger conjecture runs.
"""

from functools import wraps

def safe_generator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            for x in fn(*args, **kwargs):
                yield x
        except Exception as e:
            import traceback
            print(f"[safe_generator] Skipping due to error: {e}")
            traceback.print_exc()
            return
    return wrapper
