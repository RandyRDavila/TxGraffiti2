# txgraffiti2025/reporting.py
from __future__ import annotations

import sys
import os
import io
import datetime as _dt
from contextlib import contextmanager


class Tee(io.TextIOBase):
    """
    Write-through text stream that mirrors writes to multiple targets.

    This is used to duplicate stdout/stderr into a log file while still
    showing output in the console.
    """
    def __init__(self, *streams: io.TextIOBase):
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def _hr(ch: str = "─", n: int = 80) -> str:
    """Return a horizontal rule string of length n."""
    return ch * n


def _now_stamp() -> str:
    """Current local timestamp as YYYY-MM-DD HH:MM:SS."""
    # Assumes local timezone (e.g., America/Chicago in your environment).
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def tee_report(
    filepath: str,
    *,
    title: str = "TxGraffiti Run Report",
    include_stderr: bool = True,
):
    """
    Context manager that duplicates all prints to `filepath` (UTF-8)
    while still printing to the original stdout/stderr.

    It also writes a simple header and footer with timestamps.

    Example
    -------
    >>> with tee_report("reports/run.txt", title="My Run"):
    ...     print("Hello")
    ...     # all your usual print() calls go here
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    f = open(filepath, "w", encoding="utf-8", newline="\n")

    # Header
    header = (
        f"{_hr()}\n"
        f"{title}\n"
        f"{_hr()}\n"
        f"Started: {_now_stamp()}\n"
        f"Working dir: {os.getcwd()}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"{_hr()}\n"
    )
    f.write(header)
    f.flush()

    old_out = sys.stdout
    old_err = sys.stderr

    try:
        # Mirror stdout to console + file
        sys.stdout = Tee(old_out, f)
        # Optionally mirror stderr as well
        if include_stderr:
            sys.stderr = Tee(old_err, f)
        yield
    finally:
        # Footer
        print(_hr())
        print("END OF REPORT")
        print(f"Finished: {_now_stamp()}")
        print(_hr())

        sys.stdout.flush()
        if include_stderr:
            sys.stderr.flush()

        # Restore original streams
        sys.stdout = old_out
        sys.stderr = old_err

        f.close()
