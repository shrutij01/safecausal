"""
Rich-integrated, env-configurable breakpoints.

• `breakpoint()` everywhere in code            → pretty traceback + chosen debugger.
• `with debug_on_exception(): …` context       → debugger only if an error occurs.
• Control with:
    export PYTHONBREAKPOINT=ipdb.set_trace   # IPython debugger
    export PYTHONBREAKPOINT=pdb.post_mortem  # stdlib post-mortem (default)
    export PYTHONBREAKPOINT=0                # disable breakpoints in prod
"""

from __future__ import annotations
import importlib, os, sys
from contextlib import contextmanager
from typing import Callable, Tuple, Any

import time
import tracemalloc

from rich.console import Console
from rich.traceback import Traceback
from rich.table import Table

import signal
import textwrap

console = Console()


# --------------------------------------------------------------------------- #
# "ipdb" → ipdb.set_trace, honour PYTHONBREAKPOINT
# --------------------------------------------------------------------------- #
def _get_debugger_callable() -> Callable[[], None]:
    hook: str | None = os.getenv("PYTHONBREAKPOINT", "pdb.post_mortem")

    # Disabled?
    if hook in ("0", "None", "", None):
        return lambda: None  # no-op

    # Bare module name?  (e.g. "ipdb")  → treat as ".set_trace"
    if "." not in hook:
        hook += ".set_trace"

    mod_name, attr = hook.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


_DEBUGGER = _get_debugger_callable()  # resolved once


# --------------------------------------------------------------------------- #
# pretty printed breakpoint hook
# --------------------------------------------------------------------------- #
def rich_breakpointhook(
    *exc_info: Tuple[Any, ...]
) -> None:  # signature flexible
    """
    Replacement for sys.breakpointhook used by builtin breakpoint().

    • Pretty-prints the current exception (if any) with Rich.
    • Then invokes the chosen debugger callable.
    """
    if exc_info and exc_info[0] is not None:  # called from post-mortem
        console.print(Traceback.from_exception(*exc_info))
    else:  # called from plain breakpoint()
        console.print("[bold yellow]Breakpoint hit[/bold yellow]")

    _DEBUGGER()  # enter debugger (or no-op)


# Activate it globally
sys.breakpointhook = rich_breakpointhook


# --------------------------------------------------------------------------- #
# open debugger on exception
# --------------------------------------------------------------------------- #
@contextmanager
def debug_on_exception(exc_type: type[BaseException] = BaseException):
    """
    Usage:
        with debug_on_exception():          # any error triggers debugger
            ...
        with debug_on_exception(Exception): # exclude KeyboardInterrupt, etc.
            ...
    """
    try:
        yield
    except exc_type:
        breakpoint()  # -> rich_breakpointhook
        raise  # keep normal error propagation


@contextmanager
def profile_block(label: str = "block"):
    """
    Measure wall-clock time and Python-level RAM usage inside a with-block.

    Example
    -------
    >>> with profile_block("training-step"):
    ...     train_one_step(model, batch)
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    yield  # ← run the code inside the with-block
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    table = Table(title=f"Profiling – {label}")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("time (s)", f"{elapsed:.4f}")
    table.add_row("RAM current", f"{current/1e6:.1f} MB")
    table.add_row("RAM peak", f"{peak/1e6:.1f} MB")
    console.print(table)


# ──────────────────────────────────────────────────────────────────────────────
# Live "panic button": press *Ctrl-\*  (SIGQUIT)  or run
#                      kill -USR1 <pid>
# to drop straight into the debugger from anywhere.
# ──────────────────────────────────────────────────────────────────────────────
_DEBUG_CALL = _DEBUGGER if _DEBUGGER else lambda: None


def _live_debugger(sig, frame):  # same signature for any signal
    console.print(
        f"[bold cyan]Signal {sig} caught – entering live debugger…[/bold cyan]"
    )
    _DEBUG_CALL()  # opens ipdb/pdb/etc.


# Register SIGUSR1 (manual: `kill -USR1 <pid>`)
try:
    signal.signal(signal.SIGUSR1, _live_debugger)
except AttributeError:  # Windows has no SIGUSR1
    pass

# Register SIGQUIT, which is emitted by ⇧Ctrl-\  in the controlling TTY
try:
    signal.signal(signal.SIGQUIT, _live_debugger)
except AttributeError:  # not present on some systems
    pass


# Helpful banner so you don't have to look up the PID every time
console.print(
    textwrap.dedent(
        f"""
    [green]
    ──────────────────────────────────────────────────────────
    ▶  Live-debug hotkeys enabled
        • Press [bold]Ctrl-\\[/bold] in this terminal
        • or run  [bold]kill -USR1 {os.getpid()}[/bold]  from another shell
    ──────────────────────────────────────────────────────────
    [/green]"""
    ).strip()
)
