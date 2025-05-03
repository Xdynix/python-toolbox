"""Provides utilities for graceful shutdown handling in async applications.

This module implements signal handling mechanisms that allow for graceful shutdown of
async applications, with support for preserving existing signal handlers and nested
contexts.
"""

# ruff: noqa: T201

__all__ = (
    "GRACEFUL",
    "graceful_shutdown",
    "wait_or_timeout",
)

import asyncio
import signal
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager, suppress
from types import FrameType
from typing import Any

# Mapping from signal numbers to a list of events waiting for that signal.
signal_events: dict[int, list[asyncio.Event]] = defaultdict(list)

SignalHandler = Callable[[int, FrameType | None], Any] | int | None
LoopEntry = tuple[Callable[..., Any], tuple[Any, ...]] | None
# Store the previous handlers per signal so we can restore them later.
previous_handlers: dict[int, tuple[SignalHandler, LoopEntry]] = {}


def dispatch(signum: int) -> None:
    loop = asyncio.get_running_loop()
    for event in signal_events[signum].copy():
        loop.call_soon_threadsafe(event.set)


def add_handler(signum: int) -> None:
    # Avoid double-wrapping.
    if signum in previous_handlers:
        return

    loop = asyncio.get_running_loop()

    # 1. Capture handler from signal.signal() if one exists.
    prev_sig = signal.getsignal(signum)

    # 2. Capture handler from loop.add_signal_handler(), if present.
    prev_loop_entry = None
    handler = getattr(loop, "_signal_handlers", {}).get(signum)
    if handler is not None:
        # Extract the user callback and its args from Handle internals.
        callback = getattr(handler, "_callback", None)
        args = getattr(handler, "_args", ())
        if callback is not None:
            prev_loop_entry = (callback, args)

    def chain_and_dispatch(s: int = signum, f: FrameType | None = None) -> None:
        """Top-level handler that runs when the signal occurs.

        Order:
            1. Our `dispatch()` sets all subscribed asyncio.Events
            2. Previous `signal.signal()` handler, called with (signum, frame).
            3. Previous loop handler, called with its original args.

        We intentionally do *not* catch exceptions or fix signature mismatches,
        so developers see errors early and can correct bad handlers.
        """
        dispatch(s)

        if callable(prev_sig) and prev_sig is not chain_and_dispatch:
            prev_sig(s, f)

        if prev_loop_entry is not None:
            cb, a = prev_loop_entry
            cb(*a)

    try:
        # Try to register via asyncio (UNIX only).
        loop.add_signal_handler(signum, chain_and_dispatch)
    except (AttributeError, NotImplementedError):
        # Fallback for Windows or unsupported loops.
        signal.signal(signum, chain_and_dispatch)

    previous_handlers[signum] = (prev_sig, prev_loop_entry)


def remove_handler(signum: int) -> None:
    # Still active listeners? Don't remove yet.
    if signum not in signal_events or signal_events[signum]:
        return

    loop = asyncio.get_running_loop()

    with suppress(AttributeError, NotImplementedError):
        loop.remove_signal_handler(signum)

    prev_sig, prev_loop_entry = previous_handlers.pop(signum)
    if prev_loop_entry is not None:
        cb, args = prev_loop_entry
        loop.add_signal_handler(signum, cb, *args)
    else:
        signal.signal(signum, prev_sig)


GRACEFUL = (signal.SIGINT, signal.SIGTERM)


@asynccontextmanager
async def graceful_shutdown(
    signals: Iterable[int] = GRACEFUL,
) -> AsyncIterator[asyncio.Event]:
    """Context manager that helps handle graceful shutdown on specified signals.

    This context manager sets up signal handlers for the specified signals and yields an
    Event that will be set when any of those signals are received. This allows for a
    graceful shutdown of async applications.

    Args:
        signals: An iterable of signal numbers to handle. Defaults to GRACEFUL (SIGINT
            and SIGTERM).

    Yields:
        asyncio.Event: An event that will be set when any of the specified signals are
            received.

    Example:
        async with graceful_shutdown() as stop:
            while not stop.is_set():
                await some_async_work()
    """
    stop = asyncio.Event()

    for sig in signals:
        signal_events[sig].append(stop)
        add_handler(sig)

    try:
        yield stop
    finally:
        for sig in signals:
            signal_events[sig].remove(stop)
            remove_handler(sig)


async def wait_or_timeout(event: asyncio.Event, timeout: float) -> bool:
    """Wait for an event with a timeout.

    Args:
        event: The event to wait for.
        timeout: Maximum time to wait in seconds.

    Returns:
        bool: True if the event was set, False if the timeout occurred.
    """
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(event.wait(), timeout)
    return event.is_set()


async def demo() -> None:
    import os

    print(f"graceful_shutdown() demo: pid={os.getpid()}")
    print("Press Ctrl+C to exit...\n")

    # Existing signal handlers are preserved.
    def previous_handler(signum: int, _: FrameType | None) -> None:
        print(f"Previous Handler Called: {signum}")

    loop = asyncio.get_running_loop()
    for sig in GRACEFUL:
        try:
            loop.add_signal_handler(sig, previous_handler, sig, None)
        except (AttributeError, NotImplementedError):
            signal.signal(sig, previous_handler)

    async def nested() -> None:
        # `graceful_shutdown()` can be nested.
        async with graceful_shutdown() as nested_stop:
            print("Nested Start")
            while not nested_stop.is_set():
                # Use `wait_or_timeout()` in poll style to avoid blocking when sleeping.
                await wait_or_timeout(nested_stop, 5)
            print("Nested Stop")

    async with graceful_shutdown() as stop:
        print("Main Start")
        await asyncio.gather(nested(), stop.wait())
        print("Main Stop")


if __name__ == "__main__":
    asyncio.run(demo())
