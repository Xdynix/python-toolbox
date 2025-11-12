"""
TTL-enabled LRU cache decorator.

Examples:
    >>> from time import sleep
    >>> @ttl_cache(ttl=0.1, maxsize=2)
    ... def add(a, b): return a + b
    >>> add(1, 2)  # miss, computes
    3
    >>> add.cache_info()
    CacheInfo(hits=0, misses=1, maxsize=2, currsize=1)
    >>> add(1, 2)  # hit (fresh)
    3
    >>> add.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=2, currsize=1)
    >>> sleep(0.15); add(1, 2)  # expired, recompute
    3
    >>> add.cache_info()
    CacheInfo(hits=1, misses=2, maxsize=2, currsize=1)

    >>> @ttl_cache(ttl=10, typed=True)
    ... def echo(x): return x
    >>> echo(1)
    1
    >>> echo(1.0)
    1.0
    >>> echo.cache_info()
    CacheInfo(hits=0, misses=2, maxsize=128, currsize=2)
"""

__all__ = ("ttl_cache",)

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, NamedTuple, Protocol, TypedDict, cast

NO_VALUE = object()
SENTINEL = object()


class Hashed[T](list[T]):
    __slots__ = ("hash_value",)

    def __init__(self, seq: Iterable[T], *, hash_: Callable[..., int] = hash) -> None:
        super().__init__(seq)
        self.hash_value = hash_(seq)

    def __hash__(self) -> int:  # type: ignore[override]
        return self.hash_value


class CacheParameters(TypedDict):
    """Cache configuration parameters."""

    ttl: float
    maxsize: int | None
    typed: bool


class CacheInfo(NamedTuple):
    """Statistics about the cache."""

    hits: int
    misses: int
    maxsize: int | None
    currsize: int


class TTLCached[**P, R](Protocol):
    __wrapped__: Callable[P, R]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    # NOTE: The generic Callable[..., R] return type here is intentionally broad to
    # support usage when the cached function is a method on a class (via descriptor
    # protocol). This means positional and keyword argument types are not preserved, so
    # static type checkers like `mypy` cannot verify method call arguments when accessed
    # via an instance. The return type is still preserved.
    # FIXME: Reveal proper argument types when supported.
    def __get__(self, instance: Any, owner: Any) -> Callable[..., R]: ...

    def cache_parameters(self) -> CacheParameters: ...
    def cache_info(self) -> CacheInfo: ...
    def cache_clear(self) -> None: ...


def ttl_cache[**P, R](
    *,
    ttl: float,
    maxsize: int | None = 128,
    typed: bool = False,
) -> Callable[[Callable[P, R]], TTLCached[P, R]]:
    """TTL-enabled LRU cache.

    Similar to ``functools.lru_cache``, but entries expire after ``ttl`` seconds.
    Maintains least-recently-used eviction order and supports ``typed=True`` behavior
    for distinguishing between argument types.

    Thread-safe for concurrent use.

    The decorated function gains these attributes and methods:
        - ``__wrapped__``: The original, unwrapped function.
        - ``cache_parameters()``: Return the configuration parameters of the cache.
        - ``cache_info()``: Report cache statistics (hits, misses, maxsize, currsize).
        - ``cache_clear()``: Clear the cache and reset statistics.

    Notes:
        - ``cache_info().currsize`` may include expired entries until a later call
          triggers trimming, there is no background cleanup.
        - LRU semantics: reads move the key to MRU; inserts are MRU. Because trimming
          inspects from the LRU end, expired MRU entries can linger and still count
          toward ``currsize``; eviction at ``maxsize`` may remove a fresh LRU first.
        - All arguments must be hashable.
        - Keyword argument order matters: ``func(a=1, b=2)`` and ``func(b=2, a=1)``
          are treated as different calls and will both invoke the original function.
        - Type checking caveat: when decorating methods, the internal ``__get__``
          implementation uses a generic ``Callable[..., R]`` return type to support
          bound method behavior, which hides the actual argument types from static type
          checkers. Return types are still preserved.

    Args:
        ttl (float): Time-to-live in seconds for each cache entry.
        maxsize (int | None): Max number of entries to keep (``None`` for unbounded).
        typed (bool): If ``True``, arguments of different types are cached separately.

    Returns:
        Callable: A decorator that applies TTL-enabled caching to a function.
    """

    def decorator(func: Callable[P, R]) -> TTLCached[P, R]:
        cache = OrderedDict[Hashed[Any], tuple[R, float]]()
        lock = threading.RLock()
        hits = 0
        misses = 0

        def make_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashed[Any]:
            """Build a key for this cache."""
            # Logic adapted from `functools._make_key` in CPython.
            key = list(args)
            if kwargs:
                key.append(SENTINEL)
                for item in kwargs.items():
                    key.extend(item)
            if typed:
                key.extend(map(type, args))
                key.extend(map(type, kwargs.values()))
            # Do not use `isinstance` because we want exact type check.
            elif len(key) == 1 and type(key[0]) in {int, str}:
                return cast(Hashed[Any], key[0])
            return Hashed(tuple(key))

        def trim_cache(now: float) -> None:  # pragma: no cover
            """Opportunistic cleanup."""
            while cache:
                first_key, (_, expiry) = next(iter(cache.items()))
                if now >= expiry:
                    del cache[first_key]
                else:
                    break

        def get_value(key: Hashed[Any], now: float) -> R | object:
            """Retrieve a value from the cache and update the recency of the key."""
            record = cache.get(key)
            if record is None:
                return NO_VALUE
            value, expiry = record
            if expiry <= now:  # pragma: no cover
                del cache[key]
                return NO_VALUE
            cache.move_to_end(key)
            return value

        def store_value(key: Hashed[Any], value: R, now: float) -> R:
            """Store value to the cache and maintain max size."""
            cache[key] = value, now + ttl
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return value

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal hits, misses

            key = make_key(args, kwargs)

            with lock:
                now = time.monotonic()
                trim_cache(now)
                cached = get_value(key, now)
                if cached is not NO_VALUE:
                    hits += 1
                    return cast(R, cached)
                misses += 1

            value = func(*args, **kwargs)

            with lock:
                now = time.monotonic()
                trim_cache(now)
                # Another thread may have filled it in the meantime, use the fresh one.
                cached = get_value(key, now)
                if cached is not NO_VALUE:
                    return cast(R, cached)
                return store_value(key, value, now)

        def cache_parameters() -> CacheParameters:
            """Return the configuration parameters of the cache.

            This is for information purposes only. Mutating the values has no effect.

            Returns:
                dict: A dictionary containing the cache configuration parameters.
            """
            return {
                "ttl": ttl,
                "maxsize": maxsize,
                "typed": typed,
            }

        def cache_info() -> CacheInfo:
            """Report cache statistics.

            Returns:
                tuple: A named tuple containing the number of hits, misses, maximum
                    size, and current size of the cache.
            """
            with lock:
                return CacheInfo(
                    hits=hits,
                    misses=misses,
                    maxsize=maxsize,
                    currsize=len(cache),
                )

        def cache_clear() -> None:
            """Clear the cache and cache statistics."""
            nonlocal hits, misses
            with lock:
                cache.clear()
                hits = 0
                misses = 0

        wrapper.cache_parameters = cache_parameters  # type: ignore[attr-defined]
        wrapper.cache_info = cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]

        return cast(TTLCached[P, R], wrapper)

    return decorator
