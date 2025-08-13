from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

import pytest
from pytest_mock import MockerFixture
from ttl_cache import ttl_cache


class TestTTLCache:
    def test_function_decoration(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value=10)
        cached_func = ttl_cache(ttl=10.0)(mock_func)

        assert cached_func(5) == 10
        assert mock_func.call_count == 1

        assert cached_func(5) == 10
        assert mock_func.call_count == 1

        assert cached_func(3) == 10
        assert mock_func.call_count == 2

    def test_method_decoration(self) -> None:
        class Foobar:
            def __init__(self) -> None:
                self.count = 0

            @ttl_cache(ttl=10.0)
            def call(self) -> int:
                self.count += 1
                return self.count

        foobar = Foobar()
        assert foobar.call() == 1
        assert foobar.call() == 1
        assert Foobar.call(foobar) == 1

    def test_method_decoration_multiple_instance(self) -> None:
        class Foobar:
            def __init__(self) -> None:
                self.count = 0

            @ttl_cache(ttl=10.0)
            def echo(self, a: int) -> int:
                self.count += 1
                return a

        foobar1 = Foobar()
        foobar2 = Foobar()

        assert foobar1.echo(3) == 3
        assert foobar1.echo(3) == 3
        assert foobar1.count == 1
        assert foobar2.count == 0

        assert foobar2.echo(3) == 3
        assert foobar2.echo(5) == 5
        assert foobar1.count == 1
        assert foobar2.count == 2

    def test_class_method_decoration(self) -> None:
        count = 0

        class Foobar:
            @classmethod
            @ttl_cache(ttl=10.0)
            def call(cls) -> int:
                nonlocal count
                count += 1
                return count

        assert Foobar.call() == 1
        assert Foobar.call() == 1

    def test_cache_info(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value=42)
        cached_func = ttl_cache(ttl=10.0, maxsize=128, typed=False)(mock_func)

        info = cached_func.cache_info()
        assert info.hits == 0
        assert info.misses == 0
        assert info.maxsize == 128
        assert info.currsize == 0

        cached_func(5)
        info = cached_func.cache_info()
        assert info.hits == 0
        assert info.misses == 1
        assert info.currsize == 1

        cached_func(5)
        info = cached_func.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.currsize == 1

    def test_cache_parameters(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock()
        cached_func = ttl_cache(ttl=5.0, maxsize=64, typed=True)(mock_func)

        params = cached_func.cache_parameters()
        expected = {
            "ttl": 5.0,
            "maxsize": 64,
            "typed": True,
        }
        assert params == expected

    def test_cache_clear(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value=42)
        cached_func = ttl_cache(ttl=10.0)(mock_func)

        cached_func(1)
        cached_func(2)
        cached_func(1)

        info = cached_func.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.currsize == 2

        cached_func.cache_clear()

        info = cached_func.cache_info()
        assert info.hits == 0
        assert info.misses == 0
        assert info.currsize == 0

    def test_ttl_expiration(self, mocker: MockerFixture) -> None:
        mock_time = mocker.patch("time.monotonic")
        mock_func = mocker.Mock(return_value=10)
        cached_func = ttl_cache(ttl=1.0)(mock_func)

        mock_time.return_value = 0.0
        cached_func(5)
        assert mock_func.call_count == 1

        mock_time.return_value = 0.5
        cached_func(5)
        assert mock_func.call_count == 1

        mock_time.return_value = 1.5
        cached_func(5)
        assert mock_func.call_count == 2

    def test_lru_eviction_with_maxsize(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value=42)
        cached_func = ttl_cache(ttl=10.0, maxsize=2)(mock_func)

        cached_func(1)
        cached_func(2)
        # key order: [1, 2]
        assert mock_func.call_count == 2
        assert cached_func.cache_info().currsize == 2

        cached_func(1)
        # key order: [2, 1]
        assert mock_func.call_count == 2

        cached_func(3)
        # key order: [1, 3]
        assert mock_func.call_count == 3
        assert cached_func.cache_info().currsize == 2

        cached_func(2)
        # key order: [3, 2]
        assert mock_func.call_count == 4

        cached_func(1)
        # key order: [2, 1]
        assert mock_func.call_count == 5

    def test_typed_is_false(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value="result")
        cached_func = ttl_cache(ttl=10.0, typed=False)(mock_func)

        cached_func(1)
        cached_func(1.0)
        assert mock_func.call_count == 1

    def test_typed_is_true(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value="result")
        cached_func = ttl_cache(ttl=10.0, typed=True)(mock_func)

        cached_func(1)
        cached_func(1.0)
        assert mock_func.call_count == 2

    def test_kwargs_handling(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value=6)
        cached_func = ttl_cache(ttl=10.0)(mock_func)

        result1 = cached_func(1, 2, 3)
        result2 = cached_func(1, y=2, z=3)
        result3 = cached_func(x=1, y=2, z=3)
        assert result1 == result2 == result3 == 6
        assert mock_func.call_count == 3

    def test_thread_safety(self, mocker: MockerFixture) -> None:
        def slow() -> int:
            sleep(0.02)
            return 42

        mock_func = mocker.Mock(side_effect=slow)
        cached_func = ttl_cache(ttl=10.0)(mock_func)

        tasks = 30
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cached_func) for _ in range(tasks)]
            results = [future.result() for future in as_completed(futures)]

        assert all(result == 42 for result in results)
        assert mock_func.call_count >= 1

        info = cached_func.cache_info()
        assert info.hits >= 1
        assert info.hits + info.misses == tasks

    def test_unbounded_cache(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value=42)
        cached_func = ttl_cache(ttl=10.0, maxsize=None)(mock_func)

        for i in range(200):
            cached_func(i)

        assert mock_func.call_count == 200

        info = cached_func.cache_info()
        assert info.maxsize is None
        assert info.currsize == 200

    def test_wrapper_attributes(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock()
        mock_func.__name__ = "test_function"
        mock_func.__qualname__ = "foobar.test_function"
        mock_func.__doc__ = "Test function docstring."
        mock_func.__module__ = "test_module"

        cached_func = ttl_cache(ttl=10.0)(mock_func)

        assert cached_func.__name__ == "test_function"  # type: ignore[attr-defined]
        assert cached_func.__qualname__ == "foobar.test_function"  # type: ignore[attr-defined]
        assert cached_func.__doc__ == "Test function docstring."
        assert cached_func.__module__ == "test_module"
        assert cached_func.__wrapped__ is mock_func

    def test_zero_ttl(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value=10)
        cached_func = ttl_cache(ttl=0.0)(mock_func)

        cached_func(5)
        cached_func(5)

        assert mock_func.call_count == 2

    def test_negative_ttl(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value=10)
        cached_func = ttl_cache(ttl=-1.0)(mock_func)

        cached_func(5)
        cached_func(5)

        assert mock_func.call_count == 2

    def test_zero_maxsize(self, mocker: MockerFixture) -> None:
        mock_func = mocker.Mock(return_value=10)
        cached_func = ttl_cache(ttl=10.0, maxsize=0)(mock_func)

        cached_func(5)
        cached_func(5)

        assert mock_func.call_count == 2

        info = cached_func.cache_info()
        assert info.hits == 0
        assert info.misses == 2
        assert info.currsize == 0

    def test_function_raising_exception(self, mocker: MockerFixture) -> None:
        def error_func(x: int) -> int:
            if x < 0:
                raise ValueError("Negative value")
            return x

        mock_func = mocker.Mock(side_effect=error_func)
        cached_func = ttl_cache(ttl=10.0)(mock_func)

        cached_func(5)
        cached_func(5)

        assert mock_func.call_count == 1

        with pytest.raises(ValueError):
            cached_func(-1)

        with pytest.raises(ValueError):
            cached_func(-1)

        assert mock_func.call_count == 3
