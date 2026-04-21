"""Local pytest helpers for async tests.

The project test suite uses ``@pytest.mark.asyncio`` but this repository keeps
runtime dependencies minimal and does not require ``pytest-asyncio``.
This hook runs coroutine tests directly on an event loop so contributors can run
the bundled test suite without installing extra plugins.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register local markers used by the suite."""
    config.addinivalue_line("markers", "asyncio: run test as an asyncio coroutine")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Execute ``@pytest.mark.asyncio`` tests without pytest-asyncio plugin."""
    if "asyncio" not in pyfuncitem.keywords:
        return None

    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        sig = inspect.signature(test_func)
        accepted_args = {
            name: value
            for name, value in pyfuncitem.funcargs.items()
            if name in sig.parameters
        }
        loop.run_until_complete(test_func(**accepted_args))
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
    return True
