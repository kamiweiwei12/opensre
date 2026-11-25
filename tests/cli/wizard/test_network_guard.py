"""Regression tests for the loopback allowlist in the wizard network guard.

See ``tests/cli/wizard/conftest.py`` for the guard's full rationale. These
tests pin two properties at once:

* the guard still raises ``LiveNetworkCallBlocked`` for real outbound
  connections (its original purpose), and
* loopback connections remain usable — including the TCP loopback pair that
  ``socket.socketpair()`` falls back to on Windows, which
  ``ProactorEventLoop`` needs to create *any* asyncio event loop. Blocking
  loopback broke every ``prompt_toolkit`` prompt test on Windows.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from tests.cli.wizard.conftest import LiveNetworkCallBlocked


def test_guard_still_blocks_outbound_socket_connect() -> None:
    with pytest.raises(LiveNetworkCallBlocked):
        socket.socket().connect(("192.0.2.1", 80))  # TEST-NET-1, never routable


def test_guard_still_blocks_outbound_connect_ex() -> None:
    with pytest.raises(LiveNetworkCallBlocked):
        socket.socket().connect_ex(("192.0.2.1", 80))


def test_guard_still_blocks_outbound_create_connection() -> None:
    with pytest.raises(LiveNetworkCallBlocked):
        socket.create_connection(("192.0.2.1", 80), timeout=1)


def test_guard_allows_loopback_connect() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        client = socket.create_connection(listener.getsockname(), timeout=1)
        client.close()
    finally:
        listener.close()


def test_event_loop_creation_works_under_guard() -> None:
    """Regression: Windows' ProactorEventLoop self-pipe needs loopback.

    Fails on Windows with the previous absolute guard because
    ``_fallback_socketpair`` connects over 127.0.0.1; passes everywhere
    with the loopback allowlist.
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(asyncio.sleep(0))
    finally:
        loop.close()
