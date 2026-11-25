"""Socket-level guard: no wizard test may open a real outbound connection.

The onboarding wizard validates LLM credentials with a *live* provider request
(``validate_provider_credentials``, ``surfaces/shared/llm_setup/validation.py``). Every
wizard test that walks the API-key path must therefore stub that call out. The
danger is that a *mis-patched* stub fails open rather than closed:

* ``pytest.ini`` declares no network blocking and ``pytest-socket`` is not a
  dependency, so nothing else in this repo stops an outbound request.
* ``validation.py`` caches the provider SDK classes in module globals via its
  lazy ``_load_anthropic_client`` / ``_load_openai_client`` helpers. As
  ``tests/cli/wizard/test_validation.py`` already documents, those loaders can
  silently overwrite a monkeypatched client with the *real* SDK.
* ``tests/conftest.py`` loads ``.env`` with ``override=True``, and developer
  machines routinely export ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``. A real
  credential is therefore usually within reach of the test process.

Combined, a test that patches the wrong name (e.g. patching
``validation.validate_provider_credentials`` while ``flow`` holds its own
imported reference) would quietly call ``api.anthropic.com`` with a live key and
still go **green** — passing for the wrong reason and making the assertions
above it vacuous.

This fixture removes that failure mode: any attempt to open an outbound socket
during a wizard test raises a loud ``AssertionError`` naming the fix.

The guard allows loopback (``127.0.0.1`` / ``::1`` / ``localhost``) connections
only. That allowlist exists for a single reason: on Windows, ``asyncio``'s
default ``ProactorEventLoop`` builds its self-pipe with ``socket.socketpair()``,
which CPython emulates over a TCP loopback pair (``_fallback_socketpair`` in
``socket.py``) because Windows has no native ``socketpair`` syscall. Blocking
loopback therefore makes *any* event-loop creation fail on Windows — e.g. every
``prompt_toolkit`` ``application.run()`` in the prompt tests — while passing on
Unix, where ``socketpair`` is a native syscall that never calls ``connect()``.
Loopback traffic cannot leave the host, so allowing it does not weaken the
guard's purpose: connections to any non-loopback address still raise.
"""

from __future__ import annotations

import socket

import pytest

_MESSAGE = (
    "A wizard test attempted a real outbound network connection. "
    "Wizard tests must never touch the network: stub the live provider probe with "
    'monkeypatch.setattr(flow, "validate_provider_credentials", ...) '
    "— patching the name as imported into the module under test, not the module it "
    "was defined in. See tests/cli/wizard/conftest.py."
)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class LiveNetworkCallBlocked(BaseException):
    """Raised when a wizard test tries to open a real socket.

    Deliberately derives from ``BaseException``, **not** ``Exception``.

    The provider SDKs wrap transport failures: ``anthropic`` and ``openai`` both
    catch ``Exception`` around the request and re-raise it as
    ``APIConnectionError("Connection error.")``. An ``Exception``-derived guard
    (e.g. a plain ``AssertionError``) is therefore *swallowed* by the very code
    path it exists to police, and a mis-patched test sees a tidy
    ``ValidationResult(ok=False, detail="Validation request failed: ...")``
    instead of a crash — which a failure-path test would happily accept, passing
    for exactly the wrong reason.

    Deriving from ``BaseException`` puts this outside every ``except Exception``
    in the SDK stack, so a blocked connection always surfaces as a hard test
    error that names the fix.
    """


def _is_loopback(address: object) -> bool:
    """Return True for TCP/UDP loopback targets; Unix sockets etc. are not loopback."""
    return isinstance(address, tuple) and len(address) >= 1 and address[0] in _LOOPBACK_HOSTS


@pytest.fixture(autouse=True)
def _no_live_provider_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any wizard test opens a non-loopback socket."""

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def _guarded_connect(self: socket.socket, address: object) -> None:
        if _is_loopback(address):
            real_connect(self, address)  # type: ignore[arg-type]
            return
        raise LiveNetworkCallBlocked(_MESSAGE)

    def _guarded_connect_ex(self: socket.socket, address: object) -> int:
        if _is_loopback(address):
            return real_connect_ex(self, address)  # type: ignore[arg-type]
        raise LiveNetworkCallBlocked(_MESSAGE)

    def _guarded_create_connection(
        address: object, *args: object, **kwargs: object
    ) -> socket.socket:
        if _is_loopback(address):
            return real_create_connection(address, *args, **kwargs)  # type: ignore[arg-type]
        raise LiveNetworkCallBlocked(_MESSAGE)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", _guarded_create_connection)
