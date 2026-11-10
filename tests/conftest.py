import socket

import pytest

# The pre-refactor suite targets private APIs that no longer exist; it is kept
# for reference only and is not collected.
collect_ignore = ["legacy", "typing"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that tries to open a real network connection."""

    def guard(*args, **kwargs):
        raise RuntimeError(f"tests must not use the network (attempted: {args[1:2] or args})")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
