"""Regression test for serve.py main binding through bind_host_port()."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_main_uses_configured_bind_address(monkeypatch):
    import serve

    captured = {}

    class FakeGame:
        def loop(self):
            return None

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    class FakeServer:
        def __init__(self, address, handler):
            captured["address"] = address

        def serve_forever(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(serve, "bind_host_port", lambda: ("127.0.0.9", 9876))
    monkeypatch.setattr(serve, "load_net", lambda cfg: (object(), "test"))
    monkeypatch.setattr(serve, "Game", lambda cfg, net, source: FakeGame())
    monkeypatch.setattr(serve.threading, "Thread", FakeThread)
    monkeypatch.setattr(serve, "ThreadingHTTPServer", FakeServer)

    serve.main()

    assert captured["address"] == ("127.0.0.9", 9876)
