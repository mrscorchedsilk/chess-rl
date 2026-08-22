"""HTTP hardening tests for local control/model endpoints."""

import http.client
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _request(handler, path, body, content_type):
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request("POST", path, body=body, headers={"Content-Type": content_type})
        response = conn.getresponse()
        payload = json.loads(response.read())
        conn.close()
        return response.status, payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_train_control_rejects_simple_cross_origin_content_type():
    import train_server

    status, payload = _request(
        train_server.Handler,
        "/api/control",
        '{"action":"stop"}',
        "text/plain",
    )
    assert status == 415, payload


def test_model_control_rejects_simple_cross_origin_content_type():
    import serve

    status, payload = _request(
        serve.Handler,
        "/control",
        '{"action":"pause"}',
        "text/plain",
    )
    assert status == 415, payload
