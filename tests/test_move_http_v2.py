"""HTTP integration test for the local POST /move endpoint."""

import http.client
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_http_move_endpoint_returns_a_legal_move():
    import chess
    import serve
    from config import Config
    from model import ChessNet

    cfg = Config()
    cfg.device = "cpu"
    cfg.num_res_blocks = 1
    cfg.num_filters = 8
    cfg.num_simulations = 4
    net = ChessNet(cfg).eval()
    serve.Handler.game = serve.Game(cfg, net, "test")

    server = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps({"fen": chess.Board().fen(), "sims": 4, "seed": 7})
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        conn.request("POST", "/move", body=body, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = json.loads(response.read())
        conn.close()
        assert response.status == 200, payload
        assert chess.Move.from_uci(payload["move"]) in chess.Board().legal_moves
        assert payload["model"]["source"] == "test"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
