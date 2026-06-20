import pytest
import subprocess
import time
import socket
import json
import requests
import os

SOCKET_PATH = "/tmp/alphazero_test.sock"
PORT = 8001


@pytest.fixture(scope="module")
def inference_server():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    executable = "build/inference_server/inference_server"
    model_path = "inference_server/tests/payloads/dummy_model.pt"

    p = subprocess.Popen(
        [
            executable,
            "--socket",
            SOCKET_PATH,
            "--network-path",
            model_path,
            "--device",
            "cpu",
        ]
    )

    for _ in range(50):
        if os.path.exists(SOCKET_PATH):
            time.sleep(0.1)
            break
        time.sleep(0.1)
    else:
        p.kill()
        raise RuntimeError("Inference server failed to start")

    yield
    p.kill()


def test_inference_server_api(inference_server):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)

    request_data = {
        "game_state": {
            "board": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        }
    }

    body = json.dumps(request_data)
    http_request = (
        "POST /predict HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n" + body
    )

    sock.sendall(http_request.encode("utf-8"))

    response = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        response += chunk

    response_str = response.decode("utf-8")
    assert "200 OK" in response_str.split("\r\n")[0]

    body_part = response_str.split("\r\n\r\n")[1]
    response_json = json.loads(body_part)

    assert "policy" in response_json


@pytest.fixture(scope="module")
def game_server(inference_server):
    env = os.environ.copy()
    env["PYTHONPATH"] = "python:build/training:build/engine:build"
    p = subprocess.Popen(
        [
            "python",
            "python/game_server.py",
            "--socket",
            SOCKET_PATH,
            "--port",
            str(PORT),
        ],
        env=env,
    )

    for _ in range(50):
        try:
            resp = requests.get(
                f"http://localhost:{PORT}/game/status", headers={"Connection": "close"}
            )
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
    else:
        p.kill()
        raise RuntimeError("Game server failed to start")

    yield
    p.kill()


def test_game_server_e2e(game_server):
    status_resp = requests.get(
        f"http://localhost:{PORT}/game/status", headers={"Connection": "close"}
    )
    assert status_resp.status_code == 200

    move_resp = requests.post(
        f"http://localhost:{PORT}/game/move",
        json={"column": 3},
        headers={"Connection": "close"},
    )
    assert move_resp.status_code == 200

    resp_data = move_resp.json()
    assert resp_data["status"] == "ok"
    assert "ai_column" in resp_data
