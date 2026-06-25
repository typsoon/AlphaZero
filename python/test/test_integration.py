import pytest
import subprocess
import time
import socket
import json
import requests
import os
import signal
from pathlib import Path
from python.utils import BUILD_DIR

SOCKET_PATH = str(Path("/tmp") / "alphazero_test.sock")
PORT = 8001


@pytest.fixture(scope="module")
def inference_server():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    executable = str(BUILD_DIR / "inference_server" / "inference_server")
    model_path = str(Path("inference_server") / "tests" / "payloads" / "dummy_model.pt")

    p = subprocess.Popen(
        [
            executable,
            "--socket",
            SOCKET_PATH,
            "--network-path",
            model_path,
            "--device",
            "cpu",
        ],
        preexec_fn=os.setsid,
    )

    for _ in range(50):
        if os.path.exists(SOCKET_PATH):
            time.sleep(0.1)
            break
        time.sleep(0.1)
    else:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        raise RuntimeError("Inference server failed to start")

    yield
    os.killpg(os.getpgid(p.pid), signal.SIGKILL)


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
    env["SOCKET_PATH"] = SOCKET_PATH
    env["PORT"] = str(PORT)

    p = subprocess.Popen(
        ["npm", "run", "start"],
        cwd="gameplay_server",
        env=env,
        preexec_fn=os.setsid,
    )

    for _ in range(50):
        try:
            resp = requests.get(
                f"http://localhost:{PORT}/agents", headers={"Connection": "close"}
            )
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
    else:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        raise RuntimeError("Game server failed to start")

    yield
    os.killpg(os.getpgid(p.pid), signal.SIGKILL)


def test_game_server_e2e(game_server):
    create_resp = requests.post(
        f"http://localhost:{PORT}/game/create",
        json={"p1_type": "human", "p2_type": "ai", "p2_agent": SOCKET_PATH},
        headers={"Connection": "close"},
    )
    assert create_resp.status_code == 200
    create_data = create_resp.json()
    assert create_data["status"] == "ok"
    game_id = create_data["game_id"]
    p1_id = create_data["p1_id"]

    status_resp = requests.get(
        f"http://localhost:{PORT}/game/{game_id}/status",
        headers={"Connection": "close"},
    )
    assert status_resp.status_code == 200

    move_resp = requests.post(
        f"http://localhost:{PORT}/game/{game_id}/move",
        json={"column": 3, "player_id": p1_id},
        headers={"Connection": "close"},
    )
    assert move_resp.status_code == 200

    resp_data = move_resp.json()
    assert resp_data["status"] == "ok"


def test_human_vs_human_full_game(game_server):
    create_resp = requests.post(
        f"http://localhost:{PORT}/game/create",
        json={"p1_type": "human", "p2_type": "human"},
        headers={"Connection": "close"},
    )
    assert create_resp.status_code == 200
    create_data = create_resp.json()
    assert create_data["status"] == "ok"
    game_id = create_data["game_id"]
    p1_id = create_data["p1_id"]
    p2_id = create_data["p2_id"]

    # Play a game: Player 1 plays in column 0, Player 2 plays in column 1.
    # P1: 0, P2: 1, P1: 0, P2: 1, P1: 0, P2: 1, P1: 0 -> P1 wins vertically.
    moves = [
        (0, p1_id),
        (1, p2_id),
        (0, p1_id),
        (1, p2_id),
        (0, p1_id),
        (1, p2_id),
        (0, p1_id),
    ]

    for col, player_id in moves:
        move_resp = requests.post(
            f"http://localhost:{PORT}/game/{game_id}/move",
            json={"column": col, "player_id": player_id},
            headers={"Connection": "close"},
        )
        assert move_resp.status_code == 200
        resp_data = move_resp.json()
        assert resp_data["status"] == "ok"

    assert resp_data.get("is_terminal") is True
