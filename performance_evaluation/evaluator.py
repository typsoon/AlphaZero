import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import tempfile
import uuid
import http.client
from python.utils import PROJ_ROOT


class UnixSocketHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path, timeout=10):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def make_unix_request(socket_path, method, path, body=None):
    conn = UnixSocketHTTPConnection(socket_path)
    headers = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(body)
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    res_body = response.read().decode("utf-8")
    conn.close()
    if res_body:
        return json.loads(res_body)
    return {}


def wait_for_server(socket_path, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(socket_path):
            try:
                res = make_unix_request(socket_path, "GET", "/health")
                if res.get("status") == "healthy":
                    return True
            except Exception:
                pass
        time.sleep(0.5)
    return False


def build_game_state(game, data):
    """Build the `game_state` payload sent to /predict from a puzzle's JSON data."""
    if game == "chess":
        return {
            "board": data["board"],
            "player": data["player"],
            "en_passant": data["en_passant"],
            "castling": data["castling"],
        }
    return {"board": data["board"]}


def choose_move(game, policy):
    """Pick the argmax move from the /predict response's `policy` field.

    Connect4 returns a dense array (argmax over indices). Chess returns a sparse list of
    {"index": ..., "value": ...} entries (only legal moves have nonzero probability), so
    the chosen move is the `index` of the entry with the highest `value`.
    """
    if game == "chess":
        best = max(policy, key=lambda entry: entry["value"])
        return best["index"]
    return max(range(len(policy)), key=lambda i: policy[i])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaZero on puzzles for a given game."
    )
    parser.add_argument(
        "--network-path", required=True, help="Path to the trained network file"
    )
    parser.add_argument(
        "--inference-binary", required=True, help="Path to the inference server binary"
    )
    parser.add_argument(
        "--game",
        default="connect4",
        choices=["connect4", "chess"],
        help="Which game's puzzles to evaluate (default: connect4)",
    )
    parser.add_argument(
        "--mcts-search-depth", type=int, default=800, help="MCTS search depth"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output (individual PASS results)",
    )
    args = parser.parse_args()

    temp_dir = tempfile.mkdtemp(prefix=f"inference_service_{args.game}_")
    socket_path = str(Path(temp_dir) / f"socket_{uuid.uuid4().hex[:8]}.sock")

    cmd = [args.inference_binary]
    if args.inference_binary.endswith(".py"):
        cmd = [sys.executable, args.inference_binary]

    cmd.extend(
        [
            "--network-path",
            args.network_path,
            "--device",
            "cuda",
            "--socket",
            socket_path,
            "--mcts-search-depth",
            str(args.mcts_search_depth),
            "--game",
            args.game,
        ]
    )

    results = []
    print(f"Starting inference server: {' '.join(cmd)}")
    kwargs = {}
    if not args.verbose:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    server_process = subprocess.Popen(cmd, **kwargs)

    try:
        print(f"Waiting for inference server to start at {socket_path}...")
        if not wait_for_server(socket_path):
            stdout_data, stderr_data = server_process.communicate()
            error_msg = "Inference server failed to start or become healthy.\n"
            if not args.verbose:
                if stdout_data:
                    error_msg += f"Stdout:\n{stdout_data.decode('utf-8')}\n"
                if stderr_data:
                    error_msg += f"Stderr:\n{stderr_data.decode('utf-8')}\n"
            raise RuntimeError(error_msg)

        print("Server is ready. Starting evaluation...")

        game_files = list(
            (PROJ_ROOT / "performance_evaluation" / "games" / args.game).glob(
                "*/*.json"
            )
        )
        for file_path in sorted(game_files):
            category = file_path.parent.name
            test_name = file_path.name

            with open(file_path, "r") as f:
                data = json.load(f)

            board = data["board"]
            expected_moves = data["expected_moves"]

            payload = {"game_state": build_game_state(args.game, data)}

            response = make_unix_request(socket_path, "POST", "/predict", payload)

            if "policy" not in response:
                print(f"Error for {file_path}: Invalid response {response}")
                continue

            policy = response["policy"]
            chosen_move = choose_move(args.game, policy)

            value = response.get("value", 0.0)

            passed = chosen_move in expected_moves

            result = {
                "test_name": test_name,
                "category": category,
                "board": board,
                "expected_moves": expected_moves,
                "network_policy": policy,
                "network_value": value,
                "chosen_move": chosen_move,
                "passed": passed,
            }
            if args.game == "chess":
                result["player"] = data["player"]
            results.append(result)

            if passed:
                if args.verbose:
                    print(
                        f"\033[92m[PASS]\033[0m {category}/{test_name}: Expected {expected_moves}, Got {chosen_move}"
                    )
            else:
                print(
                    f"\033[91m[FAIL]\033[0m {category}/{test_name}: Expected {expected_moves}, Got {chosen_move}"
                )

    finally:
        print("Cleaning up inference server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Server didn't terminate gracefully, killing it...")
            server_process.kill()
            server_process.wait()

        # Write results
        results_path = PROJ_ROOT / "performance_evaluation" / "results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation complete. Results written to {results_path}")

        if results:
            passed_count = sum(1 for r in results if r["passed"])
            total_count = len(results)
            percentage = (passed_count / total_count) * 100

            if percentage == 100.0:
                color = "\033[92m"  # Green
            elif percentage > 50.0:
                color = "\033[93m"  # Yellow
            else:
                color = "\033[91m"  # Red

            print(
                f"{color}Tests Passed: {passed_count}/{total_count} ({percentage:.2f}%)\033[0m"
            )

        # Cleanup socket dir
        try:
            if os.path.exists(socket_path):
                os.remove(socket_path)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Failed to cleanup temp dir: {e}")


if __name__ == "__main__":
    main()
