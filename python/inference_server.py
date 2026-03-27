"""HTTP inference server using Unix socket for AlphaZero gameplay."""

import sys
import json
import argparse
import logging
from typing import Any
import torch
from http.server import BaseHTTPRequestHandler
import socketserver
import os
import signal
import threading

from pybind import engine_bind
from engine_bind import MCTS, Connect4  # pyright: ignore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MCTS inference."""

    protocol_version = "HTTP/1.1"  # Enable HTTP/1.1 for keep-alive support
    # Class variables to store MCTS instance and device
    mcts_instance: Any = None
    device: Any = None

    # TODO: make this method safe
    def do_POST(self):
        """Handle POST requests for inference."""
        logger.info(f"Received POST request: {self.path}")
        if self.path == "/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                logger.info(f"Received request body ({content_length} bytes)")
                request_data = json.loads(body.decode("utf-8"))

                if not hasattr(request_data, "__getitem__"):
                    raise ValueError("Invalid request format")

                # Get game state as serialized tensor
                game_state = request_data.get("game_state")
                if game_state is None:
                    raise ValueError("Missing game_state in request")

                logger.info(f"Received game state: {game_state}")
                
                # Create a Game object with the board state and device
                game = Connect4(game_state, self.device)

                logger.info("Running MCTS search...")
                # Call MCTS to get policy
                policy = self.mcts_instance.search(game)

                # Convert policy to list (ensure it's JSON serializable)
                policy_list = list(policy) if not isinstance(policy, list) else policy
                logger.info(f"MCTS search complete, policy: {policy_list}")

                response = {
                    "status": "success",
                    "policy": policy_list,
                }

                response_body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            except Exception as e:
                logger.error(f"Error processing inference request: {e}", exc_info=True)
                error_response = {
                    "status": "error",
                    "message": str(e),
                }
                response_body = json.dumps(error_response).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        else:
            logger.warning(f"404 - Unknown path: {self.path}")
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_GET(self):
        """Handle GET requests for health check."""
        logger.info(f"Received GET request: {self.path}")
        if self.path == "/health":
            response_body = json.dumps({"status": "healthy"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        else:
            logger.warning(f"404 - Unknown path: {self.path}")
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class UnixSocketHTTPServer(socketserver.UnixStreamServer):
    """HTTP server using Unix domain socket."""

    allow_reuse_address = True
    request_queue_size = 5


def start_inference_server(
    network_path: str, device: str, socket_path: str = "/tmp/alphazero.sock"
):
    """Start the inference server on a Unix socket.

    Args:
        network_path: Path to the trained network file
        device: Device to run inference on (cuda/cpu)
        socket_path: Path to Unix socket file
    """
    # Clean up existing socket file
    if os.path.exists(socket_path):
        logger.info(f"Removing existing socket file: {socket_path}")
        os.remove(socket_path)

    # Create MCTS instance
    logger.info(f"Initializing MCTS with network: {network_path}")
    device_obj = torch.device(device)
    InferenceHandler.mcts_instance = MCTS(network_path, device_obj)
    InferenceHandler.device = device_obj

    # Create and configure server
    handler = InferenceHandler
    server = UnixSocketHTTPServer(socket_path, handler)

    logger.info(f"Inference server listening on Unix socket: {socket_path}")
    logger.info(f"Network: {network_path}")
    logger.info(f"Device: {device}")

    # Shutdown flag
    shutdown_requested = threading.Event()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        shutdown_requested.set()
        # Shutdown in a separate thread to avoid deadlock
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    finally:
        logger.info("Cleaning up...")
        if os.path.exists(socket_path):
            os.remove(socket_path)
            logger.info(f"Removed socket file: {socket_path}")
        logger.info("Inference server stopped")


def main():
    parser = argparse.ArgumentParser(description="AlphaZero HTTP Inference Server")
    parser.add_argument(
        "--network-path",
        type=str,
        required=True,
        help="Path to the trained network file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--socket",
        type=str,
        default="/tmp/alphazero.sock",
        help="Unix socket path",
    )

    args = parser.parse_args()
    start_inference_server(args.network_path, args.device, args.socket)


if __name__ == "__main__":
    main()
