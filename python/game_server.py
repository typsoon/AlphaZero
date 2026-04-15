"""HTTP Game Server for Connect-4 with centralized state management."""

import json
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

from agent import AlphaZeroAgent
from connect4 import Connect4

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _to_cpp_inference_game_state(board_state: list[list[int]]) -> dict[str, list[list[int]]]:
    """Convert engine board encoding (0, 1, -1) to schema encoding (0, 1, 2)."""
    converted_board: list[list[int]] = []
    for row in board_state:
        converted_row: list[int] = []
        for cell in row:
            if cell == -1:
                converted_row.append(2)
            else:
                converted_row.append(cell)
        converted_board.append(converted_row)
    return {"board": converted_board}


class GameHandler(BaseHTTPRequestHandler):
    """HTTP request handler for game state and moves."""

    protocol_version = "HTTP/1.1"  # Enable HTTP/1.1 for keep-alive support
    game_instance = None
    ai_agent = None

    def do_GET(self):
        """Handle GET requests for game state."""
        logger.info(f"Received GET request: {self.path}")
        if self.path == "/game/status":
            try:
                board_state = self.game_instance.get_board_state()
                legal_actions = self.game_instance.get_legal_actions()
                is_terminal = self.game_instance.is_terminal

                response = {
                    "status": "ok",
                    "board": board_state,
                    "legal_actions": legal_actions,
                    "is_terminal": is_terminal,
                    "current_player": 1,  # Placeholder; could track whose turn
                }
                logger.info(
                    f"Sending status response: terminal={is_terminal}, legal_actions={legal_actions}"
                )

                response_body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            except Exception as e:
                # TODO: fix it so stack overflow attacks are not possible
                logger.error(f"An error occurred after receiving GET response: {e}")
                self._send_error(f"Failed to get game status: {e}")

        elif self.path == "/game/reset":
            try:
                self.game_instance.reset()
                response = {
                    "status": "ok",
                    "message": "Game reset",
                }
                response_body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            except Exception as e:
                # TODO: fix it so stack overflow attacks are not possible
                logger.info("An error occured after receiving a request to reset the game ", e)
                self._send_error(f"Failed to reset game: {e}")

        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        """Handle POST requests for moves."""
        logger.info(f"Received POST request: {self.path}")
        if self.path == "/game/move":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                logger.info(f"Received request body: {body.decode('utf-8')}")
                request_data = json.loads(body.decode("utf-8"))

                column = request_data.get("column")
                logger.info(f"Parsed move: column={column}")
                if column is None:
                    raise ValueError("Missing 'column' in request")

                if not isinstance(column, int) or column < 0 or column > 6:
                    raise ValueError("Column must be integer 0-6")

                legal_actions = self.game_instance.get_legal_actions()
                if column not in legal_actions:
                    raise ValueError(f"Illegal move: column {column} not available")

                # Player move
                logger.info(f"Executing player move: column={column}")
                self.game_instance.step(column)

                # Check if game is over
                if self.game_instance.is_terminal:
                    logger.info("Game over after player move")
                    response = {
                        "status": "ok",
                        "message": "Game over",
                        "board": self.game_instance.get_board_state(),
                        "is_terminal": True,
                    }
                    response_body = json.dumps(response).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(response_body)))
                    self.end_headers()
                    self.wfile.write(response_body)
                    return

                # AI move
                logger.info("Requesting AI move...")
                ai_game_state = _to_cpp_inference_game_state(
                    self.game_instance.get_board_state()
                )
                ai_move = self.ai_agent.act(ai_game_state)
                logger.info(f"AI selected column: {ai_move}")
                self.game_instance.step(ai_move)

                response = {
                    "status": "ok",
                    "player_column": column,
                    "ai_column": ai_move,
                    "board": self.game_instance.get_board_state(),
                    "legal_actions": self.game_instance.get_legal_actions(),
                    "is_terminal": self.game_instance.is_terminal,
                }
                logger.info(
                    f"Move complete. AI played: {ai_move}, terminal: {response['is_terminal']}"
                )

                response_body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            except ValueError as e:
                logger.error(f"Validation error in POST /game/move: {e}")
                self._send_error(str(e), 400)
            except Exception as e:
                logger.error(f"Error processing move: {e}", exc_info=True)
                self._send_error(f"Move failed: {e}")

        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def _send_error(self, message: str, status_code: int = 500):
        """Send error response."""
        response = {
            "status": "error",
            "message": message,
        }
        response_body = json.dumps(response).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def start_game_server(
    inference_socket: str = "/tmp/alphazero.sock",
    host: str = "localhost",
    port: int = 8000,
):
    """Start the game server.

    Args:
        inference_socket: Path to inference server Unix socket
        host: Host to bind to
        port: Port to bind to
    """
    # Initialize game and AI agent
    game = Connect4()
    game.reset()
    ai_agent = AlphaZeroAgent(socket_path=inference_socket)

    GameHandler.game_instance = game
    GameHandler.ai_agent = ai_agent

    # Create server
    server = HTTPServer((host, port), GameHandler)
    print(f"Game server listening on {host}:{port}")
    print(f"Inference server: {inference_socket}")
    print("API Endpoints:")
    print("  GET  /game/status   - Get current game state")
    print("  POST /game/move     - Make a move (send {'column': <0-6>})")
    print("  GET  /game/reset    - Reset game")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        ai_agent.close()
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Game Server")
    parser.add_argument(
        "--socket",
        type=str,
        default="/tmp/alphazero.sock",
        help="Path to inference server Unix socket",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )

    args = parser.parse_args()
    start_game_server(args.socket, args.host, args.port)


if __name__ == "__main__":
    main()
