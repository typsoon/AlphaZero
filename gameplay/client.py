"""HTTP Game Client for Connect-4 with Tkinter UI."""

import tkinter as tk
import json
import argparse
import socket
import sys
import logging
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CELL_SIZE = 80
ROWS = 6
COLS = 7


class GameClient:
    """HTTP client for communicating with game server."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        """Initialize game client.

        Args:
            host: Game server host
            port: Game server port

        Raises:
            RuntimeError: If unable to connect to server
        """
        self.host = host
        self.port = port
        self.sock = None
        self._connect()

    def _connect(self):
        """Establish connection to game server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)  # 5 second timeout
            self.sock.connect((self.host, self.port))
        except (ConnectionRefusedError, OSError) as e:
            self.sock = None
            raise RuntimeError(
                f"Failed to connect to game server at {self.host}:{self.port}. "
                f"Make sure the server is running. Error: {e}"
            )

    def _send_request(
        self, method: str, path: str, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Send HTTP request to game server.

        Args:
            method: HTTP method (GET, POST)
            path: Request path
            data: Optional JSON body for POST requests

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If request fails or response is invalid
        """
        body = json.dumps(data) if data else ""
        logger.info(f"Sending {method} request to {path}")
        if data:
            logger.info(f"Request data: {data}")

        request = (
            f"{method} {path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: keep-alive\r\n"
            "\r\n" + body
        )

        self.sock.sendall(request.encode("utf-8"))

        # Read response headers first
        response = b""
        try:
            while b"\r\n\r\n" not in response:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise RuntimeError("Connection closed while reading headers")
                response += chunk
        except socket.timeout:
            raise RuntimeError("Timeout while reading response headers")

        # Parse headers to get Content-Length
        response_str = response.decode("utf-8")
        headers_end = response_str.find("\r\n\r\n")
        headers = response_str[:headers_end]
        
        logger.info(f"Response headers:\n{headers}")
        
        content_length = None
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
                break
        
        if content_length is None:
            logger.error("No Content-Length header found in response")
            raise RuntimeError("Missing Content-Length header")
        
        logger.info(f"Content-Length: {content_length}")
        
        # Read the body based on Content-Length
        body_bytes = response[headers_end + 4:]
        try:
            while len(body_bytes) < content_length:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise RuntimeError("Connection closed while reading body")
                body_bytes += chunk
        except socket.timeout:
            raise RuntimeError("Timeout while reading response body")
        
        body = body_bytes[:content_length].decode("utf-8")
        logger.info(f"Received response body: {body[:500]}")
        
        try:
            parsed_response = json.loads(body)
            logger.info(f"Parsed response: {parsed_response}")
            return parsed_response
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in response: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current game status."""
        return self._send_request("GET", "/game/status")

    def make_move(self, column: int) -> Dict[str, Any]:
        """Make a move.

        Args:
            column: Column to place piece (0-6)

        Returns:
            Server response with updated game state
        """
        return self._send_request("POST", "/game/move", {"column": column})

    def reset_game(self) -> Dict[str, Any]:
        """Reset the game."""
        return self._send_request("GET", "/game/reset")

    def close(self):
        """Close connection to server."""
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class Connect4GUI:
    """Tkinter UI for Connect-4 game with improved styling."""

    def __init__(
        self,
        root: tk.Tk,
        client: GameClient,
        red_color: str = "#e62a50",
        yellow_color: str = "#f1c40f",
        board_color: str = "#189eeb",
        bg_color: str = "#fdfdfd",
    ):
        """Initialize GUI.

        Args:
            root: Tkinter root window
            client: Game client instance
            red_color: Color for player 1 (human)
            yellow_color: Color for player 2 (AI)
            board_color: Color for the board
            bg_color: Background color
        """
        self.root = root
        self.client = client
        self.board = None
        self.is_terminal = False
        self.waiting_for_server = False
        
        self.colors = {
            "red": red_color,
            "yellow": yellow_color,
            "board": board_color,
            "bg": bg_color,
            "outline": "#dddddd",
            "text": "black",
            "error": "red",
            "draw": "gray"
        }

        canvas_height = ROWS * CELL_SIZE + 60
        self.canvas = tk.Canvas(
            root,
            width=COLS * CELL_SIZE,
            height=canvas_height,
            bg=self.colors["bg"],
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)
        self.root.title("Connect 4 vs AlphaZero")
        
        self.status_text_id = None
        self._draw_status("Loading game state...", self.colors["text"])

        self.root.after(500, self._refresh_and_draw)

    def _draw_status(self, message: str, color: str = None):
        """Draw status message on canvas."""
        if color is None:
            color = self.colors["text"]
        if self.status_text_id is not None:
            self.canvas.delete(self.status_text_id)
        self.status_text_id = self.canvas.create_text(
            CELL_SIZE * COLS // 2,
            ROWS * CELL_SIZE + 30,
            text=message,
            font=("Helvetica", 14, "bold"),
            fill=color
        )

    def _on_click(self, event):
        """Handle canvas click."""
        # If game is over, clicking refreshes the game
        if self.is_terminal:
            logger.info("Game over - resetting game")
            self._reset_game()
            return
        
        if self.waiting_for_server:
            return

        col = event.x // CELL_SIZE
        if col < 0 or col >= COLS:
            return

        self.waiting_for_server = True
        self._draw_status("Sending move...", "blue")
        self.root.after(100, lambda: self._send_move(col))

    def _send_move(self, column: int):
        """Send move to server."""
        try:
            response = self.client.make_move(column)
            if response.get("status") == "error":
                self._draw_status(
                    f"Error: {response.get('message')}",
                    self.colors["error"],
                )
            else:
                self.board = response.get("board")
                self.is_terminal = response.get("is_terminal", False)

                if self.is_terminal:
                    self._handle_game_end()
                else:
                    self._draw_status("Your turn", "black")

                self._draw_board()
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            self._draw_status(f"Server error: {e}", self.colors["error"])
        finally:
            self.waiting_for_server = False

    def _handle_game_end(self):
        """Handle game over state."""
        # Determine winner from the board
        # In Connect4, the current_player switches after winning move
        # So we need to check who actually won
        self._draw_status("Game Over! Click to reset.", self.colors["error"])

    def _reset_game(self):
        """Reset the game."""
        try:
            response = self.client.reset_game()
            if response.get("status") == "ok":
                self.is_terminal = False
                self._draw_status("Game reset! Fetching new state...", "blue")
                # Fetch fresh game state
                self.root.after(200, self._refresh_and_draw)
            else:
                self._draw_status(
                    f"Reset failed: {response.get('message')}",
                    self.colors["error"]
                )
        except Exception as e:
            logger.error(f"Reset error: {e}", exc_info=True)
            self._draw_status(f"Reset error: {e}", self.colors["error"])

    def _refresh_and_draw(self):
        """Refresh game state and draw board."""
        logger.info("Refreshing")
        try:
            response = self.client.get_status()
            if response.get("status") == "ok":
                self.board = response.get("board")
                self.is_terminal = response.get("is_terminal", False)

                if self.is_terminal:
                    self._handle_game_end()
                else:
                    self._draw_status("Your turn - click a column", "black")

                self._draw_board()
            else:
                self._draw_status(
                    f"Error: {response.get('message')}",
                    self.colors["error"]
                )
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            self._draw_status(f"Connection error: {e}", self.colors["error"])
            self.root.after(1000, self._refresh_and_draw)
            return

        # Auto-refresh every 2 seconds
        self.root.after(2000, self._refresh_and_draw)

    def _draw_board(self):
        """Draw the game board with improved styling."""
        self.canvas.delete("board")

        if self.board is None:
            return

        for r in range(ROWS):
            for c in range(COLS):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE

                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=self.colors["board"],
                    outline=self.colors["bg"],
                    width=2,
                    tags="board"
                )

                cell = self.board[r][c]
                token_color = (
                    self.colors["bg"] if cell == 0 else
                    self.colors["red"] if cell == 1 else
                    self.colors["yellow"]
                )

                self.canvas.create_oval(
                    x0 + 10, y0 + 10, x1 - 10, y1 - 10,
                    fill=token_color,
                    outline=self.colors["outline"],
                    width=1,
                    tags="board"
                )

        # Column numbers (0-based for clarity)
        for c in range(COLS):
            x = c * CELL_SIZE + CELL_SIZE // 2
            y = ROWS * CELL_SIZE + 10
            self.canvas.create_text(
                x, y,
                text=str(c),
                font=("Helvetica", 12),
                fill="#666",
                tags="board"
            )


def main():
    parser = argparse.ArgumentParser(description="Connect-4 Game Client")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Game server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Game server port",
    )

    args = parser.parse_args()

    try:
        client = GameClient(args.host, args.port)
    except RuntimeError as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

    root = tk.Tk()
    gui = Connect4GUI(root, client)

    try:
        root.mainloop()
    finally:
        client.close()


if __name__ == "__main__":
    main()
