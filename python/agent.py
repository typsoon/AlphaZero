import json
import socket
import threading
import tkinter as tk
from abc import ABC, abstractmethod

import numpy as np
from numpy import argmax


class Agent(ABC):
    @abstractmethod
    def act(self, game_state) -> int:
        pass


class UserAgent(Agent):
    def __init__(self):
        self.selected_column = None
        self.move_ready = threading.Event()
        self._registered = False

    def _on_key(self, event):
        if event.char in "1234567":
            col = int(event.char) - 1
            self.selected_column = col
            self.move_ready.set()

    def act(self, game_state) -> int:
        if not self._registered:
            root = self._find_root_widget()
            root.bind("<Key>", self._on_key)
            self._registered = True

        self.selected_column = None
        self.move_ready.clear()

        print("Waiting for move: ")
        self.move_ready.wait()

        assert self.selected_column is not None
        return self.selected_column

    def _find_root_widget(self) -> tk.Tk:
        return tk._default_root  # pyright: ignore


class AlphaZeroAgent(Agent):
    def __init__(self, socket_path: str, player: int = -1):
        """Initialize agent with persistent Unix socket connection to inference server.

        Args:
            socket_path: Path to the Unix socket running the inference server
            player: Player identifier (-1 or 1)

        Raises:
            RuntimeError: If unable to connect to inference server
        """
        self.socket_path = socket_path
        self.player = player
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.socket_path)
        except (ConnectionRefusedError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Failed to connect to inference server at {self.socket_path}. "
                f"Make sure the server is running. Error: {e}"
            )

    def _reconnect(self):
        """Reconnect to the inference server."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.socket_path)
        except (ConnectionRefusedError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Failed to reconnect to inference server at {self.socket_path}. "
                f"Make sure the server is running. Error: {e}"
            )

    def _send_inference_request(self, game_state, retry=True) -> list:
        """Send inference request to HTTP server over persistent Unix socket.

        Args:
            game_state: Game state object to get predictions for
            retry: Whether to retry once on connection failure

        Returns:
            Policy array from the inference server

        Raises:
            RuntimeError: If server returns error or response is malformed
            BrokenPipeError: If connection is lost and retry fails
        """
        request_data = {
            "game_state": game_state,
        }

        http_request = (
            "POST /predict HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(json.dumps(request_data))}\r\n"
            "Connection: keep-alive\r\n"
            "\r\n" + json.dumps(request_data)
        )

        try:
            self.sock.sendall(http_request.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            if retry:
                # Connection lost, try to reconnect once
                print(f"Connection to inference server lost, reconnecting... ({e})")
                self._reconnect()
                return self._send_inference_request(game_state, retry=False)
            else:
                raise RuntimeError(f"Failed to send request after reconnection: {e}")

        # Read response headers first
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("Connection closed while reading headers")
            response += chunk
        
        # Parse headers to get Content-Length
        response_str = response.decode("utf-8")
        headers_end = response_str.find("\r\n\r\n")
        headers = response_str[:headers_end]
        
        content_length = None
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
                break
        
        if content_length is None:
            raise RuntimeError("Missing Content-Length header in inference response")
        
        # Read response body based on Content-Length
        body_bytes = response[headers_end + 4:]
        max_response_size = 1024 * 1024  # 1 MB limit
        
        if content_length > max_response_size:
            raise RuntimeError(f"Response too large: {content_length} bytes (max 1 MB)")
        
        while len(body_bytes) < content_length:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("Connection closed while reading body")
            body_bytes += chunk
        
        body = body_bytes[:content_length].decode("utf-8")

        # Safe deserialization with depth and type validation
        try:
            # Temporarily limit recursion to prevent stack overflow from nested JSON
            import sys

            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(100)
            try:
                response_data = json.loads(body)
            finally:
                sys.setrecursionlimit(old_limit)
        except (json.JSONDecodeError, RecursionError) as e:
            raise RuntimeError(f"Invalid or malicious JSON in response: {e}")

        if not isinstance(response_data, dict):
            raise RuntimeError("Response must be a JSON object")

        if response_data.get("status") == "error":
            raise RuntimeError(
                f"Server error: {response_data.get('message', 'Unknown error')}"
            )
        if "error" in response_data:
            raise RuntimeError(f"Server error: {response_data.get('error')}")

        policy = response_data.get("policy")
        if policy is None:
            raise RuntimeError("Missing policy in inference response")
        if not isinstance(policy, list):
            raise RuntimeError("Policy must be a list")

        if len(policy) > 1000:
            raise RuntimeError("Policy too large (max 1000 elements)")

        # Validate policy contains only numeric values (no nested structures)
        for i, p in enumerate(policy):
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                raise RuntimeError(
                    f"Policy[{i}] must be numeric, got {type(p).__name__}"
                )

        return policy

    def close(self):
        """Close the socket connection."""
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def act(self, game_state) -> int:
        """Get action from inference server.

        Args:
            game_state: Current game state

        Returns:
            Action (column index) to play
        """
        policy = self._send_inference_request(game_state)
        answer = int(argmax(np.array(policy)))
        print(f"AI chose move: {answer}")
        return answer
