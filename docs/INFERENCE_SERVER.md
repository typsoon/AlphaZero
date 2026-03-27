# AlphaZero Inference Server

The gameplay module now uses a separate HTTP inference server that communicates via Unix socket. This architecture enables process isolation and independent scaling of the inference workload.

## Architecture Overview

```
┌─────────────┐                    ┌──────────────────────┐
│  Gameplay   │                    │ Inference Server     │
│  (GUI/TUI)  │◄──HTTP/Unix Socket─┤ (MCTS + Network)    │
│  agent.py   │                    │ inference_server.py │
└─────────────┘                    └──────────────────────┘
```

## Quick Start

### Terminal 1: Start Inference Server

```bash
cd python
python inference_server.py --network-path AZNetwork.pt --device cuda
```

Output:
```
Inference server listening on Unix socket: /tmp/alphazero.sock
Network: AZNetwork.pt
Device: cuda
```

### Terminal 2: Play the Game

```bash
cd gameplay
python play_game.py
```

The gameplay will connect to the inference server automatically using the default socket path.

## Configuration

### Inference Server Options

```bash
python inference_server.py --help

# Arguments:
#   --network-path PATH    Path to the trained network file (required)
#   --device DEVICE        Device to run inference on: 'cuda' or 'cpu' (default: 'cuda')
#   --socket SOCKET_PATH   Unix socket path (default: '/tmp/alphazero.sock')
```

Examples:

```bash
# Use CPU instead of GPU
python inference_server.py --network-path AZNetwork.pt --device cpu

# Use custom socket path
python inference_server.py --network-path AZNetwork.pt --device cuda --socket /run/user/1000/alphazero.sock

# Both
python inference_server.py --network-path AZNetwork.pt --device cpu --socket /tmp/my_server.sock
```

### Gameplay Options

```bash
python play_game.py --help

# Arguments:
#   --socket SOCKET_PATH   Path to inference server Unix socket (default: '/tmp/alphazero.sock')
```

Example:

```bash
# Connect to custom socket
python play_game.py --socket /run/user/1000/alphazero.sock
```

## API Reference

### Inference Server Endpoints

#### Health Check (GET)

```http
GET /health HTTP/1.1
```

Response (200):
```json
{"status": "healthy"}
```

#### Predict (POST)

```http
POST /predict HTTP/1.1
Content-Type: application/json
Content-Length: <length>

{
  "game_state": <game_state_object>
}
```

Response (200):
```json
{
  "status": "success",
  "policy": [<float>, <float>, ..., <float>]
}
```

Response (400 - Error):
```json
{
  "status": "error",
  "message": "<error description>"
}
```

## Troubleshooting

### Connection Refused

**Error:** `Failed to connect to inference server at /tmp/alphazero.sock`

**Solution:** Ensure the inference server is running in another terminal:
```bash
python inference_server.py --network-path AZNetwork.pt --device cuda
```

### Socket File Already Exists

**Error:** `Address already in use` or socket file permissions issue

**Solution:** The server cleans up the socket file on shutdown, but if it crashes, manually remove it:
```bash
rm /tmp/alphazero.sock
```

Or use a different socket path:
```bash
python inference_server.py --network-path AZNetwork.pt --device cuda --socket /tmp/alphazero_new.sock
python play_game.py --socket /tmp/alphazero_new.sock
```

### "Missing game_state in request"

**Solution:** This is a protocol error. Ensure you're using the latest `agent.py`. The client should automatically send the game state in the correct format.

## Advanced Usage

### Multiple Gameplay Instances

Multiple gameplay instances can share a single inference server:

```bash
# Terminal 1: Start server once
python inference_server.py --network-path AZNetwork.pt --device cuda

# Terminal 2, 3, etc.: Start multiple gameplay instances
python play_game.py
python play_game.py
python play_game.py
```

All will connect to the same inference server using the default socket path.

### Container/Remote Deployment

Since this uses Unix sockets, it's well-suited for containerized setups:

- Run inference server in a container with GPU access
- Run gameplay on the host or in a separate container
- Mount the socket directory as a shared volume

Example with Docker Compose:
```yaml
services:
  inference:
    build: .
    volumes:
      - /tmp:/tmp
    command: python inference_server.py --network-path AZNetwork.pt --device cuda
  
  gameplay:
    build: .
    volumes:
      - /tmp:/tmp
    command: python play_game.py
    depends_on:
      - inference
```

### Graceful Shutdown

The server handles `SIGINT` (Ctrl+C) and `SIGTERM` signals gracefully:

```bash
# Press Ctrl+C to shutdown
# or
kill -TERM <server_pid>
```

The socket file is automatically cleaned up on exit.

## Performance Considerations

- **Serialization overhead**: Game state is JSON-serialized for HTTP. For large boards, consider protocol buffers or MessagePack in the future.
- **Socket locality**: Unix sockets are faster than TCP for local communication.
- **Server scaling**: A single inference server can handle multiple gameplay clients. Each request is processed sequentially. For parallel requests, consider adding thread pooling or running multiple servers.
