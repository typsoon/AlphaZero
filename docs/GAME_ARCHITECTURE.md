# Three-Tier Game Architecture

AlphaZero gameplay now uses a three-tier HTTP architecture for clean separation between UI, game logic, and AI inference.

## Architecture Overview

```
┌─────────────────────────┐
│  Game Client (UI)       │
│  gameplay/client.py     │
│  Tkinter GUI            │
└────────────┬────────────┘
             │ HTTP/TCP
             │ localhost:8000
             ▼
┌─────────────────────────┐
│  Game Server            │
│  python/game_server.py  │
│  • Game state           │
│  • Move validation      │
│  • AI queries           │
└────────────┬────────────┘
             │ Unix socket
             │ HTTP
             ▼
┌─────────────────────────┐
│  Inference Server       │
│  python/inference_      │
│    server.py            │
│  • MCTS model           │
│  • Policy generation    │
└─────────────────────────┘
```

## Components

### Game Client (`gameplay/client.py`)

Tkinter GUI that displays the board and handles user input.

**Responsibilities:**
- Render Connect-4 board
- Accept column clicks
- Display game state
- Handle connection errors gracefully

**Communication:**
- HTTP GET `/game/status` - Fetch current board state
- HTTP POST `/game/move` - Send player move

**Features:**
- Auto-refresh every 2 seconds
- Click column numbers (0-6) to place pieces
- Displays connection status
- Shows "Game Over" when terminal state reached

### Game Server (`python/game_server.py`)

HTTP server managing the Connect-4 game state and coordinating with the inference service.

**Responsibilities:**
- Maintain game state (board, turn, legal moves)
- Validate player moves
- Execute AI moves via inference server
- Detect terminal states

**API Endpoints:**

#### GET `/game/status`
Returns current game state.

**Response:**
```json
{
  "status": "ok",
  "board": [[1, -1, 0, ...], ...],
  "legal_actions": [0, 1, 3, 5, 6],
  "is_terminal": false,
  "current_player": 1
}
```

#### POST `/game/move`
Execute a player move and get AI response.

**Request:**
```json
{"column": 3}
```

**Response (Game continues):**
```json
{
  "status": "ok",
  "player_column": 3,
  "ai_column": 2,
  "board": [[1, -1, 0, ...], ...],
  "legal_actions": [0, 1, 4, 5, 6],
  "is_terminal": false
}
```

**Response (Game over):**
```json
{
  "status": "ok",
  "message": "Game over",
  "board": [[1, -1, 1, ...], ...],
  "is_terminal": true
}
```

#### GET `/game/reset`
Reset game to initial state.

**Response:**
```json
{
  "status": "ok",
  "message": "Game reset"
}
```

### Inference Server (`python/inference_server.py`)

See `INFERENCE_SERVER.md` for details. Game server queries this via Unix socket.

## Quick Start

### Terminal 1: Start Inference Server

```bash
cd python
python inference_server.py --network-path AZNetwork.pt --device cuda
```

### Terminal 2: Start Game Server

```bash
cd python
python game_server.py
```

### Terminal 3: Start Game Client

```bash
cd gameplay
python client.py
```

The client opens a Tkinter window showing the board. Click column numbers (0-6) at the bottom to place your pieces. The AI automatically responds.

## Configuration

### Game Server

```bash
python game_server.py --help

# Arguments:
#   --socket HOST    Path to inference server Unix socket (default: /tmp/alphazero.sock)
#   --host HOST      Host to bind server to (default: localhost)
#   --port PORT      Port to bind server to (default: 8000)
```

Examples:
```bash
# Bind to all interfaces
python game_server.py --host 0.0.0.0

# Use custom port
python game_server.py --port 9000

# Use custom inference socket
python game_server.py --socket /run/user/1000/alphazero.sock
```

### Game Client

```bash
python client.py --help

# Arguments:
#   --host HOST    Game server host (default: localhost)
#   --port PORT    Game server port (default: 8000)
```

Examples:
```bash
# Connect to remote server
python client.py --host 192.168.1.100 --port 9000

# Use custom local port
python client.py --port 9000
```

## Typical Workflow

1. **Inference Server** loads trained MCTS model on Unix socket
2. **Game Server** starts, connects to inference server, waits for client connections
3. **Game Client** connects to game server via TCP
4. Player clicks a column → Client sends HTTP POST → Server validates move → Server queries AI → Server returns new state → Client renders
5. Repeat until game over

## Troubleshooting

### Client can't connect to server

```
Connection error: Failed to connect to game server at localhost:8000
```

**Solution:** Ensure game server is running:
```bash
python python/game_server.py
```

### Game server can't connect to inference server

```
RuntimeError: Failed to connect to inference server at /tmp/alphazero.sock
```

**Solution:** Ensure inference server is running:
```bash
python python/inference_server.py --network-path AZNetwork.pt --device cuda
```

### "Illegal move: column X not available"

The column is full or invalid. Legal columns are shown at the bottom of the board.

### Moves take too long

Game server is waiting for AI inference. Check that:
1. Inference server is responsive
2. Network is not congested
3. AI model file exists and is valid

## Performance Considerations

- **Client refresh rate**: 2 seconds (configurable in `client.py`)
- **Move latency**: Network overhead + AI inference time (~100ms on modern hardware)
- **Scalability**: Single game server can handle multiple clients (sequential processing)

## Future Improvements

1. **Concurrent game instances**: Add game IDs to support multiple simultaneous games
2. **WebSocket support**: Real-time updates instead of polling
3. **Authentication**: Secure game server access
4. **Analytics**: Track game outcomes and AI performance
5. **Parallel inference**: Batch multiple game moves for efficiency
