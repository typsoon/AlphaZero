<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue';

const emit = defineEmits(['back']);
const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
const ranks = ['8', '7', '6', '5', '4', '3', '2', '1'];

const apiBase = '/api';
let ws: WebSocket | null = null;
const gameId = ref<string>('');
const playerId = ref<string>('');
const board = ref<string[][]>(
  Array.from({ length: 8 }, () => Array(8).fill(' ')),
);
const isTerminal = ref(false);
const statusMsg = ref('');
const legalActions = ref<number[]>([]);
const currentPlayer = ref<number>(1);
const selectedSquare = ref<{ row: number; col: number } | null>(null);
const promotionDialog = ref<{
  actions: number[];
  row: number;
  col: number;
} | null>(null);
const lastAction = ref<{
  from: { row: number; col: number };
  to: { row: number; col: number };
} | null>(null);

const userSessions = ref<Record<string, string>>({});

// Preload empty image to fix drag ghost issues in some browsers (e.g. Safari/Firefox showing file icons)
const emptyDragImage = new Image();
emptyDragImage.src =
  'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

function loadSessions() {
  try {
    userSessions.value = JSON.parse(
      localStorage.getItem('alphaZeroSessions') || '{}',
    );
  } catch {
    userSessions.value = {};
  }
}

function getPieceImage(cell: string) {
  if (cell === ' ') return '';
  const color = cell === cell.toUpperCase() ? 'white' : 'black';
  const nameMap: Record<string, string> = {
    p: 'pawn',
    r: 'rook',
    n: 'knight',
    b: 'bishop',
    q: 'queen',
    k: 'king',
  };
  const pieceName = nameMap[cell.toLowerCase()];
  return `/pieces/${color}_${pieceName}.svg`;
}

function saveSession(id: string, pId: string) {
  loadSessions();
  userSessions.value[id] = pId;
  localStorage.setItem('alphaZeroSessions', JSON.stringify(userSessions.value));
}

// Setup config
const p1Type = ref<'human' | 'ai'>('human');
const p1Agent = ref<string>('');
const p2Type = ref<'human' | 'ai'>('ai');
const p2Agent = ref<string>('');
const availableAgents = ref<string[]>([]);
const currentMode = ref<'setup' | 'game' | 'editor'>('setup');

// --- Chess Puzzle Editor ---
const PIECE_CHARS = [
  'K',
  'Q',
  'R',
  'B',
  'N',
  'P',
  'k',
  'q',
  'r',
  'b',
  'n',
  'p',
];
type EditorBoard = string[][];

function emptyEditorBoard(): EditorBoard {
  return Array.from({ length: 8 }, () => Array(8).fill(' '));
}

const editorBoard = ref<EditorBoard>(emptyEditorBoard());
const editorSideToMove = ref<'w' | 'b'>('w');
const editorSelectedPiece = ref<string>('P'); // the char to paint
const editorExpectedMoves = ref<string[]>([]); // uci strings like 'e2e4'
const editorHistory = ref<
  { board: EditorBoard; fen: string; expectedMoves: string[] }[]
>([]);
const editorStatusText = ref('Puzzle Editor — Ready');
const editorStatusKind = ref<'info' | 'ok' | 'error'>('info');
const editorFenInput = ref('');
const editorExpectedInput = ref(''); // comma-separated uci moves

// Editor drag-and-drop
const editorDragPiece = ref<string | null>(null); // piece being dragged
const editorDragSourceSquare = ref<{ row: number; col: number } | null>(null); // null = from palette
const editorDragGhost = ref<{ x: number; y: number; piece: string } | null>(
  null,
);

function onPaletteDragStart(pc: string, e: DragEvent) {
  editorDragPiece.value = pc;
  editorDragSourceSquare.value = null;
  editorSelectedPiece.value = pc;
  if (e.dataTransfer) e.dataTransfer.effectAllowed = 'copy';
  e.dataTransfer?.setDragImage(emptyDragImage, 0, 0);
  editorDragGhost.value = {
    x: e.clientX,
    y: e.clientY,
    piece: getPieceImage(pc),
  };
}

function onBoardPieceDragStart(row: number, col: number, e: DragEvent) {
  const pc = editorBoard.value[row]?.[col] ?? ' ';
  if (pc === ' ') return;
  editorDragPiece.value = pc;
  editorDragSourceSquare.value = { row, col };
  if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer?.setDragImage(emptyDragImage, 0, 0);
  editorDragGhost.value = {
    x: e.clientX,
    y: e.clientY,
    piece: getPieceImage(pc),
  };
}

function onEditorDrag(e: DragEvent) {
  if (editorDragGhost.value && (e.clientX !== 0 || e.clientY !== 0)) {
    editorDragGhost.value.x = e.clientX;
    editorDragGhost.value.y = e.clientY;
  }
}

function onEditorDragEnd() {
  editorDragGhost.value = null;
}

function onEditorSquareDrop(row: number, col: number) {
  const pc = editorDragPiece.value;
  if (!pc) return;
  saveEditorHistory();
  // If dragging from a board square, clear the source
  if (editorDragSourceSquare.value) {
    const { row: sr, col: sc } = editorDragSourceSquare.value;
    if (sr !== row || sc !== col) {
      editorBoard.value[sr][sc] = ' ';
    }
  }
  editorBoard.value[row][col] = pc;
  editorDragPiece.value = null;
  editorDragSourceSquare.value = null;
  editorDragGhost.value = null;
  updateFenFromBoard();
}

function setEditorStatus(msg: string, kind: 'info' | 'ok' | 'error' = 'info') {
  editorStatusText.value = msg;
  editorStatusKind.value = kind;
}

function saveEditorHistory() {
  editorHistory.value.push({
    board: editorBoard.value.map((r) => [...r]),
    fen: editorFenInput.value,
    expectedMoves: [...editorExpectedMoves.value],
  });
}

function undoEditor() {
  if (editorHistory.value.length === 0) return;
  const prev = editorHistory.value.pop()!;
  editorBoard.value = prev.board;
  editorFenInput.value = prev.fen;
  editorExpectedMoves.value = prev.expectedMoves;
  setEditorStatus('Undo successful', 'ok');
}

function editorClearBoard() {
  saveEditorHistory();
  editorBoard.value = emptyEditorBoard();
  setEditorStatus('Board cleared', 'info');
}

function editorSetupStartPosition() {
  saveEditorHistory();
  const start = [
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
  ];
  editorBoard.value = start;
  editorSideToMove.value = 'w';
  setEditorStatus('Starting position loaded', 'ok');
}

function editorSquareClick(row: number, col: number) {
  saveEditorHistory();
  const cell = editorBoard.value[row][col];
  if (
    editorSelectedPiece.value === 'eraser' ||
    cell === editorSelectedPiece.value
  ) {
    editorBoard.value[row][col] = ' ';
  } else {
    editorBoard.value[row][col] = editorSelectedPiece.value;
  }
  updateFenFromBoard();
}

function boardToFen(): string {
  const rows = editorBoard.value.map((row) => {
    let s = '';
    let empty = 0;
    for (const cell of row) {
      if (cell === ' ') {
        empty++;
      } else {
        if (empty) {
          s += empty;
          empty = 0;
        }
        s += cell;
      }
    }
    if (empty) s += empty;
    return s;
  });
  return `${rows.join('/')} ${editorSideToMove.value} - - 0 1`;
}

function parseFenToBoard(fen: string): EditorBoard | null {
  try {
    const rows = fen.trim().split(' ')[0].split('/');
    if (rows.length !== 8) return null;
    const board: EditorBoard = [];
    for (const row of rows) {
      const cells: string[] = [];
      for (const ch of row) {
        if (ch >= '1' && ch <= '8') {
          for (let i = 0; i < parseInt(ch); i++) cells.push(' ');
        } else {
          cells.push(ch);
        }
      }
      if (cells.length !== 8) return null;
      board.push(cells);
    }
    return board;
  } catch {
    return null;
  }
}

function updateFenFromBoard() {
  editorFenInput.value = boardToFen();
}

function loadFen() {
  const fen = editorFenInput.value.trim();
  const parsed = parseFenToBoard(fen);
  if (!parsed) {
    setEditorStatus('Invalid FEN string', 'error');
    return;
  }
  saveEditorHistory();
  editorBoard.value = parsed;
  const parts = fen.split(' ');
  if (parts[1] === 'b') editorSideToMove.value = 'b';
  else editorSideToMove.value = 'w';
  setEditorStatus('FEN loaded successfully', 'ok');
}

function downloadChessPuzzle() {
  updateFenFromBoard();
  const fen = boardToFen();
  const expected = editorExpectedInput.value
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const data = JSON.stringify(
    { fen, side_to_move: editorSideToMove.value, expected_moves: expected },
    null,
    2,
  );
  const blob = new Blob([data], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'chess_puzzle.json';
  a.click();
  URL.revokeObjectURL(url);
  setEditorStatus('Puzzle saved!', 'ok');
}

function handleChessPuzzleUpload(e: Event) {
  const target = e.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const json = JSON.parse(ev.target?.result as string);
      const parsed = parseFenToBoard(json.fen);
      if (!parsed) throw new Error('Invalid FEN in file');
      saveEditorHistory();
      editorBoard.value = parsed;
      editorFenInput.value = json.fen || '';
      editorSideToMove.value = json.side_to_move === 'b' ? 'b' : 'w';
      editorExpectedInput.value = (json.expected_moves || []).join(', ');
      setEditorStatus('Puzzle loaded successfully', 'ok');
    } catch (err: any) {
      setEditorStatus(`Failed to load: ${err.message}`, 'error');
    }
    target.value = '';
  };
  reader.readAsText(file);
}

const dragState = ref<{
  piece: string;
  x: number;
  y: number;
  width: number;
  height: number;
  tweakX: number;
  tweakY: number;
} | null>(null);

const isBoardFlipped = computed(() => {
  if (p1Type.value === 'human' && p2Type.value === 'human') {
    return currentPlayer.value === 1;
  }
  if (p1Type.value === 'ai' && p2Type.value === 'human') {
    return true;
  }
  return false;
});

const displayBoard = computed(() => {
  if (!board.value) return [];
  if (!isBoardFlipped.value) return board.value;
  return [...board.value].map((row) => [...row].reverse()).reverse();
});

onMounted(() => {
  fetchAgents();
});

onUnmounted(() => {
  if (ws) {
    ws.close();
  }
});

async function fetchAgents() {
  try {
    const res = await fetch(`${apiBase}/agents?game=chess`);
    const data = await res.json();
    if (data.status === 'ok') {
      availableAgents.value = data.agents;
      if (availableAgents.value.length > 0) {
        p2Agent.value = availableAgents.value[0];
      }
    }
  } catch (e) {
    console.error('Failed to load agents', e);
  }
}

async function createNewGame() {
  try {
    const res = await fetch(`${apiBase}/game/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game_type: 'chess',
        p1_type: p1Type.value,
        p1_agent: p1Type.value === 'ai' ? p1Agent.value : null,
        p2_type: p2Type.value,
        p2_agent: p2Type.value === 'ai' ? p2Agent.value : null,
      }),
    });
    const data = await res.json();
    if (data.status === 'ok') {
      gameId.value = data.game_id;
      // Depending on who we are playing, save our player id
      if (p1Type.value === 'human') {
        playerId.value = data.p1_id;
        saveSession(data.game_id, data.p1_id);
      } else if (p2Type.value === 'human') {
        playerId.value = data.p2_id;
        saveSession(data.game_id, data.p2_id);
      } else {
        playerId.value = 'spectator';
      }

      currentMode.value = 'game';
      connectWebSocket();
      fetchStatus();
    }
  } catch (e) {
    console.error(e);
    alert('Failed to start game');
  }
}

function connectWebSocket() {
  if (ws) ws.close();
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(
    `${protocol}//${window.location.host}${apiBase}/game/${gameId.value}/ws`,
  );
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'state_update') {
      board.value = msg.data.board;
      isTerminal.value = msg.data.is_terminal;
      legalActions.value = msg.data.legal_actions ?? [];
      currentPlayer.value = msg.data.current_player;
      if (msg.data.player_action !== undefined) {
        const action = msg.data.player_action;
        const act = Math.floor(action / 5);
        const actTo = act % 64;
        const actFrom = Math.floor(act / 64);
        lastAction.value = {
          from: { row: Math.floor(actFrom / 8), col: actFrom % 8 },
          to: { row: Math.floor(actTo / 8), col: actTo % 8 },
        };
      } else {
        lastAction.value = null;
      }
      if (msg.data.surrender_winner !== null) {
        if (msg.data.win_reason === 'surrender') {
          statusMsg.value = `Game Over. Player ${msg.data.surrender_winner === 1 ? '1' : '2'} wins by surrender!`;
        } else if (msg.data.win_reason === 'checkmate') {
          statusMsg.value = `Game Over. Player ${msg.data.surrender_winner === 1 ? '1' : '2'} wins by checkmate!`;
        } else if (msg.data.win_reason === 'draw') {
          statusMsg.value = 'Game Over. Draw!';
        } else {
          statusMsg.value = `Game Over. Player ${msg.data.surrender_winner === 1 ? '1' : '2'} wins!`;
        }
      } else if (isTerminal.value) {
        statusMsg.value = 'Game Over.';
      }
    }
  };
}

async function fetchStatus() {
  if (!gameId.value) return;
  try {
    const res = await fetch(`${apiBase}/game/${gameId.value}/status`);
    const data = await res.json();
    if (data.status === 'ok') {
      board.value = data.board;
      isTerminal.value = data.is_terminal;
      legalActions.value = data.legal_actions ?? [];
      currentPlayer.value = data.current_player;
      if (data.player_action !== undefined) {
        const action = data.player_action;
        const act = Math.floor(action / 5);
        const actTo = act % 64;
        const actFrom = Math.floor(act / 64);
        lastAction.value = {
          from: { row: Math.floor(actFrom / 8), col: actFrom % 8 },
          to: { row: Math.floor(actTo / 8), col: actTo % 8 },
        };
      } else {
        lastAction.value = null;
      }
      if (data.surrender_winner !== null) {
        if (data.win_reason === 'surrender') {
          statusMsg.value = `Game Over. Player ${data.surrender_winner === 1 ? '1' : '2'} wins by surrender!`;
        } else if (data.win_reason === 'checkmate') {
          statusMsg.value = `Game Over. Player ${data.surrender_winner === 1 ? '1' : '2'} wins by checkmate!`;
        } else if (data.win_reason === 'draw') {
          statusMsg.value = 'Game Over. Draw!';
        } else {
          statusMsg.value = `Game Over. Player ${data.surrender_winner === 1 ? '1' : '2'} wins!`;
        }
      } else if (isTerminal.value) {
        statusMsg.value = 'Game Over.';
      }
    }
  } catch (e) {
    console.error(e);
  }
}

function getLegalMoves(
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number,
): number[] {
  const from = fromRow * 8 + fromCol;
  const to = toRow * 8 + toCol;
  return legalActions.value.filter((action) => {
    const act = Math.floor(action / 5);
    const actTo = act % 64;
    const actFrom = Math.floor(act / 64);
    return actFrom === from && actTo === to;
  });
}

function getPromotionImage(action: number) {
  const promotion = action % 5;
  const isWhite = currentPlayer.value === 0;
  let piece = '';
  if (promotion === 1) piece = isWhite ? 'Q' : 'q';
  else if (promotion === 2) piece = isWhite ? 'R' : 'r';
  else if (promotion === 3) piece = isWhite ? 'N' : 'n';
  else if (promotion === 4) piece = isWhite ? 'B' : 'b';
  return getPieceImage(piece);
}

function selectPromotion(action: number) {
  makeMove(action);
  promotionDialog.value = null;
  selectedSquare.value = null;
}

function handleSquareClick(row: number, col: number) {
  if (isTerminal.value) return;

  if (!selectedSquare.value) {
    const from = row * 8 + col;
    const hasMoves = legalActions.value.some(
      (a) => Math.floor(Math.floor(a / 5) / 64) === from,
    );
    if (hasMoves) selectedSquare.value = { row, col };
  } else {
    if (selectedSquare.value.row === row && selectedSquare.value.col === col) {
      selectedSquare.value = null;
      return;
    }

    const actions = getLegalMoves(
      selectedSquare.value.row,
      selectedSquare.value.col,
      row,
      col,
    );
    if (actions.length > 0) {
      if (actions.length === 1 && actions[0] % 5 === 0) {
        makeMove(actions[0]);
        selectedSquare.value = null;
      } else {
        promotionDialog.value = { actions, row, col };
      }
    } else {
      const from = row * 8 + col;
      const hasMoves = legalActions.value.some(
        (a) => Math.floor(Math.floor(a / 5) / 64) === from,
      );
      if (hasMoves) selectedSquare.value = { row, col };
      else selectedSquare.value = null;
    }
  }
}

function handleDragStart(row: number, col: number, event: DragEvent) {
  if (isTerminal.value) {
    event.preventDefault();
    return;
  }
  const from = row * 8 + col;
  const hasMoves = legalActions.value.some(
    (a) => Math.floor(Math.floor(a / 5) / 64) === from,
  );
  if (hasMoves) {
    selectedSquare.value = { row, col };
    event.dataTransfer?.setData('text/plain', `${row},${col}`);
    const target = event.target as HTMLElement;
    if (target && event.dataTransfer) {
      const rect = target.getBoundingClientRect();

      // Use the preloaded transparent 1x1 image to disable the buggy native drag ghost entirely
      event.dataTransfer.setDragImage(emptyDragImage, 0, 0);

      const realCol = isBoardFlipped.value ? 7 - col : col;
      const realRow = isBoardFlipped.value ? 7 - row : row;
      const cellValue = board.value[realRow]?.[realCol] || ' ';

      dragState.value = {
        piece: getPieceImage(cellValue),
        x: event.clientX,
        y: event.clientY,
        width: rect.width,
        height: rect.height,
        tweakX: -rect.width / 2,
        tweakY: -rect.height / 2,
      };
    }
  } else {
    event.preventDefault();
  }
}

function handleDrag(event: DragEvent) {
  if (dragState.value && (event.clientX !== 0 || event.clientY !== 0)) {
    dragState.value.x = event.clientX;
    dragState.value.y = event.clientY;
  }
}

function handleDragEnd() {
  dragState.value = null;
}

function handleDrop(row: number, col: number) {
  dragState.value = null;
  if (selectedSquare.value) {
    const actions = getLegalMoves(
      selectedSquare.value.row,
      selectedSquare.value.col,
      row,
      col,
    );
    if (actions.length > 0) {
      if (actions.length === 1 && actions[0] % 5 === 0) {
        makeMove(actions[0]);
        selectedSquare.value = null;
      } else {
        promotionDialog.value = { actions, row, col };
      }
    }
  }
}

async function makeMove(action: number) {
  try {
    const res = await fetch(`${apiBase}/game/${gameId.value}/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, player_id: playerId.value }),
    });
    const data = await res.json();
    if (data.status !== 'ok') {
      alert(data.message);
    }
  } catch (e) {
    console.error(e);
  }
}
</script>

<template>
  <main class="page">
    <header class="header">
      <h1 class="title">Chess</h1>
      <div
        style="
          margin-top: 12px;
          display: flex;
          gap: 8px;
          justify-content: center;
          flex-wrap: wrap;
        "
      >
        <button class="btn" @click="emit('back')">Back to Menu</button>
        <button
          class="btn"
          :class="{ primary: currentMode === 'setup' }"
          @click="currentMode = 'setup'"
        >
          Game Setup
        </button>
        <button
          v-if="gameId"
          class="btn"
          :class="{ primary: currentMode === 'game' }"
          @click="currentMode = 'game'"
        >
          Game
        </button>
        <button
          class="btn"
          :class="{ primary: currentMode === 'editor' }"
          @click="
            currentMode = 'editor';
            updateFenFromBoard();
          "
        >
          Puzzle Editor
        </button>
      </div>
    </header>

    <div v-if="currentMode === 'setup'" class="setup-container">
      <div class="card setup-card">
        <h2>New Game</h2>
        <div class="player-config">
          <div class="config-group">
            <h3>White (Player 1)</h3>
            <select v-model="p1Type">
              <option value="human">Human</option>
              <option value="ai">AI</option>
            </select>
            <select v-if="p1Type === 'ai'" v-model="p1Agent">
              <option v-for="a in availableAgents" :key="a" :value="a">
                {{ a }}
              </option>
            </select>
          </div>
          <div class="config-group">
            <h3>Black (Player 2)</h3>
            <select v-model="p2Type">
              <option value="human">Human</option>
              <option value="ai">AI</option>
            </select>
            <select v-if="p2Type === 'ai'" v-model="p2Agent">
              <option v-for="a in availableAgents" :key="a" :value="a">
                {{ a }}
              </option>
            </select>
          </div>
        </div>
        <button class="btn start-btn" @click="createNewGame">Start Game</button>
      </div>
    </div>

    <section v-else-if="currentMode === 'game'" class="card chess-card">
      <div class="status-msg" v-if="statusMsg">{{ statusMsg }}</div>
      <div class="chess-board">
        <template v-for="(row, rIndex) in displayBoard" :key="rIndex">
          <div
            v-for="(cell, cIndex) in row"
            :key="cIndex"
            class="chess-square"
            :class="[
              (rIndex + cIndex) % 2 === 0 ? 'light' : 'dark',
              {
                selected:
                  selectedSquare?.row ===
                    (isBoardFlipped ? 7 - rIndex : rIndex) &&
                  selectedSquare?.col ===
                    (isBoardFlipped ? 7 - cIndex : cIndex),
                'last-move':
                  (lastAction?.from.row ===
                    (isBoardFlipped ? 7 - rIndex : rIndex) &&
                    lastAction?.from.col ===
                      (isBoardFlipped ? 7 - cIndex : cIndex)) ||
                  (lastAction?.to.row ===
                    (isBoardFlipped ? 7 - rIndex : rIndex) &&
                    lastAction?.to.col ===
                      (isBoardFlipped ? 7 - cIndex : cIndex)),
              },
            ]"
            @click="
              handleSquareClick(
                isBoardFlipped ? 7 - rIndex : rIndex,
                isBoardFlipped ? 7 - cIndex : cIndex,
              )
            "
            @dragover.prevent
            @drop="
              handleDrop(
                isBoardFlipped ? 7 - rIndex : rIndex,
                isBoardFlipped ? 7 - cIndex : cIndex,
              )
            "
          >
            <div
              v-if="cell !== ' '"
              class="piece-wrapper"
              :style="{ backgroundImage: `url(${getPieceImage(cell)})` }"
              draggable="true"
              @dragstart="
                handleDragStart(
                  isBoardFlipped ? 7 - rIndex : rIndex,
                  isBoardFlipped ? 7 - cIndex : cIndex,
                  $event,
                )
              "
              @drag="handleDrag"
              @dragend="handleDragEnd"
            ></div>
            <!-- Coordinates -->
            <span v-if="cIndex === 0" class="coord-rank">{{
              ranks[isBoardFlipped ? 7 - rIndex : rIndex]
            }}</span>
            <span v-if="rIndex === 7" class="coord-file">{{
              files[isBoardFlipped ? 7 - cIndex : cIndex]
            }}</span>
          </div>
        </template>
      </div>

      <!-- Custom floating drag ghost -->
      <teleport to="body">
        <div
          v-if="dragState"
          class="custom-drag-ghost"
          :style="{
            width: dragState.width + 'px',
            height: dragState.height + 'px',
            left: dragState.x + dragState.tweakX + 'px',
            top: dragState.y + dragState.tweakY + 'px',
            backgroundImage: `url(${dragState.piece})`,
          }"
        ></div>
      </teleport>

      <!-- Promotion Modal -->
      <div v-if="promotionDialog" class="modal-overlay">
        <div class="modal">
          <h3>Promote Pawn</h3>
          <div class="promotion-options">
            <div
              v-for="action in promotionDialog.actions"
              :key="action"
              class="promotion-option"
              @click="selectPromotion(action)"
            >
              <img :src="getPromotionImage(action)" alt="Promote" />
            </div>
          </div>
          <button class="btn" @click="promotionDialog = null">Cancel</button>
        </div>
      </div>
    </section>

    <!-- ===== PUZZLE EDITOR ===== -->
    <section v-else-if="currentMode === 'editor'" class="card editor-card">
      <!-- Toolbar -->
      <div class="editor-toolbar">
        <!-- Piece palette -->
        <div class="tool-group">
          <strong>Piece:</strong>
          <div class="piece-palette">
            <div
              v-for="pc in PIECE_CHARS"
              :key="pc"
              class="palette-piece"
              :class="{ 'palette-selected': editorSelectedPiece === pc }"
              :title="pc"
              draggable="true"
              @click="editorSelectedPiece = pc"
              @dragstart="onPaletteDragStart(pc, $event)"
              @drag="onEditorDrag"
              @dragend="onEditorDragEnd"
            >
              <img :src="getPieceImage(pc)" :alt="pc" draggable="false" />
            </div>
            <div
              class="palette-piece eraser"
              :class="{ 'palette-selected': editorSelectedPiece === 'eraser' }"
              title="Eraser"
              @click="editorSelectedPiece = 'eraser'"
            >
              ✕
            </div>
          </div>
        </div>

        <!-- Side to move -->
        <div class="tool-group">
          <strong>Side to move:</strong>
          <button
            class="btn"
            :class="{ primary: editorSideToMove === 'w' }"
            @click="
              editorSideToMove = 'w';
              updateFenFromBoard();
            "
          >
            White
          </button>
          <button
            class="btn"
            :class="{ primary: editorSideToMove === 'b' }"
            @click="
              editorSideToMove = 'b';
              updateFenFromBoard();
            "
          >
            Black
          </button>
        </div>

        <!-- Actions -->
        <div class="tool-group">
          <strong>Actions:</strong>
          <button class="btn" @click="editorSetupStartPosition">
            Start Position
          </button>
          <button class="btn" @click="editorClearBoard">Clear</button>
          <button
            class="btn"
            @click="undoEditor"
            :disabled="editorHistory.length === 0"
          >
            Undo
          </button>
        </div>

        <!-- File -->
        <div class="tool-group">
          <strong>File:</strong>
          <button class="btn primary" @click="downloadChessPuzzle">
            Save JSON
          </button>
          <label
            class="btn"
            style="
              cursor: pointer;
              display: inline-flex;
              align-items: center;
              margin: 0;
            "
          >
            Load JSON
            <input
              type="file"
              accept=".json"
              @change="handleChessPuzzleUpload"
              style="display: none"
            />
          </label>
        </div>
      </div>

      <!-- Status bar -->
      <div class="status" :class="editorStatusKind">{{ editorStatusText }}</div>

      <!-- FEN bar -->
      <div class="fen-bar">
        <label><strong>FEN:</strong></label>
        <input
          v-model="editorFenInput"
          class="fen-input"
          placeholder="Paste FEN…"
          @keydown.enter="loadFen"
        />
        <button class="btn" @click="loadFen">Load</button>
        <button
          class="btn"
          @click="
            updateFenFromBoard();
            setEditorStatus('FEN updated', 'ok');
          "
        >
          Refresh
        </button>
      </div>

      <!-- Expected moves -->
      <div class="fen-bar">
        <label><strong>Expected moves (UCI, comma-separated):</strong></label>
        <input
          v-model="editorExpectedInput"
          class="fen-input"
          placeholder="e.g. e2e4, d7d5"
        />
      </div>

      <!-- Board -->
      <div class="editor-board-wrap">
        <!-- Rank labels left -->
        <div class="editor-rank-labels">
          <span v-for="r in 8" :key="r">{{ 9 - r }}</span>
        </div>
        <div class="editor-chess-board">
          <template v-for="(row, rIndex) in editorBoard" :key="`er-${rIndex}`">
            <div
              v-for="(cell, cIndex) in row"
              :key="`ec-${rIndex}-${cIndex}`"
              class="editor-square"
              :class="(rIndex + cIndex) % 2 === 0 ? 'light' : 'dark'"
              :draggable="cell !== ' '"
              @click="editorSquareClick(rIndex, cIndex)"
              @dragstart="
                cell !== ' ' && onBoardPieceDragStart(rIndex, cIndex, $event)
              "
              @drag="onEditorDrag"
              @dragend="onEditorDragEnd"
              @dragover.prevent
              @drop.prevent="onEditorSquareDrop(rIndex, cIndex)"
            >
              <img
                v-if="cell !== ' '"
                :src="getPieceImage(cell)"
                :alt="cell"
                class="editor-piece-img"
                draggable="false"
              />
            </div>
          </template>
        </div>

        <!-- Editor drag ghost -->
        <teleport to="body">
          <div
            v-if="editorDragGhost"
            class="editor-drag-ghost"
            :style="{
              left: editorDragGhost.x - 24 + 'px',
              top: editorDragGhost.y - 24 + 'px',
              backgroundImage: `url(${editorDragGhost.piece})`,
            }"
          ></div>
        </teleport>
        <!-- File labels below -->
        <div class="editor-file-labels">
          <span v-for="f in files" :key="f">{{ f }}</span>
        </div>
      </div>
    </section>
  </main>
</template>

<style scoped>
.page {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  text-align: center;
  margin-bottom: 2rem;
}

.title {
  font-size: 2.5rem;
  color: #1e293b;
  margin: 0;
}

.btn {
  background: #3b82f6;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: background 0.2s;
}
.btn:hover {
  background: #2563eb;
}

.card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
  padding: 2rem;
}

.setup-container {
  display: flex;
  justify-content: center;
}
.setup-card {
  width: 100%;
  max-width: 600px;
}
.player-config {
  display: flex;
  gap: 2rem;
  margin: 1.5rem 0;
}
.config-group {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.config-group h3 {
  white-space: nowrap;
  font-size: 1.1rem;
  margin: 0;
}
select {
  padding: 0.5rem;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-size: 1rem;
}

.start-btn {
  width: 100%;
  font-size: 1.1rem;
  padding: 0.75rem;
}

.chess-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #f8fafc;
}

.status-msg {
  font-size: 1.25rem;
  font-weight: 600;
  color: #ef4444;
  margin-bottom: 1rem;
}

.chess-board {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  grid-template-rows: repeat(8, 1fr);
  width: min(80vw, 600px);
  height: min(80vw, 600px);
  border: 8px solid #334155;
  border-radius: 4px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
  user-select: none;
}

.chess-square {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  font-size: 3rem;
  font-weight: bold;
  cursor: pointer;
}

.chess-square.selected {
  background-color: rgba(255, 255, 0, 0.4) !important;
}

.chess-square.last-move {
  background-color: rgba(155, 199, 0, 0.41) !important;
}

.chess-square.light {
  background-color: #f0d9b5;
  color: #b58863;
}

.chess-square.dark {
  background-color: #b58863;
  color: #f0d9b5;
}

.piece-wrapper {
  width: 100%;
  height: 100%;
  cursor: grab;
  background-size: 90%;
  background-repeat: no-repeat;
  background-position: center;
}

.piece-wrapper:active {
  cursor: grabbing;
}

.custom-drag-ghost {
  position: fixed;
  background-size: 90%;
  background-repeat: no-repeat;
  background-position: center;
  pointer-events: none;
  z-index: 9999;
  /* Centered perfectly on the mouse pointer */
  transform: translate(-50%, -50%);
}

.coord-rank {
  position: absolute;
  top: 4px;
  left: 4px;
  font-size: 12px;
  font-weight: 600;
  opacity: 0.6;
  pointer-events: none;
}

.coord-file {
  position: absolute;
  bottom: 4px;
  right: 4px;
  font-size: 12px;
  font-weight: 600;
  opacity: 0.6;
  pointer-events: none;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10000;
}

.modal {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  text-align: center;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
}

.promotion-options {
  display: flex;
  gap: 1rem;
  margin: 1.5rem 0;
  justify-content: center;
}

.promotion-option {
  width: 60px;
  height: 60px;
  cursor: pointer;
  border-radius: 8px;
  transition: background 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.promotion-option:hover {
  background: #f1f5f9;
}

.promotion-option img {
  width: 100%;
  height: 100%;
}

/* ===== Puzzle Editor ===== */
.editor-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
}

.editor-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  justify-content: center;
  width: 100%;
}

.tool-group {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 12px;
  border-right: 1px solid #e5e7eb;
}

.tool-group:last-child {
  border-right: none;
}

.piece-palette {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  max-width: 280px;
}

.palette-piece {
  width: 36px;
  height: 36px;
  border: 2px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f1f5f9;
  transition:
    background 0.15s,
    border-color 0.15s;
}

.palette-piece img {
  width: 28px;
  height: 28px;
}

.palette-piece:hover {
  background: #e2e8f0;
}

.palette-piece.palette-selected {
  border-color: #4f46e5;
  background: #e0e7ff;
}

.palette-piece.eraser {
  font-size: 18px;
  font-weight: bold;
  color: #ef4444;
}

.fen-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  max-width: 700px;
  flex-wrap: wrap;
}

.fen-input {
  flex: 1;
  min-width: 0;
  padding: 6px 12px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  font-family: monospace;
  font-size: 13px;
  background: #f8fafc;
}

.editor-board-wrap {
  display: grid;
  grid-template-areas:
    '. board'
    'ranks board'
    '. files';
  grid-template-columns: 22px 1fr;
  grid-template-rows: 0 1fr 22px;
  width: min(80vw, 560px);
}

.editor-rank-labels {
  grid-area: ranks;
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  align-items: center;
  font-size: 11px;
  font-weight: 600;
  color: #64748b;
}

.editor-file-labels {
  grid-area: files;
  display: flex;
  justify-content: space-around;
  font-size: 11px;
  font-weight: 600;
  color: #64748b;
  padding-left: 22px;
}

.editor-chess-board {
  grid-area: board;
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  grid-template-rows: repeat(8, 1fr);
  width: 100%;
  aspect-ratio: 1 / 1;
  border: 5px solid #334155;
  border-radius: 4px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18);
  user-select: none;
}

.editor-square {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: crosshair;
  transition: opacity 0.1s;
}

.editor-square[draggable='true'] {
  cursor: grab;
}

.editor-square[draggable='true']:active {
  cursor: grabbing;
}

.editor-square:hover {
  opacity: 0.75;
}

.editor-square.light {
  background-color: #f0d9b5;
}

.editor-square.dark {
  background-color: #b58863;
}

.editor-piece-img {
  width: 88%;
  height: 88%;
  pointer-events: none;
  user-select: none;
}

.editor-drag-ghost {
  position: fixed;
  width: 48px;
  height: 48px;
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center;
  pointer-events: none;
  z-index: 99999;
}

.status.ok {
  color: #16a34a;
  font-weight: 600;
}
.status.error {
  color: #dc2626;
  font-weight: 600;
}
.status.info {
  color: #64748b;
}
</style>
