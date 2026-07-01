<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue';

const emit = defineEmits(['back']);

const currentMode = ref<'setup' | 'game' | 'editor' | 'browser'>('setup');
const gameId = ref<string>('');
const playerId = ref<string>('');

const userSessions = ref<Record<string, string>>({});

function loadSessions() {
  try {
    userSessions.value = JSON.parse(
      localStorage.getItem('alphaZeroSessions') || '{}',
    );
  } catch {
    userSessions.value = {};
  }
}

type GameInfo = {
  id: string;
  p1Type: string;
  p2Type: string;
  finished: boolean;
};
const activeGames = ref<GameInfo[]>([]);

async function fetchGames() {
  try {
    const res = await fetch(`${apiBase}/games`);
    const data = await res.json();
    if (data.status === 'ok') {
      activeGames.value = data.games;
    }
  } catch (e) {
    console.error(e);
  }
}

function joinGame(id: string) {
  gameId.value = id;
  loadSessions();
  playerId.value = userSessions.value[id] || '';
  currentMode.value = 'game';
  connectWebSocket();
  fetchStatus();
}

// Setup config
const p1Type = ref<'human' | 'ai'>('human');
const p1Agent = ref<string>('');
const p2Type = ref<'human' | 'ai'>('ai');
const p2Agent = ref<string>('');
const availableAgents = ref<string[]>([]);

// State
const board = ref<number[][]>(
  Array.from({ length: 6 }, () => Array(7).fill(0)),
);
const legalActions = ref<number[]>([]);
const isTerminal = ref(false);
const currentPlayer = ref<number>(1);
const history = ref<number[][][]>([]);
const historyIndex = ref<number>(0);
const rewindMode = ref<boolean>(false);
const showGameOver = ref<boolean>(false);
const surrenderWinner = ref<number | null>(null);
const winReason = ref<string | null>(null);
const showOptionsMenu = ref(false);
const loading = ref(false);
const showEditPlayers = ref(false);
const editGameId = ref('');
const editP1Type = ref<'human' | 'ai'>('human');
const editP1Agent = ref<string>('');
const editP2Type = ref<'human' | 'ai'>('human');
const editP2Agent = ref<string>('');
const statusText = ref('Configure game to start');
const statusKind = ref<'ok' | 'info' | 'error'>('info');
let ws: WebSocket | null = null;

const rows = 6;
const cols = 7;
const apiBase = '/api';

function setStatus(
  message: string,
  kind: 'ok' | 'info' | 'error' = 'info',
): void {
  statusText.value = message;
  statusKind.value = kind;
}

async function fetchAgents() {
  try {
    const res = await fetch(`${apiBase}/agents?game=connect4`);
    const data = await res.json();
    if (data.status === 'ok') {
      availableAgents.value = data.agents;
      if (data.agents.length > 0) {
        if (!p1Agent.value) p1Agent.value = data.agents[0];
        if (!p2Agent.value) p2Agent.value = data.agents[0];
      }
    }
  } catch (e) {
    console.error(e);
  }
}

function openEditPlayersForCurrentGame() {
  if (!gameId.value) return;
  editGameId.value = gameId.value;
  editP1Type.value = p1Type.value;
  editP1Agent.value = p1Agent.value || availableAgents.value[0] || '';
  editP2Type.value = p2Type.value;
  editP2Agent.value = p2Agent.value || availableAgents.value[0] || '';
  showEditPlayers.value = true;
}

async function saveEditedPlayers() {
  if (!editGameId.value) return;
  showEditPlayers.value = false;
  try {
    const res = await fetch(`${apiBase}/game/${editGameId.value}/players`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        p1_type: editP1Type.value,
        p1_agent: editP1Type.value === 'ai' ? editP1Agent.value : null,
        p2_type: editP2Type.value,
        p2_agent: editP2Type.value === 'ai' ? editP2Agent.value : null,
      }),
    });
    const data = await res.json();
    if (data.status === 'ok') {
      loadSessions();
      const p1Id = data.p1_id || '';
      const p2Id = data.p2_id || '';
      userSessions.value[editGameId.value] = `${p1Id},${p2Id}`;
      localStorage.setItem(
        'alphaZeroSessions',
        JSON.stringify(userSessions.value),
      );

      if (gameId.value === editGameId.value) {
        playerId.value = userSessions.value[gameId.value] || '';
        await fetchStatus();
      }

      fetchGames();
    }
  } catch (e) {
    console.error('Failed to save players', e);
  }
}

function connectWebSocket() {
  if (ws) ws.close();
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(
    `${protocol}//${window.location.host}${apiBase}/game/${gameId.value}/ws`,
  );
  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'state_update') {
        const data = msg.data;
        board.value = data.board;
        legalActions.value = data.legal_actions ?? [];
        isTerminal.value = Boolean(data.is_terminal);
        if (data.current_player !== undefined) {
          currentPlayer.value = data.current_player;
        }
        if (data.history) {
          history.value = data.history;
          historyIndex.value = data.history.length - 1;
        }
        if (data.surrender_winner !== undefined) {
          surrenderWinner.value = data.surrender_winner;
        }
        if (data.win_reason !== undefined) {
          winReason.value = data.win_reason;
        }
        if (isTerminal.value && !rewindMode.value && !showGameOver.value) {
          showGameOver.value = true;
        }
        setStatus(
          isTerminal.value ? 'Game over!' : 'Game updated via WebSocket',
          'ok',
        );
      }
    } catch (e) {}
  };
}

async function createNewGame(): Promise<void> {
  loading.value = true;
  setStatus('Creating game...', 'info');
  try {
    const response = await fetch(`${apiBase}/game/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game_type: 'connect4',
        p1_type: p1Type.value,
        p1_agent: p1Type.value === 'ai' ? p1Agent.value : null,
        p2_type: p2Type.value,
        p2_agent: p2Type.value === 'ai' ? p2Agent.value : null,
      }),
    });
    const data = await response.json();
    if (data.status !== 'ok') throw new Error(data.message ?? 'Create failed');
    gameId.value = data.game_id;

    const p1Id = data.p1_id || '';
    const p2Id = data.p2_id || '';
    playerId.value = `${p1Id},${p2Id}`;

    loadSessions();
    userSessions.value[gameId.value] = playerId.value;
    localStorage.setItem(
      'alphaZeroSessions',
      JSON.stringify(userSessions.value),
    );
    connectWebSocket();
    await fetchStatus();
    currentMode.value = 'game';
    setStatus(`Game created: ${gameId.value}.`, 'ok');
  } catch (error: any) {
    setStatus(`Create error: ${error.message}`, 'error');
  } finally {
    loading.value = false;
  }
}

async function fetchStatus(): Promise<void> {
  if (!gameId.value) return;
  loading.value = true;
  try {
    const response = await fetch(`${apiBase}/game/${gameId.value}/status`);
    const data = await response.json();
    if (data.status !== 'ok') throw new Error(data.message);
    board.value = data.board;
    legalActions.value = data.legal_actions ?? [];
    isTerminal.value = Boolean(data.is_terminal);
    if (data.current_player !== undefined) {
      currentPlayer.value = data.current_player;
    }
    if (data.history) {
      history.value = data.history;
      if (!rewindMode.value) {
        historyIndex.value = data.history.length - 1;
      }
    }
    if (data.surrender_winner !== undefined) {
      surrenderWinner.value = data.surrender_winner;
    }
    if (data.win_reason !== undefined) {
      winReason.value = data.win_reason;
    }
    if (data.p1_type) p1Type.value = data.p1_type;
    if (data.p1_agent) p1Agent.value = data.p1_agent;
    if (data.p2_type) p2Type.value = data.p2_type;
    if (data.p2_agent) p2Agent.value = data.p2_agent;

    if (isTerminal.value && !rewindMode.value && !showGameOver.value) {
      showGameOver.value = true;
    }
  } catch (e) {
    console.error(e);
  } finally {
    loading.value = false;
  }
}

function canPlay(column: number | string): boolean {
  return (
    !loading.value &&
    !isTerminal.value &&
    legalActions.value.includes(Number(column))
  );
}

async function makeMove(column: number | string): Promise<void> {
  const colNum = Number(column);
  if (!canPlay(colNum)) {
    console.warn(
      `Attempted to play illegal or blocked move in column: ${column}. Legal actions: ${legalActions.value}`,
    );
    return;
  }
  if (!gameId.value) {
    console.warn('Attempted to make a move but no game is currently active.');
    return;
  }

  const ids = playerId.value.split(',');
  const p1Id = ids[0] || '';
  const p2Id = ids[1] || '';
  const correctId = currentPlayer.value === 1 ? p1Id : p2Id;

  if (!correctId) {
    console.warn(
      `Cannot play: Missing player ID for Player ${currentPlayer.value}`,
    );
    return;
  }

  loading.value = true;
  try {
    const res = await fetch(`${apiBase}/game/${gameId.value}/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: colNum, player_id: correctId }),
    });
    const data = await res.json();
    if (data.status === 'ok') {
      return; // Success!
    } else {
      console.error(`Move rejected by server: ${data.message}`);
    }
  } catch (e) {
    console.error('Network or parsing error while making a move:', e);
  } finally {
    loading.value = false;
  }
}

async function surrender(): Promise<void> {
  if (!gameId.value || isTerminal.value) return;
  const ids = playerId.value.split(',');
  const pId = ids.find((id) => id.trim().length > 0);
  if (!pId) return;

  loading.value = true;
  try {
    const res = await fetch(`${apiBase}/game/${gameId.value}/surrender`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_id: pId }),
    });
    const data = await res.json();
    if (data.status !== 'ok') {
      setStatus(`Surrender failed: ${data.message}`, 'error');
    }
  } catch (e: any) {
    setStatus(`Surrender error: ${e.message}`, 'error');
  } finally {
    loading.value = false;
  }
}

async function resetGame(): Promise<void> {
  if (!gameId.value || !playerId.value) return;
  loading.value = true;
  try {
    const res = await fetch(`${apiBase}/game/${gameId.value}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_id: playerId.value }),
    });
    const data = await res.json();
    if (data.status !== 'ok') {
      setStatus(`Reset failed: ${data.message}`, 'error');
      return;
    }
    showGameOver.value = false;
    rewindMode.value = false;
    surrenderWinner.value = null;
    winReason.value = null;
    setStatus('Game reset.', 'ok');
  } catch (e: any) {
    setStatus(`Reset error: ${e.message}`, 'error');
  } finally {
    loading.value = false;
  }
}

function exitToMainMenu() {
  if (ws) {
    ws.close();
    ws = null;
  }
  showGameOver.value = false;
  rewindMode.value = false;
  gameId.value = '';
  playerId.value = '';
  currentMode.value = 'setup';
}

function startRewind() {
  showGameOver.value = false;
  rewindMode.value = true;
  historyIndex.value = history.value.length - 1;
}

function stepBackward() {
  if (historyIndex.value > 0) historyIndex.value--;
}

function stepForward() {
  if (historyIndex.value < history.value.length - 1) historyIndex.value++;
}

async function resumeGame() {
  if (!gameId.value) return;
  loading.value = true;
  try {
    const response = await fetch(`${apiBase}/game/${gameId.value}/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ history_index: historyIndex.value }),
    });
    const data = await response.json();
    if (data.status === 'ok') {
      rewindMode.value = false;
      setStatus('Game resumed.', 'ok');
    } else {
      setStatus(`Resume error: ${data.message}`, 'error');
    }
  } catch (e: any) {
    setStatus(`Resume error: ${e.message}`, 'error');
  } finally {
    loading.value = false;
  }
}

function getWinner(b: number[][]): number | null {
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const p = b[r][c];
      if (p === 0) continue;
      if (
        c + 3 < cols &&
        b[r][c + 1] === p &&
        b[r][c + 2] === p &&
        b[r][c + 3] === p
      )
        return p;
      if (
        r + 3 < rows &&
        b[r + 1][c] === p &&
        b[r + 2][c] === p &&
        b[r + 3][c] === p
      )
        return p;
      if (
        r + 3 < rows &&
        c + 3 < cols &&
        b[r + 1][c + 1] === p &&
        b[r + 2][c + 2] === p &&
        b[r + 3][c + 3] === p
      )
        return p;
      if (
        r - 3 >= 0 &&
        c + 3 < cols &&
        b[r - 1][c + 1] === p &&
        b[r - 2][c + 2] === p &&
        b[r - 3][c + 3] === p
      )
        return p;
    }
  }
  return null;
}

function handleKeydown(e: KeyboardEvent) {
  if (currentMode.value === 'editor' && e.ctrlKey && e.key === 'z') {
    e.preventDefault();
    undoEditor();
    return;
  }
  if (currentMode.value !== 'game') return;
  if (showGameOver.value) return;

  if (rewindMode.value) {
    if (e.key === 'ArrowLeft') {
      stepBackward();
    } else if (e.key === 'ArrowRight') {
      stepForward();
    }
    return;
  }

  let col = -1;
  if (['1', '2', '3', '4', '5', '6', '7'].includes(e.key)) {
    col = parseInt(e.key, 10) - 1;
  } else if (e.code?.startsWith('Digit') && e.code.length === 6) {
    col = parseInt(e.code.charAt(5), 10) - 1;
  } else if (e.code?.startsWith('Numpad') && e.code.length === 7) {
    col = parseInt(e.code.charAt(6), 10) - 1;
  }

  if (col >= 0 && col <= 6) {
    makeMove(col);
  }
}

// --- Editor State & Logic ---
type EditorState = { board: number[][]; expectedMoves: number[] };
const editorHistory = ref<EditorState[]>([]);
const editorBoard = ref<number[][]>(
  Array.from({ length: rows }, () => Array(cols).fill(0)),
);
const editorExpectedMoves = ref<number[]>([]);
const editorCurrentColor = ref<number>(1);

const editorStatusText = ref('Puzzle Editor - Ready');
const editorStatusKind = ref<'info' | 'error' | 'ok'>('info');

function setEditorStatus(msg: string, kind: 'info' | 'error' | 'ok' = 'info') {
  editorStatusText.value = msg;
  editorStatusKind.value = kind;
}

function saveEditorState() {
  editorHistory.value.push({
    board: editorBoard.value.map((r) => [...r]),
    expectedMoves: [...editorExpectedMoves.value],
  });
}

function undoEditor() {
  if (editorHistory.value.length === 0) return;
  const prev = editorHistory.value.pop()!;
  editorBoard.value = prev.board;
  editorExpectedMoves.value = prev.expectedMoves;
  setEditorStatus('Undid last action', 'info');
}

function editorCellClick(r: number, c: number) {
  if (editorBoard.value[r][c] === editorCurrentColor.value) return;
  saveEditorState();
  editorBoard.value[r][c] = editorCurrentColor.value;
  setEditorStatus('Cell updated', 'ok');
}

function toggleExpectedMove(c: number) {
  saveEditorState();
  const idx = editorExpectedMoves.value.indexOf(c);
  if (idx !== -1) {
    editorExpectedMoves.value.splice(idx, 1);
  } else {
    editorExpectedMoves.value.push(c);
    editorExpectedMoves.value.sort((a, b) => a - b);
  }
}

function applyGravity() {
  saveEditorState();
  let changed = false;
  for (let c = 0; c < cols; c++) {
    const pieces = [];
    for (let r = rows - 1; r >= 0; r--) {
      if (editorBoard.value[r][c] !== 0) pieces.push(editorBoard.value[r][c]);
    }
    for (let r = rows - 1; r >= 0; r--) {
      const p = rows - 1 - r < pieces.length ? pieces[rows - 1 - r] : 0;
      if (editorBoard.value[r][c] !== p) {
        editorBoard.value[r][c] = p;
        changed = true;
      }
    }
  }
  if (changed) setEditorStatus('Gravity applied', 'ok');
  else setEditorStatus('No changes from gravity', 'info');
}

function isValidConnect4(): { valid: boolean; message: string } {
  let p1 = 0,
    p2 = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (editorBoard.value[r][c] === 1) p1++;
      if (editorBoard.value[r][c] === 2) p2++;
    }
  }
  if (Math.abs(p1 - p2) > 1) {
    return { valid: false, message: 'Invalid chip count difference.' };
  }
  for (let c = 0; c < cols; c++) {
    let emptySeen = false;
    for (let r = rows - 1; r >= 0; r--) {
      if (editorBoard.value[r][c] === 0) emptySeen = true;
      else if (emptySeen)
        return { valid: false, message: `Floating chip in column ${c + 1}.` };
    }
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const p = editorBoard.value[r][c];
      if (p === 0) continue;
      if (
        c + 3 < cols &&
        editorBoard.value[r][c + 1] === p &&
        editorBoard.value[r][c + 2] === p &&
        editorBoard.value[r][c + 3] === p
      )
        return { valid: false, message: 'Terminal state (horizontal win).' };
      if (
        r + 3 < rows &&
        editorBoard.value[r + 1][c] === p &&
        editorBoard.value[r + 2][c] === p &&
        editorBoard.value[r + 3][c] === p
      )
        return { valid: false, message: 'Terminal state (vertical win).' };
      if (
        r + 3 < rows &&
        c + 3 < cols &&
        editorBoard.value[r + 1][c + 1] === p &&
        editorBoard.value[r + 2][c + 2] === p &&
        editorBoard.value[r + 3][c + 3] === p
      )
        return { valid: false, message: 'Terminal state (diag down-right).' };
      if (
        r - 3 >= 0 &&
        c + 3 < cols &&
        editorBoard.value[r - 1][c + 1] === p &&
        editorBoard.value[r - 2][c + 2] === p &&
        editorBoard.value[r - 3][c + 3] === p
      )
        return { valid: false, message: 'Terminal state (diag up-right).' };
    }
  }
  return { valid: true, message: 'Valid' };
}

function downloadPuzzle() {
  const check = isValidConnect4();
  if (!check.valid) {
    setEditorStatus(`Cannot save: ${check.message}`, 'error');
    return;
  }
  const data = JSON.stringify(
    { board: editorBoard.value, expected_moves: editorExpectedMoves.value },
    null,
    2,
  );
  const blob = new Blob([data], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'puzzle.json';
  a.click();
  URL.revokeObjectURL(url);
  setEditorStatus('Puzzle saved!', 'ok');
}

function handleFileUpload(e: Event) {
  const target = e.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const json = JSON.parse(ev.target?.result as string);
      if (!Array.isArray(json.board) || json.board.length !== rows)
        throw new Error('Invalid board format');
      saveEditorState();
      editorBoard.value = json.board;
      editorExpectedMoves.value = json.expected_moves || [];
      setEditorStatus('Loaded puzzle successfully', 'ok');
    } catch (err: any) {
      setEditorStatus(`Failed to load: ${err.message}`, 'error');
    }
    target.value = '';
  };
  reader.readAsText(file);
}

onMounted(async () => {
  loadSessions();
  window.addEventListener('keydown', handleKeydown);
  await fetchAgents();
});

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown);
});
</script>
<template>
  <main class="page">
    <header class="header">
      <h1 class="title">Connect 4 vs AlphaZero</h1>
      <div style="margin-top: 12px">
        <button class="btn" @click="emit('back')">Back to Menu</button>
      </div>
      <div
        class="mode-toggle"
        style="
          margin-top: 12px;
          display: flex;
          gap: 8px;
          justify-content: center;
        "
      >
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
          Current Game
        </button>
        <button
          class="btn"
          :class="{ primary: currentMode === 'browser' }"
          @click="
            currentMode = 'browser';
            fetchGames();
          "
        >
          Browse Games
        </button>
        <button
          class="btn"
          :class="{ primary: currentMode === 'editor' }"
          @click="currentMode = 'editor'"
        >
          Puzzle Editor
        </button>
      </div>
    </header>

    <section class="card" v-if="currentMode === 'setup'">
      <div
        class="controls"
        style="flex-direction: column; gap: 12px; align-items: center"
      >
        <div style="display: flex; gap: 12px; align-items: center">
          <label><strong>Player 1 (Red)</strong>:</label>
          <select v-model="p1Type" class="btn">
            <option value="human">Human</option>
            <option value="ai">AI</option>
          </select>
          <select v-if="p1Type === 'ai'" v-model="p1Agent" class="btn">
            <option v-for="a in availableAgents" :key="a" :value="a">
              {{ a }}
            </option>
          </select>
        </div>

        <div style="display: flex; gap: 12px; align-items: center">
          <label><strong>Player 2 (Yellow)</strong>:</label>
          <select v-model="p2Type" class="btn">
            <option value="human">Human</option>
            <option value="ai">AI</option>
          </select>
          <select v-if="p2Type === 'ai'" v-model="p2Agent" class="btn">
            <option v-for="a in availableAgents" :key="a" :value="a">
              {{ a }}
            </option>
          </select>
        </div>

        <div style="display: flex; gap: 12px; margin-top: 10px">
          <button class="btn primary" @click="createNewGame">
            Create Game
          </button>
        </div>
      </div>
    </section>

    <section
      class="card"
      v-else-if="currentMode === 'game'"
      style="position: relative"
    >
      <!-- Options Menu -->
      <div style="position: absolute; top: 16px; right: 16px" v-if="gameId">
        <button
          class="btn"
          style="
            padding: 4px 10px;
            font-weight: 900;
            letter-spacing: 1px;
            border: none;
            background: transparent;
            box-shadow: none;
          "
          @click="showOptionsMenu = !showOptionsMenu"
        >
          ...
        </button>
        <div
          v-if="showOptionsMenu"
          style="
            position: absolute;
            top: 100%;
            right: 0;
            margin-top: 4px;
            background: white;
            border: 1px solid #dbe7ff;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            padding: 6px;
            z-index: 50;
            display: flex;
            flex-direction: column;
            min-width: 140px;
          "
        >
          <button
            v-if="playerId && playerId !== ',' && !isTerminal"
            class="dropdown-item danger"
            @click="
              surrender();
              showOptionsMenu = false;
            "
          >
            <svg
              width="16"
              height="16"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M3 21v-4m0 0V5a2 2 0 012-2h6.5l1 1H21l-3 6 3 6h-8.5l-1-1H5a2 2 0 00-2 2zm9-13.5V9"
              ></path>
            </svg>
            Surrender
          </button>
          <button
            v-if="playerId"
            class="dropdown-item"
            @click="
              resetGame();
              showOptionsMenu = false;
            "
          >
            Reset Game
          </button>
          <button class="dropdown-item" @click="showOptionsMenu = false">
            Close
          </button>
        </div>
      </div>
      <div class="status" :class="statusKind">{{ statusText }}</div>

      <!-- Edit Players Popup -->
      <div v-if="showEditPlayers" class="modal-overlay">
        <div class="modal">
          <h2 style="margin-top: 0">Edit Players for {{ editGameId }}</h2>
          <div
            style="
              display: flex;
              flex-direction: column;
              gap: 16px;
              margin: 20px 0;
            "
          >
            <div style="display: flex; gap: 12px; align-items: center">
              <label><strong>Player 1 (Red)</strong>:</label>
              <select v-model="editP1Type" class="btn">
                <option value="human">Human</option>
                <option value="ai">AI</option>
              </select>
              <select
                v-if="editP1Type === 'ai'"
                v-model="editP1Agent"
                class="btn"
              >
                <option v-for="a in availableAgents" :key="a" :value="a">
                  {{ a }}
                </option>
              </select>
            </div>
            <div style="display: flex; gap: 12px; align-items: center">
              <label><strong>Player 2 (Yellow)</strong>:</label>
              <select v-model="editP2Type" class="btn">
                <option value="human">Human</option>
                <option value="ai">AI</option>
              </select>
              <select
                v-if="editP2Type === 'ai'"
                v-model="editP2Agent"
                class="btn"
              >
                <option v-for="a in availableAgents" :key="a" :value="a">
                  {{ a }}
                </option>
              </select>
            </div>
          </div>
          <div style="display: flex; gap: 12px; justify-content: flex-end">
            <button class="btn" @click="showEditPlayers = false">Cancel</button>
            <button class="btn primary" @click="saveEditedPlayers">Save</button>
          </div>
        </div>
      </div>

      <!-- Game Over Popup -->
      <div v-if="showGameOver" class="modal-overlay">
        <div class="modal-content" style="text-align: center">
          <h2 style="margin-top: 0">Game Over</h2>

          <div
            v-if="
              surrenderWinner === 1 ||
              (!surrenderWinner && getWinner(board) === 1)
            "
            style="
              color: #ef4444;
              font-size: 1.5rem;
              font-weight: bold;
              margin: 16px 0;
            "
          >
            Player 1 (Red) Wins{{
              winReason === 'surrender' ? ' by Surrender' : ''
            }}!
          </div>
          <div
            v-else-if="
              surrenderWinner === -1 ||
              (!surrenderWinner && getWinner(board) === -1)
            "
            style="
              color: #f59e0b;
              font-size: 1.5rem;
              font-weight: bold;
              margin: 16px 0;
            "
          >
            Player 2 (Yellow) Wins{{
              winReason === 'surrender' ? ' by Surrender' : ''
            }}!
          </div>
          <div
            v-else
            style="
              color: #6b7280;
              font-size: 1.5rem;
              font-weight: bold;
              margin: 16px 0;
            "
          >
            It's a Draw!
          </div>

          <div
            style="
              display: flex;
              gap: 12px;
              justify-content: center;
              margin-top: 20px;
            "
          >
            <button class="btn" @click="exitToMainMenu">
              Exit to Main Menu
            </button>
            <button class="btn primary" @click="startRewind">
              Rewind Game
            </button>
          </div>
        </div>
      </div>

      <!-- Rewind Controls -->
      <div
        v-if="rewindMode"
        class="rewind-controls"
        style="
          display: flex;
          gap: 12px;
          justify-content: center;
          margin-bottom: 12px;
          align-items: center;
        "
      >
        <button
          class="btn"
          @click="stepBackward"
          :disabled="historyIndex === 0"
        >
          ◀
        </button>
        <span style="font-weight: 600"
          >Turn {{ historyIndex + 1 }} / {{ history.length }}</span
        >
        <button
          class="btn"
          @click="stepForward"
          :disabled="historyIndex === history.length - 1"
        >
          ▶
        </button>
        <button class="btn primary" @click="resumeGame">
          Resume Game Here
        </button>
        <button
          class="btn icon-btn"
          @click="openEditPlayersForCurrentGame"
          title="Edit Players"
        >
          ...
        </button>
      </div>

      <div class="board" v-if="gameId" role="grid" aria-label="Connect4 board">
        <div
          v-for="i in 7"
          :key="`label-${i}`"
          class="col-label"
          @click="rewindMode ? null : makeMove(i - 1)"
          style="cursor: pointer"
          title="Drop piece in column"
        >
          {{ i }}
        </div>
        <template
          v-for="(row, r) in rewindMode && history.length > 0
            ? history[historyIndex]
            : board"
          :key="`r-${r}`"
        >
          <button
            v-for="(_cell, c) in row"
            :key="`c-${r}-${c}`"
            class="cell"
            :class="{
              p1:
                (rewindMode && history.length > 0
                  ? history[historyIndex]
                  : board)[r][c] === 1,
              p2:
                (rewindMode && history.length > 0
                  ? history[historyIndex]
                  : board)[r][c] === -1,
            }"
            :disabled="rewindMode || !canPlay(c)"
            :title="`Column ${c}`"
            @click="rewindMode ? null : makeMove(c)"
          />
        </template>
      </div>
    </section>

    <section class="card" v-else-if="currentMode === 'browser'">
      <div
        style="
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        "
      >
        <h2 style="margin: 0">Active Games</h2>
        <button class="btn" @click="fetchGames">Refresh</button>
      </div>
      <div
        v-if="activeGames.length === 0"
        style="text-align: center; color: #666; padding: 20px"
      >
        No active games found.
      </div>
      <div v-else style="display: flex; flex-direction: column; gap: 8px">
        <div
          v-for="g in activeGames"
          :key="g.id"
          style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            border: 1px solid #eee;
            border-radius: 8px;
          "
        >
          <div>
            <strong>{{ g.id }}</strong
            ><br />
            <span style="font-size: 0.9em; color: #666">
              {{ g.p1Type }} vs {{ g.p2Type }}
              <span v-if="g.finished" style="color: #ef4444; margin-left: 8px"
                >[Finished]</span
              >
            </span>
          </div>
          <button class="btn primary" @click="joinGame(g.id)">
            {{
              userSessions[g.id] && userSessions[g.id] !== ''
                ? 'Connect'
                : 'Spectate'
            }}
          </button>
        </div>
      </div>
    </section>

    <section class="card" v-else>
      <div class="controls editor-tools">
        <div class="tool-group">
          <strong>Paint:</strong>
          <button
            class="btn"
            :class="{ primary: editorCurrentColor === 1 }"
            @click="editorCurrentColor = 1"
          >
            P1 (Red)
          </button>
          <button
            class="btn"
            :class="{ primary: editorCurrentColor === 2 }"
            @click="editorCurrentColor = 2"
          >
            P2 (Yellow)
          </button>
          <button
            class="btn"
            :class="{ primary: editorCurrentColor === 0 }"
            @click="editorCurrentColor = 0"
          >
            Eraser
          </button>
        </div>

        <div class="tool-group">
          <strong>Actions:</strong>
          <button class="btn" @click="applyGravity">Gravitate</button>
          <button
            class="btn"
            @click="undoEditor"
            :disabled="editorHistory.length === 0"
          >
            Undo
          </button>
        </div>

        <div class="tool-group">
          <strong>File:</strong>
          <button class="btn primary" @click="downloadPuzzle">Save JSON</button>
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
              @change="handleFileUpload"
              style="display: none"
            />
          </label>
        </div>
      </div>

      <div class="status" :class="editorStatusKind">{{ editorStatusText }}</div>

      <div class="board" role="grid" aria-label="Puzzle Editor board">
        <div v-for="c in cols" :key="`header-${c}`" class="col-label">
          <button
            class="btn expected-btn"
            :class="{ selected: editorExpectedMoves.includes(c - 1) }"
            @click="toggleExpectedMove(c - 1)"
            title="Toggle as expected move"
            style="
              font-size: 11px;
              padding: 2px 4px;
              min-width: 44px;
              margin-bottom: 4px;
            "
          >
            Move
          </button>
        </div>
        <template v-for="(row, r) in editorBoard" :key="`r-${r}`">
          <button
            v-for="(_cell, c) in row"
            :key="`c-${r}-${c}`"
            class="cell"
            :class="{
              p1: editorBoard[r][c] === 1,
              p2: editorBoard[r][c] === 2,
            }"
            :title="`Row ${r}, Col ${c}`"
            @click="editorCellClick(r, c)"
          />
        </template>
      </div>
    </section>
  </main>
</template>

<style>
.editor-tools {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  justify-content: center;
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
.expected-btn {
  background: rgba(255, 255, 255, 0.2);
  color: #fff;
  border: 1px dashed rgba(255, 255, 255, 0.6);
}
.expected-btn:hover {
  background: rgba(255, 255, 255, 0.4);
}
.expected-btn.selected {
  background: #10b981;
  color: #fff;
  border: 2px solid #059669;
  font-weight: bold;
}
</style>
