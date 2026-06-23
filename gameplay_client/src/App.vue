<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

const currentMode = ref<'game' | 'editor'>('game')
const startingPlayer = ref<'human' | 'ai'>('human')

type GameStatusResponse = {
  status: 'ok' | 'error'
  board?: number[][]
  legal_actions?: number[]
  is_terminal?: boolean
  current_player?: number
  message?: string
}

type MoveResponse = {
  status: 'ok' | 'error'
  board?: number[][]
  legal_actions?: number[]
  is_terminal?: boolean
  player_column?: number
  ai_column?: number
  message?: string
}

const rows = 6
const cols = 7
const apiBase = '/api'

const board = ref<number[][]>(Array.from({ length: rows }, () => Array(cols).fill(0)))
const legalActions = ref<number[]>([])
const isTerminal = ref(false)
const loading = ref(false)
const statusText = ref('Loading game state...')
const statusKind = ref<'ok' | 'info' | 'error'>('info')

function setStatus(message: string, kind: 'ok' | 'info' | 'error' = 'info'): void {
  statusText.value = message
  statusKind.value = kind
}

function dropRowForColumn(column: number): number | null {
  for (let r = rows - 1; r >= 0; r -= 1) {
    if (board.value[r][column] === 0) return r
  }
  return null
}

function canPlay(column: number): boolean {
  return !loading.value && !isTerminal.value && legalActions.value.includes(column)
}

async function fetchStatus(): Promise<void> {
  loading.value = true
  try {
    const response = await fetch(`${apiBase}/game/status`, { method: 'GET' })
    const data = (await response.json()) as GameStatusResponse
    if (data.status !== 'ok' || !data.board) {
      throw new Error(data.message ?? 'Invalid status response')
    }
    board.value = data.board
    legalActions.value = data.legal_actions ?? []
    isTerminal.value = Boolean(data.is_terminal)
    setStatus(
      isTerminal.value ? 'Game over. Click Reset to start again.' : 'Your turn - click a column',
      isTerminal.value ? 'error' : 'ok',
    )
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    setStatus(`Failed to fetch status: ${message}`, 'error')
  } finally {
    loading.value = false
  }
}

async function makeMove(column: number): Promise<void> {
  if (!canPlay(column)) return
  loading.value = true
  setStatus(`Sending move in column ${column}...`, 'info')
  try {
    const payload = { column }
    const response = await fetch(`${apiBase}/game/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const data = (await response.json()) as MoveResponse

    if (data.status !== 'ok' || !data.board) {
      throw new Error(data.message ?? 'Move failed')
    }

    board.value = data.board
    legalActions.value = data.legal_actions ?? []
    isTerminal.value = Boolean(data.is_terminal)

    if (isTerminal.value) {
      setStatus('Game over. Click Reset to start again.', 'error')
    } else {
      setStatus(
        `You played ${data.player_column}, AI played ${data.ai_column}. Your turn.`,
        'ok',
      )
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    setStatus(`Move error: ${message}`, 'error')
  } finally {
    loading.value = false
  }
}

async function resetGame(): Promise<void> {
  loading.value = true
  setStatus('Resetting game...', 'info')
  try {
    const response = await fetch(`${apiBase}/game/reset`, { 
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ starting_player: startingPlayer.value })
    })
    const data = (await response.json()) as { status: string; message?: string }
    if (data.status !== 'ok') throw new Error(data.message ?? 'Reset failed')
    await fetchStatus()
    setStatus('Game reset. Your turn - click a column', 'ok')
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    setStatus(`Reset error: ${message}`, 'error')
  } finally {
    loading.value = false
  }
}

function handleKeydown(e: KeyboardEvent) {
  if (currentMode.value === 'editor' && e.ctrlKey && e.key === 'z') {
    e.preventDefault()
    undoEditor()
    return
  }
  if (currentMode.value !== 'game') return
  
  let col = -1
  if (['1','2','3','4','5','6','7'].includes(e.key)) {
    col = parseInt(e.key, 10) - 1
  } else if (e.code?.startsWith('Digit') && e.code.length === 6) {
    col = parseInt(e.code.charAt(5), 10) - 1
  } else if (e.code?.startsWith('Numpad') && e.code.length === 7) {
    col = parseInt(e.code.charAt(6), 10) - 1
  }

  if (col >= 0 && col <= 6) {
    makeMove(col)
  }
}

// --- Editor State & Logic ---
type EditorState = { board: number[][], expectedMoves: number[] }
const editorHistory = ref<EditorState[]>([])
const editorBoard = ref<number[][]>(Array.from({ length: rows }, () => Array(cols).fill(0)))
const editorExpectedMoves = ref<number[]>([])
const editorCurrentColor = ref<number>(1)

const editorStatusText = ref('Puzzle Editor - Ready')
const editorStatusKind = ref<'info'|'error'|'ok'>('info')

function setEditorStatus(msg: string, kind: 'info'|'error'|'ok' = 'info') {
  editorStatusText.value = msg
  editorStatusKind.value = kind
}

function saveEditorState() {
  editorHistory.value.push({
    board: editorBoard.value.map(r => [...r]),
    expectedMoves: [...editorExpectedMoves.value]
  })
}

function undoEditor() {
  if (editorHistory.value.length === 0) return
  const prev = editorHistory.value.pop()!
  editorBoard.value = prev.board
  editorExpectedMoves.value = prev.expectedMoves
  setEditorStatus('Undid last action', 'info')
}

function editorCellClick(r: number, c: number) {
  if (editorBoard.value[r][c] === editorCurrentColor.value) return
  saveEditorState()
  editorBoard.value[r][c] = editorCurrentColor.value
  setEditorStatus('Cell updated', 'ok')
}

function toggleExpectedMove(c: number) {
  saveEditorState()
  const idx = editorExpectedMoves.value.indexOf(c)
  if (idx !== -1) {
    editorExpectedMoves.value.splice(idx, 1)
  } else {
    editorExpectedMoves.value.push(c)
    editorExpectedMoves.value.sort((a, b) => a - b)
  }
}

function applyGravity() {
  saveEditorState()
  let changed = false
  for (let c = 0; c < cols; c++) {
    const pieces = []
    for (let r = rows - 1; r >= 0; r--) {
      if (editorBoard.value[r][c] !== 0) pieces.push(editorBoard.value[r][c])
    }
    for (let r = rows - 1; r >= 0; r--) {
      const p = (rows - 1 - r) < pieces.length ? pieces[rows - 1 - r] : 0
      if (editorBoard.value[r][c] !== p) {
        editorBoard.value[r][c] = p
        changed = true
      }
    }
  }
  if (changed) setEditorStatus('Gravity applied', 'ok')
  else setEditorStatus('No changes from gravity', 'info')
}

function isValidConnect4(): { valid: boolean, message: string } {
  let p1 = 0, p2 = 0
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (editorBoard.value[r][c] === 1) p1++
      if (editorBoard.value[r][c] === 2) p2++
    }
  }
  if (Math.abs(p1 - p2) > 1) {
    return { valid: false, message: 'Invalid chip count difference.' }
  }
  for (let c = 0; c < cols; c++) {
    let emptySeen = false
    for (let r = rows - 1; r >= 0; r--) {
      if (editorBoard.value[r][c] === 0) emptySeen = true
      else if (emptySeen) return { valid: false, message: `Floating chip in column ${c + 1}.` }
    }
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const p = editorBoard.value[r][c]
      if (p === 0) continue
      if (c + 3 < cols && editorBoard.value[r][c+1]===p && editorBoard.value[r][c+2]===p && editorBoard.value[r][c+3]===p)
        return { valid: false, message: 'Terminal state (horizontal win).' }
      if (r + 3 < rows && editorBoard.value[r+1][c]===p && editorBoard.value[r+2][c]===p && editorBoard.value[r+3][c]===p)
        return { valid: false, message: 'Terminal state (vertical win).' }
      if (r + 3 < rows && c + 3 < cols && editorBoard.value[r+1][c+1]===p && editorBoard.value[r+2][c+2]===p && editorBoard.value[r+3][c+3]===p)
        return { valid: false, message: 'Terminal state (diag down-right).' }
      if (r - 3 >= 0 && c + 3 < cols && editorBoard.value[r-1][c+1]===p && editorBoard.value[r-2][c+2]===p && editorBoard.value[r-3][c+3]===p)
        return { valid: false, message: 'Terminal state (diag up-right).' }
    }
  }
  return { valid: true, message: 'Valid' }
}

function downloadPuzzle() {
  const check = isValidConnect4()
  if (!check.valid) {
    setEditorStatus(`Cannot save: ${check.message}`, 'error')
    return
  }
  const data = JSON.stringify({ board: editorBoard.value, expected_moves: editorExpectedMoves.value }, null, 2)
  const blob = new Blob([data], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'puzzle.json'
  a.click()
  URL.revokeObjectURL(url)
  setEditorStatus('Puzzle saved!', 'ok')
}

function handleFileUpload(e: Event) {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = (ev) => {
    try {
      const json = JSON.parse(ev.target?.result as string)
      if (!Array.isArray(json.board) || json.board.length !== rows) throw new Error('Invalid board format')
      saveEditorState()
      editorBoard.value = json.board
      editorExpectedMoves.value = json.expected_moves || []
      setEditorStatus('Loaded puzzle successfully', 'ok')
    } catch (err: any) {
      setEditorStatus(`Failed to load: ${err.message}`, 'error')
    }
    target.value = ''
  }
  reader.readAsText(file)
}

onMounted(async () => {
  window.addEventListener('keydown', handleKeydown)
  await fetchStatus()
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <main class="page">
    <header class="header">
      <h1 class="title">Connect 4 vs AlphaZero</h1>
      <div class="mode-toggle" style="margin-top: 12px; display: flex; gap: 8px; justify-content: center;">
        <button class="btn" :class="{ primary: currentMode === 'game' }" @click="currentMode = 'game'">Game</button>
        <button class="btn" :class="{ primary: currentMode === 'editor' }" @click="currentMode = 'editor'">Puzzle Editor</button>
      </div>
    </header>

    <section class="card" v-if="currentMode === 'game'">
      <div class="controls">
        <div style="display: flex; gap: 8px; align-items: center; margin-right: auto; font-weight: 600; font-size: 14px;">
          <label for="starting-player">First turn:</label>
          <select id="starting-player" v-model="startingPlayer" class="btn" :disabled="loading" style="padding: 6px 12px; margin: 0;">
            <option value="human">You (Red)</option>
            <option value="ai">AI (Red)</option>
          </select>
        </div>
        <button class="btn primary" :disabled="loading" @click="fetchStatus">Refresh</button>
        <button class="btn" :disabled="loading" @click="resetGame">Reset</button>
      </div>

      <div class="status" :class="statusKind">{{ statusText }}</div>

      <div class="board" role="grid" aria-label="Connect4 board">
        <div v-for="i in 7" :key="`label-${i}`" class="col-label" @click="makeMove(i - 1)" style="cursor: pointer;" title="Drop piece in column">
          {{ i }}
        </div>
        <template v-for="(row, r) in board" :key="`r-${r}`">
          <button
            v-for="(_cell, c) in row"
            :key="`c-${r}-${c}`"
            class="cell"
            :class="{
              p1: board[r][c] === 1,
              p2: board[r][c] === -1
            }"
            :disabled="!canPlay(c)"
            :title="`Column ${c}`"
            @click="makeMove(c)"
          />
        </template>
      </div>
    </section>

    <section class="card" v-else>
      <div class="controls editor-tools">
        <div class="tool-group">
          <strong>Paint:</strong>
          <button class="btn" :class="{ primary: editorCurrentColor === 1 }" @click="editorCurrentColor = 1">P1 (Red)</button>
          <button class="btn" :class="{ primary: editorCurrentColor === 2 }" @click="editorCurrentColor = 2">P2 (Yellow)</button>
          <button class="btn" :class="{ primary: editorCurrentColor === 0 }" @click="editorCurrentColor = 0">Eraser</button>
        </div>
        
        <div class="tool-group">
          <strong>Actions:</strong>
          <button class="btn" @click="applyGravity">Gravitate</button>
          <button class="btn" @click="undoEditor" :disabled="editorHistory.length === 0">Undo</button>
        </div>

        <div class="tool-group">
          <strong>File:</strong>
          <button class="btn primary" @click="downloadPuzzle">Save JSON</button>
          <label class="btn" style="cursor:pointer; display:inline-flex; align-items:center; margin: 0;">
            Load JSON
            <input type="file" accept=".json" @change="handleFileUpload" style="display:none;" />
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
             style="font-size: 11px; padding: 2px 4px; min-width: 44px; margin-bottom: 4px;"
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
              p2: editorBoard[r][c] === 2
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
  border: 1px dashed rgba(255,255,255,0.6);
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
