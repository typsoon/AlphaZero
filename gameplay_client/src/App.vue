<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

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
    const response = await fetch(`${apiBase}/game/reset`, { method: 'POST' })
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
    </header>

    <section class="card">
      <div class="controls">
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
  </main>
</template>
