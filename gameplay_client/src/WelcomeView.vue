<script setup lang="ts">
defineEmits(['selectConnect4', 'selectChess']);
</script>

<template>
  <div class="welcome-container">
    <h1 class="welcome-title">AlphaZero Games</h1>
    <p class="welcome-subtitle">Select a game to play</p>
    <div class="game-cards">
      <div class="game-card" @click="$emit('selectConnect4')">
        <div class="game-icon connect4-icon">
          <div class="c4-circle"></div>
          <div class="c4-circle p1"></div>
          <div class="c4-circle p2"></div>
          <div class="c4-circle"></div>
        </div>
        <h2>Connect 4</h2>
        <p>The classic vertical checkers game.</p>
      </div>
      <div class="game-card" @click="$emit('selectChess')">
        <div class="game-icon chess-icon">
          <div class="chess-square w"></div>
          <div class="chess-square b"></div>
          <div class="chess-square b"></div>
          <div class="chess-square w"></div>
        </div>
        <h2>Chess</h2>
        <p>The ultimate strategic board game.</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.welcome-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 80vh;
  text-align: center;
}

.welcome-title {
  font-size: 3rem;
  font-weight: 800;
  background: linear-gradient(135deg, #2563eb, #3b82f6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;
}

.welcome-subtitle {
  font-size: 1.25rem;
  color: #4b5563;
  margin-bottom: 3rem;
}

.game-cards {
  display: flex;
  gap: 2rem;
  flex-wrap: wrap;
  justify-content: center;
}

.game-card {
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  border: 1px solid #dbe7ff;
  border-radius: 24px;
  padding: 2rem;
  width: 280px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 10px 30px rgba(37, 99, 235, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
}

.game-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 20px 40px rgba(37, 99, 235, 0.2);
  border-color: #93c5fd;
}

.game-card h2 {
  font-size: 1.75rem;
  margin: 1rem 0 0.5rem;
  color: #1e3a8a;
}

.game-card p {
  color: #64748b;
  margin: 0;
  line-height: 1.5;
}

.game-icon {
  width: 100px;
  height: 100px;
  border-radius: 20px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 4px;
  padding: 10px;
  background: #f1f5f9;
  box-shadow: inset 0 4px 6px rgba(0, 0, 0, 0.05);
}

.connect4-icon {
  background: #2563eb;
}

.c4-circle {
  border-radius: 50%;
  background: #1e3a8a;
}
.c4-circle.p1 {
  background: #ef4444;
}
.c4-circle.p2 {
  background: #eab308;
}

.chess-icon {
  background: #e2e8f0;
  gap: 0;
  padding: 0;
  border: 4px solid #cbd5e1;
}

.chess-square {
  width: 100%;
  height: 100%;
}
.chess-square.w {
  background: #f8fafc;
}
.chess-square.b {
  background: #64748b;
}
</style>
