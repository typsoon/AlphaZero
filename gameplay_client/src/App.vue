<script setup lang="ts">
import { ref } from 'vue';
import WelcomeView from './WelcomeView.vue';
import Connect4View from './Connect4View.vue';
import ChessView from './ChessView.vue';

type ViewName = 'welcome' | 'connect4' | 'chess';

const GAME_PATHS: Record<Exclude<ViewName, 'welcome'>, string> = {
  connect4: '/connect4',
  chess: '/chess',
};

function pathFor(view: ViewName, gameId: string): string {
  if (view === 'welcome') return '/';
  const base = GAME_PATHS[view];
  return gameId ? `${base}/${encodeURIComponent(gameId)}` : base;
}

function parseLocation(pathname: string): { view: ViewName; gameId: string } {
  const path = pathname.replace(/\/+$/, '') || '/';
  for (const [name, base] of Object.entries(GAME_PATHS) as [
    Exclude<ViewName, 'welcome'>,
    string,
  ][]) {
    if (path === base) return { view: name, gameId: '' };
    if (path.startsWith(base + '/')) {
      return { view: name, gameId: decodeURIComponent(path.slice(base.length + 1)) };
    }
  }
  return { view: 'welcome', gameId: '' };
}

const initial = parseLocation(window.location.pathname);
const currentView = ref<ViewName>(initial.view);
const currentGameId = ref<string>(initial.gameId);

function navigate(view: ViewName) {
  currentView.value = view;
  currentGameId.value = '';
  const path = pathFor(view, '');
  if (window.location.pathname !== path) {
    window.history.pushState({ view }, '', path);
  }
}

// Bubbled up from ChessView/Connect4View whenever the game actually being
// played/viewed changes, so a direct link to /chess/<id> or the browser's
// back/forward buttons can return to that exact game.
function onGameIdChange(gameId: string) {
  currentGameId.value = gameId;
  const path = pathFor(currentView.value, gameId);
  if (window.location.pathname !== path) {
    window.history.pushState({ view: currentView.value, gameId }, '', path);
  }
}

window.addEventListener('popstate', () => {
  const parsed = parseLocation(window.location.pathname);
  currentView.value = parsed.view;
  currentGameId.value = parsed.gameId;
});
</script>

<template>
  <div>
    <WelcomeView
      v-if="currentView === 'welcome'"
      @selectConnect4="navigate('connect4')"
      @selectChess="navigate('chess')"
    />
    <Connect4View
      v-else-if="currentView === 'connect4'"
      :initial-game-id="currentGameId"
      @back="navigate('welcome')"
      @game-id-change="onGameIdChange"
    />
    <ChessView
      v-else-if="currentView === 'chess'"
      :initial-game-id="currentGameId"
      @back="navigate('welcome')"
      @game-id-change="onGameIdChange"
    />
  </div>
</template>
