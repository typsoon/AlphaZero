<script setup lang="ts">
import { ref } from 'vue';
import WelcomeView from './WelcomeView.vue';
import Connect4View from './Connect4View.vue';
import ChessView from './ChessView.vue';

const currentView = ref<'welcome' | 'connect4' | 'chess'>('welcome');
</script>

<template>
  <div>
    <WelcomeView
      v-if="currentView === 'welcome'"
      @selectConnect4="currentView = 'connect4'"
      @selectChess="currentView = 'chess'"
    />
    <Connect4View
      v-else-if="currentView === 'connect4'"
      @back="currentView = 'welcome'"
    />
    <ChessView
      v-else-if="currentView === 'chess'"
      @back="currentView = 'welcome'"
    />
  </div>
</template>
