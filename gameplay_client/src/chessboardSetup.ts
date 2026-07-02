// chessboard.js (the classic chessboardjs.com widget) is a legacy jQuery
// plugin: it expects `window.jQuery`/`window.$` to already exist before its
// script runs, and it attaches itself to `window.Chessboard`. This module's
// only job is to make jQuery available on `window` as a side effect, so it
// must be imported *before* the chessboard.js script is imported anywhere.
import $ from 'jquery';

declare global {
  interface Window {
    jQuery: typeof $;
    $: typeof $;
  }
}

window.jQuery = $;
window.$ = $;

export default $;
