/**
 * A real UCI engine wrapping AlphaZeroDev's own inference_server, so it can
 * be driven by any standard UCI tool (cutechess-cli, python-chess's
 * chess.engine, a GUI) exactly like engine-zoo-uci - rather than needing a
 * bespoke protocol translator on the caller's side.
 *
 * inference_server never picks a move itself - /predict returns a policy
 * distribution + value, and move *selection* normally lives in
 * gameplay_server's playAITurn (evaluate() + pick the best *legal* action -
 * see helpers.ts's pickBestLegalAction). This wraps that exact logic (the
 * TUI's own engineMove() pattern: ChessBoard + AlphaZeroAgent +
 * pickBestLegalAction) in a UCI protocol loop instead of gameplay_server's
 * HTTP/WebSocket API.
 *
 * Unlike the TUI's per-move-fresh `new ChessBoard(chess.fen())`, this keeps
 * ONE ChessBoard alive for the whole game and step()s it through every move
 * (both sides), so its internal actionHistory - and therefore
 * get_inference_state()'s "history" field - reflects the real game, not
 * just the current position. That matters for history-encoder networks
 * (chess_encoder_history > 0, e.g. the live hist run's own checkpoints).
 *
 * Usage (point at an already-running inference_server - this doesn't spawn
 * one itself, unlike tui.ts):
 *   npm run az-uci -- --socket /tmp/alphazero-inference/chess/<net>/<hash>.sock
 */
import readline from 'node:readline';
import { AlphaZeroAgent } from '../agent.js';
import { ChessBoard } from '../game/chess.js';
import { pickBestLegalAction } from './helpers.js';

function parseArgs(argv: string[]): { socket: string } {
  let socket: string | undefined;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--socket') socket = argv[++i];
  }
  if (!socket) {
    process.stderr.write(
      'usage: az-uci --socket <inference_server .sock path>\n',
    );
    process.exit(1);
  }
  return { socket };
}

function moveToUci(move: {
  from: string;
  to: string;
  promotion?: string;
}): string {
  return move.from + move.to + (move.promotion ?? '');
}

/** Rebuilds a board from a UCI `position` command's tail, validating each
 * move against the board's own legality check as it replays - a malformed
 * or illegal move from the peer engine surfaces as an `info string` rather
 * than silently desyncing the position. */
function applyPositionCommand(parts: string[]): ChessBoard {
  let idx = 1;
  let fen: string | undefined;
  if (parts[idx] === 'startpos') {
    idx++;
  } else if (parts[idx] === 'fen') {
    idx++;
    fen = parts.slice(idx, idx + 6).join(' ');
    idx += 6;
  }

  const board = new ChessBoard(fen);
  if (parts[idx] === 'moves') {
    idx++;
    for (; idx < parts.length; idx++) {
      const uci = parts[idx]!;
      const action = board.encodeMove(uci);
      if (!board.get_legal_actions().includes(action)) {
        process.stderr.write(
          `info string illegal move in position command: ${uci}\n`,
        );
        break;
      }
      board.step(action);
    }
  }
  return board;
}

async function main() {
  const { socket } = parseArgs(process.argv.slice(2));
  const agent = new AlphaZeroAgent(socket);
  let board = new ChessBoard();

  const rl = readline.createInterface({
    input: process.stdin,
    terminal: false,
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (trimmed === 'uci') {
      process.stdout.write('id name az-uci\n');
      process.stdout.write('id author AlphaZeroDev\n');
      process.stdout.write('uciok\n');
    } else if (trimmed === 'isready') {
      process.stdout.write('readyok\n');
    } else if (trimmed === 'ucinewgame') {
      board = new ChessBoard();
    } else if (trimmed.startsWith('position')) {
      board = applyPositionCommand(trimmed.split(/\s+/));
    } else if (trimmed.startsWith('go')) {
      try {
        const evaluation = await agent.evaluate(board.get_inference_state());
        const legal = board.get_legal_actions();
        const action =
          pickBestLegalAction(evaluation.policy, legal) ?? legal[0];
        if (action === undefined) {
          process.stdout.write('bestmove 0000\n');
        } else {
          const move = board.decodeMove(action);
          process.stdout.write(`bestmove ${moveToUci(move)}\n`);
        }
      } catch (e) {
        process.stderr.write(
          `info string ${e instanceof Error ? e.message : String(e)}\n`,
        );
        process.stdout.write('bestmove 0000\n');
      }
    } else if (trimmed === 'quit') {
      process.exit(0);
    }
    // stop / setoption / unknown commands: ignored - this wrapper has no
    // tunable search parameters of its own (those live in the running
    // inference_server's own startup flags), same as engine-zoo-uci ignores
    // "stop" (search budget is fixed per "go" here too).
  }
}

main().catch((e) => {
  process.stderr.write(`${e instanceof Error ? e.message : String(e)}\n`);
  process.exit(1);
});
