/**
 * Terminal UI for playing chess against a chosen AlphaZero network.
 *
 * Spawns its own inference server (build/inference_server/inference_server)
 * on a private Unix socket for the selected network, then runs a readline
 * game loop: you type moves in SAN ("Nf3") or UCI ("g1f3"), the engine
 * answers through the same /predict + MCTS path the gameplay server uses.
 *
 * Usage (from gameplay_server/):
 *   npm run tui                          # interactive network picker
 *   npm run tui -- --network <path>      # explicit network file
 *   npm run tui -- --color b             # play Black
 *   npm run tui -- --depth 400           # MCTS simulations per engine move
 *   npm run tui -- --socket <sock>       # reuse an already-running server
 */
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline/promises';
import { fileURLToPath } from 'node:url';
import { Chess } from 'chess.js';
import { AlphaZeroAgent } from '../agent.js';
import { ChessBoard } from '../game/chess.js';
import {
  BIG_BOARD,
  COMPACT_BOARD,
  describeResult,
  listNetworks,
  pickBestLegalAction,
  renderBoard,
  type BoardStyle,
} from './helpers.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// dist/tui/ -> gameplay_server/ -> repo root
const PROJ_ROOT = path.resolve(HERE, '..', '..', '..');
const INFERENCE_BIN = path.join(
  PROJ_ROOT,
  'build',
  'inference_server',
  'inference_server',
);

type Args = {
  network?: string;
  socket?: string;
  color: 'w' | 'b';
  depth: number;
  board: BoardStyle;
  ascii: boolean;
};

function parseArgs(argv: string[]): Args {
  const args: Args = {
    color: 'w',
    depth: 800,
    board: BIG_BOARD,
    ascii: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === '--network') {
      const v = argv[++i];
      if (v !== undefined) args.network = v;
    } else if (a === '--socket') {
      const v = argv[++i];
      if (v !== undefined) args.socket = v;
    } else if (a === '--color') args.color = argv[++i] === 'b' ? 'b' : 'w';
    else if (a === '--depth') args.depth = parseInt(argv[++i] ?? '800', 10);
    else if (a === '--compact') args.board = COMPACT_BOARD;
    else if (a === '--ascii') args.ascii = true;
    else if (a === '-h' || a === '--help') {
      process.stdout.write(
        'usage: tui [--network <file>] [--socket <sock>] [--color w|b] ' +
          '[--depth <n>] [--compact] [--ascii]\n',
      );
      process.exit(0);
    }
  }
  return args;
}

async function chooseNetwork(rl: readline.Interface): Promise<string> {
  const choices = listNetworks(PROJ_ROOT);
  if (choices.length === 0) {
    throw new Error(`No chess networks found under ${PROJ_ROOT}/checkpoints`);
  }
  process.stdout.write('Available networks:\n');
  choices.forEach((c, i) => {
    process.stdout.write(`  [${i}] ${c.label}\n`);
  });
  const answer = await rl.question(
    `Pick a network [0-${choices.length - 1}]: `,
  );
  const idx = parseInt(answer, 10);
  const choice = choices[idx];
  if (!choice) throw new Error(`Invalid selection: ${answer}`);
  return choice.path;
}

/** Starts the inference server and resolves once its socket accepts requests. */
async function startInferenceServer(
  networkPath: string,
  socketPath: string,
  depth: number,
): Promise<ChildProcess> {
  if (!fs.existsSync(INFERENCE_BIN)) {
    throw new Error(
      `Inference server binary not found at ${INFERENCE_BIN} — run 'doit build' first.`,
    );
  }
  const child = spawn(
    INFERENCE_BIN,
    [
      '--network-path',
      networkPath,
      '--game',
      'chess',
      '--socket',
      socketPath,
      '--mcts-search-depth',
      String(depth),
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  );
  let stderrTail = '';
  child.stderr?.on('data', (chunk: Buffer) => {
    stderrTail = (stderrTail + chunk.toString()).slice(-2000);
  });

  process.stdout.write(
    'Starting inference server (TensorRT load can take a minute)...\n',
  );
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(
        `Inference server exited with code ${child.exitCode}:\n${stderrTail}`,
      );
    }
    if (fs.existsSync(socketPath)) return child;
    await new Promise((r) => setTimeout(r, 500));
  }
  child.kill('SIGKILL');
  throw new Error(`Inference server socket never appeared at ${socketPath}`);
}

async function engineMove(
  agent: AlphaZeroAgent,
  chess: Chess,
): Promise<{ san: string; value: number }> {
  const board = new ChessBoard(chess.fen());
  const legal = board.get_legal_actions();
  const evaluation = await agent.evaluate(board.get_inference_state());
  const action = pickBestLegalAction(evaluation.policy, legal) ?? legal[0]!;
  const move = board.decodeMove(action);
  const played = chess.move(move);
  return { san: played.san, value: evaluation.value };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  let child: ChildProcess | null = null;
  let socketPath = args.socket;
  const cleanup = () => {
    if (child && child.exitCode === null) child.kill('SIGTERM');
  };
  process.on('SIGINT', () => {
    cleanup();
    process.exit(130);
  });

  try {
    if (!socketPath) {
      const networkPath = args.network ?? (await chooseNetwork(rl));
      socketPath = `/tmp/az-tui-${process.pid}.sock`;
      child = await startInferenceServer(networkPath, socketPath, args.depth);
      process.stdout.write(`Playing against: ${path.basename(networkPath)}\n`);
    }

    const agent = new AlphaZeroAgent(socketPath);
    const chess = new Chess();
    const humanColor = args.color;
    const flip = humanColor === 'b';
    process.stdout.write(
      `You play ${humanColor === 'w' ? 'White' : 'Black'}. ` +
        'Enter moves as SAN (Nf3) or UCI (g1f3); commands: moves, fen, resign.\n\n',
    );

    while (!chess.isGameOver()) {
      process.stdout.write(
        renderBoard(chess, flip, args.board, args.ascii) + '\n',
      );
      if (chess.turn() === humanColor) {
        let input: string;
        try {
          input = (await rl.question('your move> ')).trim();
        } catch {
          process.stdout.write('\nInput closed — leaving the game.\n');
          break;
        }
        if (input === 'resign') {
          process.stdout.write('You resigned. Engine wins.\n');
          break;
        }
        if (input === 'fen') {
          process.stdout.write(chess.fen() + '\n');
          continue;
        }
        if (input === 'moves') {
          process.stdout.write(chess.moves().join(' ') + '\n');
          continue;
        }
        try {
          // chess.js accepts SAN directly; fall back to UCI from/to parsing.
          if (/^[a-h][1-8][a-h][1-8][qrnb]?$/.test(input)) {
            const uci: { from: string; to: string; promotion?: string } = {
              from: input.slice(0, 2),
              to: input.slice(2, 4),
            };
            if (input.length === 5) uci.promotion = input[4]!;
            chess.move(uci);
          } else {
            chess.move(input);
          }
        } catch {
          process.stdout.write(`Illegal or unparseable move: ${input}\n`);
          continue;
        }
      } else {
        process.stdout.write('engine thinking...\n');
        const { san, value } = await engineMove(agent, chess);
        process.stdout.write(
          `engine plays ${san}  (position value for engine: ${value.toFixed(2)})\n`,
        );
      }
    }

    if (chess.isGameOver()) {
      process.stdout.write(
        renderBoard(chess, flip, args.board, args.ascii) + '\n',
      );
      process.stdout.write(describeResult(chess) + '\n');
    }
  } finally {
    cleanup();
    rl.close();
  }
}

main().catch((e) => {
  process.stderr.write(`${e instanceof Error ? e.message : String(e)}\n`);
  process.exit(1);
});
