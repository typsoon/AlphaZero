import type { FastifyInstance } from "fastify";
import Connect4 from "../game/connect4.js";
import { AlphaZeroAgent } from "../agent.js";

const game = new Connect4();
const aiAgent = new AlphaZeroAgent(process.env.SOCKET_PATH || '/tmp/alphazero.sock');

function toCppInferenceGameState(board: number[][]) {
  const convertedBoard = board.map(row => 
    row.map(cell => cell === -1 ? 2 : cell)
  );
  return { board: convertedBoard };
}

export default async function gameRoutes(server: FastifyInstance) {
  server.get("/game/status", async (req, reply) => {
    return {
      status: "ok",
      board: game.get_board_state().board,
      legal_actions: game.get_legal_actions(),
      is_terminal: game.is_terminal(),
      current_player: 1
    };
  });

  server.post("/game/reset", async (req, reply) => {
    const body = req.body as { starting_player?: 'human' | 'ai' } | null;
    const startingPlayer = body?.starting_player || 'human';

    game.reset();

    if (startingPlayer === 'ai') {
      try {
        const aiGameState = toCppInferenceGameState(game.get_board_state().board);
        const aiMove = await aiAgent.act(aiGameState);
        game.step(aiMove);
      } catch (e: any) {
        server.log.error(e);
        reply.code(500);
        return { status: "error", message: e.message || "AI failed to make starting move" };
      }
    }

    return { status: "ok", message: "Game reset" };
  });

  server.post("/game/move", async (req, reply) => {
    const body = req.body as { column?: number };
    const column = body?.column;

    if (column === undefined || typeof column !== 'number' || column < 0 || column > 6) {
      reply.code(400);
      return { status: "error", message: "Invalid column" };
    }

    const legalActions = game.get_legal_actions();
    if (!legalActions.includes(column)) {
      reply.code(400);
      return { status: "error", message: `Illegal move: column ${column} not available` };
    }

    try {
      game.step(column);

      if (game.is_terminal()) {
        return {
          status: "ok",
          message: "Game over",
          board: game.get_board_state().board,
          is_terminal: true
        };
      }

      const aiGameState = toCppInferenceGameState(game.get_board_state().board);
      const aiMove = await aiAgent.act(aiGameState);
      
      game.step(aiMove);

      return {
        status: "ok",
        player_column: column,
        ai_column: aiMove,
        board: game.get_board_state().board,
        legal_actions: game.get_legal_actions(),
        is_terminal: game.is_terminal()
      };
    } catch (e: any) {
      server.log.error(e);
      reply.code(500);
      return { status: "error", message: e.message || "Move failed" };
    }
  });
}
