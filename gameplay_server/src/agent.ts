import http from 'node:http';

const INFERENCE_TIMEOUT_MS = 300_000; // 5 minutes – MCTS on CPU can take ~90s for chess

export type EvaluationPolicyEntry = { index: number; value: number };
export type Evaluation = {
  policy: number[] | EvaluationPolicyEntry[];
  value: number;
};

export class AlphaZeroAgent {
  private socketPath: string;

  constructor(socketPath: string = '/tmp/alphazero.sock') {
    this.socketPath = socketPath;
  }

  /** Raw policy/value from the inference server, without picking a best move. */
  async evaluate(gameState: Record<string, unknown>): Promise<Evaluation> {
    return new Promise((resolve, reject) => {
      const postData = JSON.stringify({ game_state: gameState });
      const options = {
        socketPath: this.socketPath,
        path: '/predict',
        method: 'POST',
        // Without this, node's default global agent pools/reuses sockets
        // across requests even though each call here looks like a fresh
        // http.request(). MCTS searches can easily leave 5-10s+ between
        // requests (waiting on a human's move, or - as found via
        // play_vs_engine_zoo.py - an opponent engine's own turn), long
        // enough for inference_server's HTTP layer to close an idle pooled
        // connection server-side; the next write on that now-dead socket
        // then fails with EPIPE. `agent: false` opts out of pooling
        // entirely so every request gets its own fresh connection - the
        // overhead is negligible at this request rate.
        agent: false as const,
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData),
        },
      };

      const req = http.request(options, (res) => {
        let body = '';
        res.on('data', (chunk) => {
          body += chunk;
        });
        res.on('end', () => {
          try {
            const data = JSON.parse(body);
            if (!Array.isArray(data.policy)) {
              reject(
                new Error(
                  data.error ||
                    data.message ||
                    'Invalid response from inference server',
                ),
              );
              return;
            }
            resolve({ policy: data.policy, value: data.value });
          } catch (e) {
            reject(e);
          }
        });
      });

      req.setTimeout(INFERENCE_TIMEOUT_MS, () => {
        req.destroy(
          new Error(
            `Inference server timed out after ${INFERENCE_TIMEOUT_MS / 1000}s`,
          ),
        );
      });

      req.on('error', (e) => {
        console.error(
          'Inference request failed',
          e,
          'socket:',
          this.socketPath,
        );
        reject(e);
      });

      req.write(postData);
      req.end();
    });
  }

  /** Picks the best move AND returns the policy/value it was chosen from, in
   * one inference call (evaluate() + argmax), so callers that want to show
   * the engine's evaluation don't need a second round trip to the socket. */
  async act(
    gameState: Record<string, unknown>,
  ): Promise<{ move: number } & Evaluation> {
    const evaluation = await this.evaluate(gameState);
    const policy = evaluation.policy;
    let bestMove = 0;
    let bestValue = -Infinity;

    if (policy.length > 0 && typeof policy[0] === 'object') {
      // Sparse policy array: [{index, value}, ...]
      for (const item of policy as EvaluationPolicyEntry[]) {
        if (item.value > bestValue) {
          bestValue = item.value;
          bestMove = item.index;
        }
      }
    } else {
      // Dense policy array
      const dense = policy as number[];
      for (let i = 0; i < dense.length; i++) {
        const val = dense[i]!;
        if (val > bestValue) {
          bestValue = val;
          bestMove = i;
        }
      }
    }
    return { move: bestMove, ...evaluation };
  }
}
