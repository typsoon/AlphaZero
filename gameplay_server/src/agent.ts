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

  async act(gameState: Record<string, unknown>): Promise<number> {
    return new Promise((resolve, reject) => {
      const postData = JSON.stringify({ game_state: gameState });
      const options = {
        socketPath: this.socketPath,
        path: '/predict',
        method: 'POST',
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
            const policy = data.policy;
            let bestMove = 0;
            let bestValue = -Infinity;

            if (policy.length > 0 && typeof policy[0] === 'object') {
              // Sparse policy array: [{index, value}, ...]
              for (let i = 0; i < policy.length; i++) {
                const item = policy[i];
                if (item.value > bestValue) {
                  bestValue = item.value;
                  bestMove = item.index;
                }
              }
            } else {
              // Dense policy array
              for (let i = 0; i < policy.length; i++) {
                const val = policy[i] as number;
                if (val > bestValue) {
                  bestValue = val;
                  bestMove = i;
                }
              }
            }
            resolve(bestMove);
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
}
