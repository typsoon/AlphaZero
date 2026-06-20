import http from 'node:http';

export class AlphaZeroAgent {
  private socketPath: string;

  constructor(socketPath: string = '/tmp/alphazero.sock') {
    this.socketPath = socketPath;
  }

  async act(gameState: any): Promise<number> {
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
              reject(new Error(data.message || 'Invalid response from inference server'));
              return;
            }
            const policy: number[] = data.policy;
            let bestMove = 0;
            let bestValue = -Infinity;
            for (let i = 0; i < policy.length; i++) {
              const val = policy[i] as number;
              if (val > bestValue) {
                bestValue = val;
                bestMove = i;
              }
            }
            resolve(bestMove);
          } catch (e) {
            reject(e);
          }
        });
      });

      req.on('error', (e) => {
        reject(e);
      });

      req.write(postData);
      req.end();
    });
  }
}
