import fastify from 'fastify';
import websocketPlugin from '@fastify/websocket';
import gameRoutes from './routes/game.js';
import { closeDb } from './db.js';

const server = fastify({ logger: true });

server.register(websocketPlugin);
server.register(gameRoutes, { prefix: '/:gameType' });

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 8000;

const start = async () => {
  try {
    await server.listen({ port: PORT, host: '0.0.0.0' });
  } catch (error) {
    server.log.error(error);
    process.exit(1);
  }
};

start();

// Neither SIGINT (Ctrl-C / dev restart) nor SIGTERM (systemd stop, container
// shutdown) get any special handling from Node by default - the process just
// dies immediately, mid-request, with every open WebSocket connection
// dropped without a close frame instead of the clean 1001 the
// @fastify/websocket plugin would otherwise send. server.close() stops the
// HTTP listener from accepting new connections, waits for in-flight
// requests to finish, and (via that plugin's own 'preClose' hook) closes
// every tracked WS client cleanly - see node_modules/@fastify/websocket's
// defaultPreClose. fastify's own close() has no built-in deadline, so an
// in-flight request that never resolves (e.g. AlphaZeroAgent's own
// INFERENCE_TIMEOUT_MS is 5 minutes) would otherwise hang the shutdown for
// just as long - SHUTDOWN_TIMEOUT_MS forces the process to exit instead of
// waiting that long for something outside our control.
const SHUTDOWN_TIMEOUT_MS = 10_000;
let shuttingDown = false;
async function shutdown(signal: NodeJS.Signals): Promise<void> {
  if (shuttingDown) return; // a second signal while already closing: ignore, let the first one finish
  shuttingDown = true;
  server.log.info(`Received ${signal}, shutting down gracefully...`);

  const timeout = setTimeout(() => {
    server.log.warn(
      `Graceful shutdown exceeded ${SHUTDOWN_TIMEOUT_MS}ms, forcing exit.`,
    );
    process.exit(1);
  }, SHUTDOWN_TIMEOUT_MS);
  timeout.unref(); // don't let this timer itself keep the process alive

  try {
    await server.close();
    closeDb();
    clearTimeout(timeout);
    server.log.info('Shutdown complete.');
    process.exit(0);
  } catch (error) {
    clearTimeout(timeout);
    server.log.error(error, 'Error during graceful shutdown, forcing exit.');
    process.exit(1);
  }
}

process.on('SIGINT', () => void shutdown('SIGINT'));
process.on('SIGTERM', () => void shutdown('SIGTERM'));
