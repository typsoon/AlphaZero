import fastify from 'fastify';
import websocketPlugin from '@fastify/websocket';
import gameRoutes from './routes/game.js';

const server = fastify({ logger: true });

server.register(websocketPlugin);
server.register(gameRoutes);

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
