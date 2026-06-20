import fastify from "fastify";
import gameRoutes from "./routes/game.js";

const server = fastify({ logger: true })

server.register(gameRoutes)

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 8000

let start = async () => {
  try {
    await server.listen({ port: PORT, host: '0.0.0.0' })
  }
  catch (error) {
    server.log.error(error)
    process.exit(1)
  }
}

start()
