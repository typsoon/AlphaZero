import Database from 'better-sqlite3';
import path from 'path';

const dbPath = process.env.DB_PATH || path.resolve('games.db');
const db = new Database(dbPath);

db.pragma('journal_mode = WAL');

db.exec(`
  CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    board TEXT NOT NULL,
    current_player INTEGER NOT NULL,
    finished INTEGER NOT NULL,
    p1_type TEXT NOT NULL,
    p1_agent TEXT,
    p1_id TEXT,
    p2_type TEXT NOT NULL,
    p2_agent TEXT,
    p2_id TEXT,
    history TEXT NOT NULL DEFAULT '[]',
    winner INTEGER
  );
`);

try {
  db.exec('ALTER TABLE games ADD COLUMN winner INTEGER;');
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
} catch (e) {
  // The column already exists if this throws
}

const ADJECTIVES = [
  'Silent',
  'Misty',
  'Golden',
  'Dark',
  'Bright',
  'Cold',
  'Warm',
  'Deep',
  'Ancient',
  'Wild',
  'Calm',
  'Fierce',
  'Gentle',
  'Hidden',
  'Proud',
  'Swift',
  'Lost',
  'Wandering',
  'Sacred',
  'Crystal',
];
const NOUNS = [
  'Lake',
  'Mountain',
  'Forest',
  'River',
  'Ocean',
  'Sky',
  'Stone',
  'Tree',
  'Desert',
  'Valley',
  'Wind',
  'Cloud',
  'Sun',
  'Moon',
  'Star',
  'Snow',
  'Rain',
  'Fire',
  'Ice',
  'Earth',
];

function generateGameId(): string {
  const w1 = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
  const w2 = NOUNS[Math.floor(Math.random() * NOUNS.length)];
  return `${w1}-${w2}`;
}

export function createGame(
  boardJson: string,
  currentPlayer: number,
  finished: boolean,
  p1Type: string,
  p1Agent: string | null,
  p1Id: string | null,
  p2Type: string,
  p2Agent: string | null,
  p2Id: string | null,
  historyJson: string,
) {
  let id = generateGameId();
  while (getGame(id)) {
    id = `${generateGameId()}-${Math.floor(Math.random() * 1000)}`;
  }

  const stmt = db.prepare(
    'INSERT INTO games (id, board, current_player, finished, p1_type, p1_agent, p1_id, p2_type, p2_agent, p2_id, history, winner) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
  );
  stmt.run(
    id,
    boardJson,
    currentPlayer,
    finished ? 1 : 0,
    p1Type,
    p1Agent,
    p1Id,
    p2Type,
    p2Agent,
    p2Id,
    historyJson,
    null,
  );
  return id;
}

export function getGame(id: string) {
  const stmt = db.prepare('SELECT * FROM games WHERE id = ?');
  const row = stmt.get(id) as
    | {
        id: string;
        board: string;
        current_player: number;
        finished: number;
        p1_type: string;
        p1_agent: string | null;
        p1_id: string | null;
        p2_type: string;
        p2_agent: string | null;
        p2_id: string | null;
        history: string;
        winner: number | null;
      }
    | undefined;
  if (!row) return null;
  return {
    id: row.id,
    board: JSON.parse(row.board),
    currentPlayer: row.current_player,
    finished: row.finished === 1,
    p1Type: row.p1_type,
    p1Agent: row.p1_agent,
    p1Id: row.p1_id,
    p2Type: row.p2_type,
    p2Agent: row.p2_agent,
    p2Id: row.p2_id,
    history: JSON.parse(row.history),
    winner: row.winner,
  };
}

export function updateGame(
  id: string,
  boardJson: string,
  currentPlayer: number,
  finished: boolean,
  historyJson: string,
  winner: number | null = null,
) {
  const stmt = db.prepare(
    'UPDATE games SET board = ?, current_player = ?, finished = ?, history = ?, winner = ? WHERE id = ?',
  );
  stmt.run(boardJson, currentPlayer, finished ? 1 : 0, historyJson, winner, id);
}

export function updateGamePlayers(
  id: string,
  p1Type: string,
  p1Agent: string | null,
  p1Id: string | null,
  p2Type: string,
  p2Agent: string | null,
  p2Id: string | null,
) {
  const stmt = db.prepare(
    'UPDATE games SET p1_type = ?, p1_agent = ?, p1_id = ?, p2_type = ?, p2_agent = ?, p2_id = ? WHERE id = ?',
  );
  stmt.run(p1Type, p1Agent, p1Id, p2Type, p2Agent, p2Id, id);
}

export function deleteGame(id: string) {
  const stmt = db.prepare('DELETE FROM games WHERE id = ?');
  stmt.run(id);
}

export function getGames() {
  const stmt = db.prepare(
    'SELECT id, p1_type, p1_agent, p2_type, p2_agent, finished FROM games',
  );
  const rows = stmt.all() as {
    id: string;
    p1_type: string;
    p1_agent: string | null;
    p2_type: string;
    p2_agent: string | null;
    finished: number;
  }[];
  return rows.map((r) => ({
    id: r.id,
    p1Type: r.p1_type,
    p1Agent: r.p1_agent,
    p2Type: r.p2_type,
    p2Agent: r.p2_agent,
    finished: r.finished === 1,
  }));
}
