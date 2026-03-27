import tkinter as tk
from agent import Agent

import sys

proj_root = Path(__file__).parent
build_dir = proj_root / "build"

sys.path.append(str(build_dir / "engine"))

from engine_bind import Game, Connect4  # pyright: ignore

CELL_SIZE = 80
ROWS = 6
COLS = 7


class Connect4GUI:
    def __init__(self, root, game: Game, red: Agent, yellow: Agent):
        self.root = root
        self.game = game
        self.red_agent = red
        self.yellow_agent = yellow
        self.canvas = tk.Canvas(
            root, width=COLS * CELL_SIZE, height=ROWS * CELL_SIZE + 50, bg="white"
        )
        self.canvas.pack()
        self.root.title("Connect 4 Simulation")

        self.draw_board()
        self.root.after(500, self.play_turn)

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="blue")
                cell = self.game.get_board_state()[r][c]
                if cell == 1:
                    color = "red"
                elif cell == -1:
                    color = "yellow"
                else:
                    color = "white"
                self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill=color)

        self.canvas.create_text(
            CELL_SIZE * COLS // 2,
            ROWS * CELL_SIZE + 20,
            text=f"{'Red' if self.game.current_player == 1 else 'Yellow'}'s Turn",
            font=("Arial", 16),
        )

    def play_turn(self):
        if self.game.is_terminal:
            winner_text = {1: "Red wins!", -1: "Yellow wins!", 0: "Draw!"}
            self.canvas.create_text(
                CELL_SIZE * COLS // 2,
                ROWS * CELL_SIZE + 20,
                text=winner_text[self.game.winner],
                font=("Arial", 18, "bold"),
                fill="green",
            )
            return

        agent = self.red_agent if self.game.current_player == 1 else self.yellow_agent
        try:
            print(f"Agent {self.game.current_player}")
            action = agent.act(self.game)

            print("dsasda")
            self.game.step(action)
        except Exception as e:
            print(f"Invalid move: {e}")
            return

        self.draw_board()
        self.root.after(500, self.play_turn)


def play_connect4(red: Agent, yellow: Agent):
    import tkinter as tk

    game = Connect4()
    root = tk.Tk()
    gui = Connect4GUI(root, game, red, yellow)
    root.mainloop()
