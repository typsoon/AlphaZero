import tkinter as tk
import threading
import sys
import numpy as np
import torch

sys.path.append("../build/engine/")
from engine_bind import Connect4, Game
from agent import Agent

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
            root,
            width=COLS * CELL_SIZE,
            height=ROWS * CELL_SIZE + 60,
            bg="#f0f0f0"
        )
        self.canvas.pack()
        self.root.title("Connect 4")

        self.draw_board()
        self.root.after(500, self.play_turn)

    def draw_board(self):
        self.canvas.delete("all")

        for r in range(ROWS):
            for c in range(COLS):
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE

                self.canvas.create_rectangle(x0, y0, x1, y1, fill="#3498db", outline="black")
                cell = self.game.get_board_state()[r][c]
                color = "white" if cell == 0 else ("red" if cell == 1 else "yellow")
                self.canvas.create_oval(x0 + 8, y0 + 8, x1 - 8, y1 - 8, fill=color, outline="gray")

        turn_msg = f"{'Red' if self.game.current_player == 1 else 'Yellow'}'s Turn"
        self.canvas.create_text(
            CELL_SIZE * COLS // 2,
            ROWS * CELL_SIZE + 30,
            text=turn_msg,
            font=("Helvetica", 16),
            fill="black"
        )

    def play_turn(self):
        if self.game.is_terminal:
            result = {1: "Red wins!", -1: "Yellow wins!", 0: "Draw!"}
            self.canvas.create_text(
                CELL_SIZE * COLS // 2,
                ROWS * CELL_SIZE + 30,
                text=result[self.game.current_player],
                font=("Helvetica", 18, "bold"),
                fill="green"
            )
            return

        agent = self.red_agent if self.game.current_player == 1 else self.yellow_agent

        def do_move():
            try:
                action = agent.act(self.game)
                self.game.step(action)
                self.draw_board()
                self.root.after(500, self.play_turn)
            except Exception as e:
                print(f"Error during move: {e}")
                self.root.after(1000, self.play_turn)

        threading.Thread(target=do_move).start()


def play_connect4(red: Agent, yellow: Agent):
    game = Connect4(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    root = tk.Tk()
    Connect4GUI(root, game, red, yellow)
    root.mainloop()
