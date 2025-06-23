import tkinter as tk
import threading
import sys
import numpy as np
import torch
from typing import Optional

sys.path.append("../build/engine/")
from engine_bind import Connect4, Game
from agent import Agent

CELL_SIZE = 80
ROWS = 6
COLS = 7

class Connect4GUI:
    def __init__(
        self,
        root: tk.Tk,
        game: Game,
        red_agent: Agent,
        yellow_agent: Agent,
        red_color: str = "#e62a50",
        yellow_color: str = "#f1c40f",
        board_color: str = "#189eeb",
        bg_color: str = "#fdfdfd",
    ):
        self.root = root
        self.game = game
        self.red_agent = red_agent
        self.yellow_agent = yellow_agent
        self.colors = {
            "red": red_color,
            "yellow": yellow_color,
            "board": board_color,
            "bg": bg_color,
            "outline": "#dddddd",
            "text": "black",
            "error": "red",
            "draw": "gray"
        }

        canvas_height = ROWS * CELL_SIZE + 60
        self.canvas = tk.Canvas(
            root,
            width=COLS * CELL_SIZE,
            height=canvas_height,
            bg=self.colors["bg"],
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack()
        self.root.title("Connect 4")
        self.status_text_id = None
        self.draw_board()
        self.update_turn_display()
        self.root.after(500, self.play_turn)

    def draw_board(self):
        self.canvas.delete("board")
        for row in range(ROWS):
            for col in range(COLS):
                x0, y0 = col * CELL_SIZE, row * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE

                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=self.colors["board"],
                    outline=self.colors["bg"],
                    width=2,
                    tags="board"
                )

                cell_value = self.game.get_board_state()[row][col]
                token_color = (
                    self.colors["bg"] if cell_value == 0 else
                    self.colors["red"] if cell_value == 1 else
                    self.colors["yellow"]
                )

                self.canvas.create_oval(
                    x0 + 10, y0 + 10, x1 - 10, y1 - 10,
                    fill=token_color,
                    outline=self.colors["outline"],
                    width=1,
                    tags="board"
                )

        # ✅ Add column numbers (1-based)
        for col in range(COLS):
            x = col * CELL_SIZE + CELL_SIZE // 2
            y = ROWS * CELL_SIZE + 10
            self.canvas.create_text(
                x, y,
                text=str(col + 1),
                font=("Helvetica", 12),
                fill="#666",
                tags="board"
            )


    def draw_status(self, message: str, color: str = None):
        if color is None:
            color = self.colors["text"]
        if self.status_text_id is not None:
            self.canvas.delete(self.status_text_id)
        self.status_text_id = self.canvas.create_text(
            CELL_SIZE * COLS // 2,
            ROWS * CELL_SIZE + 30,
            text=message,
            font=("Helvetica", 14, "bold"),
            fill=color
        )

    def update_turn_display(self):
        player_color = self.colors["red"] if self.game.current_player == 1 else self.colors["yellow"]
        player_name = "Red" if self.game.current_player == 1 else "Yellow"
        self.draw_status(f"{player_name}'s Turn", "Black")

    def play_turn(self):
        if self.game.is_terminal:
            self.handle_game_end()
            return

        current_agent = self.red_agent if self.game.current_player == 1 else self.yellow_agent
        threading.Thread(target=self.execute_move, args=(current_agent,)).start()

    def handle_game_end(self):
        winner = self.game.current_player
        if winner == 0:
            self.draw_status("It's a draw.", self.colors["draw"])
        elif winner == 1:
            self.draw_status("Red wins!", self.colors["red"])
        else:
            self.draw_status("Yellow wins!", self.colors["yellow"])

    def execute_move(self, agent: Agent):
        try:
            action = agent.act(self.game)
            self.game.step(action)
            self.draw_board()
            if not self.game.is_terminal:
                self.update_turn_display() 
                self.root.after(300, self.play_turn)
            else:
                self.root.after(300, self.handle_game_end)
        except Exception as e:
            print(f"Move error: {e}")
            self.draw_status("Error. Skipping turn.", self.colors["error"])
            self.root.after(1000, self.play_turn)


def play_connect4(red: Agent, yellow: Agent):
    game = Connect4(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    root = tk.Tk()
    Connect4GUI(root, game, red, yellow)
    root.mainloop()
