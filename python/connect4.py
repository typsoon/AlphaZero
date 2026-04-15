"""Pure-Python Connect4 game state for gameplay server."""

from __future__ import annotations


class Connect4:
    ROWS = 6
    COLS = 7

    def __init__(self, initial_board: list[list[int]] | None = None):
        self.board: list[list[int]] = []
        self.current_player = 1
        self._finished = False
        self._reward = 0.0

        if initial_board is None:
            self.reset()
            return

        self._validate_board_shape(initial_board)
        self.board = [
            [self._normalize_cell_value(cell) for cell in row] for row in initial_board
        ]

        player1_count = 0
        player2_count = 0
        for row in self.board:
            for cell in row:
                if cell == 1:
                    player1_count += 1
                elif cell == -1:
                    player2_count += 1

        self.current_player = 1 if player1_count == player2_count else -1
        self._finished = False
        self._reward = 0.0

        for row in range(self.ROWS):
            for col in range(self.COLS):
                cell = self.board[row][col]
                if cell != 0 and self._check_win_for_player(row, col, cell):
                    self._finished = True
                    self._reward = 1.0 if cell == self.current_player else -1.0
                    return

        board_full = all(self.board[0][col] != 0 for col in range(self.COLS))
        if board_full:
            self._finished = True
            self._reward = 0.0

    @property
    def is_terminal(self) -> bool:
        return self._finished

    def reset(self) -> None:
        self.board = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.current_player = 1
        self._finished = False
        self._reward = 0.0

    def get_legal_actions(self) -> list[int]:
        return [col for col in range(self.COLS) if self.board[0][col] == 0]

    def step(self, action: int) -> None:
        if (
            self._finished
            or action < 0
            or action >= self.COLS
            or self.board[0][action] != 0
        ):
            raise ValueError(f"Invalid action {self._finished} {action}")

        placed_row = -1
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                placed_row = row
                break

        if placed_row == -1:
            raise ValueError(f"Invalid action {self._finished} {action}")

        if self._check_win_for_player(placed_row, action, self.current_player):
            self._finished = True
            self._reward = 1.0
        elif not self.get_legal_actions():
            self._finished = True
            self._reward = 0.0
        else:
            self.current_player = -self.current_player

    def get_board_state(self) -> list[list[int]]:
        return [row[:] for row in self.board]

    def reward(self) -> float:
        return self._reward if self._finished else 0.0

    def _validate_board_shape(self, board: list[list[int]]) -> None:
        if len(board) != self.ROWS:
            raise ValueError(f"Board must have {self.ROWS} rows")

        for row in board:
            if len(row) != self.COLS:
                raise ValueError(f"Board must have {self.COLS} columns")

    def _normalize_cell_value(self, cell: int) -> int:
        if cell == 2:
            return -1
        if cell in (-1, 0, 1):
            return cell
        raise ValueError(f"Invalid cell value: {cell}")

    def _check_win_for_player(self, row: int, col: int, player: int) -> bool:
        return (
            self._check_direction_for_player(row, col, 1, 0, player)
            or self._check_direction_for_player(row, col, 0, 1, player)
            or self._check_direction_for_player(row, col, 1, 1, player)
            or self._check_direction_for_player(row, col, 1, -1, player)
        )

    def _check_direction_for_player(
        self, row: int, col: int, d_row: int, d_col: int, player: int
    ) -> bool:
        count = 0
        for i in range(-3, 4):
            r = row + i * d_row
            c = col + i * d_col
            in_bounds = 0 <= r < self.ROWS and 0 <= c < self.COLS
            if in_bounds and self.board[r][c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
        return False
