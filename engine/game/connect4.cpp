#include "connect4.hpp"
#include "game.hpp"
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <iostream>
#include <stdexcept> // for invalid_argument
#include <string>
#include <unistd.h>
#include <vector>

using std::vector;

Connect4::Connect4(torch::Device device) : device(device) {
    reset();
}

Connect4::Connect4(const std::vector<std::vector<int>>& initial_board, 
                   torch::Device device) 
    : device(device) {
    // Validate board dimensions
    if (initial_board.size() != ROWS) {
        throw std::invalid_argument("Board must have " + std::to_string(ROWS) + " rows");
    }
    for (const auto& row : initial_board) {
        if (row.size() != COLS) {
            throw std::invalid_argument("Board must have " + std::to_string(COLS) + " columns");
        }
    }
    
    // Copy the board
    board = initial_board;
    
    // Determine current player by counting pieces
    int player1_count = 0;
    int player2_count = 0;
    for (const auto& row : board) {
        for (int cell : row) {
            if (cell == 1) player1_count++;
            else if (cell == -1) player2_count++;
        }
    }
    
    // Player 1 goes first, so if counts are equal, it's player 1's turn
    currentPlayer = (player1_count == player2_count) ? 1 : -1;
    
    // Check if game is already finished
    finished = false;
    _reward = 0.0f;
    
    // Check for wins
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
            if (board[row][col] != 0 && checkWin(row, col)) {
                finished = true;
                _reward = (board[row][col] == currentPlayer) ? 1.0f : -1.0f;
                return;
            }
        }
    }
    
    // Check for draw
    bool boardFull = true;
    for (int col = 0; col < COLS; col++) {
        if (board[0][col] == 0) {
            boardFull = false;
            break;
        }
    }
    if (boardFull) {
        finished = true;
        _reward = 0.0f;
    }
}

void Connect4::reset() {
    board = vector<vector<int>>(ROWS, vector<int>(COLS, 0));
    currentPlayer = 1; // Player 1 starts
    finished = false;
    _reward = 0.0f;
}

int Connect4::getActionSize() const {
    return COLS;
}

vector<int> Connect4::get_legal_actions() const {
    vector<int> legalActions;
    for (int col = 0; col < COLS; col++) {
        if (board[0][col] == 0) { // Column not full
            legalActions.push_back(col);
        }
    }
    return legalActions;
}

void Connect4::step(int action) {
    if (finished || action < 0 || action >= COLS || board[0][action] != 0) {
        throw std::invalid_argument("Invalid action " +
                                    std::to_string(finished) + " " +
                                    std::to_string(action));
    }

    int placedRow = -1;
    int placedCol = action;
    for (int row = ROWS - 1; row >= 0; row--) {
        if (board[row][action] == 0) {
            board[row][action] = currentPlayer;
            placedRow = row;
            break;
        }
    }

    if (checkWin(placedRow, placedCol)) {
        finished = true;
        _reward = 1.0f; // winner reward always positive
    } else if (get_legal_actions().empty()) {
        finished = true; // Draw
        _reward = 0.0f;
    } else {
        currentPlayer = -currentPlayer; // Switch player
    }
}

bool Connect4::is_terminal() const {
    return finished;
}

int Connect4::get_current_player() const {
    return currentPlayer;
}

float Connect4::reward() const {
    // Return reward from current player's perspective when finished
    // If not finished, 0.0f
    if (!finished)
        return 0.0f;
    return _reward;
}

GameState Connect4::get_canonical_state() const {
    torch::Tensor state = torch::zeros({1, ROWS, COLS}, torch::kFloat32);
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
            state[0][row][col] = board[row][col] * currentPlayer;
        }
    }

    if (state.device() != device) {
        state = state.to(device);
    }

    return GameState(std::move(state));
}

std::unique_ptr<Game> Connect4::clone() const {
    auto newGame = std::make_unique<Connect4>(device);
    newGame->board = board;
    newGame->currentPlayer = currentPlayer;
    newGame->finished = finished;
    newGame->_reward = _reward;
    return newGame;
}

void Connect4::render() const {
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
            char piece;
            if (board[row][col] == 1)
                piece = 'X';
            else if (board[row][col] == -1)
                piece = 'O';
            else
                piece = '.';
            std::cout << piece << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Current Player: " << (currentPlayer == 1 ? "X" : "O")
              << std::endl;
}

bool Connect4::checkWin(int row, int col) const {
    return checkDirection(row, col, 1, 0) || // Horizontal
           checkDirection(row, col, 0, 1) || // Vertical
           checkDirection(row, col, 1, 1) || // Diagonal
           checkDirection(row, col, 1, -1);  // Diagonal
}

bool Connect4::checkDirection(int row, int col, int dRow, int dCol) const {
    int count = 0;
    for (int i = -3; i <= 3; i++) {
        int r = row + i * dRow;
        int c = col + i * dCol;
        if (r >= 0 && r < ROWS && c >= 0 && c < COLS &&
            board[r][c] == currentPlayer) {
            count++;
            if (count == 4)
                return true;
        } else {
            count = 0;
        }
    }
    return false;
}

vector<vector<int>> Connect4::get_board_state() const {
    return board;
}
