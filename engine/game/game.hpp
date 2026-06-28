#ifndef GAME_HPP
#define GAME_HPP

#include <ATen/core/TensorBody.h>
#include <array>
#include <cstdint>
#include <memory>
#include <torch/torch.h>
#include <vector>
using torch::Tensor;
class GameState {
  public:
    virtual ~GameState() = default;
    virtual void write_canonical_state(float *out_buffer) const = 0;
    virtual std::vector<int64_t> get_state_shape() const = 0;
};

// Abstract base class for Games
class Game : public GameState, public std::enable_shared_from_this<Game> {
  protected:
    Game(const Game &) = default;
    Game &operator=(const Game &) = default;

  public:
    Game() = default;
    virtual ~Game() = default;

    // Reset game to initial state
    virtual void reset() = 0;

    // Return the dimension of the action space (number of possible distinct actions)
    virtual int getActionSize() const = 0;

    // Return a list of legal action indices (returns a vector of valid actions)
    virtual std::vector<int> get_legal_actions() const = 0;

    // Apply the given action, modifying the game state
    virtual void step(int action) = 0;

    // Returns true if the game has ended (win, loss, or draw)
    virtual bool is_terminal() const = 0;

    virtual int get_current_player() const = 0;

    // Compute and return the reward for the current state (note: must handle wins, losses, and
    // draws)
    virtual float reward() const = 0;

    // Get the canonical representation of the state for neural network input
    virtual std::shared_ptr<const GameState> get_canonical_state() const = 0;

    // Creates a deep copy of the current game state
    virtual std::shared_ptr<Game> clone() const = 0;

    // Optional: render the current state (e.g., for debugging/visualization)
    virtual void render() const = 0;
};

// Abstract base class for 2D board games
template <int Rows, int Cols, typename CellT = int8_t> class Game2D : public Game {
  public:
    static constexpr int ROWS = Rows;
    static constexpr int COLS = Cols;

    using cell_t = CellT;
    using row_t = std::array<cell_t, COLS>;
    using board_t = std::array<row_t, ROWS>;

    // The format matches what get_canonical_state computes
    virtual board_t get_board_state() const = 0;
    // Concrete 2D games can use board_t internally
};

#endif // GAME_HPP
