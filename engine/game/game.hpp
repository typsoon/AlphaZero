#ifndef GAME_HPP
#define GAME_HPP

#include <ATen/core/TensorBody.h>
#include <array>
#include <memory>
#include <torch/torch.h>
#include <vector>

using torch::Tensor;
// Abstract base class for game states, enabling hashing and comparison if
// needed
class GameState {
  private:
    Tensor state_tensor;

  public:
    GameState(Tensor state_tensor) : state_tensor(std::move(state_tensor)) {}

    // const Tensor &get_tensor() const {
    //     return state_tensor;
    // };

    operator Tensor() && { return std::move(state_tensor); }
    // operator Tensor() {
    //     return state_tensor;
    // };

    // Equality comparison for hashing or state lookup
    bool operator==(const GameState &other) const;
};

// Abstract base class for Games
class Game {
  public:
    Game() = default;
    Game(const Game &) = delete;
    Game &operator=(const Game &) = delete;
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
    virtual GameState get_canonical_state() const = 0;

    // Write the canonical state directly to a pre-allocated raw memory buffer
    virtual void write_canonical_state(float *out_buffer) const = 0;

    // Creates a deep copy of the current game state
    virtual std::unique_ptr<Game> clone() const = 0;

    // Optional: render the current state (e.g., for debugging/visualization)
    virtual void render() const = 0;
};

// Abstract base class for 2D board games
template <int Rows, int Cols> class Game2D : public Game {
  public:
    static constexpr int ROWS = Rows;
    static constexpr int COLS = Cols;

    using row_t = std::array<int, COLS>;
    using board_t = std::array<row_t, ROWS>;

    // The format matches what get_canonical_state computes
    virtual std::vector<std::vector<int>> get_board_state() const = 0;
    // Concrete 2D games can use board_t internally
};

#endif // GAME_HPP
