#ifndef GAME_HPP
#define GAME_HPP

#include <ATen/core/TensorBody.h>
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
    GameState(Tensor &&state_tensor) : state_tensor(state_tensor) {}

    // const Tensor &get_tensor() const {
    //     return state_tensor;
    // };

    operator Tensor() && {
        return std::move(state_tensor);
    }
    // operator Tensor() {
    //     return state_tensor;
    // };

    // Equality comparison for hashing or state lookup
    bool operator==(const GameState &other) const;
};

// Abstract base class for Games
class Game {
  public:
    virtual ~Game() = default;

    // Reset game to initial state
    virtual void reset() = 0;

    // Return the dimension of the action space (number of possible discrete
    // actions)
    virtual int getActionSize() const = 0;

    // Return a list of legal action indices in the current state
    virtual std::vector<int> get_legal_actions() const = 0;

    // Apply the given action, modifying the game state
    virtual void step(int action) = 0;

    // Check if the current state is terminal (game over)
    virtual bool is_terminal() const = 0;

    virtual int get_current_player() const = 0;

    // Compute and return the reward for the current state (from perspective of
    // current player)
    virtual float reward() const = 0;

    // Get the canonical representation of the state for neural network input
    virtual GameState get_canonical_state() const = 0;

    // Get board state
    virtual std::vector<std::vector<int>> get_board_state() const = 0;

    // Produce a deep copy of the game (for tree search branching)
    virtual std::unique_ptr<Game> clone() const = 0;

    // Optional: render the current state (e.g., for debugging/visualization)
    virtual void render() const = 0;
};

#endif // GAME_HPP
