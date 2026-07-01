export default interface Game {
  reset(): void;

  getActionSize(): number;

  // Return a list of legal action indices in the current state
  get_legal_actions(): number[];

  // Apply the given action, modifying the game state
  step(action: number): void;

  get_board_state(): { board: (number | string)[][] };

  get_current_player(): number;

  is_terminal(): boolean;
}
