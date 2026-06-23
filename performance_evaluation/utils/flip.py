import argparse
import json
import sys
from pathlib import Path


def flip_puzzle(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    if "board" not in data or "expected_moves" not in data:
        print(
            f"Error: {input_path} is not in the correct puzzle format.", file=sys.stderr
        )
        sys.exit(1)

    # Flip board horizontally
    flipped_board = []
    for row in data["board"]:
        flipped_board.append(list(reversed(row)))

    # Flip expected moves
    flipped_moves = []
    for move in data["expected_moves"]:
        flipped_moves.append(6 - move)
    flipped_moves.sort()

    data["board"] = flipped_board
    data["expected_moves"] = flipped_moves

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Successfully flipped puzzle and saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Flip a Connect4 puzzle horizontally.")
    parser.add_argument(
        "input_file", type=Path, help="Path to the input JSON puzzle file"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Path to the output JSON puzzle file (default: appends _flipped to the input filename)",
    )

    args = parser.parse_args()

    input_path = args.input_file
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output_file
    if not output_path:
        output_path = input_path.with_name(
            f"{input_path.stem}_flipped{input_path.suffix}"
        )

    flip_puzzle(input_path, output_path)


if __name__ == "__main__":
    main()
