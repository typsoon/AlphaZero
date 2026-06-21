#!/usr/bin/env python3
import json
import glob
import os
import sys

from .format_puzzles import format_json_string


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(base_dir, "games", "connect4", "*", "*.json")
    files = glob.glob(pattern)

    all_correct = True
    for filepath in files:
        with open(filepath, "r") as f:
            original_content = f.read()

        data = json.loads(original_content)
        json_str = json.dumps(data, indent=2)
        expected_content = format_json_string(json_str) + "\n"

        if original_content != expected_content:
            print(f"File {filepath} is not formatted correctly.")
            all_correct = False

    if not all_correct:
        sys.exit(1)
    else:
        print("All files are correctly formatted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
