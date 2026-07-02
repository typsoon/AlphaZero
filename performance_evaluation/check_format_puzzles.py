#!/usr/bin/env python3
import json
import sys
from pathlib import Path

from .format_puzzles import format_json_string


def main():
    base_dir = Path(__file__).resolve().parent
    files = (base_dir / "games").glob("*/*/*.json")

    all_correct = True
    for filepath in files:
        original_content = filepath.read_text()

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
