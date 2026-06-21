#!/usr/bin/env python3
import json
import glob
import re
import os


def format_json_string(json_str):
    def replacer(match):
        content = match.group(1)
        items = [x.strip() for x in content.split(",") if x.strip()]
        return "[" + ", ".join(items) + "]"

    return re.sub(r"\[([\s0-9,-]+)\]", replacer, json_str)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(base_dir, "games", "connect4", "*", "*.json")
    files = glob.glob(pattern)

    for filepath in files:
        with open(filepath, "r") as f:
            data = json.load(f)

        json_str = json.dumps(data, indent=2)
        formatted_str = format_json_string(json_str)

        with open(filepath, "w") as f:
            f.write(formatted_str + "\n")


if __name__ == "__main__":
    main()
