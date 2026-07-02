#!/usr/bin/env python3
import json
import re
from pathlib import Path


def format_json_string(json_str):
    def replacer(match):
        content = match.group(1)
        items = [x.strip() for x in content.split(",") if x.strip()]
        return "[" + ", ".join(items) + "]"

    return re.sub(r"\[([\s0-9,-]+)\]", replacer, json_str)


def main():
    base_dir = Path(__file__).resolve().parent
    files = (base_dir / "games").glob("*/*/*.json")

    for filepath in files:
        data = json.loads(filepath.read_text())

        json_str = json.dumps(data, indent=2)
        formatted_str = format_json_string(json_str)

        filepath.write_text(formatted_str + "\n")


if __name__ == "__main__":
    main()
