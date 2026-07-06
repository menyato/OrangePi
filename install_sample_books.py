#!/usr/bin/env python3
"""Install the Book Reader practice books from the terminal.

Run on the Orange Pi, from the repo folder:

    python3 install_sample_books.py            # add the 3 practice books
    python3 install_sample_books.py --clear    # delete ALL saved books first

Installs garden, ocean, and mountain into ~/ocr_sessions (same as saying
"samples" inside the Book Reader). The Book Reader keeps only the 3 newest
books, so if you already have saved books, use --clear to start from a
clean shelf — otherwise the oldest books get pruned at the next startup.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.ocr_reader import SESSIONS_DIR, _install_sample_books


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Install the Book Reader practice books "
                    "(garden, ocean, mountain)")
    ap.add_argument("--clear", action="store_true",
                    help="delete every saved book first")
    args = ap.parse_args()

    if args.clear and os.path.isdir(SESSIONS_DIR):
        for path in glob.glob(os.path.join(SESSIONS_DIR, "*.json")):
            os.remove(path)
            print(f"deleted {path}")

    _install_sample_books()

    print(f"\nBooks now in {SESSIONS_DIR}:")
    for fname in sorted(os.listdir(SESSIONS_DIR)):
        if fname.endswith(".json"):
            print(f"  {fname}")


if __name__ == "__main__":
    main()
