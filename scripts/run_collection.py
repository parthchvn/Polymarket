#!/usr/bin/env python3
"""Thin wrapper: python scripts/run_collection.py --db ... --surface trades --condition-id ..."""
import sys

from polymarket.cli import main

if __name__ == "__main__":
    sys.exit(main(["collect", *sys.argv[1:]]))
