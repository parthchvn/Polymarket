#!/usr/bin/env python3
"""Thin wrapper: python scripts/init_db.py --db data/polymarket.sqlite"""
import sys

from polymarket.cli import main

if __name__ == "__main__":
    sys.exit(main(["init-db", *sys.argv[1:]]))
