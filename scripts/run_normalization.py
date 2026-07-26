#!/usr/bin/env python3
"""Thin wrapper: python scripts/run_normalization.py --db data/polymarket.sqlite"""
import sys

from polymarket.cli import main

if __name__ == "__main__":
    sys.exit(main(["normalize", *sys.argv[1:]]))
