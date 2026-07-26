#!/usr/bin/env python3
"""Thin wrapper: python scripts/run_analysis.py --db ... --output outputs/run"""
import sys

from polymarket.cli import main

if __name__ == "__main__":
    sys.exit(main(["run-analysis", *sys.argv[1:]]))
