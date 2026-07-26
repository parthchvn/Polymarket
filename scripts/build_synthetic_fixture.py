#!/usr/bin/env python3
"""Thin wrapper: python scripts/build_synthetic_fixture.py --db fixtures/synthetic_normalized.sqlite --overwrite"""
import sys

from polymarket.cli import main

if __name__ == "__main__":
    sys.exit(main(["build-synthetic", *sys.argv[1:]]))
