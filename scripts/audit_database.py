#!/usr/bin/env python3
"""Thin wrapper: python scripts/audit_database.py --db data/polymarket.sqlite [--json]"""
import sys

from polymarket.cli import main

if __name__ == "__main__":
    sys.exit(main(["audit", *sys.argv[1:]]))
