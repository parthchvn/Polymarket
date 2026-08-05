#!/bin/bash
# Overnight worker — ZERO writes to the collector's database.
# The collector owns runs/forward.sqlite exclusively; this script
# snapshots it (python backup API, safe against a live writer) and
# merges new raw responses into runs/nightly/work.sqlite, where all
# normalization, LLM scoring, and analysis happen with no contention.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/nightly

snapshot_and_merge() {
python3 - <<'PY'
import sqlite3

import os, sys

def backup(src, dst):
    a = sqlite3.connect(src, timeout=60)
    b = sqlite3.connect(dst)
    a.backup(b)
    b.close(); a.close()

# GUARD: sqlite3.connect silently CREATES a missing file — an empty
# database must never masquerade as the collector's. Refuse loudly.
if not os.path.exists("runs/forward.sqlite"):
    sys.exit("runs/forward.sqlite does not exist — is the collector "
             "running, and is this script running from the repo root?")
probe = sqlite3.connect("runs/forward.sqlite", timeout=60)
tables = {r[0] for r in probe.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")}
probe.close()
if "raw_responses" not in tables:
    sys.exit("runs/forward.sqlite has no raw_responses table — "
             "refusing to snapshot an empty or foreign database")

backup("runs/forward.sqlite", "runs/nightly/snap.sqlite")
import os
if not os.path.exists("runs/nightly/work.sqlite"):
    backup("runs/nightly/snap.sqlite", "runs/nightly/work.sqlite")
    print("seeded work.sqlite from first snapshot")
else:
    w = sqlite3.connect("runs/nightly/work.sqlite", timeout=60)
    w.execute("ATTACH 'runs/nightly/snap.sqlite' AS snap")
    added = 0
    for table in ("collector_runs", "collector_gaps", "raw_responses"):
        cur = w.execute(
            f"INSERT OR IGNORE INTO {table} SELECT * FROM snap.{table}"
        )
        added += cur.rowcount
    w.commit()
    w.execute("DETACH snap")
    w.close()
    print(f"merged snapshot: {added} new raw-layer rows")
PY
}

while true; do
  TS=$(date +%m%d-%H%M)
  echo "=== cycle $TS ==="
  snapshot_and_merge
  python -m polymarket.cli normalize --db runs/nightly/work.sqlite
  python -m polymarket.cli rescore-news --db runs/nightly/work.sqlite \
    --method ollama --model qwen3:8b --limit 300
  python -m polymarket.cli normalize --db runs/nightly/work.sqlite \
    --news-llm --llm-model qwen3:8b --llm-limit 30 --llm-score-limit 150
  python -m polymarket.cli reasoning-pipeline --db runs/nightly/work.sqlite \
    --output "runs/nightly/$TS" \
    --reasoning-model runs/reasoning_artifact/reasoning_model.json
  echo "=== cycle $TS done, sleeping 2h ==="
  sleep 7200
done
