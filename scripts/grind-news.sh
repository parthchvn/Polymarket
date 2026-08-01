#!/bin/bash
# Continuous news enrichment: download real article bodies, then
# LLM-extract claims from them (prioritized queue: newest, real
# bodies, plausibly relevant).  Everything is resumable; Ctrl-C is
# always safe.
set -u
cd "$(dirname "$0")/.."
DB="${1:-runs/nightly/work.sqlite}"
while true; do
  python -m polymarket.cli fetch-article-bodies --db "$DB" --limit 20
  python -m polymarket.cli extract-claims --db "$DB" --model qwen3:8b --limit 5
  sleep 10
done
