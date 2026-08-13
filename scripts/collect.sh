#!/bin/bash
# Forward collector with the current market set. Add markets by
# appending --condition-id / --news-query lines; then restart.
set -u
cd "$(dirname "$0")/.."
exec caffeinate -i python -m polymarket.cli collect-loop \
  --db runs/forward.sqlite \
  --condition-id 0x5db999fad322cea2914535aae5517060c3f80ad6d8c0231cde2124a434d16846 \
  --condition-id 0x60c2c085ee8c16bc8f2419739a94971d4c9d00f637ead10fc0f540afa1be64e8 \
  --condition-id 0x3d675f1c88099a57c12abca632cf926be1bf430125168321de06234e9930fe1a \
  --news-query "Iran nuclear" \
  --news-query "Strait of Hormuz" \
  --news-query "Federal Reserve rates" \
  --book-every 60
