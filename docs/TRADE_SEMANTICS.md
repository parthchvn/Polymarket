# Trade semantics

**Everything below is a WORKING ASSUMPTION supported by prior live
probes and requires continued validation** (see RESEARCH_ASSUMPTIONS.md).

## takerOnly=true

The `/trades` view with `takerOnly=true` appears to return approximately
one taker-side record per transaction.  It is the leading source for
canonical executions, subject to continuing audit.  Canonical executions
are the ONLY basis for market-wide volume, trade arrival, price paths,
market-state reconstruction and event windows.

## takerOnly=false

The `takerOnly=false` view expands trades into wallet-side/counterparty
records and reveals maker activity.  It feeds `actor_trade_legs`.  The
two views are NOT interchangeable, and expanded legs are never used for
market-wide volume because counterparties can double-count executions.

## Actor legs and deduplication cautions

The candidate fingerprint
`(transactionHash, proxyWallet, asset, side, size, price)` is stored as
a diagnostic only — it is NOT proven globally unique.  `actor_leg_id` is
constructed from raw provenance (raw_response_id + record index), so
legitimate repeated source records survive.  A shared transaction hash
alone never deduplicates.

## Positive-proposition convention

All analysis uses the price of the positive proposition
(`outcome_sign = +1`).  For a record on a negative token
(`outcome_sign = -1`):

```python
positive_price = 1.0 - observed_price
positive_side  = "SELL" if observed_side == "BUY" else "BUY"
```

Prices outside `[0, 1]` (beyond a 1e-9 tolerance) are rejected as
unresolved.  `reconciliation_status` records whether a canonical row was
`direct` or `complemented`.

## Maker/taker reconciliation

Matching order: source record ID → transaction hash → log index →
transaction-local occurrence → condition → positive-converted
size/price → 2-second timestamp tolerance.  An expanded leg matching a
taker-only record is `taker`; other confidently matched legs in the same
execution are `maker`; ambiguous matches stay `unknown` with an explicit
`role_confidence`.  Roles are never silently forced.
