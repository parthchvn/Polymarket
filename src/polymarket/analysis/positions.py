"""Position reconciliation: reconstructed ledger vs platform snapshots.

The comparison runs over the UNION of platform-reported assets and
reconstructed nonzero assets, with an absolute-plus-relative tolerance.
Incomplete histories remain flagged, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polymarket.analysis.reader import SQLiteNormalizedReader


@dataclass
class PositionReconciliation:
    wallet: str
    asof: float
    platform_asset_count: int = 0
    reconstructed_nonzero_asset_count: int = 0
    union_asset_count: int = 0
    match_count: int = 0
    false_reconstructed_count: int = 0
    missing_reconstructed_count: int = 0
    max_absolute_error: float = 0.0
    max_relative_error: float = 0.0
    unresolved_event_count: int = 0
    unresolved_event_quantity: float = 0.0
    mismatches: list[dict] = field(default_factory=list)


def reconcile_wallet_positions(
    reader: SQLiteNormalizedReader,
    wallet: str,
    asof: float,
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-6,
) -> PositionReconciliation:
    result = PositionReconciliation(wallet=wallet, asof=asof)

    # latest platform snapshot per asset strictly before asof
    platform_balances: dict[str, float] = {}
    for row in reader.position_snapshots_before(asof, wallet=wallet):
        platform_balances[row["asset"]] = row["reported_size"]
    result.platform_asset_count = len(platform_balances)

    # reconstructed balances across all conditions strictly before asof
    reconstructed_balances: dict[str, float] = {}
    for event in reader.position_events_before(asof, wallet=wallet):
        if event["accounting_confidence"] == "unresolved":
            result.unresolved_event_count += 1
            if event["signed_token_change"] is not None:
                result.unresolved_event_quantity += abs(
                    event["signed_token_change"]
                )
            continue
        asset = event["asset"]
        if asset is None or event["signed_token_change"] is None:
            continue
        reconstructed_balances[asset] = (
            reconstructed_balances.get(asset, 0.0) + event["signed_token_change"]
        )
    reconstructed_nonzero = {
        a: v for a, v in reconstructed_balances.items() if abs(v) > 1e-12
    }
    result.reconstructed_nonzero_asset_count = len(reconstructed_nonzero)

    union_assets = set(platform_balances) | set(reconstructed_nonzero)
    result.union_asset_count = len(union_assets)

    for asset in sorted(union_assets):
        reported = platform_balances.get(asset, 0.0)
        reconstructed = reconstructed_balances.get(asset, 0.0)
        absolute_error = abs(reconstructed - reported)
        allowed_error = absolute_tolerance + relative_tolerance * abs(reported)
        match = absolute_error <= allowed_error
        if match:
            result.match_count += 1
        else:
            if asset not in platform_balances:
                result.false_reconstructed_count += 1
            elif asset not in reconstructed_nonzero:
                result.missing_reconstructed_count += 1
            result.mismatches.append(
                {"asset": asset, "reported": reported,
                 "reconstructed": reconstructed}
            )
        result.max_absolute_error = max(result.max_absolute_error, absolute_error)
        if abs(reported) > 0:
            result.max_relative_error = max(
                result.max_relative_error, absolute_error / abs(reported)
            )
    return result
