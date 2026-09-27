# Input contract

[Home](../README.md) · [Framework](FRAMEWORK.md)

## Daily portfolio panel

One row per UTC signal date and symbol. Duplicate keys are rejected.

| Field | Meaning | Available when |
| :--- | :--- | :--- |
| `date` | UTC signal date | Signal formation |
| `symbol` | Contract identifier | Signal formation |
| `signal` | Oriented score; higher means long | Signal formation |
| `tradable_at_signal` | Point-in-time eligibility | Signal formation |
| `ex_ante_vol` | Positive risk estimate from past data | Signal formation |
| `fwd_price_return` | Exact entry-to-exit price return | After exit |
| `fwd_funding_rate` | Long-side funding payment divided by entry notional | After exit |

Returns are decimals: `0.01` means 1%. Costs are bps per traded notional. Positive `fwd_funding_rate` is a payment by longs. Outcomes are accounting inputs, never selection inputs.

## Hourly bars

`symbol`, `timestamp`, `open`, `high`, `low`, `close`, `quote_volume`, `taker_buy_quote_volume`, `trades_count`.

Timestamps identify exact UTC hour starts. Production daily inputs require all 24 distinct bars. Prices must be positive and OHLC bounds valid. Volumes and trade counts must be nonnegative. Supply valid finite numeric values; do not forward-fill execution prices.

## Funding events

`symbol`, `funding_time`, `funding_rate`, `mark_price`.

One event per contract and settlement timestamp. Coverage uses `symbol`, `coverage_start`, `coverage_end`; each interval must certify an uninterrupted download. Windows need `symbol`, `entry_time`, `exit_time`, `entry_price`.

## Point-in-time metadata

Provide historical trading status, listing and delisting announcements, liquidity observations and required trading constraints through your own adapter. The public package contains no database connection, API key, historical market dataset or exchange metadata archive.
