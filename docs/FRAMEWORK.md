# Backtesting framework

[Home](../README.md) · [中文概览](../README.zh-CN.md) · [Input contract](DATA_CONTRACT.md)

## 1. The information clock

All timestamps use UTC. A daily feature uses the 24 hourly bars beginning on day `t`. It becomes observable at `t+1 00:00`. The default execution helper waits one hour, enters at `t+1 01:00`, and exits 24 hours later. Entry and exit prices are exact timestamp joins. A missing hour remains missing; the label never jumps to a later observation.

The price label is `exit_price / entry_price - 1`. OHLC opens are a research execution proxy; spread, slippage and impact are separate assumptions.

## 2. Selection before outcomes

`tradable_at_signal`, `signal` and `ex_ante_vol` determine the basket. Build eligibility from historically available exchange status, listing age, past liquidity and input completeness. Neither the existence of a future price nor the existence of future funding may determine eligibility.

Once a basket is fixed, the engine joins outcomes. Missing selected-name exit or funding data raises an error by default. Relaxed policies are available for exploratory work; they do not establish that a position could be closed.

## 3. Portfolio constraints

Scores are sorted deterministically. Top and bottom quantiles form separate long and short legs, each with half of total gross exposure. Inverse-volatility allocations are redistributed iteratively under a hard per-name cap. Both legs use the same feasible exposure after volatility scaling.

The quantile is a target. The engine may enlarge baskets to satisfy minimum-name and weight-cap requirements. With 40 names, 20% baskets contain 8 names each; with fewer names, the hard cap can require larger baskets. Dollar neutrality means signed notionals sum to zero; it does not guarantee zero BTC beta or zero sector exposure.

## 4. Funding cash flows

Funding is an event stream, not a fixed daily fee. Positive funding is paid by longs. For each event strictly inside the holding window, the data helper computes `funding_rate × mark_price / entry_price` when prices are supplied. The portfolio's funding contribution is the negative signed-weighted sum.

Pass `coverage` and set both `require_complete_coverage=True` and `require_mark_price=True` for complete event accounting. A coverage interval must represent a verified complete download; its presence alone cannot detect missing interior events. Exact-boundary settlements are excluded by convention. Place execution away from settlement boundaries.

## 5. Turnover and costs

At the next rebalance, prior holdings drift with price and account equity:

```text
pretrade_weight = previous_target × (1 + asset_price_return)
                  / (1 + previous_net_portfolio_return)
trade_weight = new_target - pretrade_weight
turnover = sum(abs(trade_weight))
cost = turnover × (commission + half_spread + slippage + impact) / 10,000
net_return = price_pnl + funding_pnl - cost
```

Opening positions incurs costs. Names leaving the basket generate closing trades. The ledger records pretrade, target and trade weights for reconciliation. The final daily NAV marks open positions to market; terminal liquidation costs are not automatically charged.

## 6. Walk-forward evaluation

Keep feature discovery, direction selection, transformations and model tuning inside training windows. A label can be used for training only after its exit time. Purge overlapping labels and apply an embargo where required. Rolling IC weights must use matured outcomes only. Compare all model scores using the same eligibility, execution and accounting rules.

The package accepts scores produced elsewhere. It does not implement the research factor library, Lasso, LightGBM or symbolic search.

## 7. Outputs and boundaries

The engine returns a daily PnL table and a position ledger. `performance_stats` reports CAGR, annualized volatility, zero-risk-free-rate Sharpe, maximum drawdown, final NAV and average turnover, using 365 daily observations per year.

This is a daily linear-notional accounting model. It does not simulate exchange orders, partial fills, intraday margin, liquidation or ADL. The bankruptcy guard stops a daily portfolio loss of 100% or more; it is not an intraday margin model. Point-in-time data quality remains the caller's responsibility.

