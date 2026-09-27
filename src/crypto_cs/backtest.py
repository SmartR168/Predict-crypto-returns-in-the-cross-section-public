"""Point-in-time cross-sectional portfolio backtest.

The engine deliberately separates the *selection set* from future returns. Positions
are selected using only ``tradable_at_signal``, signal values, and ex-ante risk. The
forward return is joined only after positions have been fixed. Missing exits therefore
cannot silently remove a delisted or otherwise problematic contract from a basket.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np
import pandas as pd


MissingReturnPolicy = Literal["raise", "drop_day", "zero"]
InvalidCrossSectionPolicy = Literal["raise", "skip"]


@dataclass(frozen=True)
class BacktestConfig:
    annualization: int = 365
    long_quantile: float = 0.20
    short_quantile: float = 0.20
    min_names_per_side: int = 3
    min_cross_section: int = 20
    gross_exposure: float = 1.0
    max_name_weight: float = 0.08
    # ``cost_bps_per_traded_notional`` is kept as the base commission/fee input for
    # backward compatibility. Spread, slippage and impact are charged separately.
    cost_bps_per_traded_notional: float = 2.0
    half_spread_bps_per_traded_notional: float = 0.0
    slippage_bps_per_traded_notional: float = 0.0
    impact_bps_per_traded_notional: float = 0.0
    target_annual_vol: float | None = 0.12
    vol_lookback_days: int = 30
    vol_min_observations: int = 15
    min_vol_scale: float = 0.50
    max_vol_scale: float = 1.50
    missing_return_policy: MissingReturnPolicy = "raise"
    invalid_cross_section_policy: InvalidCrossSectionPolicy = "raise"
    require_consecutive_dates: bool = True
    fail_on_bankruptcy: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.long_quantile < 0.5:
            raise ValueError("long_quantile must be between 0 and 0.5")
        if not 0 < self.short_quantile < 0.5:
            raise ValueError("short_quantile must be between 0 and 0.5")
        if self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be positive")
        if self.max_name_weight <= 0:
            raise ValueError("max_name_weight must be positive")
        cost_inputs = (
            self.cost_bps_per_traded_notional,
            self.half_spread_bps_per_traded_notional,
            self.slippage_bps_per_traded_notional,
            self.impact_bps_per_traded_notional,
        )
        if any(value < 0 for value in cost_inputs):
            raise ValueError("transaction cost inputs cannot be negative")
        if not 0 < self.min_vol_scale <= self.max_vol_scale:
            raise ValueError("volatility scale bounds must be positive and ordered")


def _capped_proportional_weights(
    risk_score: pd.Series,
    total_abs_weight: float,
    cap: float,
) -> pd.Series:
    """Allocate positive weights with a true hard cap via iterative redistribution."""
    if risk_score.empty:
        raise ValueError("cannot allocate an empty basket")
    if len(risk_score) * cap + 1e-12 < total_abs_weight:
        raise ValueError(
            f"weight cap infeasible: {len(risk_score)} names * {cap:.4f} "
            f"< side exposure {total_abs_weight:.4f}"
        )

    score = pd.to_numeric(risk_score, errors="coerce").replace([np.inf, -np.inf], np.nan)
    positive = score[score > 0]
    fallback = float(positive.median()) if not positive.empty else 1.0
    score = score.fillna(fallback).clip(lower=np.finfo(float).eps)

    result = pd.Series(0.0, index=score.index, dtype=float)
    active = list(score.index)
    remaining = float(total_abs_weight)

    while active:
        active_score = score.loc[active]
        proposal = active_score / active_score.sum() * remaining
        breached = proposal[proposal > cap + 1e-12]
        if breached.empty:
            result.loc[active] = proposal
            remaining = 0.0
            break
        result.loc[breached.index] = cap
        remaining -= cap * len(breached)
        active = [name for name in active if name not in set(breached.index)]

    if remaining > 1e-9:
        raise RuntimeError("capped allocation did not exhaust the requested exposure")
    if result.max() > cap + 1e-10:
        raise RuntimeError("weight cap was violated")
    return result


def _basket_size(n_names: int, quantile: float, config: BacktestConfig) -> int:
    side_exposure = config.gross_exposure / 2.0
    names_required_by_cap = ceil(side_exposure / config.max_name_weight)
    return max(
        int(np.floor(n_names * quantile)),
        config.min_names_per_side,
        names_required_by_cap,
    )


def build_positions_for_date(
    cross_section: pd.DataFrame,
    config: BacktestConfig,
    *,
    signal_col: str = "signal",
    tradable_col: str = "tradable_at_signal",
    risk_col: str = "ex_ante_vol",
) -> pd.Series:
    """Build base weights without consulting any future-return column."""
    required = {"symbol", signal_col, tradable_col, risk_col}
    missing = required - set(cross_section.columns)
    if missing:
        raise KeyError(f"missing position inputs: {sorted(missing)}")

    eligible = cross_section.loc[
        cross_section[tradable_col].fillna(False)
        & cross_section[signal_col].notna()
        & cross_section[risk_col].notna()
        & (cross_section[risk_col] > 0),
        ["symbol", signal_col, risk_col],
    ].copy()
    eligible = eligible.drop_duplicates("symbol", keep="last")
    if len(eligible) < config.min_cross_section:
        raise ValueError(f"cross-section too small: {len(eligible)}")

    long_n = _basket_size(len(eligible), config.long_quantile, config)
    short_n = _basket_size(len(eligible), config.short_quantile, config)
    if long_n + short_n > len(eligible):
        raise ValueError("long and short baskets would overlap")

    ordered = eligible.sort_values([signal_col, "symbol"], ascending=[False, True])
    long_leg = ordered.head(long_n).set_index("symbol")
    short_leg = ordered.tail(short_n).set_index("symbol")
    side_exposure = config.gross_exposure / 2.0

    long_weight = _capped_proportional_weights(
        1.0 / long_leg[risk_col], side_exposure, config.max_name_weight
    )
    short_weight = _capped_proportional_weights(
        1.0 / short_leg[risk_col], side_exposure, config.max_name_weight
    )
    weights = pd.concat([long_weight, -short_weight]).sort_index()
    if weights.index.has_duplicates:
        raise RuntimeError("a symbol appeared in both portfolio legs")
    return weights


def _vol_scale(realized_net_returns: list[float], config: BacktestConfig) -> float:
    if config.target_annual_vol is None:
        return 1.0
    history = pd.Series(realized_net_returns, dtype=float).tail(config.vol_lookback_days)
    if len(history) < config.vol_min_observations:
        return 1.0
    realized = history.std(ddof=1) * np.sqrt(config.annualization)
    if not np.isfinite(realized) or realized <= 0:
        return 1.0
    return float(
        np.clip(
            config.target_annual_vol / realized,
            config.min_vol_scale,
            config.max_vol_scale,
        )
    )


def _apply_vol_scale_with_neutrality(
    base_weight: pd.Series,
    requested_scale: float,
    config: BacktestConfig,
) -> tuple[pd.Series, float]:
    """Scale both legs equally while preserving neutrality and the hard name cap.

    A simple ``(weight * scale).clip(cap)`` can clip the two legs by different
    amounts and create unintended market exposure. This routine finds one feasible
    side exposure, then redistributes each leg independently under the same cap.
    """
    long_score = base_weight[base_weight > 0]
    short_score = -base_weight[base_weight < 0]
    if long_score.empty or short_score.empty:
        raise ValueError("both long and short legs are required")

    requested_side = config.gross_exposure * requested_scale / 2.0
    capacity = min(
        len(long_score) * config.max_name_weight,
        len(short_score) * config.max_name_weight,
    )
    actual_side = min(requested_side, capacity)
    long_weight = _capped_proportional_weights(
        long_score, actual_side, config.max_name_weight
    )
    short_weight = _capped_proportional_weights(
        short_score, actual_side, config.max_name_weight
    )
    target = pd.concat([long_weight, -short_weight]).sort_index()
    effective_scale = (2.0 * actual_side) / config.gross_exposure
    if abs(float(target.sum())) > 1e-10:
        raise RuntimeError("scaled portfolio lost dollar neutrality")
    return target, float(effective_scale)


def _drift_weights_to_next_rebalance(
    target_weight: pd.Series,
    price_return: pd.Series,
    net_portfolio_return: float,
) -> pd.Series:
    """Mark positions to the next rebalance before computing the next trade.

    Turnover must compare the new target with the *drifted live book*, not with
    yesterday's target. Funding and costs change account equity (the denominator),
    while price moves also change each contract's marked notional (the numerator).
    """
    denominator = 1.0 + net_portfolio_return
    if denominator <= 0:
        raise RuntimeError("cannot drift weights after non-positive portfolio equity")
    aligned_return = price_return.reindex(target_weight.index)
    if aligned_return.isna().any():
        raise ValueError("cannot drift positions with missing price returns")
    return target_weight * (1.0 + aligned_return) / denominator


def run_cross_sectional_backtest(
    panel: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
    *,
    date_col: str = "date",
    signal_col: str = "signal",
    tradable_col: str = "tradable_at_signal",
    risk_col: str = "ex_ante_vol",
    price_return_col: str = "fwd_price_return",
    funding_col: str = "fwd_funding_rate",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a daily long-short backtest and return daily PnL plus target weights.

    Funding follows Binance's long-position convention: a positive funding rate is
    paid by longs, so portfolio funding PnL is ``-weight * funding_rate``.
    """
    required = {
        date_col,
        "symbol",
        signal_col,
        tradable_col,
        risk_col,
        price_return_col,
        funding_col,
    }
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"missing backtest columns: {sorted(missing)}")

    data = panel.copy()
    data[date_col] = pd.to_datetime(data[date_col], utc=True)
    for column in (signal_col, risk_col, price_return_col, funding_col):
        data[column] = pd.to_numeric(data[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    data = data.sort_values([date_col, "symbol"])
    if data.duplicated([date_col, "symbol"]).any():
        raise ValueError("duplicate date/symbol rows found in backtest panel")
    dates = pd.Index(data[date_col].drop_duplicates().sort_values())
    if config.require_consecutive_dates and len(dates) > 1:
        gaps = dates.to_series().diff().dropna()
        if (gaps != pd.Timedelta(days=1)).any():
            raise ValueError("backtest dates are not consecutive UTC days")

    pretrade_weight = pd.Series(dtype=float)
    realized_net_returns: list[float] = []
    daily_rows: list[dict[str, float | int | pd.Timestamp]] = []
    weight_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for signal_date, cross_section in data.groupby(date_col, sort=True):
        try:
            base_weight = build_positions_for_date(
                cross_section,
                config,
                signal_col=signal_col,
                tradable_col=tradable_col,
                risk_col=risk_col,
            )
        except ValueError:
            if config.invalid_cross_section_policy == "skip":
                continue
            raise

        selected = cross_section.set_index("symbol").reindex(base_weight.index)
        missing_outcome = selected[[price_return_col, funding_col]].isna().any(axis=1)
        if missing_outcome.any():
            names = selected.index[missing_outcome].tolist()
            if config.missing_return_policy == "raise":
                raise ValueError(
                    f"selected names have missing exit/funding data on {signal_date.date()}: {names}"
                )
            if config.missing_return_policy == "drop_day":
                continue
            selected.loc[missing_outcome, [price_return_col, funding_col]] = 0.0
        if (selected[price_return_col] <= -1.0).any():
            raise ValueError(
                f"selected names contain price returns <= -100% on {signal_date.date()}"
            )

        requested_scale = _vol_scale(realized_net_returns, config)
        target_weight, effective_scale = _apply_vol_scale_with_neutrality(
            base_weight,
            requested_scale,
            config,
        )

        all_names = target_weight.index.union(pretrade_weight.index)
        trade_weight = (
            target_weight.reindex(all_names, fill_value=0.0)
            - pretrade_weight.reindex(all_names, fill_value=0.0)
        )
        turnover = float(trade_weight.abs().sum())
        commission_cost = turnover * config.cost_bps_per_traded_notional / 10_000.0
        spread_cost = turnover * config.half_spread_bps_per_traded_notional / 10_000.0
        slippage_cost = turnover * config.slippage_bps_per_traded_notional / 10_000.0
        impact_cost = turnover * config.impact_bps_per_traded_notional / 10_000.0
        cost = commission_cost + spread_cost + slippage_cost + impact_cost

        price_pnl = float((target_weight * selected[price_return_col]).sum())
        funding_pnl = float((-target_weight * selected[funding_col]).sum())
        gross_return = price_pnl + funding_pnl
        net_return = gross_return - cost
        if config.fail_on_bankruptcy and net_return <= -1:
            raise RuntimeError(
                f"portfolio return <= -100% on {signal_date.date()}; "
                "a linear-return backtest is invalid without liquidation/margin modelling"
            )

        daily_rows.append(
            {
                "date": signal_date,
                "price_pnl": price_pnl,
                "funding_pnl": funding_pnl,
                "gross_return": gross_return,
                "transaction_cost": cost,
                "commission_cost": commission_cost,
                "spread_cost": spread_cost,
                "slippage_cost": slippage_cost,
                "impact_cost": impact_cost,
                "net_return": net_return,
                "turnover": turnover,
                "requested_vol_scale": requested_scale,
                "vol_scale": effective_scale,
                "gross_exposure": float(target_weight.abs().sum()),
                "net_exposure": float(target_weight.sum()),
                "n_long": int((target_weight > 0).sum()),
                "n_short": int((target_weight < 0).sum()),
            }
        )
        for symbol in all_names:
            target = float(target_weight.get(symbol, 0.0))
            before = float(pretrade_weight.get(symbol, 0.0))
            trade = target - before
            weight_rows.append(
                {
                    "date": signal_date,
                    "symbol": symbol,
                    "weight": target,
                    "target_weight": target,
                    "pretrade_weight": before,
                    "trade_weight": trade,
                }
            )
        realized_net_returns.append(net_return)
        pretrade_weight = _drift_weights_to_next_rebalance(
            target_weight,
            selected[price_return_col],
            net_return,
        )

    daily = pd.DataFrame(daily_rows)
    weights = pd.DataFrame(weight_rows)
    if not daily.empty:
        daily["nav"] = (1.0 + daily["net_return"]).cumprod()
    return daily, weights


def performance_stats(
    daily: pd.DataFrame,
    *,
    return_col: str = "net_return",
    annualization: int = 365,
) -> dict[str, float | int | bool]:
    """Compute transparent performance statistics and flag invalid wealth paths."""
    if daily.empty or return_col not in daily:
        return {"days": 0, "bankrupt": False}
    returns = pd.to_numeric(daily[return_col], errors="coerce").dropna()
    if returns.empty:
        return {"days": 0, "bankrupt": False}
    wealth = (1.0 + returns).cumprod()
    bankrupt = bool((returns <= -1).any() or (wealth <= 0).any())
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(annualization))
        if returns.std(ddof=1) > 0
        else np.nan
    )
    cagr = np.nan if bankrupt else float(wealth.iloc[-1] ** (annualization / len(returns)) - 1)
    # Starting equity is 1.0: a loss on the first day is a drawdown too.
    drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1.0
    return {
        "days": int(len(returns)),
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_rf": sharpe,
        "max_drawdown": float(drawdown.min()),
        "final_nav": float(wealth.iloc[-1]),
        "average_turnover": float(daily["turnover"].mean()) if "turnover" in daily else np.nan,
        "bankrupt": bankrupt,
    }
