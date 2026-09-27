"""Data preparation helpers with explicit timestamps and completeness checks."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_HOURLY_COLUMNS = {
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "quote_volume",
    "taker_buy_quote_volume",
    "trades_count",
}


def _normalise_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_HOURLY_COLUMNS - set(hourly.columns)
    if missing:
        raise KeyError(f"missing hourly columns: {sorted(missing)}")
    data = hourly.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(["symbol", "timestamp"])
    if data.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("duplicate symbol/timestamp bars found")
    if not data["timestamp"].eq(data["timestamp"].dt.floor("h")).all():
        raise ValueError("hourly timestamps must be aligned to exact UTC hours")
    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be strictly positive")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("invalid high price")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("invalid low price")
    if (data[["quote_volume", "taker_buy_quote_volume", "trades_count"]] < 0).any().any():
        raise ValueError("volume and trade-count fields cannot be negative")
    if (data["taker_buy_quote_volume"] > data["quote_volume"] + 1e-9).any():
        raise ValueError("taker-buy quote volume cannot exceed total quote volume")
    return data


def aggregate_daily_features(hourly: pd.DataFrame, *, min_hours: int = 24) -> pd.DataFrame:
    """Aggregate complete UTC days and record when each signal becomes observable."""
    data = _normalise_hourly(hourly)
    data["date"] = data["timestamp"].dt.floor("D")
    data["hourly_log_return"] = np.log(data["close"] / data["open"])

    daily = (
        data.groupby(["symbol", "date"], sort=True)
        .agg(
            open_d=("open", "first"),
            high_d=("high", "max"),
            low_d=("low", "min"),
            close_d=("close", "last"),
            quote_volume_d=("quote_volume", "sum"),
            taker_buy_quote_volume_d=("taker_buy_quote_volume", "sum"),
            trades_count_d=("trades_count", "sum"),
            hourly_realized_vol=("hourly_log_return", "std"),
            n_hours=("timestamp", "size"),
            first_hour=("timestamp", "min"),
            last_hour=("timestamp", "max"),
        )
        .reset_index()
    )
    daily["complete_utc_day"] = (
        daily["n_hours"].eq(24)
        & daily["first_hour"].eq(daily["date"])
        & daily["last_hour"].eq(daily["date"] + pd.Timedelta(hours=23))
    )
    # ``min_hours`` remains available for exploratory studies. Production runs must
    # select ``complete_utc_day`` (the default of 24 hours), not a relaxed threshold.
    daily["complete_signal_day"] = (
        daily["complete_utc_day"]
        if min_hours == 24
        else daily["n_hours"].ge(min_hours)
    )
    daily["signal_ready_time"] = daily["date"] + pd.Timedelta(days=1)
    return daily


def attach_hourly_execution_labels(
    daily_signals: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    entry_delay_hours: int = 1,
    holding_hours: int = 24,
) -> pd.DataFrame:
    """Attach exact-timestamp open-to-open labels after a conservative entry delay.

    A feature built from all 24 bars of UTC day ``t`` is known at ``t+1 00:00``.
    The default waits one full hour and enters at ``t+1 01:00``; this avoids assuming
    a fill at the same instant the last input candle becomes final. Missing entry or
    exit bars stay missing rather than shifting to the next available row.
    """
    if entry_delay_hours < 0 or holding_hours <= 0:
        raise ValueError("entry delay must be non-negative and holding period positive")
    signals = daily_signals.copy()
    if "signal_ready_time" not in signals:
        if "date" not in signals:
            raise KeyError("daily_signals needs date or signal_ready_time")
        signals["date"] = pd.to_datetime(signals["date"], utc=True)
        signals["signal_ready_time"] = signals["date"] + pd.Timedelta(days=1)
    signals["signal_ready_time"] = pd.to_datetime(signals["signal_ready_time"], utc=True)
    signals["entry_time"] = signals["signal_ready_time"] + pd.Timedelta(hours=entry_delay_hours)
    signals["exit_time"] = signals["entry_time"] + pd.Timedelta(hours=holding_hours)

    bars = _normalise_hourly(hourly)[["symbol", "timestamp", "open"]]
    entry = bars.rename(columns={"timestamp": "entry_time", "open": "entry_price"})
    exit_ = bars.rename(columns={"timestamp": "exit_time", "open": "exit_price"})
    out = signals.merge(entry, on=["symbol", "entry_time"], how="left")
    out = out.merge(exit_, on=["symbol", "exit_time"], how="left")
    out["fwd_price_return"] = out["exit_price"] / out["entry_price"] - 1.0
    out["has_exact_execution_window"] = out[["entry_price", "exit_price"]].notna().all(axis=1)
    return out


def aggregate_funding_events_for_windows(
    windows: pd.DataFrame,
    funding_events: pd.DataFrame,
    *,
    coverage: pd.DataFrame | None = None,
    require_complete_coverage: bool = False,
    require_mark_price: bool = False,
) -> pd.DataFrame:
    """Sum funding events strictly inside each position window.

    Exact-boundary events are excluded. Research code should place entry/exit away
    from settlement timestamps because Binance documents a small timing deviation.

    If ``entry_price`` and event-level ``mark_price`` are available, the function
    converts each funding payment to an entry-equity return using
    ``funding_rate * mark_price / entry_price``. Production runs should set
    ``require_mark_price=True``.

    ``coverage`` can prove that a zero event sum means "no payment in a fully
    collected interval" rather than "the downloader missed data". It must contain
    ``symbol``, ``coverage_start`` and ``coverage_end``. Production runs should pass
    it and set ``require_complete_coverage=True``.
    """
    required_windows = {"symbol", "entry_time", "exit_time"}
    required_funding = {"symbol", "funding_time", "funding_rate"}
    if required_windows - set(windows) or required_funding - set(funding_events):
        raise KeyError("missing execution-window or funding-event columns")

    win = windows.copy().reset_index(names="_window_id")
    events = funding_events.copy()
    win["entry_time"] = pd.to_datetime(win["entry_time"], utc=True)
    win["exit_time"] = pd.to_datetime(win["exit_time"], utc=True)
    events["funding_time"] = pd.to_datetime(events["funding_time"], utc=True)
    if events.duplicated(["symbol", "funding_time"]).any():
        raise ValueError("duplicate symbol/funding_time events found")
    events["funding_rate"] = pd.to_numeric(events["funding_rate"], errors="coerce")
    if events["funding_rate"].isna().any():
        raise ValueError("funding events contain missing or invalid rates")
    if require_mark_price and ("entry_price" not in win or "mark_price" not in events):
        raise KeyError("exact funding cashflow requires entry_price and mark_price")
    window_columns = ["_window_id", "symbol", "entry_time", "exit_time"]
    if "entry_price" in win:
        win["entry_price"] = pd.to_numeric(win["entry_price"], errors="coerce")
        if (win["entry_price"] <= 0).any() or win["entry_price"].isna().any():
            raise ValueError("funding windows contain invalid entry prices")
        window_columns.append("entry_price")
    if "mark_price" in events:
        events["mark_price"] = pd.to_numeric(events["mark_price"], errors="coerce")
        if require_mark_price and (
            events["mark_price"].isna().any() or (events["mark_price"] <= 0).any()
        ):
            raise ValueError("funding events contain invalid mark prices")
    merged = win[window_columns].merge(events, on="symbol", how="left")
    inside = (merged["funding_time"] > merged["entry_time"]) & (
        merged["funding_time"] < merged["exit_time"]
    )
    inside_events = merged.loc[inside].copy()
    use_mark_price = "entry_price" in inside_events and "mark_price" in inside_events
    if use_mark_price:
        inside_events["funding_cashflow_return"] = (
            inside_events["funding_rate"]
            * inside_events["mark_price"]
            / inside_events["entry_price"]
        )
        if require_mark_price and inside_events["funding_cashflow_return"].isna().any():
            raise ValueError("funding events inside position windows lack mark prices")
    else:
        inside_events["funding_cashflow_return"] = inside_events["funding_rate"]
    sums = inside_events.groupby("_window_id")["funding_cashflow_return"].sum()
    counts = inside_events.groupby("_window_id")["funding_rate"].size()
    win["fwd_funding_rate"] = win["_window_id"].map(sums).fillna(0.0)
    win["funding_event_count"] = win["_window_id"].map(counts).fillna(0).astype(int)
    win["funding_uses_mark_price"] = bool(use_mark_price)

    if coverage is None:
        win["funding_data_complete"] = pd.Series(pd.NA, index=win.index, dtype="boolean")
    else:
        required_coverage = {"symbol", "coverage_start", "coverage_end"}
        missing_coverage = required_coverage - set(coverage.columns)
        if missing_coverage:
            raise KeyError(f"missing funding coverage columns: {sorted(missing_coverage)}")
        cov = coverage[list(required_coverage)].copy()
        cov["coverage_start"] = pd.to_datetime(cov["coverage_start"], utc=True)
        cov["coverage_end"] = pd.to_datetime(cov["coverage_end"], utc=True)
        coverage_join = win[["_window_id", "symbol", "entry_time", "exit_time"]].merge(
            cov, on="symbol", how="left"
        )
        covered = (
            coverage_join["coverage_start"].le(coverage_join["entry_time"])
            & coverage_join["coverage_end"].ge(coverage_join["exit_time"])
        )
        covered_ids = set(coverage_join.loc[covered, "_window_id"])
        win["funding_data_complete"] = win["_window_id"].isin(covered_ids)

    if require_complete_coverage and not win["funding_data_complete"].fillna(False).all():
        bad = win.loc[
            ~win["funding_data_complete"].fillna(False), ["symbol", "entry_time", "exit_time"]
        ]
        raise ValueError(f"funding data coverage is incomplete for {len(bad)} windows")
    return win.drop(columns="_window_id")
