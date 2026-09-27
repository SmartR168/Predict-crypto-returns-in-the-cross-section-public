import numpy as np
import pandas as pd
import unittest

from crypto_cs.backtest import (
    BacktestConfig,
    _apply_vol_scale_with_neutrality,
    _capped_proportional_weights,
    run_cross_sectional_backtest,
    performance_stats,
)
from crypto_cs.data import (
    aggregate_daily_features,
    aggregate_funding_events_for_windows,
    attach_hourly_execution_labels,
)


def _panel_with_missing_top_return():
    symbols = [f"S{i:02d}" for i in range(20)]
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2025-01-01", tz="UTC"),
            "symbol": symbols,
            "signal": np.arange(20.0),
            "tradable_at_signal": True,
            "ex_ante_vol": 0.02,
            "fwd_price_return": [0.0] * 19 + [np.nan],
            "fwd_funding_rate": 0.0,
        }
    )


class BacktestTests(unittest.TestCase):
    def test_drawdown_includes_starting_equity(self):
        daily = pd.DataFrame({"net_return": [-0.10, 0.0]})
        self.assertAlmostEqual(performance_stats(daily)["max_drawdown"], -0.10)

    def test_hard_weight_cap_survives_redistribution(self):
        score = pd.Series([100.0] + [1.0] * 9, index=[f"S{i}" for i in range(10)])
        weights = _capped_proportional_weights(score, total_abs_weight=0.5, cap=0.08)
        self.assertAlmostEqual(weights.sum(), 0.5)
        self.assertLessEqual(weights.max(), 0.08 + 1e-12)

    def test_exact_execution_window_does_not_jump_over_missing_hour(self):
        timestamps = pd.date_range("2025-01-02 00:00", periods=27, freq="h", tz="UTC")
        hourly = pd.DataFrame(
            {
                "symbol": "AUSDT",
                "timestamp": timestamps,
                "open": np.arange(100.0, 127.0),
                "high": np.arange(101.0, 128.0),
                "low": np.arange(99.0, 126.0),
                "close": np.arange(100.5, 127.5),
                "quote_volume": 1_000_000.0,
                "taker_buy_quote_volume": 500_000.0,
                "trades_count": 100,
            }
        )
        hourly = hourly[hourly["timestamp"] != pd.Timestamp("2025-01-03 01:00", tz="UTC")]
        signals = pd.DataFrame({"symbol": ["AUSDT"], "date": ["2025-01-01"]})
        labelled = attach_hourly_execution_labels(signals, hourly)
        self.assertFalse(labelled.loc[0, "has_exact_execution_window"])
        self.assertTrue(pd.isna(labelled.loc[0, "fwd_price_return"]))

    def test_funding_boundary_events_are_excluded(self):
        windows = pd.DataFrame(
            {
                "symbol": ["AUSDT"],
                "entry_time": ["2025-01-01 00:00:00+00:00"],
                "exit_time": ["2025-01-02 00:00:00+00:00"],
            }
        )
        events = pd.DataFrame(
            {
                "symbol": ["AUSDT"] * 4,
                "funding_time": pd.to_datetime(
                    [
                        "2025-01-01 00:00:00+00:00",
                        "2025-01-01 08:00:00+00:00",
                        "2025-01-01 16:00:00+00:00",
                        "2025-01-02 00:00:00+00:00",
                    ]
                ),
                "funding_rate": [0.1, 0.01, 0.02, 0.2],
            }
        )
        result = aggregate_funding_events_for_windows(windows, events)
        self.assertAlmostEqual(result.loc[0, "fwd_funding_rate"], 0.03)
        self.assertEqual(result.loc[0, "funding_event_count"], 2)

    def test_funding_coverage_is_required_in_production_mode(self):
        windows = pd.DataFrame(
            {
                "symbol": ["AUSDT"],
                "entry_time": ["2025-01-01 01:00:00+00:00"],
                "exit_time": ["2025-01-02 01:00:00+00:00"],
            }
        )
        events = pd.DataFrame(
            {
                "symbol": ["AUSDT"],
                "funding_time": ["2025-01-01 08:00:00+00:00"],
                "funding_rate": [0.01],
            }
        )
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            aggregate_funding_events_for_windows(
                windows,
                events,
                require_complete_coverage=True,
            )

    def test_funding_cashflow_uses_event_mark_price(self):
        windows = pd.DataFrame(
            {
                "symbol": ["AUSDT"],
                "entry_time": ["2025-01-01 01:00:00+00:00"],
                "exit_time": ["2025-01-02 01:00:00+00:00"],
                "entry_price": [100.0],
            }
        )
        events = pd.DataFrame(
            {
                "symbol": ["AUSDT", "AUSDT"],
                "funding_time": [
                    "2025-01-01 08:00:00+00:00",
                    "2025-01-01 16:00:00+00:00",
                ],
                "funding_rate": [0.01, 0.01],
                "mark_price": [110.0, 90.0],
            }
        )
        result = aggregate_funding_events_for_windows(
            windows,
            events,
            require_mark_price=True,
        )
        self.assertAlmostEqual(result.loc[0, "fwd_funding_rate"], 0.02)
        self.assertTrue(result.loc[0, "funding_uses_mark_price"])

    def test_hourly_bars_must_be_on_exact_utc_hours(self):
        hourly = pd.DataFrame(
            {
                "symbol": ["AUSDT"],
                "timestamp": ["2025-01-01 00:30:00+00:00"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "quote_volume": [1_000_000.0],
                "taker_buy_quote_volume": [500_000.0],
                "trades_count": [100],
            }
        )
        with self.assertRaisesRegex(ValueError, "exact UTC hours"):
            aggregate_daily_features(hourly)

    def test_future_return_missing_does_not_remove_name_before_ranking(self):
        config = BacktestConfig(target_annual_vol=None, missing_return_policy="raise")
        with self.assertRaisesRegex(ValueError, "missing exit/funding"):
            run_cross_sectional_backtest(_panel_with_missing_top_return(), config)

    def test_turnover_uses_drifted_live_weights(self):
        rows = []
        symbols = [f"S{i:02d}" for i in range(20)]
        for day in pd.date_range("2025-01-01", periods=2, tz="UTC"):
            for i, symbol in enumerate(symbols):
                rows.append(
                    {
                        "date": day,
                        "symbol": symbol,
                        "signal": float(i),
                        "tradable_at_signal": True,
                        "ex_ante_vol": 0.02,
                        "fwd_price_return": 0.001 if i >= 13 else -0.001,
                        "fwd_funding_rate": 0.0,
                    }
                )
        config = BacktestConfig(target_annual_vol=None, cost_bps_per_traded_notional=2.0)
        daily, weights = run_cross_sectional_backtest(pd.DataFrame(rows), config)
        first_weights = weights[weights["date"] == daily.loc[0, "date"]]["weight"].abs().sum()
        self.assertAlmostEqual(daily.loc[0, "turnover"], first_weights)
        self.assertGreater(daily.loc[1, "turnover"], 0.0)

    def test_vol_scaling_preserves_neutrality_and_hard_cap(self):
        base = pd.Series(
            [0.08, 0.08, 0.08, 0.08, 0.06, 0.06, 0.06]
            + [-0.08, -0.08, -0.08, -0.07, -0.07, -0.06, -0.06],
            index=[f"S{i:02d}" for i in range(14)],
        )
        config = BacktestConfig(target_annual_vol=0.12)
        target, effective_scale = _apply_vol_scale_with_neutrality(base, 1.5, config)
        self.assertAlmostEqual(target.sum(), 0.0)
        self.assertLessEqual(target.abs().max(), config.max_name_weight + 1e-12)
        self.assertLess(effective_scale, 1.5)

    def test_cost_components_are_reported_separately(self):
        config = BacktestConfig(
            target_annual_vol=None,
            cost_bps_per_traded_notional=2.0,
            half_spread_bps_per_traded_notional=1.0,
            slippage_bps_per_traded_notional=0.5,
            impact_bps_per_traded_notional=0.25,
        )
        daily, _ = run_cross_sectional_backtest(_panel_with_missing_top_return().fillna(0.0), config)
        components = daily.loc[0, [
            "commission_cost", "spread_cost", "slippage_cost", "impact_cost"
        ]].sum()
        self.assertAlmostEqual(daily.loc[0, "transaction_cost"], components)

    def test_impossible_price_return_stops_backtest(self):
        panel = _panel_with_missing_top_return().fillna(0.0)
        panel.loc[panel["signal"].idxmax(), "fwd_price_return"] = -1.0
        config = BacktestConfig(target_annual_vol=None)
        with self.assertRaisesRegex(ValueError, "price returns <= -100%"):
            run_cross_sectional_backtest(panel, config)

    def test_date_gap_stops_production_backtest(self):
        first = _panel_with_missing_top_return().fillna(0.0)
        second = first.copy()
        second["date"] = pd.Timestamp("2025-01-03", tz="UTC")
        with self.assertRaisesRegex(ValueError, "not consecutive"):
            run_cross_sectional_backtest(
                pd.concat([first, second], ignore_index=True),
                BacktestConfig(target_annual_vol=None),
            )


if __name__ == "__main__":
    unittest.main()
