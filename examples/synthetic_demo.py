"""Generated demonstration inputs; no research data or private signals."""
import numpy as np
import pandas as pd

from crypto_cs import BacktestConfig, performance_stats, run_cross_sectional_backtest


def make_synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for day in pd.date_range("2025-01-01", periods=90, tz="UTC"):
        for i in range(40):
            rows.append({
                "date": day,
                "symbol": f"DEMO{i:02d}",
                "signal": rng.normal(),
                "tradable_at_signal": True,
                "ex_ante_vol": rng.uniform(0.01, 0.04),
                "fwd_price_return": rng.normal(0, 0.015),
                "fwd_funding_rate": rng.normal(0.0001, 0.00005),
            })
    return pd.DataFrame(rows)


def main() -> None:
    daily, ledger = run_cross_sectional_backtest(
        make_synthetic_panel(),
        BacktestConfig(target_annual_vol=None),
    )
    print("SYNTHETIC DEMONSTRATION - generated inputs, not research performance")
    for name, value in performance_stats(daily).items():
        print(f"{name}: {value}")
    print("\nDaily accounting:")
    print(daily.head().to_string(index=False))
    print(f"\nPosition ledger: {len(ledger)} rows")


if __name__ == "__main__":
    main()

