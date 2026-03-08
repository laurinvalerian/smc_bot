import pandas as pd
import numpy as np
from tqdm import tqdm

from data_loader import load_all_pairs
from smc_strategy import get_smc_features, generate_signals

# ---------------------------------------------------------------------------
# OHLC aggregation mapping used when resampling
# ---------------------------------------------------------------------------
_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


# ---------------------------------------------------------------------------

def backtest_pair(df: pd.DataFrame, pair_name: str) -> dict:
    """Backtest a single currency pair.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with a ``DatetimeIndex`` (any granularity).
    pair_name : str
        Human-readable name used only for labelling.

    Returns
    -------
    dict
        Keys: ``pair``, ``trades``, ``winrate``, ``profit_factor``,
        ``net_profit``, ``max_dd``, ``total_pnl``.
    """
    # ------------------------------------------------------------------
    # 1. Resample to M5
    # ------------------------------------------------------------------
    agg_map = {col: func for col, func in _OHLCV_AGG.items() if col in df.columns}
    df_m5 = df.resample("5min").agg(agg_map).dropna(subset=["close"])

    if len(df_m5) < 50:
        return {
            "pair": pair_name,
            "trades": 0,
            "wins": 0,
            "winrate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "max_dd": 0.0,
            "total_pnl": 0.0,
        }

    # ------------------------------------------------------------------
    # 2. Generate signals (TP = next Liquidity Pool, RR >= 2.0 enforced
    #    inside generate_signals via min_rr=2.0)
    # ------------------------------------------------------------------
    sig_df = generate_signals(df_m5, min_rr=2.0)

    # ------------------------------------------------------------------
    # 3. Simulate trades
    # ------------------------------------------------------------------
    high_arr = sig_df["high"].values
    low_arr = sig_df["low"].values
    signal_arr = sig_df["signal"].values
    entry_arr = sig_df["entry"].values
    sl_arr = sig_df["sl"].values
    tp_arr = sig_df["tp"].values

    n = len(sig_df)
    pnl_list: list[float] = []
    in_trade = False
    trade_signal = 0
    trade_entry = 0.0
    trade_sl = 0.0
    trade_tp = 0.0

    for i in range(n):
        if in_trade:
            # Check exit for long trade
            if trade_signal == 1:
                if low_arr[i] <= trade_sl:
                    pnl_list.append(trade_sl - trade_entry)
                    in_trade = False
                elif high_arr[i] >= trade_tp:
                    pnl_list.append(trade_tp - trade_entry)
                    in_trade = False
            # Check exit for short trade
            elif trade_signal == -1:
                if high_arr[i] >= trade_sl:
                    pnl_list.append(trade_entry - trade_sl)
                    in_trade = False
                elif low_arr[i] <= trade_tp:
                    pnl_list.append(trade_entry - trade_tp)
                    in_trade = False

        if not in_trade and signal_arr[i] != 0 and not np.isnan(entry_arr[i]):
            in_trade = True
            trade_signal = int(signal_arr[i])
            trade_entry = entry_arr[i]
            trade_sl = sl_arr[i]
            trade_tp = tp_arr[i]

    # Close any open trade at the last close
    if in_trade:
        last_close = sig_df["close"].iloc[-1]
        if trade_signal == 1:
            pnl_list.append(last_close - trade_entry)
        else:
            pnl_list.append(trade_entry - last_close)

    # ------------------------------------------------------------------
    # 4. Compute statistics
    # ------------------------------------------------------------------
    total_trades = len(pnl_list)
    if total_trades == 0:
        return {
            "pair": pair_name,
            "trades": 0,
            "wins": 0,
            "winrate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "max_dd": 0.0,
            "total_pnl": 0.0,
        }

    pnl_arr = np.array(pnl_list, dtype=np.float64)
    win_mask = pnl_arr > 0
    wins_arr = pnl_arr[win_mask]
    losses_arr = pnl_arr[~win_mask]

    win_count = int(win_mask.sum())
    winrate = win_count / total_trades
    gross_profit = float(wins_arr.sum()) if len(wins_arr) > 0 else 0.0
    gross_loss = float(abs(losses_arr.sum())) if len(losses_arr) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    net_profit = float(pnl_arr.sum())

    # Max drawdown on cumulative PnL curve
    cum_pnl = np.cumsum(pnl_arr)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = float(drawdowns.max())

    return {
        "pair": pair_name,
        "trades": total_trades,
        "wins": win_count,
        "winrate": winrate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "net_profit": net_profit,
        "max_dd": max_dd,
        "total_pnl": net_profit,
    }


# ---------------------------------------------------------------------------

def run_backtest(folder_path: str, last_years: int = None) -> dict:
    """Run a backtest across all pairs found in *folder_path*.

    Parameters
    ----------
    folder_path : str
        Directory that contains ``*.csv`` files readable by
        :func:`data_loader.load_all_pairs`.
    last_years : int, optional
        When given, each pair's DataFrame is trimmed to the last
        *last_years* calendar years before backtesting.

    Returns
    -------
    dict
        ``summary`` key contains overall aggregated statistics;
        ``results`` key maps each pair name to its individual stats dict.
    """
    data_dict = load_all_pairs(folder_path)
    pair_names = list(data_dict.keys())

    results: dict[str, dict] = {}

    for pair_name in tqdm(
        pair_names,
        desc="Backtesting Pairs",
        unit="it",
        bar_format="{desc}: {n}/{total} [{bar}] {percentage:.0f}% | {rate_fmt}",
    ):
        df = data_dict[pair_name]

        if last_years is not None:
            cutoff = df.index.max() - pd.DateOffset(years=last_years)
            df = df[df.index >= cutoff]

        results[pair_name] = backtest_pair(df, pair_name)

    # ------------------------------------------------------------------
    # Aggregate overall statistics
    # ------------------------------------------------------------------
    total_trades = 0
    total_wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for stats in results.values():
        total_trades += stats["trades"]
        total_wins += stats["wins"]
        gross_profit += stats["gross_profit"]
        gross_loss += stats["gross_loss"]

    overall_winrate = total_wins / total_trades if total_trades > 0 else 0.0
    overall_profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    # Max DD on combined PnL stream (approximated from per-pair net profits)
    net_profits = np.array(
        [stats["net_profit"] for stats in results.values()], dtype=np.float64
    )
    cum_total = np.cumsum(net_profits)
    if len(cum_total) > 0:
        running_max = np.maximum.accumulate(cum_total)
        overall_max_dd = float((running_max - cum_total).max())
    else:
        overall_max_dd = 0.0

    total_net_pnl = float(cum_total[-1]) if len(cum_total) > 0 else 0.0

    summary = {
        "total_trades": total_trades,
        "overall_winrate": overall_winrate,
        "profit_factor": overall_profit_factor,
        "max_dd": overall_max_dd,
        "total_return_pct": total_net_pnl,
    }

    return {"summary": summary, "results": results}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import tempfile

    print("Running backtester self-test with dummy data …")

    np.random.seed(42)
    n_bars = 5_000
    dates = pd.date_range("2022-01-01", periods=n_bars, freq="1min")
    close = 1.1000 + np.cumsum(np.random.randn(n_bars) * 0.0005)
    dummy_df = pd.DataFrame(
        {
            "open": close + np.random.randn(n_bars) * 0.0002,
            "high": close + np.abs(np.random.randn(n_bars) * 0.0005),
            "low": close - np.abs(np.random.randn(n_bars) * 0.0005),
            "close": close,
            "volume": np.random.randint(100, 1000, n_bars).astype(float),
        },
        index=dates,
    )
    dummy_df.index.name = "timestamp"

    # ---- Single-pair test ----
    print("\n[1] backtest_pair …")
    stats = backtest_pair(dummy_df, "DUMMY")
    print(stats)

    # ---- Multi-pair test via run_backtest ----
    print("\n[2] run_backtest …")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write dummy CSV in histdata format
        rows = []
        for ts, row in dummy_df.iterrows():
            rows.append(
                f"{ts.strftime('%Y.%m.%d')},{ts.strftime('%H:%M')},"
                f"{row['open']:.5f},{row['high']:.5f},"
                f"{row['low']:.5f},{row['close']:.5f},{int(row['volume'])}"
            )
        csv_path = os.path.join(tmp_dir, "DUMMY.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(rows))

        bt_result = run_backtest(tmp_dir)

    print("\nSummary:")
    for k, v in bt_result["summary"].items():
        print(f"  {k}: {v}")
    print("\nPer-pair results:")
    for pair, res in bt_result["results"].items():
        print(f"  {pair}: {res}")
