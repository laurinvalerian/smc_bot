import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from data_loader import load_all_pairs
from smc_strategy import get_smc_features, generate_signals


def get_pip_size(pair_name: str) -> float:
    """Return pip size: 0.01 for JPY pairs, 0.0001 for all others.

    Used by both the backtester and the optimizer.  For the optimizer a
    fixed **0.5 pip round-trip** transaction cost (realistic OANDA-level)
    is applied per trade via::

        cost_pct = (0.5 * pip_size / sl_distance) * risk_percent

    Pair names follow data_loader's format: ``XXXYYY_YEAR``
    (e.g. ``USDJPY_2023``).  ``split("_")[0]`` yields the full 6-character
    pair code (``USDJPY``), and ``endswith("JPY")`` correctly identifies
    JPY quote-currency pairs.

    The current ``backtest_pair`` function does *not* apply explicit pip
    costs so that existing backtest behaviour is preserved.  Use
    ``optimizer.py`` for cost-aware walk-forward analysis.

    ``ROUND_TRIP_PIPS`` defines the realistic transaction cost used when
    the backtester is extended to apply spread/commission deductions.
    """
    base = pair_name.split("_")[0].upper()
    return 0.01 if base.endswith("JPY") else 0.0001

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

_TRADES_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades_log.csv")

_REASON_LONG = "Bull FVG + Bull BOS + Liq Sweep Below + Discount Zone"
_REASON_SHORT = "Bear FVG + Bear BOS + Liq Sweep Above + Premium Zone"

# Realistic round-trip transaction cost (OANDA-level: 0.5 pip total per trade)
ROUND_TRIP_PIPS = 0.5

# Fixed risk per trade in account percentage (like a real trader risking 1% per trade)
_RISK_PERCENT = 1.0


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
        Keys: ``pair``, ``trades``, ``wins``, ``winrate``, ``gross_profit``,
        ``gross_loss``, ``profit_factor``, ``net_profit``, ``max_dd``,
        ``total_pnl``, ``trade_records``.
    """
    # ------------------------------------------------------------------
    # 1. Resample to M5
    # ------------------------------------------------------------------
    agg_map = {col: func for col, func in _OHLCV_AGG.items() if col in df.columns}
    df_m5 = df.resample("5min").agg(agg_map).dropna(subset=["close"])

    _empty = {
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
        "trade_records": [],
    }

    if len(df_m5) < 50:
        return _empty

    # ------------------------------------------------------------------
    # 2. Generate signals (TP = next Liquidity Pool, RR >= 2.0 enforced
    #    inside generate_signals via min_rr=2.0)
    # ------------------------------------------------------------------
    sig_df = generate_signals(df_m5, min_rr=2.0)

    # ------------------------------------------------------------------
    # 3. Simulate trades (full TP / full SL — no partials)
    # ------------------------------------------------------------------
    timestamps = sig_df.index
    high_arr = sig_df["high"].values
    low_arr = sig_df["low"].values
    close_arr = sig_df["close"].values
    signal_arr = sig_df["signal"].values
    entry_arr = sig_df["entry"].values
    sl_arr = sig_df["sl"].values
    tp_arr = sig_df["tp"].values
    rr_arr = sig_df["rr_ratio"].values

    n = len(sig_df)
    pnl_list: list[float] = []
    trade_records: list[dict] = []

    in_trade = False
    trade_signal = 0
    trade_entry = 0.0
    trade_sl = 0.0
    trade_tp = 0.0
    trade_rr = 0.0
    trade_ts = None
    trade_direction = ""
    trade_reason = ""

    # Running equity curve (account %, starting at 100) for max drawdown
    equity = 100.0
    equity_curve: list[float] = [100.0]

    def _record_trade(result: str, pnl_pct: float) -> None:
        nonlocal equity
        equity += pnl_pct
        equity_curve.append(equity)
        pnl_list.append(pnl_pct)
        trade_records.append(
            {
                "timestamp": trade_ts,
                "pair": pair_name,
                "direction": trade_direction,
                "entry": trade_entry,
                "sl": trade_sl,
                "tp": trade_tp,
                "rr": round(trade_rr, 4),
                "reason_entry": trade_reason,
                "result": result,
                "pnl": round(pnl_pct, 4),
                "risk_pct_used": _RISK_PERCENT,
                "pnl_pct": round(pnl_pct, 4),
                "equity_after": round(equity, 4),
            }
        )
        tqdm.write(
            f"[{pair_name}] {trade_direction} @ {trade_entry:.5f}"
            f" → TP {trade_tp:.5f} (RR {trade_rr:.2f})"
            f" → {result} {pnl_pct:+.4f}%  Equity: {equity:.2f}%"
        )

    for i in range(n):
        if in_trade:
            # Full SL exit (long)
            if trade_signal == 1:
                if low_arr[i] <= trade_sl:
                    _record_trade("LOSS", -_RISK_PERCENT)
                    in_trade = False
                # Full TP exit (long)
                elif high_arr[i] >= trade_tp:
                    _record_trade("WIN", _RISK_PERCENT * trade_rr)
                    in_trade = False
            # Full SL exit (short)
            elif trade_signal == -1:
                if high_arr[i] >= trade_sl:
                    _record_trade("LOSS", -_RISK_PERCENT)
                    in_trade = False
                # Full TP exit (short)
                elif low_arr[i] <= trade_tp:
                    _record_trade("WIN", _RISK_PERCENT * trade_rr)
                    in_trade = False

        if not in_trade and signal_arr[i] != 0 and not np.isnan(entry_arr[i]):
            in_trade = True
            trade_signal = int(signal_arr[i])
            trade_entry = entry_arr[i]
            trade_sl = sl_arr[i]
            trade_tp = tp_arr[i]
            trade_rr = float(rr_arr[i]) if not np.isnan(rr_arr[i]) else 0.0
            trade_ts = timestamps[i]
            trade_direction = "LONG" if trade_signal == 1 else "SHORT"
            trade_reason = _REASON_LONG if trade_signal == 1 else _REASON_SHORT

    # Close any open trade at the last close price (proportional account PnL)
    if in_trade:
        last_close = sig_df["close"].iloc[-1]
        sl_dist = abs(trade_entry - trade_sl)
        if sl_dist > 0:
            if trade_signal == 1:
                price_pnl = last_close - trade_entry
            else:
                price_pnl = trade_entry - last_close
            pnl_pct = price_pnl / sl_dist * _RISK_PERCENT
        else:
            pnl_pct = 0.0
        result = "WIN" if pnl_pct > 0 else "LOSS"
        _record_trade(result, pnl_pct)

    # ------------------------------------------------------------------
    # 4. Compute statistics
    # ------------------------------------------------------------------
    total_trades = len(pnl_list)
    if total_trades == 0:
        return _empty

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

    # Real max drawdown: peak-to-trough on the running equity curve (account %)
    eq_arr = np.array(equity_curve, dtype=np.float64)
    running_peak = np.maximum.accumulate(eq_arr)
    drawdowns = running_peak - eq_arr
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
        "trade_records": trade_records,
    }


# ---------------------------------------------------------------------------

def run_backtest(last_years: int = None) -> dict:
    """Run a backtest across all pairs found in the project's data folder.

    Parameters
    ----------
    last_years : int, optional
        When given, each pair's DataFrame is trimmed to the last
        *last_years* calendar years before backtesting.  Filtering is
        applied per-pair as::

            df = df[df.index >= df.index.max() - pd.DateOffset(years=last_years)]

    Returns
    -------
    dict
        ``summary`` key contains overall aggregated statistics;
        ``results`` key maps each pair name to its individual stats dict.
    """
    data_dict = load_all_pairs()
    pair_names = list(data_dict.keys())

    results: dict[str, dict] = {}
    all_trade_records: list[dict] = []

    for pair_name in tqdm(
        pair_names,
        desc="Backtesting Pairs",
        unit="it",
        bar_format="{desc}: {n}/{total} [{bar}] {percentage:.0f}% | {rate_fmt}",
    ):
        df = data_dict[pair_name]

        # Fix 1: filter truly the last X years per pair using each pair's own max date
        if last_years is not None:
            df = df[df.index >= df.index.max() - pd.DateOffset(years=last_years)]

        stats = backtest_pair(df, pair_name)
        results[pair_name] = stats
        all_trade_records.extend(stats.get("trade_records", []))

    # ------------------------------------------------------------------
    # Write trades_log.csv with extended columns
    # ------------------------------------------------------------------
    if all_trade_records:
        trades_df = pd.DataFrame(
            all_trade_records,
            columns=["timestamp", "pair", "direction", "entry", "sl", "tp",
                     "rr", "reason_entry", "result", "pnl",
                     "risk_pct_used", "pnl_pct", "equity_after"],
        )
        trades_df.to_csv(_TRADES_LOG_FILE, index=False)

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

    # Overall max DD: peak-to-trough on the combined account PnL stream
    all_pnl: list[float] = []
    for s in results.values():
        for r in s.get("trade_records", []):
            all_pnl.append(r["pnl_pct"])
    if all_pnl:
        pnl_cumsum = np.cumsum(np.array(all_pnl, dtype=np.float64))
        eq_combined = np.concatenate([[100.0], 100.0 + pnl_cumsum])
        running_max = np.maximum.accumulate(eq_combined)
        overall_max_dd = float((running_max - eq_combined).max())
        total_net_pnl = float(eq_combined[-1] - 100.0)
    else:
        overall_max_dd = 0.0
        total_net_pnl = 0.0

    summary = {
        "total_trades": total_trades,
        "overall_winrate": overall_winrate,
        "profit_factor": overall_profit_factor,
        "max_dd": overall_max_dd,
        "total_return_pct": total_net_pnl,
    }

    print("\nDetaillierte Trades + Equity in trades_log.csv gespeichert")

    return {"summary": summary, "results": results}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
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
    # Print stats without the full trade_records list for readability
    display = {k: v for k, v in stats.items() if k != "trade_records"}
    print(display)
    print(f"Trade records logged: {len(stats['trade_records'])}")
    if stats["trade_records"]:
        print("\nFirst trade record columns:", list(stats["trade_records"][0].keys()))
        print("First trade record:", stats["trade_records"][0])

    print("\nSelf-test complete. Use run_backtest() after populating the data/ folder for multi-pair tests.")
