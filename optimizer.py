"""
optimizer.py
============

Extreme-width Grid-Search Optimizer for the SMC Trading Strategy.

Walk-Forward Validation
-----------------------
  Training  (IS)  : 2016-01-01 – 2023-12-31
  Out-of-Sample   : 2024-01-01 – 2025-12-31

Transaction Costs (fixed, no slippage parameter)
-------------------------------------------------
  1.0 pip round-trip per trade (0.5 pip entry + 0.5 pip exit) is
  deducted from every trade's PnL in units of risk-percent:

      cost = (ROUND_TRIP_PIPS * pip_size / sl_distance) * risk_percent

Parameter Grid
--------------
  session_start_hour : [5, 6, 7, 8, 9]
  session_end_hour   : [16, 17, 18, 19, 20, 21]
  max_spread_pips    : [0.8, 1.2, 1.5, 1.8, 2.2, 2.5, 3.0, 4.0]
  min_rr             : [1.5, 2.0, 3.0, 4.0, 5.0]
  lookback           : [5, 8, 12, 15, 20, 25, 30]
  risk_percent       : [0.5, 1.0, 2.0, 3.0]

Outputs
-------
  optimization_results.csv  – all combinations with Train + Test metrics,
                              sorted descending by test_net_profit_pct
  top_20_net_profit.txt     – top 20 rows by Net Profit (Test period)

Usage
-----
  from optimizer import run_optimization
  run_optimization()
"""

from __future__ import annotations

import itertools
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_START = "2016-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-31"

ROUND_TRIP_PIPS = 1.0       # Fixed 1.0 pip round-trip cost (0.5 entry + 0.5 exit)
NUM_WORKERS     = 10        # Max parallel workers
MAX_LOOKFORWARD = 1000      # Max M5 bars to scan forward for SL/TP (~3.5 days)

PARAM_GRID: dict[str, list] = {
    "session_start_hour": [5, 6, 7, 8, 9],
    "session_end_hour":   [16, 17, 18, 19, 20, 21],
    "max_spread_pips":    [0.8, 1.2, 1.5, 1.8, 2.2, 2.5, 3.0, 4.0],
    "min_rr":             [1.5, 2.0, 3.0, 4.0, 5.0],
    "lookback":           [5, 8, 12, 15, 20, 25, 30],
    "risk_percent":       [0.5, 1.0, 2.0, 3.0],
}

_PARAM_KEYS = list(PARAM_GRID.keys())

_OHLCV_AGG = {
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}

# ---------------------------------------------------------------------------
# Module-level cache – populated in the main process and inherited by workers
# ---------------------------------------------------------------------------
_PRECOMP_TRAIN: dict[str, dict] = {}
_PRECOMP_TEST:  dict[str, dict] = {}
_PIP_SIZES:     dict[str, float] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pip_size(pair_name: str) -> float:
    """Return pip size: 0.01 for JPY pairs, 0.0001 for all others.

    Pair names follow data_loader's format: ``XXXYYY_YEAR``
    (e.g. ``USDJPY_2023``).  ``split("_")[0]`` therefore yields the
    full 6-character pair code (``USDJPY``), and ``endswith("JPY")``
    correctly identifies JPY quote-currency pairs.
    """
    base = pair_name.split("_")[0].upper()
    return 0.01 if base.endswith("JPY") else 0.0001


def _empty_stats() -> dict:
    return {
        "trades": 0, "wins": 0, "winrate": 0.0,
        "gross_profit": 0.0, "gross_loss": 0.0,
        "profit_factor": 0.0, "net_profit": 0.0, "max_dd": 0.0,
    }


# ---------------------------------------------------------------------------
# Pre-computation (runs once per pair, in the main process)
# ---------------------------------------------------------------------------

def _compute_ob_sl(
    ob_active: np.ndarray,
    ob_dir: np.ndarray,
    ob_bot: np.ndarray,
    ob_top: np.ndarray,
    close_arr: np.ndarray,
    swing_low: np.ndarray,
    swing_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute per-bar OB-based SL levels (with swing-level fallback).

    sl_long[i]  = highest active bull-OB bottom that lies below close[i]
    sl_short[i] = lowest  active bear-OB top   that lies above close[i]
    Falls back to swing_low / swing_high when no suitable OB exists.
    """
    n = len(close_arr)
    sl_long  = np.where(np.isnan(swing_low), np.nan, swing_low).copy()
    sl_short = np.where(np.isnan(swing_high), np.nan, swing_high).copy()

    ob_positions = np.where(ob_active.astype(bool))[0]
    if len(ob_positions) == 0:
        return sl_long, sl_short

    ob_dirs = ob_dir[ob_positions]
    ob_bots = ob_bot[ob_positions]
    ob_tops = ob_top[ob_positions]

    for idx in range(len(ob_positions)):
        pos = int(ob_positions[idx])
        if ob_dirs[idx] == 1:                          # Bull OB → Long SL
            b = ob_bots[idx]
            if np.isnan(b):
                continue
            # All bars from pos onward where close > OB bottom
            future = np.where(close_arr[pos:] > b)[0] + pos
            sl_long[future] = np.fmax(sl_long[future], b)
        elif ob_dirs[idx] == -1:                       # Bear OB → Short SL
            t = ob_tops[idx]
            if np.isnan(t):
                continue
            # All bars from pos onward where close < OB top
            future = np.where(close_arr[pos:] < t)[0] + pos
            sl_short[future] = np.fmin(sl_short[future], t)

    return sl_long, sl_short


def _precompute_pair(df_m5: pd.DataFrame, pair_name: str) -> dict:
    """Pre-compute SMC features, SL arrays, and rolling conditions for a pair.

    Returns a compact dict of numpy arrays used by worker processes.
    Rolling conditions are pre-computed for ALL lookback values in PARAM_GRID
    so each worker only does cheap array indexing, not rolling-window work.
    """
    from smc_strategy import get_smc_features

    if len(df_m5) < 100:
        return {}

    feat = get_smc_features(df_m5)
    n    = len(feat)

    close_arr  = feat["close"].values.astype(np.float64)
    high_arr   = feat["high"].values.astype(np.float64)
    low_arr    = feat["low"].values.astype(np.float64)
    open_arr   = feat["open"].values.astype(np.float64)

    swing_low  = feat["swing_low_level"].values.astype(np.float64)
    swing_high = feat["swing_high_level"].values.astype(np.float64)

    # OB-based SL arrays (expensive – done once here)
    sl_long, sl_short = _compute_ob_sl(
        feat["ob_active"].values,
        feat["ob_direction"].values,
        feat["ob_bottom"].values,
        feat["ob_top"].values,
        close_arr, swing_low, swing_high,
    )

    # Estimated spread: gap between bar open and previous close (price units)
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = close_arr[0]
    prev_close[1:] = close_arr[:-1]
    est_spread_raw = np.abs(open_arr - prev_close)

    # Rolling conditions for every lookback value in the grid
    rolling_cache: dict[int, dict[str, np.ndarray]] = {}
    for lb in PARAM_GRID["lookback"]:
        bull_fvg = (
            feat["fvg_bullish"].rolling(lb, min_periods=1).max().astype(bool).values
        )
        bear_fvg = (
            feat["fvg_bearish"].rolling(lb, min_periods=1).max().astype(bool).values
        )
        bull_bos = (
            (feat["bos_direction"] == 1)
            .astype(int)
            .rolling(lb, min_periods=1)
            .max()
            .astype(bool)
            .values
        )
        bear_bos = (
            (feat["bos_direction"] == -1)
            .astype(int)
            .rolling(lb, min_periods=1)
            .max()
            .astype(bool)
            .values
        )
        liq_below = (
            feat["liquidity_swept_below"].rolling(lb, min_periods=1).max().astype(bool).values
        )
        liq_above = (
            feat["liquidity_swept_above"].rolling(lb, min_periods=1).max().astype(bool).values
        )
        rolling_cache[lb] = {
            "bull_fvg":   bull_fvg,
            "bear_fvg":   bear_fvg,
            "bull_bos":   bull_bos,
            "bear_bos":   bear_bos,
            "liq_below":  liq_below,
            "liq_above":  liq_above,
        }

    return {
        "close":          close_arr,
        "high":           high_arr,
        "low":            low_arr,
        "open":           open_arr,
        "hours":          feat.index.hour.values.astype(np.int8),
        "sl_long":        sl_long,
        "sl_short":       sl_short,
        "liq_above":      feat["liquidity_pool_above"].values.astype(np.float64),
        "liq_below":      feat["liquidity_pool_below"].values.astype(np.float64),
        "discount_50":    feat["discount_50"].values.astype(np.float64),
        "est_spread_raw": est_spread_raw,
        "rolling":        rolling_cache,
        "pip_size":       get_pip_size(pair_name),
    }


# ---------------------------------------------------------------------------
# Trade simulation (runs inside each worker)
# ---------------------------------------------------------------------------

def _simulate_pair(pdata: dict, params: dict) -> dict:
    """Simulate trades for one pair with the given parameter combination.

    1.0 pip round-trip cost (0.5 pip entry + 0.5 pip exit) is deducted
    from every trade's PnL in units of risk-percent.
    """
    if not pdata:
        return _empty_stats()

    session_start    = int(params["session_start_hour"])
    session_end      = int(params["session_end_hour"])
    max_spread_raw   = float(params["max_spread_pips"]) * pdata["pip_size"]
    min_rr           = float(params["min_rr"])
    lookback         = int(params["lookback"])
    risk_percent     = float(params["risk_percent"])
    pip_size         = pdata["pip_size"]

    # ------------------------------------------------------------------
    # 1. Session filter (where we allow trade entries)
    # ------------------------------------------------------------------
    hours     = pdata["hours"]
    sess_mask = (hours >= session_start) & (hours < session_end)

    close = pdata["close"][sess_mask]
    high  = pdata["high"][sess_mask]
    low   = pdata["low"][sess_mask]

    sl_long   = pdata["sl_long"][sess_mask]
    sl_short  = pdata["sl_short"][sess_mask]
    liq_above = pdata["liq_above"][sess_mask]
    liq_below = pdata["liq_below"][sess_mask]
    disc_50   = pdata["discount_50"][sess_mask]
    est_sprd  = pdata["est_spread_raw"][sess_mask]

    roll = pdata["rolling"][lookback]
    bull_fvg  = roll["bull_fvg"][sess_mask]
    bear_fvg  = roll["bear_fvg"][sess_mask]
    bull_bos  = roll["bull_bos"][sess_mask]
    bear_bos  = roll["bear_bos"][sess_mask]
    liq_sw_below = roll["liq_below"][sess_mask]
    liq_sw_above = roll["liq_above"][sess_mask]

    n = len(close)
    if n < 50:
        return _empty_stats()

    # ------------------------------------------------------------------
    # 2. Signal conditions
    # ------------------------------------------------------------------
    in_discount = close < disc_50
    in_premium  = close > disc_50
    spread_ok   = est_sprd <= max_spread_raw

    long_cond  = bull_fvg & bull_bos  & liq_sw_below & in_discount & spread_ok
    short_cond = bear_fvg & bear_bos  & liq_sw_above & in_premium  & spread_ok

    # Validate SL / TP availability
    long_valid = (
        long_cond
        & ~np.isnan(sl_long)  & (sl_long  < close)
        & ~np.isnan(liq_above) & (liq_above > close)
    )
    short_valid = (
        short_cond
        & ~np.isnan(sl_short) & (sl_short  > close)
        & ~np.isnan(liq_below) & (liq_below < close)
    )

    # RR check (avoid divide-by-zero from np.where evaluating both branches)
    sl_dist_long  = close - sl_long
    sl_dist_short = sl_short - close

    with np.errstate(divide="ignore", invalid="ignore"):
        rr_long  = np.where(sl_dist_long  > 0, (liq_above - close) / sl_dist_long,  0.0)
        rr_short = np.where(sl_dist_short > 0, (close - liq_below)  / sl_dist_short, 0.0)

    long_valid  = long_valid  & (rr_long  >= min_rr)
    short_valid = short_valid & (rr_short >= min_rr)

    # Skip ambiguous bars (both long and short valid simultaneously)
    both        = long_valid & short_valid
    long_valid  = long_valid  & ~both
    short_valid = short_valid & ~both

    # ------------------------------------------------------------------
    # 3. Trade simulation (loop only over signal bars – fast in practice)
    # ------------------------------------------------------------------
    signal_bars = np.where(long_valid | short_valid)[0]

    pnl_list     : list[float] = []
    wins          = 0
    gross_profit  = 0.0
    gross_loss    = 0.0
    equity        = 100.0
    equity_curve  : list[float] = [100.0]
    next_available = 0

    for sig_i in signal_bars:
        if sig_i < next_available:
            continue

        is_long = bool(long_valid[sig_i])

        if is_long:
            sl       = float(sl_long[sig_i])
            tp       = float(liq_above[sig_i])
            rr       = float(rr_long[sig_i])
            sl_dist  = float(sl_dist_long[sig_i])
        else:
            sl       = float(sl_short[sig_i])
            tp       = float(liq_below[sig_i])
            rr       = float(rr_short[sig_i])
            sl_dist  = float(sl_dist_short[sig_i])

        # 1.0 pip round-trip cost (0.5 entry + 0.5 exit) in risk-percent units
        cost = (ROUND_TRIP_PIPS * pip_size / sl_dist) * risk_percent

        # Bounded forward search for first SL / TP hit
        end_idx   = min(sig_i + 1 + MAX_LOOKFORWARD, n)
        fut_high  = high[sig_i + 1 : end_idx]
        fut_low   = low[sig_i + 1  : end_idx]

        if is_long:
            sl_mask = fut_low  <= sl
            tp_mask = fut_high >= tp
        else:
            sl_mask = fut_high >= sl
            tp_mask = fut_low  <= tp

        sl_hit = int(np.argmax(sl_mask)) if sl_mask.any() else -1
        tp_hit = int(np.argmax(tp_mask)) if tp_mask.any() else -1

        if sl_hit == -1 and tp_hit == -1:
            # Neither hit within lookforward; mark as proportional close
            last_close = float(close[end_idx - 1])
            entry_px   = float(close[sig_i])
            price_pnl  = (last_close - entry_px) if is_long else (entry_px - last_close)
            pnl        = price_pnl / sl_dist * risk_percent - cost
            exit_bar   = end_idx - 1
        elif sl_hit == -1:
            pnl      = risk_percent * rr - cost
            exit_bar = sig_i + 1 + tp_hit
        elif tp_hit == -1:
            pnl      = -risk_percent - cost
            exit_bar = sig_i + 1 + sl_hit
        else:
            if sl_hit <= tp_hit:
                pnl      = -risk_percent - cost
                exit_bar = sig_i + 1 + sl_hit
            else:
                pnl      = risk_percent * rr - cost
                exit_bar = sig_i + 1 + tp_hit

        equity += pnl
        equity_curve.append(equity)
        pnl_list.append(pnl)

        if pnl > 0:
            wins         += 1
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)

        next_available = exit_bar + 1

    # ------------------------------------------------------------------
    # 4. Statistics
    # ------------------------------------------------------------------
    total_trades = len(pnl_list)
    if total_trades == 0:
        return _empty_stats()

    winrate = wins / total_trades
    pf      = (gross_profit / gross_loss
               if gross_loss > 0
               else (float("inf") if gross_profit > 0 else 0.0))
    net_profit = float(sum(pnl_list))

    eq_arr       = np.array(equity_curve, dtype=np.float64)
    running_peak = np.maximum.accumulate(eq_arr)
    max_dd       = float((running_peak - eq_arr).max())

    return {
        "trades":        total_trades,
        "wins":          wins,
        "winrate":       winrate,
        "gross_profit":  gross_profit,
        "gross_loss":    gross_loss,
        "profit_factor": pf,
        "net_profit":    net_profit,
        "max_dd":        max_dd,
    }


def _agg_pairs(precomp_dict: dict, params: dict) -> dict:
    """Aggregate simulation results across all pairs."""
    total_trades = 0
    total_wins   = 0
    total_gp     = 0.0
    total_gl     = 0.0
    total_net    = 0.0
    max_dd       = 0.0

    for pdata in precomp_dict.values():
        s            = _simulate_pair(pdata, params)
        total_trades += s["trades"]
        total_wins   += s["wins"]
        total_gp     += s["gross_profit"]
        total_gl     += s["gross_loss"]
        total_net    += s["net_profit"]
        max_dd        = max(max_dd, s["max_dd"])

    if total_trades == 0:
        return {"trades": 0, "winrate": 0.0, "profit_factor": 0.0,
                "net_profit": 0.0, "max_dd": 0.0}

    return {
        "trades":        total_trades,
        "winrate":       total_wins / total_trades,
        "profit_factor": total_gp / total_gl if total_gl > 0 else float("inf"),
        "net_profit":    total_net,
        "max_dd":        max_dd,
    }


# ---------------------------------------------------------------------------
# Worker (runs in child processes)
# ---------------------------------------------------------------------------

def _worker_init(train_data: dict, test_data: dict) -> None:
    """Initializer: populate module-level caches in each worker process."""
    global _PRECOMP_TRAIN, _PRECOMP_TEST
    _PRECOMP_TRAIN = train_data
    _PRECOMP_TEST  = test_data


def _run_combo(params_tuple: tuple) -> dict | None:
    """Worker function: evaluate one parameter combination on train + test data."""
    params = dict(zip(_PARAM_KEYS, params_tuple))

    # Skip invalid session window
    if params["session_start_hour"] >= params["session_end_hour"]:
        return None

    train_agg = _agg_pairs(_PRECOMP_TRAIN, params)
    test_agg  = _agg_pairs(_PRECOMP_TEST,  params)

    # Primary score = net profit of the test period (final account balance in %)
    score = test_agg["net_profit"]

    # stable_score helps filter later: reward profit relative to drawdown
    # The +5 offset prevents extreme values when drawdown is near zero
    stable_score = test_agg["net_profit"] / (test_agg["max_dd"] + 5)

    return {
        # Parameters
        "session_start_hour":  params["session_start_hour"],
        "session_end_hour":    params["session_end_hour"],
        "max_spread_pips":     params["max_spread_pips"],
        "min_rr":              params["min_rr"],
        "lookback":            params["lookback"],
        "risk_percent":        params["risk_percent"],
        # Training metrics
        "train_trades":        train_agg["trades"],
        "train_winrate":       round(train_agg["winrate"] * 100, 2),
        "train_profit_factor": round(train_agg["profit_factor"], 4),
        "train_net_profit_pct": round(train_agg["net_profit"], 4),
        "train_max_dd":        round(train_agg["max_dd"], 4),
        # Out-of-sample metrics
        "test_trades":         test_agg["trades"],
        "test_winrate":        round(test_agg["winrate"] * 100, 2),
        "test_profit_factor":  round(test_agg["profit_factor"], 4),
        "test_net_profit_pct": round(test_agg["net_profit"], 4),
        "test_max_dd":         round(test_agg["max_dd"], 4),
        # Ranking score = test_net_profit_pct (primary) and stable_score (secondary)
        "score":               round(score, 6),
        "stable_score":        round(stable_score, 6),
    }


# ---------------------------------------------------------------------------
# Main optimisation routine
# ---------------------------------------------------------------------------

def run_optimization() -> pd.DataFrame:
    """Run the full walk-forward grid-search optimisation.

    Returns
    -------
    pd.DataFrame
        All results sorted by ``test_net_profit_pct`` descending.
    """
    from data_loader import load_all_pairs

    base_dir     = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, "optimization_results.csv")
    top20_path   = os.path.join(base_dir, "top_20_net_profit.txt")

    print("=" * 70)
    print("SMC Grid-Search Optimizer")
    print(f"  Train : {TRAIN_START} → {TRAIN_END}")
    print(f"  Test  : {TEST_START}  → {TEST_END}")
    print(f"  Fixed transaction cost: {ROUND_TRIP_PIPS} pip round-trip per trade")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading pair data …")
    data_dict = load_all_pairs()
    print(f"      {len(data_dict)} pairs loaded.")

    # ------------------------------------------------------------------
    # 2. Pre-compute features for every pair / period
    # ------------------------------------------------------------------
    print("\n[2/4] Pre-computing M5 features (this may take a few minutes) …")
    t0 = time.time()

    train_precomp: dict[str, dict] = {}
    test_precomp:  dict[str, dict] = {}

    for pair_name, raw_df in data_dict.items():
        agg_map = {c: f for c, f in _OHLCV_AGG.items() if c in raw_df.columns}

        # Train split
        train_raw = raw_df.loc[TRAIN_START:TRAIN_END]
        if len(train_raw) > 0:
            train_m5 = train_raw.resample("5min").agg(agg_map).dropna(subset=["close"])
            if len(train_m5) >= 100:
                print(f"      {pair_name} train ({len(train_m5):,} M5 bars) …", end=" ")
                try:
                    train_precomp[pair_name] = _precompute_pair(train_m5, pair_name)
                    print("✓")
                except Exception as exc:
                    print(f"✗ ({exc})")
                    train_precomp[pair_name] = {}

        # Test split
        test_raw = raw_df.loc[TEST_START:TEST_END]
        if len(test_raw) > 0:
            test_m5 = test_raw.resample("5min").agg(agg_map).dropna(subset=["close"])
            if len(test_m5) >= 100:
                print(f"      {pair_name} test  ({len(test_m5):,} M5 bars) …", end=" ")
                try:
                    test_precomp[pair_name] = _precompute_pair(test_m5, pair_name)
                    print("✓")
                except Exception as exc:
                    print(f"✗ ({exc})")
                    test_precomp[pair_name] = {}

    elapsed = time.time() - t0
    print(f"      Pre-computation done in {elapsed:.1f}s")
    print(f"      Train pairs: {sum(1 for v in train_precomp.values() if v)}"
          f"  Test pairs: {sum(1 for v in test_precomp.values() if v)}")

    # ------------------------------------------------------------------
    # 3. Build parameter combinations
    # ------------------------------------------------------------------
    all_values   = [PARAM_GRID[k] for k in _PARAM_KEYS]
    all_combos   = list(itertools.product(*all_values))
    total_combos = len(all_combos)
    print(f"\n[3/4] Grid has {total_combos:,} parameter combinations.")
    print(f"      Running with {NUM_WORKERS} workers …\n")

    # ------------------------------------------------------------------
    # 4. Parallel grid search
    # ------------------------------------------------------------------
    # Use 'fork' on Unix/macOS for zero-copy data sharing;
    # fall back to the default start method on Windows.
    if sys.platform != "win32":
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = multiprocessing.get_context()
    else:
        ctx = multiprocessing.get_context()

    t1 = time.time()
    with ctx.Pool(
        processes=NUM_WORKERS,
        initializer=_worker_init,
        initargs=(train_precomp, test_precomp),
    ) as pool:
        raw_results = list(
            tqdm(
                pool.imap(_run_combo, all_combos, chunksize=20),
                total=total_combos,
                desc="Optimising",
                unit="combo",
            )
        )

    elapsed_grid = time.time() - t1
    print(f"\n      Grid search finished in {elapsed_grid:.1f}s")

    # ------------------------------------------------------------------
    # 5. Collect and save results
    # ------------------------------------------------------------------
    valid_rows = [r for r in raw_results if r is not None]
    results_df = pd.DataFrame(valid_rows)

    if results_df.empty:
        print("No valid results. Check your data / parameter grid.")
        return results_df

    results_df.sort_values("test_net_profit_pct", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    results_df.to_csv(results_path, index=False)
    print(f"\n[4/4] All {len(results_df):,} results saved to: {results_path}")

    # ------------------------------------------------------------------
    # 6. Top 20 by Net Profit
    # ------------------------------------------------------------------
    top20 = results_df.head(20).copy()
    top20.insert(0, "rank", range(1, len(top20) + 1))

    # Build display-friendly table
    # Note: 'score' == 'test_net_profit_pct' (redundant), so only the latter is shown
    display_cols = [
        "rank",
        "session_start_hour", "session_end_hour",
        "max_spread_pips", "min_rr", "lookback", "risk_percent",
        "test_trades", "test_winrate", "test_profit_factor",
        "test_net_profit_pct", "test_max_dd",
        "stable_score",
    ]
    top20_display = top20[display_cols].to_string(index=False)

    separator = "=" * 80
    top20_text = (
        f"{separator}\n"
        f"TOP 20 PARAMETER COMBINATIONS (sorted by Net Profit)\n"
        f"  Score  =  test_net_profit_pct  (final account balance in %)\n"
        f"  stable_score = test_net_profit_pct / (test_max_dd + 5)\n"
        f"  Period : Out-of-Sample  {TEST_START} → {TEST_END}\n"
        f"  Cost   : {ROUND_TRIP_PIPS} pip round-trip per trade (fixed)\n"
        f"{separator}\n\n"
        f"{top20_display}\n\n"
        f"{separator}\n"
    )

    print("\n" + top20_text)
    with open(top20_path, "w", encoding="utf-8") as fh:
        fh.write(top20_text)
    print(f"Top 20 saved to: {top20_path}")

    return results_df


# ---------------------------------------------------------------------------
# CLI entry-point (also callable from main_optimizer.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()   # Required on Windows
    run_optimization()
