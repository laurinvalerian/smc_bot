import pandas as pd
import numpy as np
from smartmoneyconcepts import smc


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Smart Money Concepts features and attach them as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume`` and a ``DatetimeIndex``.

    Returns
    -------
    pd.DataFrame
        Extended copy of *df* with the following additional columns:

        * ``swing_hl``               – 1 = swing high, -1 = swing low
        * ``swing_level``            – price level of the swing point
        * ``fvg_bullish``            – 1 where a bullish Fair-Value-Gap exists
        * ``fvg_bearish``            – 1 where a bearish Fair-Value-Gap exists
        * ``fvg_top`` / ``fvg_bottom`` – FVG price bounds
        * ``bos_direction``          – BOS/CHOCH direction (1 / -1 / 0)
        * ``bos_level``              – price level of the break
        * ``ob_direction``           – 1 = bullish OB, -1 = bearish OB, 0 = none
        * ``ob_top`` / ``ob_bottom`` – Order-Block price bounds
        * ``ob_active``              – 1 when OB is not yet mitigated
        * ``ob_level``               – mid-point of the *nearest* active OB
        * ``liquidity_pool_above``   – nearest unswept liquidity level above close
        * ``liquidity_pool_below``   – nearest unswept liquidity level below close
        * ``liquidity_swept``        – 1 on candles where any liquidity is swept
        * ``liquidity_swept_below``  – 1 when a *below* pool is swept (stop-hunt low)
        * ``liquidity_swept_above``  – 1 when an *above* pool is swept (stop-hunt high)
        * ``swing_high_level``       – most-recent swing-high price (forward-filled)
        * ``swing_low_level``        – most-recent swing-low price (forward-filled)
        * ``discount_50``            – 50 % level between last swing high and low
    """
    result = df.copy()
    n = len(df)
    close_arr = df["close"].values

    # ------------------------------------------------------------------
    # 1. Swing Highs / Lows
    # NOTE: smc functions return a RangeIndex DataFrame regardless of the
    # input index.  Always use .values when assigning back to *result*
    # (which preserves the original DatetimeIndex) to prevent pandas from
    # silently filling every row with NaN due to index misalignment.
    # ------------------------------------------------------------------
    swing = smc.swing_highs_lows(df, swing_length=10)
    result["swing_hl"] = swing["HighLow"].values
    result["swing_level"] = swing["Level"].values

    # Forward-fill most-recent swing high / low level for discount calculation
    swing_high_series = pd.Series(
        swing["Level"].where(swing["HighLow"] == 1).ffill().values, index=df.index
    )
    swing_low_series = pd.Series(
        swing["Level"].where(swing["HighLow"] == -1).ffill().values, index=df.index
    )
    result["swing_high_level"] = swing_high_series
    result["swing_low_level"] = swing_low_series
    result["discount_50"] = (swing_high_series + swing_low_series) / 2

    # ------------------------------------------------------------------
    # 2. Fair Value Gaps
    # ------------------------------------------------------------------
    fvg = smc.fvg(df)
    result["fvg_top"] = fvg["Top"].values
    result["fvg_bottom"] = fvg["Bottom"].values
    result["fvg_bullish"] = (fvg["FVG"].values == 1).astype(np.int8)
    result["fvg_bearish"] = (fvg["FVG"].values == -1).astype(np.int8)

    # ------------------------------------------------------------------
    # 3. Break of Structure / Change of Character
    # ------------------------------------------------------------------
    bos = smc.bos_choch(df, swing)
    # Combine BOS and CHOCH – non-zero values mark the direction
    bos_dir = (
        pd.Series(bos["BOS"].values).fillna(0)
        + pd.Series(bos["CHOCH"].values).fillna(0)
    )
    result["bos_direction"] = bos_dir.clip(-1, 1).astype(np.int8).values
    result["bos_level"] = bos["Level"].values

    # ------------------------------------------------------------------
    # 4. Order Blocks  (MitigatedIndex == 0 → NOT yet mitigated → active)
    # ------------------------------------------------------------------
    ob = smc.ob(df, swing)
    result["ob_direction"] = pd.Series(ob["OB"].values).fillna(0).astype(np.int8).values
    result["ob_top"] = ob["Top"].values
    result["ob_bottom"] = ob["Bottom"].values
    result["ob_active"] = (
        (ob["OB"].notna().values) & (ob["MitigatedIndex"].values == 0)
    ).astype(np.int8)

    ob_mask = ob["OB"].notna().values
    ob_positions = np.where(ob_mask)[0]
    ob_tops_k = ob["Top"].values[ob_mask]
    ob_bottoms_k = ob["Bottom"].values[ob_mask]
    ob_mitigated_k = ob["MitigatedIndex"].values[ob_mask]

    ob_level_arr = np.full(n, np.nan, dtype=np.float64)
    if len(ob_positions) > 0:
        pos_mat = ob_positions[:, np.newaxis]           # (k, 1)
        i_range = np.arange(n)[np.newaxis, :]           # (1, n)
        active_ob = (pos_mat <= i_range) & (ob_mitigated_k[:, np.newaxis] == 0)
        midpoints = ((ob_tops_k + ob_bottoms_k) / 2)[:, np.newaxis]
        dist = np.abs(midpoints - close_arr[np.newaxis, :])
        dist_masked = np.where(active_ob, dist, np.inf)
        nearest_idx = np.argmin(dist_masked, axis=0)
        has_ob = active_ob.any(axis=0)
        ob_level_arr = np.where(
            has_ob,
            (ob_tops_k[nearest_idx] + ob_bottoms_k[nearest_idx]) / 2,
            np.nan,
        )
    result["ob_level"] = ob_level_arr

    # ------------------------------------------------------------------
    # 5. Liquidity Pools & Sweeps  (Liquidity 1 = above, -1 = below)
    # ------------------------------------------------------------------
    liq = smc.liquidity(df, swing)

    liq_mask = liq["Liquidity"].notna().values
    liq_positions = np.where(liq_mask)[0]

    pool_above_arr = np.full(n, np.nan, dtype=np.float64)
    pool_below_arr = np.full(n, np.nan, dtype=np.float64)
    liq_swept_arr = np.zeros(n, dtype=np.int8)
    liq_swept_below_arr = np.zeros(n, dtype=np.int8)
    liq_swept_above_arr = np.zeros(n, dtype=np.int8)

    if len(liq_positions) > 0:
        liq_levels_k = liq["Level"].values[liq_mask].astype(np.float64)
        liq_types_k = liq["Liquidity"].values[liq_mask].astype(np.float64)
        liq_swept_k = liq["Swept"].values[liq_mask].astype(np.float64)

        pos_mat = liq_positions[:, np.newaxis]              # (k, 1)
        i_range = np.arange(n, dtype=np.float64)[np.newaxis, :]  # (1, n)

        created_before = pos_mat <= i_range                 # (k, n)
        not_swept = (
            np.isnan(liq_swept_k)[:, np.newaxis]
            | (liq_swept_k[:, np.newaxis] > i_range)
        )
        active = created_before & not_swept                 # (k, n)

        levels_mat = liq_levels_k[:, np.newaxis]            # (k, 1)
        close_mat = close_arr[np.newaxis, :]                # (1, n)

        above = (levels_mat > close_mat) & active           # (k, n)
        below = (levels_mat < close_mat) & active           # (k, n)

        pool_above_arr = np.where(
            above.any(axis=0),
            np.min(np.where(above, levels_mat, np.inf), axis=0),
            np.nan,
        )
        pool_below_arr = np.where(
            below.any(axis=0),
            np.max(np.where(below, levels_mat, -np.inf), axis=0),
            np.nan,
        )

        # Mark candles where liquidity was swept
        swept_valid_mask = ~np.isnan(liq_swept_k)
        if swept_valid_mask.any():
            swept_indices = liq_swept_k[swept_valid_mask].astype(int)
            swept_types = liq_types_k[swept_valid_mask]
            valid = swept_indices < n
            for idx, ltype in zip(swept_indices[valid], swept_types[valid]):
                liq_swept_arr[idx] = 1
                if ltype < 0:
                    liq_swept_below_arr[idx] = 1
                else:
                    liq_swept_above_arr[idx] = 1

    result["liquidity_pool_above"] = pool_above_arr
    result["liquidity_pool_below"] = pool_below_arr
    result["liquidity_swept"] = liq_swept_arr
    result["liquidity_swept_below"] = liq_swept_below_arr
    result["liquidity_swept_above"] = liq_swept_above_arr

    return result


# ---------------------------------------------------------------------------

def calculate_rr(entry_price: float, sl: float, tp: float) -> float:
    """Return the Risk-to-Reward ratio for a trade.

    Parameters
    ----------
    entry_price : float
    sl          : float  Stop-Loss level
    tp          : float  Take-Profit level

    Returns
    -------
    float
        abs(tp - entry) / abs(entry - sl), or 0.0 if sl == entry.
    """
    if sl == entry_price:
        return 0.0
    return abs(tp - entry_price) / abs(entry_price - sl)


# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    risk_percent: float = 1.0,
    account_balance: float = 10_000.0,
    lookback: int = 10,
    min_rr: float = 2.0,
) -> pd.DataFrame:
    """Generate SMC-based trading signals.

    Conditions
    ----------
    Long  : Bullish FVG + Bullish BOS/CHOCH + Liquidity sweep below
            + price in discount (< 50 % of swing range)
    Short : Bearish FVG + Bearish BOS/CHOCH + Liquidity sweep above
            + price in premium  (> 50 % of swing range)

    SL    : Bottom of nearest active bullish OB (Long) or top of nearest
            active bearish OB (Short).  Falls back to the most-recent
            swing low / high when no matching OB is available.
    TP    : Nearest unswept liquidity pool above (Long) / below (Short).

    Parameters
    ----------
    df             : pd.DataFrame
        Raw OHLCV DataFrame (see ``get_smc_features`` for format).
    risk_percent   : float
        Percentage of *account_balance* to risk per trade (default 1 %).
    account_balance: float
        Nominal account size in account currency (default 10 000).
    lookback       : int
        Rolling-window size (in bars) used to detect *recent* signals.
    min_rr         : float
        Minimum Risk:Reward ratio required to emit a signal (default 2.0).

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:
        ``signal``   – 1 = Long, -1 = Short, 0 = no trade
        ``entry``    – entry price (close of signal bar)
        ``sl``       – stop-loss level
        ``tp``       – take-profit level
        ``rr_ratio`` – reward / risk
        ``lot_size`` – position size (units at risk / SL distance)
    """
    feat = get_smc_features(df)
    n = len(feat)
    close_arr = feat["close"].values

    # ------------------------------------------------------------------
    # Pre-compute rolling "was condition true in last <lookback> bars?"
    # ------------------------------------------------------------------
    roll = lookback

    recent_bull_fvg = feat["fvg_bullish"].rolling(roll, min_periods=1).max().astype(bool)
    recent_bear_fvg = feat["fvg_bearish"].rolling(roll, min_periods=1).max().astype(bool)
    recent_bull_bos = (
        (feat["bos_direction"] == 1).astype(int).rolling(roll, min_periods=1).max().astype(bool)
    )
    recent_bear_bos = (
        (feat["bos_direction"] == -1).astype(int).rolling(roll, min_periods=1).max().astype(bool)
    )
    recent_liq_sweep_below = (
        feat["liquidity_swept_below"].rolling(roll, min_periods=1).max().astype(bool)
    )
    recent_liq_sweep_above = (
        feat["liquidity_swept_above"].rolling(roll, min_periods=1).max().astype(bool)
    )

    discount_50 = feat["discount_50"].values
    in_discount = close_arr < discount_50
    in_premium = close_arr > discount_50

    liq_pool_above = feat["liquidity_pool_above"].values
    liq_pool_below = feat["liquidity_pool_below"].values
    swing_low_level = feat["swing_low_level"].values
    swing_high_level = feat["swing_high_level"].values

    # ------------------------------------------------------------------
    # Determine per-candle SL levels from Order Blocks
    # ------------------------------------------------------------------
    ob_mask = feat["ob_active"].values.astype(bool) & feat["ob_direction"].values.astype(bool)
    ob_bull_mask = feat["ob_active"].values.astype(bool) & (feat["ob_direction"].values == 1)
    ob_bear_mask = feat["ob_active"].values.astype(bool) & (feat["ob_direction"].values == -1)

    ob_bottom_arr = feat["ob_bottom"].values
    ob_top_arr = feat["ob_top"].values
    ob_positions = np.where(feat["ob_active"].values)[0]

    # For each candle: nearest active bullish OB bottom below price → Long SL
    # For each candle: nearest active bearish OB top above price → Short SL
    sl_long_arr = np.full(n, np.nan, dtype=np.float64)
    sl_short_arr = np.full(n, np.nan, dtype=np.float64)

    if len(ob_positions) > 0:
        ob_dirs_all = feat["ob_direction"].values[ob_positions]
        ob_mit_all = feat["ob_active"].values[ob_positions]
        ob_bots_all = ob_bottom_arr[ob_positions]
        ob_tops_all = ob_top_arr[ob_positions]

        pos_mat = ob_positions[:, np.newaxis]               # (k, 1)
        i_range = np.arange(n)[np.newaxis, :]               # (1, n)
        created_before = pos_mat <= i_range                 # (k, n)
        active_all = created_before & (ob_mit_all[:, np.newaxis] == 1)  # ob_active=1

        # --- Long SL: nearest bullish OB whose bottom is below close ---
        is_bull = (ob_dirs_all == 1)[:, np.newaxis]
        bull_active = active_all & is_bull                  # (k, n)
        bots_mat = ob_bots_all[:, np.newaxis]               # (k, 1)
        below_price = bots_mat < close_arr[np.newaxis, :]   # (k, n)
        bull_below = bull_active & below_price              # (k, n)
        if bull_below.any():
            # Highest bottom among those below price (closest SL to entry)
            bots_candidate = np.where(bull_below, bots_mat, -np.inf)
            has_cand = bull_below.any(axis=0)
            sl_long_arr = np.where(has_cand, np.max(bots_candidate, axis=0), np.nan)

        # --- Short SL: nearest bearish OB whose top is above close ---
        is_bear = (ob_dirs_all == -1)[:, np.newaxis]
        bear_active = active_all & is_bear                  # (k, n)
        tops_mat = ob_tops_all[:, np.newaxis]               # (k, 1)
        above_price = tops_mat > close_arr[np.newaxis, :]   # (k, n)
        bear_above = bear_active & above_price              # (k, n)
        if bear_above.any():
            tops_candidate = np.where(bear_above, tops_mat, np.inf)
            has_cand = bear_above.any(axis=0)
            sl_short_arr = np.where(has_cand, np.min(tops_candidate, axis=0), np.nan)

    # Fall back to recent swing low (Long) / swing high (Short) when no OB
    sl_long_arr = np.where(np.isnan(sl_long_arr), swing_low_level, sl_long_arr)
    sl_short_arr = np.where(np.isnan(sl_short_arr), swing_high_level, sl_short_arr)

    # ------------------------------------------------------------------
    # Build signal arrays
    # ------------------------------------------------------------------
    signal_arr = np.zeros(n, dtype=np.int8)
    entry_arr = np.full(n, np.nan, dtype=np.float64)
    sl_arr = np.full(n, np.nan, dtype=np.float64)
    tp_arr = np.full(n, np.nan, dtype=np.float64)
    rr_arr = np.full(n, np.nan, dtype=np.float64)
    lot_arr = np.full(n, np.nan, dtype=np.float64)

    risk_amount = account_balance * risk_percent / 100.0

    long_cond = (
        recent_bull_fvg.values
        & recent_bull_bos.values
        & recent_liq_sweep_below.values
        & in_discount
    )
    short_cond = (
        recent_bear_fvg.values
        & recent_bear_bos.values
        & recent_liq_sweep_above.values
        & in_premium
    )

    for i in range(n):
        entry = close_arr[i]

        if long_cond[i]:
            sl = sl_long_arr[i]
            tp = liq_pool_above[i]
            if not (np.isnan(sl) or np.isnan(tp) or sl >= entry or tp <= entry):
                rr = calculate_rr(entry, sl, tp)
                if rr >= min_rr:
                    sl_dist = abs(entry - sl)
                    signal_arr[i] = 1
                    entry_arr[i] = entry
                    sl_arr[i] = sl
                    tp_arr[i] = tp
                    rr_arr[i] = rr
                    lot_arr[i] = risk_amount / sl_dist if sl_dist > 0 else np.nan

        elif short_cond[i]:
            sl = sl_short_arr[i]
            tp = liq_pool_below[i]
            if not (np.isnan(sl) or np.isnan(tp) or sl <= entry or tp >= entry):
                rr = calculate_rr(entry, sl, tp)
                if rr >= min_rr:
                    sl_dist = abs(entry - sl)
                    signal_arr[i] = -1
                    entry_arr[i] = entry
                    sl_arr[i] = sl
                    tp_arr[i] = tp
                    rr_arr[i] = rr
                    lot_arr[i] = risk_amount / sl_dist if sl_dist > 0 else np.nan

    feat["signal"] = signal_arr
    feat["entry"] = entry_arr
    feat["sl"] = sl_arr
    feat["tp"] = tp_arr
    feat["rr_ratio"] = rr_arr
    feat["lot_size"] = lot_arr

    return feat


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("Generating dummy OHLCV data …")
    # seed=1 reliably produces setups that satisfy R:R >= 2.0 on 2 000 bars
    np.random.seed(1)
    n_bars = 2_000
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
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

    # Optionally load real data when a CSV path is provided as first argument
    if len(sys.argv) > 1:
        try:
            from data_loader import load_histdata_csv

            dummy_df = load_histdata_csv(sys.argv[1])
            print(f"Loaded {len(dummy_df)} bars from {sys.argv[1]}")
        except Exception as exc:
            print(f"Could not load {sys.argv[1]}: {exc} – using dummy data.")

    print(f"Running generate_signals on {len(dummy_df)} bars …")
    result_df = generate_signals(dummy_df, risk_percent=1.0, account_balance=10_000.0, lookback=20)

    signals = result_df[result_df["signal"] != 0][
        ["signal", "entry", "sl", "tp", "rr_ratio", "lot_size"]
    ]
    total = len(signals)
    longs = (signals["signal"] == 1).sum()
    shorts = (signals["signal"] == -1).sum()
    print(f"\nTotal signals: {total}  (Long: {longs}, Short: {shorts})")
    print("\nFirst 5 signals:")
    print(signals.head(5).to_string())
