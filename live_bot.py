"""live_bot.py – MetaTrader5 live / paper trading bot for the SMC strategy.

Usage
-----
    python live_bot.py --symbol EURUSD --timeframe M5 [--magic 123456]

Environment variables (optional)
----------------------------------
    MT5_LOGIN       – MT5 account number  (int)
    MT5_PASSWORD    – MT5 account password
    MT5_SERVER      – MT5 broker server name
    TELEGRAM_TOKEN  – Bot token for Telegram notifications
    TELEGRAM_CHAT_ID– Chat / channel ID for Telegram notifications

Safeguards
----------
    * Max 3 open trades at any time
    * Max 3 % daily loss (relative to start-of-day equity)
    * Only trades during London + NY session: 08:00–17:00 London time
    * Spread filter: < 0.6 pips

Rules (same as backtester / smc_strategy)
------------------------------------------
    * Liquidity-Pool TP
    * RR >= 2.0
    * 1 % account risk per trade
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional imports – graceful fallback so the module can be imported/tested
# without the MT5 terminal installed.
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5  # type: ignore
    _MT5_AVAILABLE = True
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore
    _MT5_AVAILABLE = False

try:
    import pytz  # type: ignore
    _LONDON_TZ = pytz.timezone("Europe/London")
except ImportError:  # pragma: no cover
    _LONDON_TZ = None  # type: ignore

from smc_strategy import generate_signals

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SESSION_START_HOUR = 8   # 08:00 London time
_SESSION_END_HOUR = 17    # 17:00 London time (exclusive)
_MAX_SPREAD_PIPS = 0.6
_RISK_PERCENT = 1.0        # % of account balance risked per trade
_MIN_RR = 2.0
_MAX_OPEN_TRADES = 3
_MAX_DAILY_LOSS_PCT = 3.0  # % of start-of-day equity
_POLL_INTERVAL_SECONDS = 60

# Number of M5 bars to fetch for signal computation (must be > lookback used
# inside generate_signals, which defaults to 10; 200 bars gives comfortable
# history while staying fast).
_BARS_NEEDED = 200

# MT5 timeframe map
_TF_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
}

# ---------------------------------------------------------------------------
# Logging setup – one compact line per event
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("live_bot")

# ---------------------------------------------------------------------------
# Optional Telegram helper
# ---------------------------------------------------------------------------
_TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def _send_telegram(message: str) -> None:
    """Send *message* via Telegram if credentials are configured."""
    if not (_TELEGRAM_TOKEN and _TELEGRAM_CHAT_ID):
        return
    try:
        import json as _json
        import urllib.request
        payload = _json.dumps(
            {"chat_id": _TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        ).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                log.warning("Telegram delivery failed: HTTP %s", resp.status)
    except Exception as exc:  # noqa: BLE001
        log.warning("Telegram error: %s", exc)


# ---------------------------------------------------------------------------
# Session filter
# ---------------------------------------------------------------------------

def _in_session(dt_utc: datetime) -> bool:
    """Return True when *dt_utc* falls within 08:00–17:00 London time."""
    if _LONDON_TZ is None:
        # pytz not installed – London is UTC+0 (GMT) or UTC+1 (BST).
        # Without accurate DST data we use UTC+0 as the safe fallback, which
        # means the session window is slightly conservative during BST.  Install
        # pytz for correct behaviour: pip install pytz
        log.warning(
            "pytz not installed; using UTC as approximation for London time."
        )
        london_hour = dt_utc.hour
    else:
        london_dt = dt_utc.astimezone(_LONDON_TZ)
        london_hour = london_dt.hour
    return _SESSION_START_HOUR <= london_hour < _SESSION_END_HOUR


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------

def _mt5_init(login: int | None, password: str | None, server: str | None) -> bool:
    """Initialise the MT5 connection.  Returns True on success."""
    if not _MT5_AVAILABLE:
        log.error("MetaTrader5 package is not installed.")
        return False
    kwargs: dict = {}
    if login:
        kwargs["login"] = login
    if password:
        kwargs["password"] = password
    if server:
        kwargs["server"] = server
    if not mt5.initialize(**kwargs):
        log.error("MT5 initialize() failed: %s", mt5.last_error())
        return False
    info = mt5.account_info()
    if info is None:
        log.error("MT5 account_info() returned None: %s", mt5.last_error())
        return False
    log.info(
        "MT5 connected – account %s  balance %.2f  currency %s",
        info.login,
        info.balance,
        info.currency,
    )
    return True


def _get_equity() -> float:
    """Return current account equity from MT5."""
    info = mt5.account_info()
    return info.equity if info else 0.0


def _get_balance() -> float:
    """Return current account balance from MT5."""
    info = mt5.account_info()
    return info.balance if info else 0.0


def _count_open_trades(magic: int) -> int:
    """Count open positions opened by this bot (identified by *magic*)."""
    positions = mt5.positions_get(magic=magic)
    return len(positions) if positions else 0


def _get_spread_pips(symbol: str) -> float:
    """Return the current spread in pips for *symbol*."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return float("inf")
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return float("inf")
    spread_pts = tick.ask - tick.bid
    # point = smallest price change; 1 pip = 10 * point for 5-digit brokers
    point = sym_info.point
    digits = sym_info.digits
    pip_size = point * 10 if digits in (3, 5) else point
    return spread_pts / pip_size if pip_size > 0 else float("inf")


def _fetch_ohlcv(symbol: str, timeframe_id: int, n_bars: int) -> pd.DataFrame | None:
    """Fetch the last *n_bars* bars for *symbol* and return as OHLCV DataFrame."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe_id, 0, n_bars)
    if rates is None or len(rates) == 0:
        log.warning("No rates returned for %s: %s", symbol, mt5.last_error())
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    return df


def _place_order(
    symbol: str,
    direction: int,
    entry: float,
    sl: float,
    tp: float,
    lot_size: float,
    magic: int,
    comment: str = "smc_bot",
) -> bool:
    """Send a market order to MT5.  Returns True on success."""
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        log.error("Symbol info unavailable for %s", symbol)
        return False

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("Could not retrieve tick for %s", symbol)
        return False
    price = tick.ask if direction == 1 else tick.bid

    # Round lot to broker step
    volume_step = sym_info.volume_step
    lot_size = max(sym_info.volume_min, round(lot_size / volume_step) * volume_step)
    lot_size = min(lot_size, sym_info.volume_max)

    # Use the first filling mode accepted by this symbol (broker-dependent)
    filling_modes = [
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_RETURN,
    ]
    filling = mt5.ORDER_FILLING_IOC  # safe default
    if hasattr(sym_info, "filling_mode"):
        for mode in filling_modes:
            if sym_info.filling_mode & mode:
                filling = mode
                break

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": round(sl, sym_info.digits),
        "tp": round(tp, sym_info.digits),
        "deviation": 20,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        log.error("Order failed for %s  retcode=%s", symbol, retcode)
        return False
    return True


# ---------------------------------------------------------------------------
# Daily loss / equity tracking
# ---------------------------------------------------------------------------

class DailyGuard:
    """Track start-of-day equity and enforce the max daily loss limit."""

    def __init__(self, max_loss_pct: float = _MAX_DAILY_LOSS_PCT) -> None:
        self.max_loss_pct = max_loss_pct
        self._day: int | None = None
        self._start_equity: float = 0.0

    def update(self, equity: float, today: int) -> None:
        if self._day != today:
            self._day = today
            self._start_equity = equity
            log.info("New trading day – start equity: %.2f", equity)
            _send_telegram(f"📊 Daily equity update: <b>{equity:.2f}</b>")

    def is_limit_hit(self, equity: float) -> bool:
        if self._start_equity <= 0:
            return False
        loss_pct = (self._start_equity - equity) / self._start_equity * 100.0
        return loss_pct >= self.max_loss_pct


# ---------------------------------------------------------------------------
# Main bot loop
# ---------------------------------------------------------------------------

def run_bot(
    symbol: str,
    timeframe: str = "M5",
    magic: int = 234567,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
) -> None:
    """Start the live/paper trading loop.

    Parameters
    ----------
    symbol    : MT5 symbol string, e.g. ``"EURUSD"``.
    timeframe : Timeframe string key from ``_TF_MAP`` (default ``"M5"``).
    magic     : Unique magic number used to identify bot orders.
    login     : MT5 account login (int).  Falls back to ``MT5_LOGIN`` env var.
    password  : MT5 password.  Falls back to ``MT5_PASSWORD`` env var.
    server    : MT5 server name.  Falls back to ``MT5_SERVER`` env var.
    """
    if not _MT5_AVAILABLE:
        log.error("MetaTrader5 package is not installed.  Run: pip install MetaTrader5")
        return

    # Resolve credentials from env vars if not passed directly
    if login is None:
        env_login = os.getenv("MT5_LOGIN", "").strip()
        if env_login.isdigit():
            login = int(env_login)
    password = password or os.getenv("MT5_PASSWORD") or None
    server = server or os.getenv("MT5_SERVER") or None

    if not _mt5_init(login, password, server):
        return

    tf_id = _TF_MAP.get(timeframe.upper())
    if tf_id is None:
        log.error("Unknown timeframe '%s'.  Valid: %s", timeframe, list(_TF_MAP))
        mt5.shutdown()
        return

    daily_guard = DailyGuard(max_loss_pct=_MAX_DAILY_LOSS_PCT)
    log.info("Bot started – symbol=%s  tf=%s  magic=%s", symbol, timeframe, magic)
    _send_telegram(f"🤖 SMC Bot started – <b>{symbol}</b> {timeframe}")

    last_bar_time: pd.Timestamp | None = None

    try:
        while True:
            now_utc = datetime.now(timezone.utc)
            today_int = now_utc.date().toordinal()

            equity = _get_equity()
            daily_guard.update(equity, today_int)

            # ----------------------------------------------------------
            # Safeguard 1: daily loss limit
            # ----------------------------------------------------------
            if daily_guard.is_limit_hit(equity):
                log.warning(
                    "Daily loss limit (%.1f%%) reached – no new trades today.",
                    _MAX_DAILY_LOSS_PCT,
                )
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # ----------------------------------------------------------
            # Session filter
            # ----------------------------------------------------------
            if not _in_session(now_utc):
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # ----------------------------------------------------------
            # Spread filter
            # ----------------------------------------------------------
            spread = _get_spread_pips(symbol)
            if spread >= _MAX_SPREAD_PIPS:
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # ----------------------------------------------------------
            # Safeguard 2: max open trades
            # ----------------------------------------------------------
            open_count = _count_open_trades(magic)
            if open_count >= _MAX_OPEN_TRADES:
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # ----------------------------------------------------------
            # Fetch OHLCV + compute SMC signals
            # ----------------------------------------------------------
            df = _fetch_ohlcv(symbol, tf_id, _BARS_NEEDED)
            if df is None or len(df) < 50:
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # Only process a new signal when a new bar has closed
            current_bar_time = df.index[-1]
            if current_bar_time == last_bar_time:
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue
            last_bar_time = current_bar_time

            balance = _get_balance()
            try:
                sig_df = generate_signals(
                    df,
                    risk_percent=_RISK_PERCENT,
                    account_balance=balance,
                    min_rr=_MIN_RR,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("generate_signals error: %s", exc)
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # Evaluate only the last completed bar (index -2; -1 is the live
            # candle still forming)
            last_closed = sig_df.iloc[-2] if len(sig_df) >= 2 else sig_df.iloc[-1]
            signal = int(last_closed["signal"])

            if signal == 0 or np.isnan(last_closed["entry"]):
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            direction_str = "LONG" if signal == 1 else "SHORT"
            entry = float(last_closed["entry"])
            sl = float(last_closed["sl"])
            tp = float(last_closed["tp"])
            rr = float(last_closed["rr_ratio"])
            lot = float(last_closed["lot_size"])

            if np.isnan(sl) or np.isnan(tp) or np.isnan(lot) or lot <= 0:
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # ----------------------------------------------------------
            # Place the order
            # ----------------------------------------------------------
            success = _place_order(symbol, signal, entry, sl, tp, lot, magic)
            if success:
                log.info(
                    "TRADE  %s %s  entry=%.5f  sl=%.5f  tp=%.5f  RR=%.2f  lot=%.2f",
                    direction_str,
                    symbol,
                    entry,
                    sl,
                    tp,
                    rr,
                    lot,
                )
                _send_telegram(
                    f"📈 {direction_str} {symbol}\n"
                    f"Entry: <b>{entry:.5f}</b>  SL: {sl:.5f}  TP: {tp:.5f}  RR: {rr:.2f}"
                )

            time.sleep(_POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
        _send_telegram("🛑 SMC Bot stopped.")
    finally:
        mt5.shutdown()
        log.info("MT5 connection closed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMC live/paper trading bot for MT5")
    p.add_argument("--symbol", default="EURUSD", help="MT5 symbol (default: EURUSD)")
    p.add_argument(
        "--timeframe",
        default="M5",
        choices=list(_TF_MAP.keys()),
        help="Candle timeframe (default: M5)",
    )
    p.add_argument("--magic", type=int, default=234567, help="Magic number for orders")
    p.add_argument("--login", type=int, default=None, help="MT5 account login")
    p.add_argument("--password", default=None, help="MT5 account password")
    p.add_argument("--server", default=None, help="MT5 broker server")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_bot(
        symbol=args.symbol,
        timeframe=args.timeframe,
        magic=args.magic,
        login=args.login,
        password=args.password,
        server=args.server,
    )
