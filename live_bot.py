"""live_bot.py – OANDA live / paper trading bot for the SMC strategy.

.env Setup
----------
Create a file named ``.env`` in the same directory as this script and fill in
the following variables (one per line, no spaces around ``=``):

    ACCOUNT_ID=101-001-12345678-001
    ACCESS_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ENVIRONMENT=practice

    # Optional – Telegram trade notifications
    TELEGRAM_TOKEN=123456789:AAxxxxxx...
    TELEGRAM_CHAT_ID=-1001234567890

    ENVIRONMENT must be either ``practice`` (OANDA fxTrade Practice) or
    ``live`` (real-money fxTrade).  Default is ``practice`` when the variable
    is absent.

    Get your ACCESS_TOKEN and ACCOUNT_ID from
    https://www.oanda.com/demo-account/tpa/personal_token  (practice) or
    https://www.oanda.com/account/tpa/personal_token        (live).

Usage
-----
    python live_bot.py [--symbols EURUSD,GBPUSD,USDJPY] [--timeframe M5]

    # Trade all 10 default major pairs simultaneously:
    python live_bot.py

    # Trade a custom set of symbols:
    python live_bot.py --symbols EURUSD,GBPUSD,USDJPY

Environment variables
---------------------
    ACCOUNT_ID      – OANDA account ID  (required)
    ACCESS_TOKEN    – OANDA API access token  (required)
    ENVIRONMENT     – "practice" (default) or "live"
    TELEGRAM_TOKEN  – Bot token for Telegram notifications  (optional)
    TELEGRAM_CHAT_ID– Chat / channel ID for Telegram notifications  (optional)

Safeguards (all GLOBAL across all symbols)
------------------------------------------
    * Max 3 open trades at any time (across all symbols)
    * Max 3 % daily loss (relative to start-of-day equity)
    * Only trades during London + NY session: 08:00–17:00 London time
    * Spread filter: < 1.8 pips (evaluated per symbol)

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
# Load .env file if present (python-dotenv optional – fallback parser included)
# ---------------------------------------------------------------------------
def _load_dotenv(path: str = ".env") -> None:
    """Parse a simple KEY=VALUE .env file and set missing env vars."""
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    _load_dotenv()

# ---------------------------------------------------------------------------
# Optional imports – graceful fallback so the module can be imported/tested
# without oandapyV20 installed.
# ---------------------------------------------------------------------------
try:
    from oandapyV20 import API as _OandaAPI  # type: ignore
    import oandapyV20.endpoints.accounts as _ep_accounts  # type: ignore
    import oandapyV20.endpoints.instruments as _ep_instruments  # type: ignore
    import oandapyV20.endpoints.orders as _ep_orders  # type: ignore
    import oandapyV20.endpoints.pricing as _ep_pricing  # type: ignore
    import oandapyV20.endpoints.trades as _ep_trades  # type: ignore
    _OANDA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OandaAPI = None  # type: ignore
    _ep_accounts = _ep_instruments = _ep_orders = _ep_pricing = _ep_trades = None
    _OANDA_AVAILABLE = False

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
_MAX_SPREAD_PIPS = 1.8
_RISK_PERCENT = 1.0        # % of account balance risked per trade
_MIN_RR = 2.0
_MAX_OPEN_TRADES = 3
_MAX_DAILY_LOSS_PCT = 3.0  # % of start-of-day equity
_POLL_INTERVAL_SECONDS = 60

# Default list of major pairs traded simultaneously in multi-symbol mode
_DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "NZDUSD", "EURJPY", "GBPJPY", "USDCHF", "EURGBP",
]

# Number of M5 bars to fetch for signal computation (must be > lookback used
# inside generate_signals, which defaults to 10; 200 bars gives comfortable
# history while staying fast).
_BARS_NEEDED = 200

# OANDA granularity map (matches common timeframe strings)
_TF_MAP = {
    "M1": "M1",
    "M5": "M5",
    "M15": "M15",
    "M30": "M30",
    "H1": "H1",
    "H4": "H4",
    "D1": "D",  # OANDA uses "D" for daily; our CLI/TF map uses "D1"
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# All log messages go to live_bot.log only – the terminal stays clean.
# A file handler is attached in run_bot() once the log path is known.
logging.basicConfig(handlers=[logging.NullHandler()])
log = logging.getLogger("live_bot")
log.setLevel(logging.INFO)

# Global file handle used by _journal() for the structured journal blocks
_journal_fh: object | None = None  # TextIO; set to an open file in run_bot()

_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_BOT_DIR, "live_bot.log")


def _setup_logging() -> None:
    """Attach a rotating-capable FileHandler to the module logger."""
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-5s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    log.addHandler(fh)


def _journal(text: str) -> None:
    """Append *text* followed by a newline to the journal file."""
    global _journal_fh  # noqa: PLW0603
    if _journal_fh is not None:
        try:
            _journal_fh.write(text + "\n")
            _journal_fh.flush()
        except OSError:
            pass

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
        # pytz not installed – fall back to UTC (slightly conservative during BST)
        london_hour = dt_utc.hour
    else:
        london_dt = dt_utc.astimezone(_LONDON_TZ)
        london_hour = london_dt.hour
    return _SESSION_START_HOUR <= london_hour < _SESSION_END_HOUR


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------

def _to_oanda_symbol(symbol: str) -> str:
    """Convert ``EURUSD`` → ``EUR_USD`` for the OANDA REST API."""
    s = symbol.upper().replace("_", "")
    if len(s) == 6:
        return f"{s[:3]}_{s[3:]}"
    return s


def _pip_size(symbol: str) -> float:
    """Return the pip size for *symbol* (0.01 for JPY pairs, else 0.0001)."""
    s = symbol.upper().replace("_", "")
    if "JPY" in s:
        return 0.01
    return 0.0001


# ---------------------------------------------------------------------------
# OANDA helpers
# ---------------------------------------------------------------------------

def _oanda_init(account_id: str, access_token: str, environment: str) -> "_OandaAPI | None":
    """Create and verify an OANDA API client.  Returns the client or None."""
    if not _OANDA_AVAILABLE:
        log.error("oandapyV20 package is not installed.  Run: pip install oandapyV20")
        return None
    client = _OandaAPI(access_token=access_token, environment=environment)
    try:
        r = _ep_accounts.AccountSummary(accountID=account_id)
        client.request(r)
        acct = r.response["account"]
        log.info(
            "OANDA connected – account %s  balance %s  currency %s  env=%s",
            acct.get("id", account_id),
            acct.get("balance", "?"),
            acct.get("currency", "?"),
            environment,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("OANDA connection failed: %s", exc)
        return None
    return client


def _get_account_summary(client: "_OandaAPI", account_id: str) -> dict:
    """Return the raw account summary dict from OANDA."""
    try:
        r = _ep_accounts.AccountSummary(accountID=account_id)
        client.request(r)
        return r.response.get("account", {})
    except Exception as exc:  # noqa: BLE001
        log.error("AccountSummary error: %s", exc)
        return {}


def _get_equity(client: "_OandaAPI", account_id: str) -> float:
    """Return current account equity (NAV) from OANDA.

    NAV (Net Asset Value) represents unrealised P&L + balance.  It falls back
    to balance when NAV is absent, and to 0.0 when the account summary cannot
    be retrieved (connection error).
    """
    acct = _get_account_summary(client, account_id)
    return float(acct.get("NAV", acct.get("balance", 0.0)))


def _get_balance(client: "_OandaAPI", account_id: str) -> float:
    """Return current account balance from OANDA."""
    acct = _get_account_summary(client, account_id)
    return float(acct.get("balance", 0.0))


def _count_open_trades(client: "_OandaAPI", account_id: str) -> int:
    """Count all currently open trades on the account."""
    try:
        r = _ep_trades.TradesList(accountID=account_id, params={"state": "OPEN"})
        client.request(r)
        return len(r.response.get("trades", []))
    except Exception as exc:  # noqa: BLE001
        log.error("TradesList error: %s", exc)
        return 0


def _get_spread_pips(client: "_OandaAPI", account_id: str, symbol: str) -> float:
    """Return the current spread in pips for *symbol* via OANDA PricingInfo."""
    oanda_sym = _to_oanda_symbol(symbol)
    try:
        r = _ep_pricing.PricingInfo(
            accountID=account_id,
            params={"instruments": oanda_sym},
        )
        client.request(r)
        prices = r.response.get("prices", [])
        if not prices:
            return float("inf")
        price_data = prices[0]
        bids = price_data.get("bids", [])
        asks = price_data.get("asks", [])
        if not bids or not asks:
            return float("inf")
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
        spread = ask - bid
        pip = _pip_size(symbol)
        return spread / pip if pip > 0 else float("inf")
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] PricingInfo error: %s", symbol, exc)
        return float("inf")


def _fetch_ohlcv(
    client: "_OandaAPI", symbol: str, granularity: str, n_bars: int
) -> "pd.DataFrame | None":
    """Fetch the last *n_bars* completed candles and return as OHLCV DataFrame."""
    oanda_sym = _to_oanda_symbol(symbol)
    try:
        r = _ep_instruments.InstrumentsCandles(
            instrument=oanda_sym,
            params={"count": n_bars, "granularity": granularity, "price": "M"},
        )
        client.request(r)
        candles = r.response.get("candles", [])
        if not candles:
            log.warning("[%s] No candles returned", symbol)
            return None
        rows = []
        for c in candles:
            mid = c.get("mid", {})
            rows.append(
                {
                    "time": c["time"],
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": float(c.get("volume", 0)),
                }
            )
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] InstrumentsCandles error: %s", symbol, exc)
        return None


def _place_order(
    client: "_OandaAPI",
    account_id: str,
    symbol: str,
    direction: int,
    sl: float,
    tp: float,
    units: float,
) -> bool:
    """Send a market order to OANDA with attached SL and TP.

    Parameters
    ----------
    direction : 1 = buy (long), -1 = sell (short)
    units     : Absolute number of units; sign is applied from *direction*.
    """
    oanda_sym = _to_oanda_symbol(symbol)
    pip = _pip_size(symbol)
    # OANDA requires 5 decimal places for most pairs, 3 for JPY pairs
    price_decimals = 3 if pip == 0.01 else 5

    signed_units = int(round(abs(units))) * direction
    if signed_units == 0:
        log.error("[%s] Calculated units = 0; skipping order.", symbol)
        return False

    data = {
        "order": {
            "type": "MARKET",
            "instrument": oanda_sym,
            "units": str(signed_units),
            "stopLossOnFill": {
                "price": f"{sl:.{price_decimals}f}",
                "timeInForce": "GTC",
            },
            "takeProfitOnFill": {
                "price": f"{tp:.{price_decimals}f}",
                "timeInForce": "GTC",
            },
        }
    }

    try:
        r = _ep_orders.Orders(accountID=account_id, data=data)
        client.request(r)
        resp = r.response
        # A rejected order populates 'orderRejectTransaction' instead of 'orderFillTransaction'
        if "orderRejectTransaction" in resp:
            reason = resp["orderRejectTransaction"].get("rejectReason", "unknown")
            log.error("[%s] Order rejected by OANDA: %s", symbol, reason)
            return False
        if "orderFillTransaction" not in resp:
            log.error("[%s] Order response had no fill transaction: %s", symbol, resp)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("[%s] Order placement failed: %s", symbol, exc)
        return False


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
    symbols: list[str] | None = None,
    timeframe: str = "M5",
) -> None:
    """Start the live/paper trading loop for multiple symbols simultaneously.

    Parameters
    ----------
    symbols   : List of symbol strings, e.g. ``["EURUSD", "GBPUSD"]``.
                Defaults to all 10 major pairs when ``None``.
    timeframe : Timeframe string key from ``_TF_MAP`` (default ``"M5"``).
    """
    if not _OANDA_AVAILABLE:
        log.error("oandapyV20 package is not installed.  Run: pip install oandapyV20")
        return

    if not symbols:
        symbols = list(_DEFAULT_SYMBOLS)

    # Resolve OANDA credentials from environment
    account_id = os.getenv("ACCOUNT_ID", "").strip()
    access_token = os.getenv("ACCESS_TOKEN", "").strip()
    environment = os.getenv("ENVIRONMENT", "practice").strip().lower()

    if not account_id or not access_token:
        log.error(
            "ACCOUNT_ID and ACCESS_TOKEN must be set in the environment or .env file."
        )
        return

    if environment not in ("practice", "live"):
        log.error("ENVIRONMENT must be 'practice' or 'live', got: %r", environment)
        return

    # ------------------------------------------------------------------
    # Set up file logging + journal
    # ------------------------------------------------------------------
    global _journal_fh  # noqa: PLW0603
    _setup_logging()

    client = _oanda_init(account_id, access_token, environment)
    if client is None:
        return

    granularity = _TF_MAP.get(timeframe.upper())
    if granularity is None:
        log.error("Unknown timeframe '%s'.  Valid: %s", timeframe, list(_TF_MAP))
        return

    # One-time pytz warning (to log file only)
    if _LONDON_TZ is None:
        log.warning(
            "pytz not installed; using UTC as approximation for London time.  "
            "Install with: pip install pytz"
        )

    daily_guard = DailyGuard(max_loss_pct=_MAX_DAILY_LOSS_PCT)
    log.info(
        "Bot started – symbols=%s  tf=%s  env=%s",
        ",".join(symbols),
        timeframe,
        environment,
    )
    _send_telegram(
        f"🤖 SMC Bot started – <b>{','.join(symbols)}</b> {timeframe}"
    )

    print(
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}] "
        f"Bot started – {len(symbols)} symbols – tf={timeframe} – env={environment}"
    )

    # Per-symbol tracking of the last processed bar time
    last_bar_times: dict[str, "pd.Timestamp | None"] = {sym: None for sym in symbols}

    # Open journal file inside the try/finally so it is always closed on exit
    try:
        _journal_fh = open(_LOG_FILE, "a", encoding="utf-8")  # noqa: SIM115
        while True:
            now_utc = datetime.now(timezone.utc)
            today_int = now_utc.date().toordinal()
            cycle_ts = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

            equity = _get_equity(client, account_id)
            daily_guard.update(equity, today_int)

            # ----------------------------------------------------------
            # Safeguard 1: daily loss limit (global)
            # ----------------------------------------------------------
            if daily_guard.is_limit_hit(equity):
                log.warning(
                    "Daily loss limit (%.1f%%) reached – no new trades today.",
                    _MAX_DAILY_LOSS_PCT,
                )
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # ----------------------------------------------------------
            # Session filter (global)
            # ----------------------------------------------------------
            if not _in_session(now_utc):
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # ----------------------------------------------------------
            # Safeguard 2: max open trades (global, across all symbols)
            # ----------------------------------------------------------
            open_count = _count_open_trades(client, account_id)
            if open_count >= _MAX_OPEN_TRADES:
                log.info(
                    "Max open trades (%d) reached – skipping cycle.", open_count
                )
                print(
                    f"[{now_utc.strftime('%Y-%m-%d %H:%M')}] "
                    f"Checked {len(symbols)} symbols – "
                    f"0 new signals – "
                    f"{open_count} open trade{'s' if open_count != 1 else ''} – "
                    f"equity {equity:.2f}"
                )
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # ----------------------------------------------------------
            # Per-cycle journal header
            # ----------------------------------------------------------
            _journal("=" * 80)
            _journal(f"[{cycle_ts}] CYCLE")
            _journal("-" * 80)
            _journal(f"{'Symbol':<12} {'Spread(pips)':>13}  {'New Bar':>7}  Signal")
            _journal("-" * 80)

            new_signals = 0
            cycle_rows: list[str] = []
            # Each item is a list of lines for the detailed trade block
            trade_blocks: list[list[str]] = []

            # ----------------------------------------------------------
            # Iterate over every symbol in one loop pass
            # ----------------------------------------------------------
            for symbol in symbols:
                # Re-check global trade cap inside the loop so we don't
                # open more trades than allowed if earlier symbols filled it.
                open_count = _count_open_trades(client, account_id)
                if open_count >= _MAX_OPEN_TRADES:
                    cycle_rows.append(
                        f"{symbol:<12} {'--':>13}  {'--':>7}  max open trades reached"
                    )
                    break

                # Spread filter (per symbol)
                spread = _get_spread_pips(client, account_id, symbol)
                spread_str = f"{spread:.3f}" if spread != float("inf") else "inf"
                if spread >= _MAX_SPREAD_PIPS:
                    cycle_rows.append(
                        f"{symbol:<12} {spread_str:>13}  {'--':>7}  spread too wide – skipped"
                    )
                    continue

                # Fetch OHLCV + compute SMC signals
                df = _fetch_ohlcv(client, symbol, granularity, _BARS_NEEDED)
                if df is None or len(df) < 50:
                    cycle_rows.append(
                        f"{symbol:<12} {spread_str:>13}  {'--':>7}  no data"
                    )
                    continue

                # Only process a new signal when a new bar has closed
                current_bar_time = df.index[-1]
                new_bar = current_bar_time != last_bar_times[symbol]
                new_bar_str = "YES" if new_bar else "NO"
                if not new_bar:
                    cycle_rows.append(
                        f"{symbol:<12} {spread_str:>13}  {new_bar_str:>7}  (waiting for new bar)"
                    )
                    continue
                last_bar_times[symbol] = current_bar_time

                # Fetch fresh balance before sizing each trade so that a
                # position placed for an earlier symbol is reflected here.
                balance = _get_balance(client, account_id)

                try:
                    sig_df = generate_signals(
                        df,
                        risk_percent=_RISK_PERCENT,
                        account_balance=balance,
                        min_rr=_MIN_RR,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error("[%s] generate_signals error: %s", symbol, exc)
                    cycle_rows.append(
                        f"{symbol:<12} {spread_str:>13}  {new_bar_str:>7}  generate_signals error"
                    )
                    continue

                # Evaluate only the last completed bar (index -2; -1 is the
                # live candle still forming)
                last_closed = sig_df.iloc[-2] if len(sig_df) >= 2 else sig_df.iloc[-1]
                signal = int(last_closed["signal"])

                if signal == 0 or np.isnan(last_closed["entry"]):
                    cycle_rows.append(
                        f"{symbol:<12} {spread_str:>13}  {new_bar_str:>7}  no signal"
                    )
                    continue

                direction_str = "LONG" if signal == 1 else "SHORT"
                trade_emoji = "📈" if signal == 1 else "📉"
                entry = float(last_closed["entry"])
                sl = float(last_closed["sl"])
                tp = float(last_closed["tp"])
                rr = float(last_closed["rr_ratio"])
                lot = float(last_closed["lot_size"])

                if np.isnan(sl) or np.isnan(tp) or np.isnan(lot) or lot <= 0:
                    cycle_rows.append(
                        f"{symbol:<12} {spread_str:>13}  {new_bar_str:>7}  invalid signal values"
                    )
                    continue

                cycle_rows.append(
                    f"{symbol:<12} {spread_str:>13}  {new_bar_str:>7}  "
                    f"{direction_str} entry={entry:.5f} sl={sl:.5f} tp={tp:.5f}"
                )

                # Place the order
                success = _place_order(
                    client, account_id, symbol, signal, sl, tp, lot
                )
                if success:
                    new_signals += 1
                    equity_after = _get_equity(client, account_id)

                    # ── Terminal: bold trade alert ──────────────────────
                    print(
                        f"\033[1m[{symbol}] TRADE {direction_str} @ {entry:.5f}"
                        f"  sl={sl:.5f}  tp={tp:.5f}  RR={rr:.2f}  lot={lot:.2f}\033[0m"
                    )

                    # Collect trade block lines (written to journal after cycle table)
                    trade_blocks.append([
                        "",
                        "─" * 80,
                        f"[TRADE] [{cycle_ts}] {symbol} {direction_str}",
                        f"  Entry    : {entry:.5f}",
                        f"  Stop Loss: {sl:.5f}",
                        f"  Take Profit: {tp:.5f}",
                        f"  RR Ratio : {rr:.2f}",
                        f"  Lot Size : {lot:.2f}",
                        "  Reason   : FVG + BOS + Liquidity + Discount Zone",
                        f"  Equity   : {equity:.2f} → {equity_after:.2f}",
                        "─" * 80,
                        "",
                    ])

                    log.info(
                        "[%s] TRADE %s @ %.5f  sl=%.5f  tp=%.5f  RR=%.2f  lot=%.2f",
                        symbol,
                        direction_str,
                        entry,
                        sl,
                        tp,
                        rr,
                        lot,
                    )
                    _send_telegram(
                        f"[{symbol}] {trade_emoji} {direction_str}\n"
                        f"Entry: <b>{entry:.5f}</b>  SL: {sl:.5f}  TP: {tp:.5f}  RR: {rr:.2f}"
                    )

            # ── Write cycle table rows to journal ────────────────────────
            for row in cycle_rows:
                _journal(row)
            _journal("-" * 80)

            open_count = _count_open_trades(client, account_id)
            _journal(
                f"Summary: {len(symbols)} symbols | {new_signals} new signal(s) | "
                f"{open_count} open trade(s) | equity={equity:.2f}"
            )
            _journal("=" * 80)

            # ── Write detailed trade blocks below the cycle summary ──────
            for block in trade_blocks:
                for line in block:
                    _journal(line)

            _journal("")

            # ── Terminal: one clean summary line per cycle ───────────────
            print(
                f"[{now_utc.strftime('%Y-%m-%d %H:%M')}] "
                f"Checked {len(symbols)} symbols – "
                f"{new_signals} new signal{'s' if new_signals != 1 else ''} – "
                f"{open_count} open trade{'s' if open_count != 1 else ''} – "
                f"equity {equity:.2f}"
            )

            time.sleep(_POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
        _send_telegram("🛑 SMC Bot stopped.")
        print(
            f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}] "
            "Bot stopped by user."
        )
    finally:
        if _journal_fh is not None:
            try:
                _journal_fh.close()
            except OSError as exc:
                log.warning("Error closing journal file: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMC live/paper trading bot for OANDA")
    p.add_argument(
        "--symbols",
        default=None,
        help=(
            "Comma-separated list of symbols to trade simultaneously "
            "(default: all 10 major pairs: "
            + ",".join(_DEFAULT_SYMBOLS)
            + ")"
        ),
    )
    p.add_argument(
        "--timeframe",
        default="M5",
        choices=list(_TF_MAP.keys()),
        help="Candle timeframe (default: M5)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    symbols: list[str] | None = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run_bot(
        symbols=symbols,
        timeframe=args.timeframe,
    )
