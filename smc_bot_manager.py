"""smc_bot_manager.py – Manager for 20 SMC bots on a single OANDA Demo account.

Overview
--------
20 bots are derived from the **Top-5 optimised parameter sets**, each expanded
into **4 filter variants** (Sehr_Locker / Locker / Original / Streng).
Every bot is identified by a unique **Magic Number** (1001–1020) embedded in
the OANDA order's ``clientExtensions.id``.  All 20 bots run in parallel via
:mod:`multiprocessing`.

A single **DataManager** process (from ``data_manager.py``) preloads history
once on start-up (500 bars each for M5, H4, D1) and then streams live prices
via the OANDA v20 Pricing Stream.  Whenever a bar closes the DataManager
pushes ``(symbol, tf, dataframe)`` tuples to each bot's dedicated
:class:`multiprocessing.Queue`.  Bots drain their queues every cycle instead
of making their own REST calls for candle data.

Bot variants (per Top-5 set)
-----------------------------
* **Sehr_Locker** – min_rr=1.5, lookback=50, daily_bias=off,  liq_pool=off
* **Locker**      – min_rr=1.8, lookback=40, daily_bias=weak, liq_pool=off
* **Original**    – exact optimised values,  daily_bias=normal,liq_pool=on
* **Streng**      – min_rr=2.5, lookback=20, daily_bias=very_strong,liq_pool=on

Bot names & IDs
---------------
    Bot_01_Sehr_Locker (magic=1001) … Bot_01_Streng (magic=1004)
    Bot_02_Sehr_Locker (magic=1005) … Bot_05_Streng (magic=1020)

Symbols traded by every bot
----------------------------
    EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD,
    NZDUSD, EURJPY, GBPJPY, USDCHF, EURGBP

Folder structure (auto-created on start)
-----------------------------------------
* ``logs/manager.log``           central Manager + DataManager log
* ``logs/bot_01.log`` …          one log file per bot
* ``trades/trades_bot_01.csv`` … one CSV per bot

Telegram format (on every filled trade)
----------------------------------------
    Bot_02_Original (EURUSD) LONG @ 1.0854 | RR 3.0 | Magic 1007
    PnL: +12.34 USD | Daily PnL: +34.56 USD

Usage
-----
    # Start all 20 bots:
    python smc_bot_manager.py

    # Start only a subset of bots by id:
    python smc_bot_manager.py --bot-ids 1,3,7

    # List available bot configurations and exit:
    python smc_bot_manager.py --list-bots

.env Setup
----------
Create a ``.env`` file in the same directory with:

    ACCOUNT_ID=101-001-12345678-001
    ACCESS_TOKEN=xxxxxxxx-xxxx...
    ENVIRONMENT=practice          # practice or live (default: practice)

    # Optional – Telegram trade notifications
    TELEGRAM_TOKEN=123456789:AAxxxxxx...
    TELEGRAM_CHAT_ID=-1001234567890

Ubuntu / tmux Quick-Start  (single screen session)
----------------------------------------------------
    # 1. Install dependencies (once):
    pip install oandapyV20 requests pandas numpy smartmoneyconcepts pytz python-dotenv

    # 2. Create / edit your .env file with credentials.

    # 3. Start a detached tmux session so the bots survive SSH disconnect:
    tmux new-session -d -s smc 'python smc_bot_manager.py'

    # 4. Attach to watch live output:
    tmux attach -t smc

    # 5. Detach without stopping (keep bots running):
    Ctrl-b  d

    # 6. Stop all bots:
    tmux send-keys -t smc C-c
    # or kill the session entirely:
    tmux kill-session -t smc
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load .env file (python-dotenv optional – built-in fallback parser)
# ---------------------------------------------------------------------------

def _load_dotenv(path: str = ".env") -> None:
    """Parse a simple KEY=VALUE .env file and populate missing env vars."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).parent / path
    if not resolved.is_file():
        return
    with resolved.open(encoding="utf-8") as fh:
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
    from dotenv import load_dotenv as _dotenv_load  # type: ignore
    _dotenv_load(dotenv_path=Path(__file__).parent / ".env")
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
from data_manager import run_data_manager

# ---------------------------------------------------------------------------
# Global constants (mirrors live_bot.py defaults)
# ---------------------------------------------------------------------------
_SESSION_START_HOUR = 8    # 08:00 London time
_SESSION_END_HOUR = 17     # 17:00 London time (exclusive)
_MAX_SPREAD_PIPS = 3.0
_RISK_PERCENT = 1.0        # % of account balance risked per trade
_MIN_RR = 2.0
_LOOKBACK = 30             # Rolling-window size (bars) for signal detection
_MAX_DAILY_LOSS_PCT = 3.0  # % of start-of-day equity
_MAX_OPEN_TRADES = 3       # Max open trades per bot (counted from CSV)
_POLL_INTERVAL_SECONDS = 60
_BARS_NEEDED = 200

_MAJOR_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "NZDUSD", "EURJPY", "GBPJPY", "USDCHF", "EURGBP",
]

# OANDA granularity map
_TF_MAP = {
    "M1": "M1", "M5": "M5", "M15": "M15",
    "M30": "M30", "H1": "H1", "H4": "H4",
    "D1": "D",
}

# Base directory – same folder as this script
_BOT_DIR = Path(__file__).parent

# Sub-directories for logs and trade CSVs (auto-created on start)
_LOGS_DIR   = _BOT_DIR / "logs"
_TRADES_DIR = _BOT_DIR / "trades"

# ---------------------------------------------------------------------------
# Top-5 optimised parameter sets (base configurations)
# ---------------------------------------------------------------------------

_TOP5_PARAMS = [
    # These are the exact Top-5 parameter sets from the optimisation run.
    # Some share the same min_rr/lookback values but are treated as independent
    # seeds so that each of the 4 variants (Sehr_Locker … Streng) produces a
    # distinctly-named and independently-tracked bot.
    {"set_id": 1, "min_rr": 2.5, "lookback": 30},
    {"set_id": 2, "min_rr": 3.0, "lookback": 30},
    {"set_id": 3, "min_rr": 2.5, "lookback": 30},
    {"set_id": 4, "min_rr": 3.0, "lookback": 30},
    {"set_id": 5, "min_rr": 2.5, "lookback": 30},
]

# ---------------------------------------------------------------------------
# 4 filter variants applied to every Top-5 set
# ---------------------------------------------------------------------------
# daily_bias_filter values:
#   "off"         – skip BOS-direction check entirely
#   "weak"        – require any BOS in lookback window
#   "normal"      – require BOS aligned with trade direction
#   "very_strong" – require BOS + FVG + price-position alignment + liq sweep
# liquidity_pool_filter: True = require nearby pool for TP, False = skip

_VARIANT_DEFS = [
    {
        "variant": "Sehr_Locker",
        "min_rr": 1.5, "lookback": 50,
        "daily_bias_filter": "off",
        "liquidity_pool_filter": False,
    },
    {
        "variant": "Locker",
        "min_rr": 1.8, "lookback": 40,
        "daily_bias_filter": "weak",
        "liquidity_pool_filter": False,
    },
    {
        "variant": "Original",
        "min_rr": None, "lookback": None,   # inherit from Top-5 set
        "daily_bias_filter": "normal",
        "liquidity_pool_filter": True,
    },
    {
        "variant": "Streng",
        "min_rr": 2.5, "lookback": 20,
        "daily_bias_filter": "very_strong",
        "liquidity_pool_filter": True,
    },
]

# ---------------------------------------------------------------------------
# Build 20 bot configs (5 sets × 4 variants)
# Bot IDs: 1-4 → Set-1 variants, 5-8 → Set-2, …, 17-20 → Set-5
# Magic numbers: 1001–1020
# ---------------------------------------------------------------------------

BOT_CONFIGS: list[dict] = []
_bot_counter = 1

for _top_set in _TOP5_PARAMS:
    for _var in _VARIANT_DEFS:
        _min_rr = (
            _top_set["min_rr"] if _var["min_rr"] is None else _var["min_rr"]
        )
        _lookback = (
            _top_set["lookback"] if _var["lookback"] is None else _var["lookback"]
        )
        _bot_name = f"Bot_{_top_set['set_id']:02d}_{_var['variant']}"
        BOT_CONFIGS.append(
            {
                "bot_id":               _bot_counter,
                "bot_name":             _bot_name,
                "magic":                1000 + _bot_counter,
                "symbols":              list(_MAJOR_PAIRS),
                "timeframe":            "M5",
                "session_start":        7,
                "session_end":          19,
                "max_spread":           3.0,
                "min_rr":               _min_rr,
                "lookback":             _lookback,
                "risk":                 0.5,
                "daily_bias_filter":    _var["daily_bias_filter"],
                "liquidity_pool_filter": _var["liquidity_pool_filter"],
            }
        )
        _bot_counter += 1


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_TRADE_CSV_COLUMNS = [
    "timestamp",
    "bot_id",
    "magic",
    "symbol",
    "timeframe",
    "direction",
    "entry",
    "sl",
    "tp",
    "rr_ratio",
    "lot_size",
    "oanda_trade_id",
    "status",           # OPEN | CLOSED | CANCELLED
    "realized_pnl",
    "close_time",
]


def _csv_path(bot_id: int) -> Path:
    return _TRADES_DIR / f"trades_bot_{bot_id:02d}.csv"


def _ensure_csv(bot_id: int) -> None:
    """Create the trades CSV (and its parent directory) if not present yet."""
    path = _csv_path(bot_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_TRADE_CSV_COLUMNS)
            writer.writeheader()


def _append_trade(bot_id: int, row: dict) -> None:
    """Append a trade row to this bot's CSV file."""
    path = _csv_path(bot_id)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TRADE_CSV_COLUMNS)
        writer.writerow(row)


def _update_trade_pnl(
    bot_id: int,
    oanda_trade_id: str,
    realized_pnl: float,
    close_time: str,
) -> None:
    """Update the realized_pnl + close_time for a specific OANDA trade ID."""
    path = _csv_path(bot_id)
    if not path.exists():
        return
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r["oanda_trade_id"] == oanda_trade_id and r["status"] == "OPEN":
                r["realized_pnl"] = f"{realized_pnl:.2f}"
                r["close_time"] = close_time
                r["status"] = "CLOSED"
            rows.append(r)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TRADE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _get_open_trade_ids(bot_id: int) -> list[str]:
    """Return OANDA trade IDs that still have status=OPEN in the CSV."""
    path = _csv_path(bot_id)
    if not path.exists():
        return []
    ids: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("status") == "OPEN" and r.get("oanda_trade_id"):
                ids.append(r["oanda_trade_id"])
    return ids


def _get_daily_pnl(bot_id: int) -> float:
    """Sum of realized_pnl for all trades closed today for this bot."""
    path = _csv_path(bot_id)
    if not path.exists():
        return 0.0
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total = 0.0
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("status") == "CLOSED" and r.get("close_time", "").startswith(today_str):
                try:
                    total += float(r["realized_pnl"])
                except (ValueError, KeyError):
                    pass
    return total


def _get_total_pnl(bot_id: int) -> float:
    """Sum of all realized_pnl entries for this bot."""
    path = _csv_path(bot_id)
    if not path.exists():
        return 0.0
    total = 0.0
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("status") == "CLOSED":
                try:
                    total += float(r["realized_pnl"])
                except (ValueError, KeyError):
                    pass
    return total


# ---------------------------------------------------------------------------
# Logging setup (per-bot) and central manager log
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    """Create logs/ and trades/ directories if they do not exist yet."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _TRADES_DIR.mkdir(parents=True, exist_ok=True)


def _setup_manager_logging() -> None:
    """Configure the root logger to write to logs/manager.log + stdout."""
    _ensure_dirs()
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return  # already set up
    root.setLevel(logging.INFO)
    log_path = _LOGS_DIR / "manager.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(name)s  %(levelname)-5s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(name)s  %(levelname)-5s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(fh)
    root.addHandler(ch)


def _make_logger(bot_id: int, bot_name: str = "") -> logging.Logger:
    """Return a Logger that writes to logs/bot_XX.log."""
    label = bot_name or f"Bot {bot_id:02d}"
    name  = f"smc_bot_{bot_id:02d}"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (e.g. after fork on macOS)
    logger.setLevel(logging.DEBUG)
    _ensure_dirs()
    log_path = _LOGS_DIR / f"bot_{bot_id:02d}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            f"%(asctime)s  [{label}]  %(levelname)-5s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Telegram helper
# ---------------------------------------------------------------------------

def _send_telegram(message: str, token: str, chat_id: str, logger: logging.Logger) -> None:
    """Send *message* via Telegram if credentials are configured."""
    if not (token and chat_id):
        return
    try:
        import json as _json
        import urllib.request

        payload = _json.dumps(
            {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        ).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                logger.warning("Telegram delivery failed: HTTP %s", resp.status)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram error: %s", exc)


# ---------------------------------------------------------------------------
# Session filter (shared with live_bot)
# ---------------------------------------------------------------------------

def _in_session(
    dt_utc: datetime,
    start_hour: int = _SESSION_START_HOUR,
    end_hour: int = _SESSION_END_HOUR,
) -> bool:
    """Return True when *dt_utc* falls within *start_hour*–*end_hour* London time."""
    if _LONDON_TZ is None:
        london_hour = dt_utc.hour
    else:
        london_dt = dt_utc.astimezone(_LONDON_TZ)
        london_hour = london_dt.hour
    return start_hour <= london_hour < end_hour


# ---------------------------------------------------------------------------
# Symbol helpers (shared with live_bot)
# ---------------------------------------------------------------------------

def _to_oanda_symbol(symbol: str) -> str:
    s = symbol.upper().replace("_", "")
    if len(s) == 6:
        return f"{s[:3]}_{s[3:]}"
    return s


def _pip_size(symbol: str) -> float:
    s = symbol.upper().replace("_", "")
    return 0.01 if "JPY" in s else 0.0001


# ---------------------------------------------------------------------------
# OANDA API helpers
# ---------------------------------------------------------------------------

def _oanda_init(
    account_id: str,
    access_token: str,
    environment: str,
    logger: logging.Logger,
) -> "_OandaAPI | None":
    if not _OANDA_AVAILABLE:
        logger.error("oandapyV20 not installed.  Run: pip install oandapyV20")
        return None
    client = _OandaAPI(access_token=access_token, environment=environment)
    try:
        r = _ep_accounts.AccountSummary(accountID=account_id)
        client.request(r)
        acct = r.response["account"]
        logger.info(
            "OANDA connected – account=%s  balance=%s  currency=%s  env=%s",
            acct.get("id", account_id),
            acct.get("balance", "?"),
            acct.get("currency", "?"),
            environment,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("OANDA connection failed: %s", exc)
        return None
    return client


def _get_account_summary(
    client: "_OandaAPI", account_id: str, logger: logging.Logger
) -> dict:
    try:
        r = _ep_accounts.AccountSummary(accountID=account_id)
        client.request(r)
        return r.response.get("account", {})
    except Exception as exc:  # noqa: BLE001
        logger.error("AccountSummary error: %s", exc)
        return {}


def _get_equity(client: "_OandaAPI", account_id: str, logger: logging.Logger) -> float:
    acct = _get_account_summary(client, account_id, logger)
    return float(acct.get("NAV", acct.get("balance", 0.0)))


def _get_balance(client: "_OandaAPI", account_id: str, logger: logging.Logger) -> float:
    acct = _get_account_summary(client, account_id, logger)
    return float(acct.get("balance", 0.0))


def _get_spread_pips(
    client: "_OandaAPI",
    account_id: str,
    symbol: str,
    logger: logging.Logger,
) -> float:
    oanda_sym = _to_oanda_symbol(symbol)
    try:
        r = _ep_pricing.PricingInfo(
            accountID=account_id, params={"instruments": oanda_sym}
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
        logger.warning("PricingInfo error: %s", exc)
        return float("inf")


def _fetch_ohlcv(
    client: "_OandaAPI",
    symbol: str,
    granularity: str,
    n_bars: int,
    logger: logging.Logger,
) -> "pd.DataFrame | None":
    oanda_sym = _to_oanda_symbol(symbol)
    try:
        r = _ep_instruments.InstrumentsCandles(
            instrument=oanda_sym,
            params={"count": n_bars, "granularity": granularity, "price": "M"},
        )
        client.request(r)
        candles = r.response.get("candles", [])
        if not candles:
            logger.warning("No candles returned for %s", symbol)
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
        logger.warning("InstrumentsCandles error: %s", exc)
        return None


def _place_order(
    client: "_OandaAPI",
    account_id: str,
    symbol: str,
    direction: int,
    sl: float,
    tp: float,
    units: float,
    magic: int,
    logger: logging.Logger,
) -> "str | None":
    """Send a market order to OANDA with attached SL, TP, and clientExtensions.

    Returns the OANDA trade ID on success, ``None`` on failure.
    """
    oanda_sym = _to_oanda_symbol(symbol)
    pip = _pip_size(symbol)
    price_decimals = 3 if pip == 0.01 else 5

    signed_units = int(round(abs(units))) * direction
    if signed_units == 0:
        logger.error("Calculated units = 0 for %s; skipping order.", symbol)
        return None

    # clientExtensions.id encodes the magic number so trades can be filtered
    # per bot.  The id must be 1–128 alphanumeric / hyphen / underscore chars.
    client_ext_id = f"magic_{magic}"

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
            "tradeClientExtensions": {
                "id": client_ext_id,
                "tag": f"bot_{magic}",
                "comment": f"SMC Bot magic={magic}",
            },
        }
    }

    try:
        r = _ep_orders.Orders(accountID=account_id, data=data)
        client.request(r)
        resp = r.response
        if "orderRejectTransaction" in resp:
            reason = resp["orderRejectTransaction"].get("rejectReason", "unknown")
            logger.error("Order rejected by OANDA: %s", reason)
            return None
        if "orderFillTransaction" not in resp:
            logger.error("Order response had no fill transaction: %s", resp)
            return None
        trade_id = resp["orderFillTransaction"].get("tradeOpened", {}).get("tradeID")
        if not trade_id:
            # Fallback: try the first tradeID from relatedTransactionIDs
            related = resp.get("relatedTransactionIDs", [])
            trade_id = str(related[0]) if related else ""
        return str(trade_id) if trade_id else None
    except Exception as exc:  # noqa: BLE001
        logger.error("Order placement failed: %s", exc)
        return None


def _check_closed_trades(
    client: "_OandaAPI",
    account_id: str,
    bot_id: int,
    logger: logging.Logger,
) -> None:
    """Check open trades recorded in the CSV and update PnL for those closed."""
    open_ids = _get_open_trade_ids(bot_id)
    if not open_ids:
        return
    for trade_id in open_ids:
        try:
            r = _ep_trades.TradeDetails(accountID=account_id, tradeID=trade_id)
            client.request(r)
            trade_data = r.response.get("trade", {})
            state = trade_data.get("state", "OPEN")
            if state in ("CLOSED", "CLOSE_WHEN_TRADABLE"):
                realized_pnl = float(trade_data.get("realizedPL", 0.0))
                close_time = trade_data.get("closeTime", datetime.now(timezone.utc).isoformat())
                _update_trade_pnl(bot_id, trade_id, realized_pnl, close_time)
                logger.info(
                    "Trade %s CLOSED  realizedPL=%.2f", trade_id, realized_pnl
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not check trade %s: %s", trade_id, exc)


# ---------------------------------------------------------------------------
# Daily loss guard (per-bot equity tracking)
# ---------------------------------------------------------------------------

class DailyGuard:
    """Track start-of-day equity and enforce the max daily loss limit."""

    def __init__(self, max_loss_pct: float = _MAX_DAILY_LOSS_PCT) -> None:
        self.max_loss_pct = max_loss_pct
        self._day: int | None = None
        self._start_equity: float = 0.0

    def update(self, equity: float, today_int: int) -> None:
        if self._day != today_int:
            self._day = today_int
            self._start_equity = equity

    def is_limit_hit(self, equity: float) -> bool:
        if self._start_equity <= 0:
            return False
        loss_pct = (self._start_equity - equity) / self._start_equity * 100.0
        return loss_pct >= self.max_loss_pct


# ---------------------------------------------------------------------------
# Single-bot worker (runs in its own process)
# ---------------------------------------------------------------------------

def run_single_bot(config: dict) -> None:
    """Entry point for one bot process.

    Parameters
    ----------
    config : dict
        Keys: ``bot_id``, ``bot_name``, ``magic``, ``symbols``, ``timeframe``,
        ``session_start``, ``session_end``, ``max_spread``,
        ``min_rr``, ``lookback``, ``risk``,
        ``daily_bias_filter``, ``liquidity_pool_filter``,
        ``data_queue``.
        ``data_queue`` is a :class:`multiprocessing.Queue` fed by the
        DataManager process.  Each queue item is a
        ``(symbol, tf, dataframe)`` tuple pushed on every bar close.
        The bot caches the latest DataFrame per symbol and uses it
        instead of making its own REST calls for candle data.
    """
    bot_id: int = config["bot_id"]
    bot_name: str = config.get("bot_name", f"Bot {bot_id:02d}")
    magic: int = config["magic"]
    symbols: list[str] = config["symbols"]
    timeframe: str = config["timeframe"]
    session_start: int = config.get("session_start", _SESSION_START_HOUR)
    session_end: int = config.get("session_end", _SESSION_END_HOUR)
    max_spread: float = config.get("max_spread", _MAX_SPREAD_PIPS)
    min_rr: float = config.get("min_rr", _MIN_RR)
    lookback: int = config.get("lookback", _LOOKBACK)
    risk_percent: float = config.get("risk", _RISK_PERCENT)
    daily_bias_filter: str = config.get("daily_bias_filter", "normal")
    liq_pool_filter: bool = config.get("liquidity_pool_filter", True)
    data_queue = config.get("data_queue")
    bot_label = bot_name

    # Set up per-bot logger
    logger = _make_logger(bot_id, bot_name)

    # Each child process must re-read env vars (needed after os.fork on Unix)
    try:
        from dotenv import load_dotenv as _dl  # type: ignore
        _dl(dotenv_path=Path(__file__).parent / ".env", override=False)
    except ImportError:
        _load_dotenv()

    account_id = os.getenv("ACCOUNT_ID", "").strip()
    access_token = os.getenv("ACCESS_TOKEN", "").strip()
    environment = os.getenv("ENVIRONMENT", "practice").strip().lower()
    tg_token = os.getenv("TELEGRAM_TOKEN", "").strip()
    tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    def telegram(msg: str) -> None:
        _send_telegram(msg, tg_token, tg_chat_id, logger)

    if not account_id or not access_token:
        logger.error(
            "ACCOUNT_ID and ACCESS_TOKEN must be set in the .env file."
        )
        return

    if environment not in ("practice", "live"):
        logger.error(
            "Invalid ENVIRONMENT value in .env: %r (must be 'practice' or 'live')",
            environment,
        )
        return

    # Ensure CSV exists
    _ensure_csv(bot_id)

    client = _oanda_init(account_id, access_token, environment, logger)
    if client is None:
        return

    daily_guard = DailyGuard()

    # Local DataFrame cache populated from data_queue: {symbol: DataFrame}
    local_dfs: dict[str, "pd.DataFrame | None"] = {sym: None for sym in symbols}

    # Per-symbol tracking of the last processed bar time (new-bar guard)
    last_bar_times: dict[str, "pd.Timestamp | None"] = {sym: None for sym in symbols}

    logger.info(
        "%s started – symbols=%s  tf=%s  magic=%d  env=%s  "
        "session=%02d-%02dh  max_spread=%.1f  min_rr=%.1f  lookback=%d  risk=%.1f%%",
        bot_label, ",".join(symbols), timeframe, magic, environment,
        session_start, session_end, max_spread, min_rr, lookback, risk_percent,
    )
    telegram(
        f"🤖 {bot_label} started – {len(symbols)} pairs – tf={timeframe} – magic={magic} – "
        f"session={session_start:02d}-{session_end:02d}h – spread≤{max_spread} – RR≥{min_rr}"
    )

    try:
        while True:
            now_utc = datetime.now(timezone.utc)
            today_int = now_utc.date().toordinal()

            # ── Drain the DataManager queue into our local cache ─────────
            if data_queue is not None:
                while True:
                    try:
                        sym_key, tf_key, df_update = data_queue.get_nowait()
                        if sym_key in local_dfs and tf_key == timeframe:
                            local_dfs[sym_key] = df_update
                    except Exception:  # noqa: BLE001
                        break  # queue empty

            # Single AccountSummary call per cycle – provides both equity and balance.
            # NAV (Net Asset Value) is used as equity; balance is the cash component.
            # If the API call fails, _get_account_summary returns {} and both
            # values default to 0.0 (the error is already logged inside that helper).
            _acct = _get_account_summary(client, account_id, logger)
            balance = float(_acct.get("balance", 0.0))
            equity = float(_acct.get("NAV", balance))  # NAV primary, balance as fallback
            daily_guard.update(equity, today_int)

            # ── Update PnL for any trades that closed since last cycle ───
            _check_closed_trades(client, account_id, bot_id, logger)

            # ── Daily loss guard ─────────────────────────────────────────
            if daily_guard.is_limit_hit(equity):
                logger.warning(
                    "Daily loss limit (%.1f%%) reached – pausing until next day.",
                    _MAX_DAILY_LOSS_PCT,
                )
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # ── Max open trades guard (per bot, counted from CSV) ────────
            open_count = len(_get_open_trade_ids(bot_id))
            if open_count >= _MAX_OPEN_TRADES:
                logger.info(
                    "Max open trades (%d) reached – skipping cycle.", open_count
                )
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # ── Iterate over every symbol in one loop pass ───────────────
            for symbol in symbols:
                # Re-check open trades cap before each symbol so we don't
                # exceed the limit if a previous symbol in this cycle filled it.
                if open_count >= _MAX_OPEN_TRADES:
                    logger.info(
                        "[%s] Max open trades reached mid-cycle – stopping symbol loop.",
                        symbol,
                    )
                    break

                # Use DataFrame from local cache (populated by DataManager queue)
                df = local_dfs.get(symbol)
                if df is None or len(df) < 50:
                    logger.debug("[%s] Awaiting DataManager data.", symbol)
                    continue

                # New-bar guard (per symbol)
                current_bar_time = df.index[-1]
                if current_bar_time == last_bar_times[symbol]:
                    continue
                last_bar_times[symbol] = current_bar_time

                # ── Generate signals (needed to detect potential FVG/BOS) ─
                try:
                    sig_df = generate_signals(
                        df,
                        risk_percent=risk_percent,
                        account_balance=balance,
                        lookback=lookback,
                        min_rr=min_rr,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("[%s] generate_signals error: %s", symbol, exc)
                    continue

                # Evaluate only the last *closed* bar (index -2;
                # index -1 is the live candle still forming)
                last_closed = sig_df.iloc[-2] if len(sig_df) >= 2 else sig_df.iloc[-1]
                signal = int(last_closed["signal"])

                # ── Check for recent FVG/BOS patterns in lookback window ──
                lb_start = max(0, len(sig_df) - lookback - 1)
                lb_slice = sig_df.iloc[lb_start:-1]
                has_bull_fvg = bool(lb_slice["fvg_bullish"].max())
                has_bear_fvg = bool(lb_slice["fvg_bearish"].max())
                has_bull_bos = bool((lb_slice["bos_direction"] == 1).any())
                has_bear_bos = bool((lb_slice["bos_direction"] == -1).any())
                has_potential = has_bull_fvg or has_bear_fvg or has_bull_bos or has_bear_bos

                if not has_potential:
                    logger.info("[%s] No signal on latest closed M5 bar", symbol)
                    continue

                # ── Potential FVG/BOS setup detected ─────────────────────
                logger.info("Potential FVG/BOS detected on %s M5", symbol)

                # ── Filter: Trading session ───────────────────────────────
                if not _in_session(now_utc, session_start, session_end):
                    logger.info("[%s] Filtered: Not in trading session", symbol)
                    continue

                # ── Skip if no actionable signal on last closed bar ───────
                # (checked before the spread API call to avoid a redundant
                #  _get_spread_pips request when the signal is already 0)
                if signal == 0 or np.isnan(last_closed.get("entry", np.nan)):
                    continue

                # ── Filter: Spread (API call only when signal is actionable) ─
                spread = _get_spread_pips(client, account_id, symbol, logger)
                if spread >= max_spread:
                    logger.info(
                        "[%s] Filtered: Spread %.1f > %.1f", symbol, spread, max_spread
                    )
                    continue

                # ── daily_bias_filter ─────────────────────────────────────
                # "off"         → always pass
                # "weak"        → any BOS in lookback window
                # "normal"      → BOS aligned with trade direction
                # "very_strong" → BOS + FVG + price-position + liq sweep
                if daily_bias_filter != "off":
                    if daily_bias_filter in ("weak", "normal", "very_strong"):
                        if not (has_bull_bos or has_bear_bos):
                            logger.info(
                                "[%s] Filtered (daily_bias=%s): No BOS detected",
                                symbol, daily_bias_filter,
                            )
                            continue

                    if daily_bias_filter in ("normal", "very_strong"):
                        if signal == 1 and not has_bull_bos:
                            logger.info(
                                "[%s] Filtered (daily_bias=%s): No bullish BOS for LONG",
                                symbol, daily_bias_filter,
                            )
                            continue
                        if signal == -1 and not has_bear_bos:
                            logger.info(
                                "[%s] Filtered (daily_bias=%s): No bearish BOS for SHORT",
                                symbol, daily_bias_filter,
                            )
                            continue

                    if daily_bias_filter == "very_strong":
                        discount_50_raw = last_closed.get("discount_50", np.nan)
                        discount_50_f = (
                            float(discount_50_raw)
                            if not pd.isna(discount_50_raw) else np.nan
                        )
                        close_price = float(last_closed["close"])
                        in_discount = (
                            close_price < discount_50_f
                            if not np.isnan(discount_50_f) else False
                        )
                        in_premium = (
                            close_price > discount_50_f
                            if not np.isnan(discount_50_f) else False
                        )
                        liq_below = bool(lb_slice["liquidity_swept_below"].max())
                        liq_above = bool(lb_slice["liquidity_swept_above"].max())

                        meets_bull_criteria = (
                            has_bull_fvg and has_bull_bos and liq_below and in_discount
                        )
                        meets_bear_criteria = (
                            has_bear_fvg and has_bear_bos and liq_above and in_premium
                        )

                        if signal == 1 and not meets_bull_criteria:
                            logger.info(
                                "[%s] Filtered (daily_bias=very_strong): "
                                "Incomplete LONG alignment "
                                "(bull_fvg=%s bull_bos=%s liq_below=%s discount=%s)",
                                symbol, has_bull_fvg, has_bull_bos, liq_below, in_discount,
                            )
                            continue
                        if signal == -1 and not meets_bear_criteria:
                            logger.info(
                                "[%s] Filtered (daily_bias=very_strong): "
                                "Incomplete SHORT alignment "
                                "(bear_fvg=%s bear_bos=%s liq_above=%s premium=%s)",
                                symbol, has_bear_fvg, has_bear_bos, liq_above, in_premium,
                            )
                            continue

                # ── liquidity_pool_filter ─────────────────────────────────
                if liq_pool_filter:
                    pool_above_raw = last_closed.get("liquidity_pool_above", np.nan)
                    pool_below_raw = last_closed.get("liquidity_pool_below", np.nan)
                    pool_above_f = (
                        float(pool_above_raw)
                        if not pd.isna(pool_above_raw) else np.nan
                    )
                    pool_below_f = (
                        float(pool_below_raw)
                        if not pd.isna(pool_below_raw) else np.nan
                    )
                    if signal == 1 and np.isnan(pool_above_f):
                        logger.info(
                            "[%s] Filtered (liq_pool=on): No liquidity pool above for LONG",
                            symbol,
                        )
                        continue
                    if signal == -1 and np.isnan(pool_below_f):
                        logger.info(
                            "[%s] Filtered (liq_pool=on): No liquidity pool below for SHORT",
                            symbol,
                        )
                        continue

                direction_str = "LONG" if signal == 1 else "SHORT"
                entry = float(last_closed["entry"])
                sl = float(last_closed["sl"])
                tp = float(last_closed["tp"])
                rr = float(last_closed["rr_ratio"])
                lot = float(last_closed["lot_size"])

                if np.isnan(sl) or np.isnan(tp) or np.isnan(lot) or lot <= 0:
                    logger.warning("[%s] Invalid signal values; skipping.", symbol)
                    continue

                logger.info(
                    "[%s] Signal valid! Placing order... Magic %d", symbol, magic
                )
                logger.info(
                    "[%s] Signal: %s @ %.5f  sl=%.5f  tp=%.5f  RR=%.2f  lot=%.2f",
                    symbol, direction_str, entry, sl, tp, rr, lot,
                )

                # ── Place the order ──────────────────────────────────────
                oanda_trade_id = _place_order(
                    client, account_id, symbol, signal, sl, tp, lot, magic, logger
                )

                if oanda_trade_id is None:
                    continue

                # ── Record in CSV ────────────────────────────────────────
                ts_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")
                _append_trade(
                    bot_id,
                    {
                        "timestamp": ts_str,
                        "bot_id": bot_id,
                        "magic": magic,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "direction": direction_str,
                        "entry": f"{entry:.5f}",
                        "sl": f"{sl:.5f}",
                        "tp": f"{tp:.5f}",
                        "rr_ratio": f"{rr:.2f}",
                        "lot_size": f"{lot:.2f}",
                        "oanda_trade_id": oanda_trade_id,
                        "status": "OPEN",
                        "realized_pnl": "",
                        "close_time": "",
                    },
                )

                # Increment open-count after successful fill
                open_count += 1

                # ── Compute PnL figures for Telegram ─────────────────────
                total_pnl = _get_total_pnl(bot_id)
                daily_pnl = _get_daily_pnl(bot_id)
                pnl_sign = "+" if total_pnl >= 0 else ""
                daily_sign = "+" if daily_pnl >= 0 else ""

                # ── Telegram notification (bot number + symbol) ──────────
                tg_msg = (
                    f"<b>{bot_label} ({symbol}) {direction_str} @ {entry:.5f}</b>\n"
                    f"RR {rr:.1f} | Magic {magic}\n"
                    f"SL: {sl:.5f} | TP: {tp:.5f}\n"
                    f"PnL: {pnl_sign}{total_pnl:.2f} USD | "
                    f"Daily PnL: {daily_sign}{daily_pnl:.2f} USD"
                )
                telegram(tg_msg)

                logger.info(
                    "TRADE PLACED – [%s] %s @ %.5f | RR %.2f | Magic %d | "
                    "trade_id=%s | PnL %.2f | Daily PnL %.2f",
                    symbol, direction_str, entry, rr, magic,
                    oanda_trade_id, total_pnl, daily_pnl,
                )

            time.sleep(_POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("%s stopped by user.", bot_label)
        telegram(f"🛑 {bot_label} stopped.")


# ---------------------------------------------------------------------------
# Manager entry point
# ---------------------------------------------------------------------------

def main(bot_ids: list[int] | None = None) -> None:
    """Start the DataManager and all (or a subset of) bot processes.

    Parameters
    ----------
    bot_ids : list[int] | None
        If given, only start bots whose ``bot_id`` is in this list.
        Defaults to all 20 bots.
    """
    # ── Create folder structure and set up central manager log ──────────────
    _ensure_dirs()
    _setup_manager_logging()
    manager_log = logging.getLogger("manager")

    # Resolve credentials so they can be passed to the DataManager process
    try:
        from dotenv import load_dotenv as _dl  # type: ignore
        _dl(dotenv_path=Path(__file__).parent / ".env")
    except ImportError:
        _load_dotenv()

    account_id = os.getenv("ACCOUNT_ID", "").strip()
    access_token = os.getenv("ACCESS_TOKEN", "").strip()
    environment = os.getenv("ENVIRONMENT", "practice").strip().lower()

    if not account_id or not access_token:
        manager_log.error(
            "ACCOUNT_ID and ACCESS_TOKEN must be set in the .env file."
        )
        print("ERROR: ACCOUNT_ID and ACCESS_TOKEN must be set in the .env file.")
        return

    configs = BOT_CONFIGS
    if bot_ids:
        configs = [c for c in BOT_CONFIGS if c["bot_id"] in bot_ids]
    if not configs:
        manager_log.error("No matching bot configurations found.")
        print("No matching bot configurations found.")
        return

    manager_log.info(
        "Starting DataManager + %d bot(s). Logs → %s, Trades → %s",
        len(configs), _LOGS_DIR, _TRADES_DIR,
    )
    print(f"Starting DataManager + {len(configs)} bot(s) …")
    print(f"  Logs   → {_LOGS_DIR}")
    print(f"  Trades → {_TRADES_DIR}")
    for cfg in configs:
        _bname = cfg.get("bot_name") or f"Bot {cfg['bot_id']:02d}"
        line = (
            f"  {_bname:<28}  "
            f"magic={cfg['magic']}  "
            f"RR≥{cfg.get('min_rr', _MIN_RR):.1f}  "
            f"lookback={cfg.get('lookback', _LOOKBACK)}  "
            f"bias={cfg.get('daily_bias_filter', 'normal'):<12}  "
            f"liq_pool={'on' if cfg.get('liquidity_pool_filter', True) else 'off'}"
        )
        print(line)
        manager_log.info(line.strip())

    # ── Create one queue per bot ─────────────────────────────────────────────
    # Each queue receives (symbol, tf, df) tuples from the DataManager whenever
    # a new bar closes.  maxsize prevents unbounded memory growth if a bot lags.
    bot_queues = [
        multiprocessing.Queue(maxsize=2000) for _ in configs
    ]

    # ── Start the DataManager process ────────────────────────────────────────
    logs_dir_str = str(_LOGS_DIR)
    dm_process = multiprocessing.Process(
        target=run_data_manager,
        args=(account_id, access_token, list(_MAJOR_PAIRS), environment, bot_queues,
              logs_dir_str),
        name="DataManager",
        daemon=False,
    )
    dm_process.start()
    manager_log.info("DataManager process started (preloading history …)")
    print("DataManager process started (preloading history …)")

    # ── Start each bot process ───────────────────────────────────────────────
    processes: list[multiprocessing.Process] = [dm_process]
    for cfg, q in zip(configs, bot_queues):
        cfg_with_queue = {**cfg, "data_queue": q}
        p = multiprocessing.Process(
            target=run_single_bot,
            args=(cfg_with_queue,),
            name=cfg.get("bot_name", f"SMCBot-{cfg['bot_id']:02d}"),
            daemon=False,
        )
        p.start()
        processes.append(p)
        # Small stagger to avoid all bots hammering the OANDA account API
        time.sleep(0.5)

    msg = f"{len(processes) - 1} bot process(es) started.  Press Ctrl+C to stop all."
    manager_log.info(msg)
    print(msg)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nShutting down DataManager and all bots …")
        manager_log.info("KeyboardInterrupt – shutting down all processes.")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        print("All processes stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SMC Bot Manager – run 20 SMC bots on a single OANDA Demo account."
    )
    parser.add_argument(
        "--list-bots",
        action="store_true",
        help="Print all bot configurations and exit.",
    )
    parser.add_argument(
        "--bot-ids",
        default=None,
        help=(
            "Comma-separated list of bot IDs to start (1–20).  "
            "Starts all 20 when omitted.  Example: --bot-ids 1,3,7"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Required on Windows / macOS with spawn start method
    multiprocessing.freeze_support()

    args = _parse_args()

    if args.list_bots:
        print(
            f"{'ID':>4}  {'Name':<28}  {'Magic':>6}  {'MinRR':>6}  "
            f"{'Lookback':>8}  {'DailyBias':<12}  {'LiqPool':>7}  "
            f"{'Risk%':>6}  TF"
        )
        print("-" * 100)
        for cfg in BOT_CONFIGS:
            _bname = cfg.get("bot_name") or f"Bot {cfg['bot_id']:02d}"
            print(
                f"{cfg['bot_id']:>4}  "
                f"{_bname:<28}  "
                f"{cfg['magic']:>6}  "
                f"{cfg.get('min_rr', _MIN_RR):>6.1f}  "
                f"{cfg.get('lookback', _LOOKBACK):>8}  "
                f"{cfg.get('daily_bias_filter', 'normal'):<12}  "
                f"{'on' if cfg.get('liquidity_pool_filter', True) else 'off':>7}  "
                f"{cfg.get('risk', _RISK_PERCENT):>6.1f}  "
                f"{cfg['timeframe']}"
            )
    else:
        selected_ids: list[int] | None = None
        if args.bot_ids:
            selected_ids = [
                int(x.strip()) for x in args.bot_ids.split(",") if x.strip().isdigit()
            ]
        main(bot_ids=selected_ids)
