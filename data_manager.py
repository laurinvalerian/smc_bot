"""data_manager.py – Central data manager for the SMC bot cluster.

Overview
--------
A single ``DataManager`` instance preloads historical OHLCV data once on
start-up and then maintains live candles by consuming the OANDA v20 Pricing
Stream.  When a new bar closes for any symbol / timeframe, the updated
``pandas.DataFrame`` is pushed to every registered bot queue so that the
individual bot processes never need to call the OANDA REST API for candle
data themselves.

History preload (done once, at start)
--------------------------------------
    M5  : 500 bars
    H4  : 500 bars
    D1  : 500 bars

Streaming
---------
* One persistent HTTP streaming connection (OANDA ``/v3/accounts/.../pricing/stream``)
* Receives ``PRICE`` (tick) and ``HEARTBEAT`` JSON messages
* Mid-price (average of best bid & ask) is used for candle building
* Automatic reconnect on any disconnect / error (5-second back-off)

Distribution
------------
* One ``multiprocessing.Queue`` per bot is registered via ``add_bot_queue()``.
  Messages are ``(symbol, tf, dataframe_copy)`` tuples published to **all**
  registered queues whenever a bar closes.  Bots are responsible for draining
  their own queue.

Usage inside smc_bot_manager.py
---------------------------------
    from data_manager import run_data_manager

    q = multiprocessing.Queue(maxsize=2000)
    p = multiprocessing.Process(
        target=run_data_manager,
        args=(account_id, access_token, symbols, environment, [q]),
        name="DataManager",
        daemon=False,
    )
    p.start()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Optional import: oandapyV20 (only used for history preload)
# ---------------------------------------------------------------------------
try:
    from oandapyV20 import API as _OandaAPI  # type: ignore
    import oandapyV20.endpoints.instruments as _ep_instruments  # type: ignore
    _OANDA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OandaAPI = None  # type: ignore
    _ep_instruments = None  # type: ignore
    _OANDA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("data_manager")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of bars to preload for each timeframe (done once at start)
PRELOAD_BARS: Dict[str, int] = {
    "M5": 500,
    "H4": 500,
    "D1": 500,
}

# Mapping from our TF key to OANDA granularity string
_TF_GRAN: Dict[str, str] = {
    "M5": "M5",
    "H4": "H4",
    "D1": "D",
}

# Bar duration in seconds (used to derive bar open time from close time)
_TF_SECONDS: Dict[str, int] = {
    "M5": 300,
    "H4": 14_400,
    "D1": 86_400,
}

# OANDA streaming base URLs
_STREAM_URLS: Dict[str, str] = {
    "practice": "https://stream-fxpractice.oanda.com",
    "live":     "https://stream-fxtrade.oanda.com",
}

# Maximum rows kept in each in-memory DataFrame
_MAX_ROWS = 500

# Reconnect back-off on stream error (seconds)
_RECONNECT_DELAY = 5

# Streaming connection timeout (seconds); OANDA sends heartbeats every ~5 s
_STREAM_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_oanda_symbol(symbol: str) -> str:
    """Convert ``EURUSD`` → ``EUR_USD`` for OANDA REST/stream endpoints."""
    s = symbol.upper().replace("_", "")
    if len(s) == 6:
        return f"{s[:3]}_{s[3:]}"
    return s


def _bar_open_time(ts: datetime, tf: str) -> datetime:
    """Return the bar *open* time (start of bar) for *ts* and *tf*."""
    if tf == "M5":
        bucket = (ts.minute // 5) * 5
        return ts.replace(minute=bucket, second=0, microsecond=0)
    if tf == "H4":
        bucket = (ts.hour // 4) * 4
        return ts.replace(hour=bucket, minute=0, second=0, microsecond=0)
    if tf == "D1":
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    return ts


# ---------------------------------------------------------------------------
# DataManager class
# ---------------------------------------------------------------------------

class DataManager:
    """Preloads history and maintains live OHLCV DataFrames via OANDA stream.

    Parameters
    ----------
    account_id   : OANDA account ID
    access_token : OANDA API access token
    symbols      : List of symbol strings (e.g. ``["EURUSD", "GBPUSD"]``)
    environment  : ``"practice"`` or ``"live"``
    """

    def __init__(
        self,
        account_id: str,
        access_token: str,
        symbols: List[str],
        environment: str = "practice",
    ) -> None:
        self.account_id = account_id
        self.access_token = access_token
        self.symbols = list(symbols)
        self.environment = environment

        # Latest DataFrames: {symbol: {tf: DataFrame}}
        self._data: Dict[str, Dict[str, pd.DataFrame]] = {
            sym: {} for sym in self.symbols
        }

        # Currently-forming candle per symbol/tf:
        # {symbol: {tf: {"open", "high", "low", "close", "volume", "bar_open"}}}
        self._current_candle: Dict[str, Dict[str, Optional[dict]]] = {
            sym: {tf: None for tf in _TF_GRAN} for sym in self.symbols
        }

        # Queues to publish bar-close events to all bots
        self.bot_queues: list = []

        # Thread control
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_bot_queue(self, q) -> None:
        """Register *q* to receive ``(symbol, tf, df)`` updates on bar close."""
        self.bot_queues.append(q)

    def start(self) -> None:
        """Preload history, then launch the streaming thread (non-blocking)."""
        self._preload_history()
        t = threading.Thread(target=self._stream_loop, daemon=True, name="oanda-stream")
        t.start()
        log.info("DataManager streaming thread started.")

    def get_latest_data(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        """Return a copy of the latest DataFrame for *symbol* / *tf*, or ``None``."""
        with self._lock:
            df = self._data.get(symbol, {}).get(tf)
            return df.copy() if df is not None else None

    def stop(self) -> None:
        """Signal the streaming thread to stop gracefully."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # History preload
    # ------------------------------------------------------------------

    def _preload_history(self) -> None:
        """Fetch PRELOAD_BARS candles for every symbol × timeframe via REST."""
        if not _OANDA_AVAILABLE:
            log.error(
                "oandapyV20 not installed – history preload skipped.  "
                "Run: pip install oandapyV20"
            )
            return

        client = _OandaAPI(
            access_token=self.access_token,
            environment=self.environment,
        )

        for sym in self.symbols:
            oanda_sym = _to_oanda_symbol(sym)
            for tf, gran in _TF_GRAN.items():
                n_bars = PRELOAD_BARS[tf]
                try:
                    r = _ep_instruments.InstrumentsCandles(
                        instrument=oanda_sym,
                        params={
                            "count": n_bars,
                            "granularity": gran,
                            "price": "M",
                        },
                    )
                    client.request(r)
                    candles = r.response.get("candles", [])
                    if not candles:
                        log.warning("No candles returned for %s %s", sym, tf)
                        continue

                    rows = []
                    for c in candles:
                        mid = c.get("mid", {})
                        rows.append(
                            {
                                "time":   c["time"],
                                "open":   float(mid["o"]),
                                "high":   float(mid["h"]),
                                "low":    float(mid["l"]),
                                "close":  float(mid["c"]),
                                "volume": float(c.get("volume", 0)),
                            }
                        )

                    df = pd.DataFrame(rows)
                    df["time"] = pd.to_datetime(df["time"], utc=True)
                    df.set_index("time", inplace=True)

                    with self._lock:
                        self._data[sym][tf] = df

                    log.info(
                        "Preloaded %d %s %s bars (last: %s)",
                        len(df), sym, tf, df.index[-1],
                    )

                except Exception as exc:  # noqa: BLE001
                    log.error("Preload error for %s %s: %s", sym, tf, exc)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _stream_loop(self) -> None:
        """Reconnect loop – reconnects after any error with a back-off."""
        while not self._stop_event.is_set():
            try:
                self._connect_and_stream()
            except Exception as exc:  # noqa: BLE001
                if not self._stop_event.is_set():
                    log.warning(
                        "Stream disconnected: %s – reconnecting in %ds",
                        exc,
                        _RECONNECT_DELAY,
                    )
                    time.sleep(_RECONNECT_DELAY)

    def _connect_and_stream(self) -> None:
        """Open one HTTP streaming connection and process incoming messages."""
        base_url = _STREAM_URLS.get(self.environment, _STREAM_URLS["practice"])
        instruments = ",".join(_to_oanda_symbol(s) for s in self.symbols)
        url = (
            f"{base_url}/v3/accounts/{self.account_id}"
            f"/pricing/stream?instruments={instruments}&snapshot=true"
        )
        headers = {"Authorization": f"Bearer {self.access_token}"}

        with requests.get(
            url, headers=headers, stream=True, timeout=_STREAM_TIMEOUT
        ) as resp:
            resp.raise_for_status()
            log.info("Connected to OANDA pricing stream (%s).", self.environment)

            for raw_line in resp.iter_lines():
                if self._stop_event.is_set():
                    break
                if not raw_line:
                    continue
                try:
                    msg = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")
                if msg_type == "PRICE":
                    self._handle_price(msg)
                # HEARTBEAT messages are intentionally ignored

    # ------------------------------------------------------------------
    # Tick / candle processing
    # ------------------------------------------------------------------

    def _handle_price(self, msg: dict) -> None:
        """Extract mid price from a PRICE message and update candles."""
        instrument = msg.get("instrument", "")
        # OANDA sends "EUR_USD"; convert to "EURUSD"
        symbol = instrument.replace("_", "")
        if symbol not in self.symbols:
            return

        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        if not bids or not asks:
            return

        mid = (float(bids[0]["price"]) + float(asks[0]["price"])) / 2.0

        time_str = msg.get("time", "")
        try:
            ts = pd.to_datetime(time_str, utc=True).to_pydatetime()
        except Exception:  # noqa: BLE001
            ts = datetime.now(timezone.utc)

        for tf in _TF_GRAN:
            self._update_candle(symbol, tf, ts, mid)

    def _update_candle(
        self, symbol: str, tf: str, ts: datetime, price: float
    ) -> None:
        """Update the forming candle; append to DataFrame when a bar closes."""
        bar_open = _bar_open_time(ts, tf)

        with self._lock:
            current = self._current_candle[symbol][tf]

            if current is None:
                # First tick – initialise the current bar
                self._current_candle[symbol][tf] = {
                    "open":     price,
                    "high":     price,
                    "low":      price,
                    "close":    price,
                    "volume":   1,
                    "bar_open": bar_open,
                }
                return

            if bar_open != current["bar_open"]:
                # A new bar has started → finalise the previous one
                closed = dict(current)
                self._current_candle[symbol][tf] = {
                    "open":     price,
                    "high":     price,
                    "low":      price,
                    "close":    price,
                    "volume":   1,
                    "bar_open": bar_open,
                }
                # _append_candle expects the lock already held
                self._append_candle_locked(symbol, tf, closed)
            else:
                # Same bar – update OHLC
                current["high"] = max(current["high"], price)
                current["low"] = min(current["low"], price)
                current["close"] = price
                current["volume"] += 1

    def _append_candle_locked(
        self, symbol: str, tf: str, candle: dict
    ) -> None:
        """Append *candle* to the in-memory DataFrame and notify bot queues.

        Must be called while ``self._lock`` is already held.
        """
        bar_open: datetime = candle["bar_open"]
        idx = pd.DatetimeIndex([pd.Timestamp(bar_open)], name="time")
        new_row = pd.DataFrame(
            [
                {
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": float(candle["volume"]),
                }
            ],
            index=idx,
        )

        df = self._data.get(symbol, {}).get(tf)
        if df is not None:
            df = pd.concat([df, new_row])
            df = df.iloc[-_MAX_ROWS:]  # keep rolling window
        else:
            df = new_row

        self._data[symbol][tf] = df

        log.debug("New %s %s bar  open=%s", tf, symbol, bar_open)

        # Publish to all bot queues (non-blocking; drop if full)
        df_copy = df.copy()
        for q in self.bot_queues:
            try:
                q.put_nowait((symbol, tf, df_copy))
            except Exception:  # noqa: BLE001
                pass  # queue full – bot will catch up on next bar


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def run_data_manager(
    account_id: str,
    access_token: str,
    symbols: List[str],
    environment: str,
    queues: list,
    log_dir: str = "",
) -> None:
    """Run the DataManager in its own process.

    Parameters
    ----------
    account_id   : OANDA account ID
    access_token : OANDA API access token
    symbols      : List of traded symbol strings
    environment  : ``"practice"`` or ``"live"``
    queues       : List of :class:`multiprocessing.Queue` instances – one per
                   bot.  The manager pushes ``(symbol, tf, df)`` tuples to
                   every queue whenever a bar closes.
    log_dir      : Optional path to the logs directory.  When provided the
                   DataManager appends to ``<log_dir>/manager.log`` instead of
                   writing to stdout only.
    """
    from pathlib import Path as _Path

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_dir:
        _ld = _Path(log_dir)
        _ld.mkdir(parents=True, exist_ok=True)
        _lp = _ld / "manager.log"
        handlers.append(logging.FileHandler(str(_lp), encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [DataManager]  %(levelname)-5s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    dm = DataManager(
        account_id=account_id,
        access_token=access_token,
        symbols=symbols,
        environment=environment,
    )
    for q in queues:
        dm.add_bot_queue(q)

    dm.start()

    # Keep the process alive (streaming runs in a daemon thread)
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        dm.stop()
        log.info("DataManager stopped.")
