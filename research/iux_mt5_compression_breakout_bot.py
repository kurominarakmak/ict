"""
IUX MT5 demo forward-test bot: XAUUSD M15 compression breakout-following.

This implements only the validated compression breakout strategy. OB+FVG can be
added later as another Strategy subclass with magic number 1001.

Backtest mapping:
- `trailing_atr_cutoff` ports research/volatility_compression_breakout_audit.py::trailing_atr_cutoff.
- `is_compression_end` ports research/volatility_compression_breakout_audit.py::is_compression_end.
- Compression range is the same 16-bar high/low window used by the audit.
- Breakout following is the simple-follow variant tested in
  research/simple_breakout_atr_exit_audit.py: enter the broken range edge in
  the breakout direction, stop = 1.0 * ATR, target = configurable RR.

Live/backtest comparability note:
The backtest can mark a range-edge fill after seeing a completed breakout bar.
Live trading cannot know a future breakout bar's completed ATR before the fill.
To match the range-edge entry as closely as possible without look-ahead, this
bot places OCO pending stop orders at the range edges immediately after a
closed compression bar. SL/TP use the ATR known at setup time. All intended vs
actual fills and costs are logged so any residual divergence is measurable.

Forward testing is still required. The bot removes the need to watch charts; it
does not remove the need for weeks/months of out-of-sample trade count.
"""

from __future__ import annotations

import csv
import math
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError as exc:  # pragma: no cover - depends on local MT5 install
    raise SystemExit("MetaTrader5 package is required: pip install MetaTrader5") from exc


SYMBOL = os.getenv("IUX_SYMBOL", "XAUUSD")
TIMEFRAME = mt5.TIMEFRAME_M15
TIMEFRAME_SECONDS = 15 * 60
HISTORY_BARS = int(os.getenv("IUX_HISTORY_BARS", "220"))

COMPRESSION_WINDOW = 16
ATR_PERIOD = 14
ATR_TRAIL = 100
ATR_TERCILE_Q = 1 / 3
COMPRESSION_MIN_FRACTION = 0.75

LOT_SIZE = float(os.getenv("IUX_LOT_SIZE", "0.01"))
RR_TARGET = float(os.getenv("IUX_RR_TARGET", "1.5"))
FORCE_CLOSE_BARS = int(os.getenv("IUX_FORCE_CLOSE_BARS", "10"))
MAGIC_COMPRESSION = 1002
BACKTEST_ROUNDTRIP_SPREAD = 0.20
SESSION_FLATTEN_ENABLED = os.getenv("IUX_SESSION_FLATTEN", "1") != "0"
SESSION_FLATTEN_HOUR = int(os.getenv("IUX_SESSION_FLATTEN_HOUR_UTC", "21"))
SESSION_FLATTEN_MINUTE = int(os.getenv("IUX_SESSION_FLATTEN_MINUTE_UTC", "45"))

POLL_SECONDS = int(os.getenv("IUX_POLL_SECONDS", "5"))
MAX_DEVIATION_POINTS = int(os.getenv("IUX_DEVIATION_POINTS", "50"))
LOG_PATH = Path(os.getenv("IUX_COMPRESSION_LOG", "research/iux_compression_breakout_live_log.csv"))

ORDER_COMMENT = "compression_breakout"


@dataclass
class LiveBar:
    index: int
    time: datetime
    open: float
    high: float
    low: float
    close: float
    atr14: Optional[float] = None


@dataclass
class CompressionSetup:
    setup_time: datetime
    setup_index: int
    range_high: float
    range_low: float
    atr_at_setup: float
    buy_order: Optional[int] = None
    sell_order: Optional[int] = None


@dataclass
class ActiveTrade:
    position_ticket: int
    direction: int
    entry_time: datetime
    entry_bar_time: datetime
    intended_entry: float
    actual_entry: float
    atr_at_entry: float
    sl: float
    tp: float
    range_high: float
    range_low: float
    setup_time: datetime
    bars_elapsed: int = 0
    reconstructed: bool = False


def utc_from_timestamp(raw: int | float) -> datetime:
    return datetime.fromtimestamp(int(raw), tz=timezone.utc)


def quantile(vals: list[float], q: float) -> float:
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def add_atr14(bars: list[LiveBar]) -> None:
    trs: list[float] = []
    for i, bar in enumerate(bars):
        if i == 0:
            tr = bar.high - bar.low
        else:
            prev_close = bars[i - 1].close
            tr = max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))
        trs.append(tr)
        if i + 1 >= ATR_PERIOD:
            bar.atr14 = mean(trs[i - ATR_PERIOD + 1 : i + 1])


def trailing_atr_cutoff(bars: list[LiveBar], i: int) -> float | None:
    """Port of volatility_compression_breakout_audit.py::trailing_atr_cutoff."""

    vals: list[float] = []
    j = i - 1
    while j >= 0 and len(vals) < ATR_TRAIL:
        if bars[j].atr14 is not None:
            vals.append(bars[j].atr14)
        j -= 1
    if len(vals) < ATR_TRAIL:
        return None
    return quantile(vals, ATR_TERCILE_Q)


def is_compression_end(bars: list[LiveBar], i: int) -> bool:
    """Port of volatility_compression_breakout_audit.py::is_compression_end."""

    start = i - COMPRESSION_WINDOW + 1
    if start < 0:
        return False
    compressed = 0
    checked = 0
    for j in range(start, i + 1):
        cutoff = trailing_atr_cutoff(bars, j)
        if cutoff is None or bars[j].atr14 is None:
            return False
        checked += 1
        if bars[j].atr14 <= cutoff:
            compressed += 1
    return checked == COMPRESSION_WINDOW and compressed / checked >= COMPRESSION_MIN_FRACTION


def load_closed_m15_bars(symbol: str, count: int) -> list[LiveBar]:
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, count + 5)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
    now = datetime.now(timezone.utc)
    current_bucket = int(now.timestamp()) // TIMEFRAME_SECONDS * TIMEFRAME_SECONDS
    bars: list[LiveBar] = []
    for row in rates:
        ts = int(row["time"])
        if ts >= current_bucket:
            continue  # forming bar; never use it for signals
        bars.append(
            LiveBar(
                index=0,
                time=utc_from_timestamp(ts),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
        )
    bars.sort(key=lambda b: b.time)
    bars = bars[-count:]
    for idx, bar in enumerate(bars):
        bar.index = idx
    add_atr14(bars)
    return bars


class CsvTradeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = [
            "event",
            "timestamp_utc",
            "strategy",
            "magic",
            "symbol",
            "signal_time",
            "setup_time",
            "range_high",
            "range_low",
            "breakout_direction",
            "intended_entry",
            "actual_fill_price",
            "sl",
            "tp",
            "atr_at_entry",
            "exit_price",
            "exit_time",
            "exit_reason",
            "gross_r",
            "net_r_vs_020_spread",
            "realized_spread_or_slippage",
            "real_spread_at_entry",
            "real_spread_at_exit",
            "is_intersection",
            "bars_held",
            "ticket",
            "notes",
        ]
        if self.path.exists():
            self.ensure_header_has_fields()
        else:
            with self.path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self.fields).writeheader()

    def ensure_header_has_fields(self) -> None:
        with self.path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            old_fields = reader.fieldnames or []
            rows = list(reader)
        if all(field in old_fields for field in self.fields):
            return
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in self.fields})

    def write(self, **kwargs: object) -> None:
        row = {field: kwargs.get(field, "") for field in self.fields}
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writerow(row)

    def read_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def rewrite_rows(self, rows: list[dict[str, object]]) -> None:
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in self.fields})


class Strategy:
    magic: int
    name: str

    def on_new_closed_bar(self, bars: list[LiveBar]) -> None:
        raise NotImplementedError

    def reconcile(self, bars: list[LiveBar]) -> None:
        raise NotImplementedError

    def on_poll(self, bars: list[LiveBar]) -> None:
        raise NotImplementedError


class CompressionBreakoutStrategy(Strategy):
    name = "compression_breakout"
    magic = MAGIC_COMPRESSION

    def __init__(self, symbol: str, lot: float, rr: float, force_close_bars: int, logger: CsvTradeLogger) -> None:
        self.symbol = symbol
        self.lot = lot
        self.rr = rr
        self.force_close_bars = force_close_bars
        self.logger = logger
        self.setup: Optional[CompressionSetup] = None
        self.active: Optional[ActiveTrade] = None

    def positions(self):
        return list(mt5.positions_get(symbol=self.symbol) or [])

    def own_positions(self):
        return [p for p in self.positions() if int(p.magic) == self.magic]

    def own_orders(self):
        orders = list(mt5.orders_get(symbol=self.symbol) or [])
        return [o for o in orders if int(o.magic) == self.magic]

    def reconcile(self, bars: list[LiveBar]) -> None:
        own_positions = self.own_positions()
        if own_positions:
            pos = own_positions[0]
            already_active = self.active is not None and self.active.position_ticket == int(pos.ticket)
            direction = 1 if pos.type == mt5.POSITION_TYPE_BUY else -1
            # Use the broker position open time, never the current/restart bar,
            # when reconstructing state after a restart.
            entry_time = utc_from_timestamp(pos.time)
            entry_bar = self.find_bar_at_or_before(bars, entry_time) or bars[-1]
            if self.setup is not None:
                intended_entry = self.setup.range_high if direction == 1 else self.setup.range_low
                atr = self.setup.atr_at_setup
                range_high = self.setup.range_high
                range_low = self.setup.range_low
                setup_time = self.setup.setup_time
            else:
                recovered = self.recover_setup_from_csv(int(pos.ticket), direction)
                intended_entry = float(pos.price_open)
                atr = abs(float(pos.price_open) - float(pos.sl)) if float(pos.sl) else (entry_bar.atr14 or bars[-1].atr14 or 0.0)
                range_high = recovered.range_high if recovered is not None else 0.0
                range_low = recovered.range_low if recovered is not None else 0.0
                setup_time = recovered.setup_time if recovered is not None else entry_bar.time
                if recovered is not None:
                    intended_entry = range_high if direction == 1 else range_low
                    atr = recovered.atr_at_setup or atr
            self.active = ActiveTrade(
                position_ticket=int(pos.ticket),
                direction=direction,
                entry_time=entry_time,
                entry_bar_time=entry_bar.time,
                intended_entry=float(intended_entry),
                actual_entry=float(pos.price_open),
                atr_at_entry=atr,
                sl=float(pos.sl),
                tp=float(pos.tp),
                range_high=range_high,
                range_low=range_low,
                setup_time=setup_time,
                bars_elapsed=self.closed_bars_since(bars, entry_bar.time),
                reconstructed=self.setup is None,
            )
            self.cancel_all_pending("reconcile_open_position")
            if not already_active:
                print(
                    "WARNING: reconstructed open position after startup/restart. "
                    "Avoid restarting while a position is open; orphan-exit recovery will backfill if needed.",
                    flush=True,
                )
                self.log_entry(self.active)
        else:
            self.active = None
            existing_orders = self.own_orders()
            if existing_orders:
                # Keep existing pending orders after restart; they represent a
                # live compression setup already submitted.
                self.reconstruct_setup_from_orders(existing_orders, bars)

    def on_poll(self, bars: list[LiveBar]) -> None:
        self.recover_orphaned_exits(bars)
        self.reconcile_exit_if_position_closed(bars)
        self.reconcile(bars)
        self.cancel_opposite_if_one_side_filled()
        if self.handle_session_flatten(bars):
            return
        if self.active is not None:
            self.force_close_if_due(bars)

    def on_new_closed_bar(self, bars: list[LiveBar]) -> None:
        self.recover_orphaned_exits(bars)
        self.reconcile_exit_if_position_closed(bars)
        self.reconcile(bars)
        if self.handle_session_flatten(bars):
            return
        if self.active is not None:
            self.force_close_if_due(bars)
            return
        if self.own_orders():
            self.cancel_opposite_if_one_side_filled()
            return
        self.setup = None
        i = len(bars) - 1
        if is_compression_end(bars, i):
            self.create_setup_and_orders(bars, i)

    def create_setup_and_orders(self, bars: list[LiveBar], i: int) -> None:
        window = bars[i - COMPRESSION_WINDOW + 1 : i + 1]
        atr = bars[i].atr14
        if atr is None or atr <= 0:
            return
        range_high = max(b.high for b in window)
        range_low = min(b.low for b in window)
        if range_high <= range_low:
            return
        if self.is_session_flatten_window():
            self.logger.write(
                event="order_skip",
                strategy=self.name,
                magic=self.magic,
                symbol=self.symbol,
                signal_time=bars[i].time.isoformat(),
                setup_time=bars[i].time.isoformat(),
                range_high=range_high,
                range_low=range_low,
                atr_at_entry=atr,
                notes="session flatten window; no new pending orders submitted",
            )
            return
        if not self.market_is_open():
            self.logger.write(
                event="order_skip",
                strategy=self.name,
                magic=self.magic,
                symbol=self.symbol,
                signal_time=bars[i].time.isoformat(),
                setup_time=bars[i].time.isoformat(),
                range_high=range_high,
                range_low=range_low,
                atr_at_entry=atr,
                notes="market closed or stale tick; pending range-edge orders not submitted",
            )
            return
        self.setup = CompressionSetup(
            setup_time=bars[i].time,
            setup_index=i,
            range_high=range_high,
            range_low=range_low,
            atr_at_setup=atr,
        )
        buy_ticket = self.place_pending(direction=1, entry=range_high, atr=atr, setup=self.setup)
        sell_ticket = self.place_pending(direction=-1, entry=range_low, atr=atr, setup=self.setup)
        self.setup.buy_order = buy_ticket
        self.setup.sell_order = sell_ticket
        self.logger.write(
            event="signal",
            strategy=self.name,
            magic=self.magic,
            symbol=self.symbol,
            signal_time=bars[i].time.isoformat(),
            setup_time=bars[i].time.isoformat(),
            range_high=range_high,
            range_low=range_low,
            intended_entry=f"buy_stop={range_high};sell_stop={range_low}",
            sl=f"buy={range_high - atr};sell={range_low + atr}",
            tp=f"buy={range_high + self.rr * atr};sell={range_low - self.rr * atr}",
            atr_at_entry=atr,
            ticket=f"buy={buy_ticket};sell={sell_ticket}",
            notes="compression confirmed; OCO pending stops placed at range edges",
        )

    def place_pending(self, direction: int, entry: float, atr: float, setup: CompressionSetup) -> Optional[int]:
        if not self.pending_stop_is_valid(direction, entry, setup):
            return None
        order_type = mt5.ORDER_TYPE_BUY_STOP if direction == 1 else mt5.ORDER_TYPE_SELL_STOP
        sl = entry - direction * atr
        tp = entry + direction * self.rr * atr
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": self.lot,
            "type": order_type,
            "price": self.normalize_price(entry),
            "sl": self.normalize_price(sl),
            "tp": self.normalize_price(tp),
            "deviation": MAX_DEVIATION_POINTS,
            "magic": self.magic,
            "comment": ORDER_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result = mt5.order_send(request)
        ok_retcodes = {mt5.TRADE_RETCODE_DONE, getattr(mt5, "TRADE_RETCODE_PLACED", mt5.TRADE_RETCODE_DONE)}
        if result is None or result.retcode not in ok_retcodes:
            retcode = getattr(result, "retcode", None)
            if retcode == getattr(mt5, "TRADE_RETCODE_MARKET_CLOSED", 10018):
                event = "order_skip"
                note = f"market closed; pending order skipped quietly: {result}; last_error={mt5.last_error()}"
            else:
                event = "order_error"
                note = f"pending order failed: {result}; last_error={mt5.last_error()}"
            self.logger.write(
                event=event,
                strategy=self.name,
                magic=self.magic,
                symbol=self.symbol,
                signal_time=setup.setup_time.isoformat(),
                setup_time=setup.setup_time.isoformat(),
                range_high=setup.range_high,
                range_low=setup.range_low,
                breakout_direction="long" if direction == 1 else "short",
                intended_entry=entry,
                sl=sl,
                tp=tp,
                atr_at_entry=atr,
                notes=note,
            )
            return None
        return int(result.order)

    def market_is_open(self) -> bool:
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return False
        full_mode = getattr(mt5, "SYMBOL_TRADE_MODE_FULL", None)
        if full_mode is not None and int(info.trade_mode) != int(full_mode):
            return False
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None or not getattr(tick, "time", 0):
            return False
        tick_time = utc_from_timestamp(tick.time)
        return (datetime.now(timezone.utc) - tick_time) <= timedelta(minutes=20)

    def pending_stop_is_valid(self, direction: int, entry: float, setup: CompressionSetup) -> bool:
        info = mt5.symbol_info(self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)
        if info is None or tick is None:
            self.log_order_skip(direction, entry, setup, "missing symbol info or tick; pending order skipped")
            return False
        point = float(getattr(info, "point", 0.0) or 0.0)
        stops_level_points = float(getattr(info, "trade_stops_level", 0.0) or 0.0)
        min_distance = stops_level_points * point
        if direction == 1:
            threshold = float(tick.ask) + min_distance
            if entry <= threshold:
                self.log_order_skip(
                    direction,
                    entry,
                    setup,
                    f"buy_stop invalid or missed: entry={entry} <= ask+stops={threshold}; price already too close/beyond edge",
                )
                return False
        else:
            threshold = float(tick.bid) - min_distance
            if entry >= threshold:
                self.log_order_skip(
                    direction,
                    entry,
                    setup,
                    f"sell_stop invalid or missed: entry={entry} >= bid-stops={threshold}; price already too close/beyond edge",
                )
                return False
        return True

    def log_order_skip(self, direction: int, entry: float, setup: CompressionSetup, note: str) -> None:
        self.logger.write(
            event="order_skip",
            strategy=self.name,
            magic=self.magic,
            symbol=self.symbol,
            signal_time=setup.setup_time.isoformat(),
            setup_time=setup.setup_time.isoformat(),
            range_high=setup.range_high,
            range_low=setup.range_low,
            breakout_direction="long" if direction == 1 else "short",
            intended_entry=entry,
            atr_at_entry=setup.atr_at_setup,
            notes=note,
        )

    def cancel_opposite_if_one_side_filled(self) -> None:
        if self.own_positions():
            self.cancel_all_pending("oco_position_filled")

    def cancel_all_pending(self, reason: str) -> None:
        for order in self.own_orders():
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order.ticket),
                "symbol": self.symbol,
                "magic": self.magic,
                "comment": reason,
            }
            mt5.order_send(request)

    def reconstruct_setup_from_orders(self, orders, bars: list[LiveBar]) -> None:
        if self.setup is not None:
            return
        buy = next((o for o in orders if o.type == mt5.ORDER_TYPE_BUY_STOP), None)
        sell = next((o for o in orders if o.type == mt5.ORDER_TYPE_SELL_STOP), None)
        if buy is None or sell is None:
            return
        buy_price = float(buy.price_open)
        sell_price = float(sell.price_open)
        buy_sl = float(buy.sl)
        atr = abs(buy_price - buy_sl) if buy_sl else (bars[-1].atr14 or 0.0)
        self.setup = CompressionSetup(
            setup_time=bars[-1].time,
            setup_index=bars[-1].index,
            range_high=buy_price,
            range_low=sell_price,
            atr_at_setup=atr,
            buy_order=int(buy.ticket),
            sell_order=int(sell.ticket),
        )

    def force_close_if_due(self, bars: list[LiveBar]) -> None:
        if self.active is None:
            return
        elapsed = self.closed_bars_since(bars, self.active.entry_bar_time)
        self.active.bars_elapsed = elapsed
        if elapsed < self.force_close_bars:
            return
        pos = next((p for p in self.own_positions() if int(p.ticket) == self.active.position_ticket), None)
        if pos is None:
            return
        self.close_position(pos, "force_close", bars[-1].time.isoformat(), f"elapsed_closed_bars={elapsed}")

    def handle_session_flatten(self, bars: list[LiveBar]) -> bool:
        if not self.is_session_flatten_window():
            return False
        if self.own_orders():
            self.cancel_all_pending("session_flatten_cancel_pending")
            self.logger.write(
                event="order_skip",
                strategy=self.name,
                magic=self.magic,
                symbol=self.symbol,
                signal_time=bars[-1].time.isoformat(),
                notes="session flatten window; pending compression orders cancelled before market gap",
            )
        if self.active is None:
            return True
        pos = next((p for p in self.own_positions() if int(p.ticket) == self.active.position_ticket), None)
        if pos is None:
            return True
        self.close_position(pos, "session_flatten", bars[-1].time.isoformat(), "pre-session-gap flatten to match backtest and avoid unmanaged gap risk")
        return True

    def is_session_flatten_window(self) -> bool:
        if not SESSION_FLATTEN_ENABLED:
            return False
        now = datetime.now(timezone.utc)
        if now.weekday() > 4:
            return False
        return (now.hour, now.minute) >= (SESSION_FLATTEN_HOUR, SESSION_FLATTEN_MINUTE)

    def current_real_spread(self) -> float | str:
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return ""
        return float(tick.ask) - float(tick.bid)

    def close_position(self, pos, reason: str, signal_time: str, note: str) -> None:
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(pos.volume),
            "type": close_type,
            "position": int(pos.ticket),
            "price": self.normalize_price(price),
            "deviation": MAX_DEVIATION_POINTS,
            "magic": self.magic,
            "comment": reason,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        self.logger.write(
            event="force_close_sent",
            strategy=self.name,
            magic=self.magic,
            symbol=self.symbol,
            signal_time=signal_time,
            actual_fill_price=price,
            real_spread_at_exit=self.current_real_spread(),
            ticket=int(pos.ticket),
            notes=f"{note}; result={result}",
        )

    def reconcile_exit_if_position_closed(self, bars: list[LiveBar]) -> None:
        if self.active is None:
            return
        if any(int(p.ticket) == self.active.position_ticket for p in self.own_positions()):
            return
        self.log_closed_trade(self.active, bars)
        self.active = None

    def log_closed_trade(self, trade: ActiveTrade, bars: list[LiveBar]) -> None:
        close_deal, match_method = self.find_closing_deal(trade)
        if close_deal is None:
            exit_price: float | str = ""
            exit_time: datetime | str = ""
            exit_reason = "unknown"
            gross_r: float | str = ""
            net_r: float | str = ""
            bars_held: int | str = ""
            notes = "closed position detected but no closing deal found in MT5 history"
        else:
            exit_price = float(close_deal.price)
            exit_time = utc_from_timestamp(getattr(close_deal, "time", int(datetime.now(timezone.utc).timestamp())))
            exit_reason = self.classify_exit_reason(trade, exit_price)
            gross_r = trade.direction * (exit_price - trade.actual_entry) / trade.atr_at_entry
            net_r = gross_r - BACKTEST_ROUNDTRIP_SPREAD / trade.atr_at_entry
            bars_held = self.closed_bars_since(bars, trade.entry_bar_time)
            notes = (
                f"closed position reconciled from MT5 history; match_method={match_method}; "
                f"deal={getattr(close_deal, 'ticket', '')}; mt5_reason={getattr(close_deal, 'reason', '')}; "
                f"comment={getattr(close_deal, 'comment', '')}"
            )
        self.logger.write(
            event="exit",
            strategy=self.name,
            magic=self.magic,
            symbol=self.symbol,
            signal_time=trade.entry_bar_time.isoformat(),
            setup_time=trade.setup_time.isoformat(),
            range_high=trade.range_high,
            range_low=trade.range_low,
            breakout_direction="long" if trade.direction == 1 else "short",
            intended_entry=trade.intended_entry,
            actual_fill_price=trade.actual_entry,
            sl=trade.sl,
            tp=trade.tp,
            atr_at_entry=trade.atr_at_entry,
            exit_price=exit_price,
            exit_time=exit_time.isoformat() if isinstance(exit_time, datetime) else "",
            exit_reason=exit_reason,
            gross_r=gross_r,
            net_r_vs_020_spread=net_r,
            realized_spread_or_slippage=trade.direction * (trade.actual_entry - trade.intended_entry),
            real_spread_at_exit=self.current_real_spread(),
            is_intersection=False,
            bars_held=bars_held,
            ticket=trade.position_ticket,
            notes=notes,
        )

    def find_closing_deal(self, trade: ActiveTrade):
        true_open_time = self.position_open_time_from_history(trade.position_ticket)
        start = (true_open_time - timedelta(minutes=5)) if true_open_time is not None else (datetime.now(timezone.utc) - timedelta(days=30))
        end = datetime.now(timezone.utc) + timedelta(minutes=5)
        deals = mt5.history_deals_get(start, end)
        if deals is None:
            return None, "history_deals_get_none"
        out_entry_codes = {
            getattr(mt5, "DEAL_ENTRY_OUT", 1),
            getattr(mt5, "DEAL_ENTRY_INOUT", 2),
        }
        symbol_exits = [
            d for d in list(deals)
            if getattr(d, "symbol", "") == self.symbol
            and int(getattr(d, "entry", -1)) in out_entry_codes
            and (true_open_time is None or utc_from_timestamp(getattr(d, "time", 0)) >= true_open_time)
        ]
        # Broker-side TP/SL closes can have magic=0. The position_id is the
        # reliable key, so it is primary and does not depend on magic.
        strict = [d for d in symbol_exits if int(getattr(d, "position_id", -1)) == trade.position_ticket]
        if strict:
            strict.sort(key=lambda d: int(getattr(d, "time", 0)))
            return strict[-1], "position_id"
        fallback = [
            d for d in symbol_exits
            if true_open_time is None or utc_from_timestamp(getattr(d, "time", 0)) >= true_open_time
        ]
        if fallback:
            anchor = true_open_time or trade.entry_time
            fallback.sort(key=lambda d: abs(int(getattr(d, "time", 0)) - int(anchor.timestamp())))
            return fallback[0], "symbol_time_fallback"
        return None, "no_match"

    def position_open_time_from_history(self, position_ticket: int) -> Optional[datetime]:
        start = datetime.now(timezone.utc) - timedelta(days=30)
        end = datetime.now(timezone.utc) + timedelta(minutes=5)
        deals = mt5.history_deals_get(start, end)
        if deals is None:
            return None
        in_entry_codes = {getattr(mt5, "DEAL_ENTRY_IN", 0), getattr(mt5, "DEAL_ENTRY_INOUT", 2)}
        candidates = [
            d for d in list(deals)
            if getattr(d, "symbol", "") == self.symbol
            and int(getattr(d, "position_id", -1)) == position_ticket
            and int(getattr(d, "entry", -1)) in in_entry_codes
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda d: int(getattr(d, "time", 0)))
        return utc_from_timestamp(getattr(candidates[0], "time", 0))

    def recover_setup_from_csv(self, position_ticket: int, direction: int) -> Optional[CompressionSetup]:
        rows = self.logger.read_rows()
        for row in reversed(rows):
            if str(row.get("ticket", "")) != str(position_ticket):
                continue
            if row.get("event", "").lower() not in {"entry", "exit"}:
                continue
            try:
                range_high = float(row.get("range_high", "") or 0)
                range_low = float(row.get("range_low", "") or 0)
                atr = float(row.get("atr_at_entry", "") or 0)
            except ValueError:
                continue
            if range_high > range_low > 0 and atr > 0:
                setup_raw = row.get("setup_time") or row.get("signal_time") or row.get("timestamp_utc")
                setup_time = datetime.fromisoformat(str(setup_raw).replace("Z", "+00:00"))
                return CompressionSetup(setup_time, 0, range_high, range_low, atr)
        open_order = self.opening_order_ticket(position_ticket)
        if open_order is not None:
            token = str(open_order)
            for row in reversed(rows):
                if row.get("event", "").lower() != "signal":
                    continue
                if token not in str(row.get("ticket", "")):
                    continue
                try:
                    range_high = float(row.get("range_high", "") or 0)
                    range_low = float(row.get("range_low", "") or 0)
                    atr = float(row.get("atr_at_entry", "") or 0)
                except ValueError:
                    continue
                if range_high > range_low > 0 and atr > 0:
                    setup_raw = row.get("setup_time") or row.get("signal_time") or row.get("timestamp_utc")
                    setup_time = datetime.fromisoformat(str(setup_raw).replace("Z", "+00:00"))
                    return CompressionSetup(setup_time, 0, range_high, range_low, atr)
        return None

    def opening_order_ticket(self, position_ticket: int) -> Optional[int]:
        start = datetime.now(timezone.utc) - timedelta(days=30)
        end = datetime.now(timezone.utc) + timedelta(minutes=5)
        deals = mt5.history_deals_get(start, end)
        if deals is None:
            return None
        in_entry_codes = {getattr(mt5, "DEAL_ENTRY_IN", 0), getattr(mt5, "DEAL_ENTRY_INOUT", 2)}
        candidates = [
            d for d in list(deals)
            if getattr(d, "symbol", "") == self.symbol
            and int(getattr(d, "position_id", -1)) == position_ticket
            and int(getattr(d, "entry", -1)) in in_entry_codes
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda d: int(getattr(d, "time", 0)))
        return int(getattr(candidates[0], "order", 0) or 0) or None

    def recover_orphaned_exits(self, bars: list[LiveBar]) -> None:
        rows = self.logger.read_rows()
        if not rows:
            return
        open_tickets = {str(int(p.ticket)) for p in self.own_positions()}
        entry_by_ticket = {}
        valid_exit_tickets = set()
        bad_exit_tickets = set()
        for row in rows:
            ticket = str(row.get("ticket", "")).strip()
            if not ticket:
                continue
            event = row.get("event", "").lower()
            if event == "entry":
                entry_by_ticket[ticket] = row
            elif event == "exit":
                has_r = row.get("gross_r") not in ("", None) and row.get("net_r_vs_020_spread") not in ("", None)
                reason = str(row.get("exit_reason", "")).lower()
                notes = str(row.get("notes", "")).lower()
                if has_r and reason != "unknown" and "no closing deal found" not in notes:
                    valid_exit_tickets.add(ticket)
                else:
                    bad_exit_tickets.add(ticket)
        candidates = sorted((set(entry_by_ticket) - valid_exit_tickets) | bad_exit_tickets)
        for ticket in candidates:
            if ticket in open_tickets:
                continue
            try:
                position_ticket = int(ticket)
            except ValueError:
                continue
            recovered = self.recovered_exit_row(position_ticket, entry_by_ticket.get(ticket), bars)
            if recovered is None:
                continue
            rows = [
                row for row in rows
                if not (str(row.get("ticket", "")).strip() == ticket and row.get("event", "").lower() == "exit")
            ]
            rows.append(recovered)
            self.logger.rewrite_rows(rows)
            print(f"Recovered orphaned exit for ticket={ticket}", flush=True)

    def recovered_exit_row(self, position_ticket: int, entry_row: Optional[dict[str, str]], bars: list[LiveBar]) -> Optional[dict[str, object]]:
        if entry_row is None:
            return None
        try:
            direction = 1 if str(entry_row.get("breakout_direction", "")).lower() == "long" else -1
            actual_entry = float(entry_row.get("actual_fill_price", "") or entry_row.get("intended_entry", ""))
            intended_entry = float(entry_row.get("intended_entry", "") or actual_entry)
            atr = float(entry_row.get("atr_at_entry", "") or 0)
            sl = float(entry_row.get("sl", "") or 0)
            tp = float(entry_row.get("tp", "") or 0)
            range_high = float(entry_row.get("range_high", "") or 0)
            range_low = float(entry_row.get("range_low", "") or 0)
            setup_raw = entry_row.get("setup_time") or entry_row.get("signal_time") or entry_row.get("timestamp_utc")
            signal_raw = entry_row.get("signal_time") or entry_row.get("timestamp_utc")
            setup_time = datetime.fromisoformat(str(setup_raw).replace("Z", "+00:00"))
            entry_bar_time = datetime.fromisoformat(str(signal_raw).replace("Z", "+00:00"))
            entry_time_raw = entry_row.get("timestamp_utc") or signal_raw
            entry_time = datetime.fromisoformat(str(entry_time_raw).replace("Z", "+00:00"))
        except Exception:
            return None
        if atr <= 0:
            return None
        trade = ActiveTrade(
            position_ticket=position_ticket,
            direction=direction,
            entry_time=entry_time,
            entry_bar_time=entry_bar_time,
            intended_entry=intended_entry,
            actual_entry=actual_entry,
            atr_at_entry=atr,
            sl=sl,
            tp=tp,
            range_high=range_high,
            range_low=range_low,
            setup_time=setup_time,
        )
        close_deal, match_method = self.find_closing_deal(trade)
        if close_deal is None:
            return None
        exit_price = float(close_deal.price)
        exit_time = utc_from_timestamp(getattr(close_deal, "time", int(datetime.now(timezone.utc).timestamp())))
        gross_r = direction * (exit_price - actual_entry) / atr
        net_r = gross_r - BACKTEST_ROUNDTRIP_SPREAD / atr
        return {
            "event": "exit",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "strategy": self.name,
            "magic": self.magic,
            "symbol": self.symbol,
            "signal_time": entry_bar_time.isoformat(),
            "setup_time": setup_time.isoformat(),
            "range_high": range_high,
            "range_low": range_low,
            "breakout_direction": "long" if direction == 1 else "short",
            "intended_entry": intended_entry,
            "actual_fill_price": actual_entry,
            "sl": sl,
            "tp": tp,
            "atr_at_entry": atr,
            "exit_price": exit_price,
            "exit_time": exit_time.isoformat(),
            "exit_reason": self.classify_exit_reason(trade, exit_price),
            "gross_r": gross_r,
            "net_r_vs_020_spread": net_r,
            "realized_spread_or_slippage": direction * (actual_entry - intended_entry),
            "real_spread_at_exit": "",
            "is_intersection": False,
            "bars_held": self.closed_bars_since(bars, entry_bar_time),
            "ticket": position_ticket,
            "notes": f"orphaned exit recovered from MT5 history; match_method={match_method}; deal={getattr(close_deal, 'ticket', '')}; mt5_reason={getattr(close_deal, 'reason', '')}; comment={getattr(close_deal, 'comment', '')}",
        }

    def classify_exit_reason(self, trade: ActiveTrade, exit_price: float) -> str:
        tolerance = self.price_tolerance(trade)
        if trade.direction == 1:
            if exit_price >= trade.tp - tolerance:
                return "target"
            if exit_price <= trade.sl + tolerance:
                return "stop"
        else:
            if exit_price <= trade.tp + tolerance:
                return "target"
            if exit_price >= trade.sl - tolerance:
                return "stop"
        return "force_close"

    def price_tolerance(self, trade: ActiveTrade) -> float:
        info = mt5.symbol_info(self.symbol)
        point = float(getattr(info, "point", 0.01) or 0.01) if info is not None else 0.01
        return max(point * 10, trade.atr_at_entry * 0.02)

    def bars_between_times(self, start: datetime, end: datetime) -> int:
        if end <= start:
            return 0
        return max(0, int((end.timestamp() - start.timestamp()) // TIMEFRAME_SECONDS))

    def log_entry(self, trade: ActiveTrade) -> None:
        self.logger.write(
            event="entry",
            strategy=self.name,
            magic=self.magic,
            symbol=self.symbol,
            signal_time=trade.entry_bar_time.isoformat(),
            setup_time=trade.setup_time.isoformat(),
            range_high=trade.range_high,
            range_low=trade.range_low,
            breakout_direction="long" if trade.direction == 1 else "short",
            intended_entry=trade.intended_entry,
            actual_fill_price=trade.actual_entry,
            sl=trade.sl,
            tp=trade.tp,
            atr_at_entry=trade.atr_at_entry,
            realized_spread_or_slippage=trade.direction * (trade.actual_entry - trade.intended_entry),
            real_spread_at_entry=self.current_real_spread(),
            is_intersection=False,
            ticket=trade.position_ticket,
            notes=(
                "pending range-edge order filled; opposite pending cancelled"
                + ("; reconstructed after restart from open MT5 position" if trade.reconstructed else "")
            ),
        )

    def find_bar_at_or_before(self, bars: list[LiveBar], ts: datetime) -> Optional[LiveBar]:
        prior = [b for b in bars if b.time <= ts]
        return prior[-1] if prior else None

    def closed_bars_since(self, bars: list[LiveBar], ts: datetime) -> int:
        return sum(1 for b in bars if b.time > ts)

    def normalize_price(self, price: float) -> float:
        info = mt5.symbol_info(self.symbol)
        digits = int(info.digits) if info is not None else 2
        return round(float(price), digits)


class TradingBot:
    def __init__(self, strategies: list[Strategy]) -> None:
        self.strategies = strategies
        self.last_closed_bar_time: Optional[datetime] = None
        self.running = True

    def stop(self, *_args: object) -> None:
        self.running = False

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        while self.running:
            try:
                ensure_connected_demo()
                bars = load_closed_m15_bars(SYMBOL, HISTORY_BARS)
                if len(bars) < ATR_TRAIL + COMPRESSION_WINDOW + ATR_PERIOD:
                    print(f"Waiting for enough bars: have {len(bars)}", flush=True)
                    time.sleep(POLL_SECONDS)
                    continue
                closed_time = bars[-1].time
                if self.last_closed_bar_time is None:
                    self.last_closed_bar_time = closed_time
                    for strategy in self.strategies:
                        if hasattr(strategy, "recover_orphaned_exits"):
                            strategy.recover_orphaned_exits(bars)
                        strategy.reconcile(bars)
                    print(f"Initialized on last closed bar {closed_time.isoformat()}", flush=True)
                elif closed_time > self.last_closed_bar_time:
                    self.last_closed_bar_time = closed_time
                    print(f"New closed M15 bar: {closed_time.isoformat()}", flush=True)
                    for strategy in self.strategies:
                        strategy.on_new_closed_bar(bars)
                else:
                    for strategy in self.strategies:
                        strategy.on_poll(bars)
                time.sleep(POLL_SECONDS)
            except Exception as exc:
                print(f"Loop error: {exc}; retrying after backoff", flush=True)
                shutdown_mt5()
                time.sleep(15)


def ensure_connected_demo() -> None:
    if mt5.terminal_info() is None:
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    account = mt5.account_info()
    if account is None:
        raise RuntimeError(f"MT5 account_info failed: {mt5.last_error()}")
    demo_value = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
    if int(account.trade_mode) != int(demo_value):
        raise SystemExit(
            "REFUSING TO RUN: connected MT5 account is not DEMO. "
            f"account={account.login}, trade_mode={account.trade_mode}, expected_demo={demo_value}"
        )
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        raise RuntimeError(f"Symbol not found: {SYMBOL}")
    if not info.visible and not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"Could not select symbol {SYMBOL}: {mt5.last_error()}")


def shutdown_mt5() -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass


def main() -> None:
    print("Starting IUX MT5 DEMO compression breakout bot", flush=True)
    print(f"symbol={SYMBOL}, timeframe=M15, lot={LOT_SIZE}, rr={RR_TARGET}, force_close_bars={FORCE_CLOSE_BARS}", flush=True)
    ensure_connected_demo()
    logger = CsvTradeLogger(LOG_PATH)
    # OB+FVG extension seam:
    # Add a future OrderBlockFvgStrategy(magic=1001) to this list. It should
    # own its own detection, orders, reconciliation, and logging so strategy
    # state/magic numbers do not collide with compression magic 1002.
    strategies: list[Strategy] = [
        CompressionBreakoutStrategy(SYMBOL, LOT_SIZE, RR_TARGET, FORCE_CLOSE_BARS, logger),
    ]
    TradingBot(strategies).run()


if __name__ == "__main__":
    main()
