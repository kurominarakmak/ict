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
from datetime import datetime, timezone
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
            "exit_reason",
            "gross_r",
            "net_r_vs_020_spread",
            "realized_spread_or_slippage",
            "ticket",
            "notes",
        ]
        if not self.path.exists():
            with self.path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self.fields).writeheader()

    def write(self, **kwargs: object) -> None:
        row = {field: kwargs.get(field, "") for field in self.fields}
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writerow(row)


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
            entry_time = utc_from_timestamp(pos.time)
            entry_bar = self.find_bar_at_or_before(bars, entry_time) or bars[-1]
            if self.setup is not None:
                intended_entry = self.setup.range_high if direction == 1 else self.setup.range_low
                atr = self.setup.atr_at_setup
                range_high = self.setup.range_high
                range_low = self.setup.range_low
                setup_time = self.setup.setup_time
            else:
                intended_entry = float(pos.price_open)
                atr = abs(float(pos.price_open) - float(pos.sl)) if float(pos.sl) else (entry_bar.atr14 or bars[-1].atr14 or 0.0)
                range_high = 0.0
                range_low = 0.0
                setup_time = entry_bar.time
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
            )
            self.cancel_all_pending("reconcile_open_position")
            if not already_active:
                self.log_entry(self.active)
        else:
            self.active = None
            existing_orders = self.own_orders()
            if existing_orders:
                # Keep existing pending orders after restart; they represent a
                # live compression setup already submitted.
                self.reconstruct_setup_from_orders(existing_orders, bars)

    def on_poll(self, bars: list[LiveBar]) -> None:
        self.reconcile_exit_if_position_closed()
        self.reconcile(bars)
        self.cancel_opposite_if_one_side_filled()
        if self.active is not None:
            self.force_close_if_due(bars)

    def on_new_closed_bar(self, bars: list[LiveBar]) -> None:
        self.reconcile_exit_if_position_closed()
        self.reconcile(bars)
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
            self.logger.write(
                event="order_error",
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
                notes=f"pending order failed: {result}; last_error={mt5.last_error()}",
            )
            return None
        return int(result.order)

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
            "comment": "force_close",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        self.logger.write(
            event="force_close_sent",
            strategy=self.name,
            magic=self.magic,
            symbol=self.symbol,
            signal_time=bars[-1].time.isoformat(),
            actual_fill_price=price,
            ticket=int(pos.ticket),
            notes=f"elapsed_closed_bars={elapsed}; result={result}",
        )

    def reconcile_exit_if_position_closed(self) -> None:
        if self.active is None:
            return
        if any(int(p.ticket) == self.active.position_ticket for p in self.own_positions()):
            return
        self.log_closed_trade(self.active)
        self.active = None

    def log_closed_trade(self, trade: ActiveTrade) -> None:
        now = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(trade.entry_time, now)
        own_deals = [
            d for d in list(deals or [])
            if int(getattr(d, "magic", 0)) == self.magic and getattr(d, "symbol", "") == self.symbol
        ]
        exit_deals = [
            d for d in own_deals
            if int(getattr(d, "position_id", 0)) == trade.position_ticket
            and int(getattr(d, "entry", -1)) in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)
        ]
        exit_price = float(exit_deals[-1].price) if exit_deals else ""
        exit_reason = "unknown"
        if exit_deals:
            comment = str(getattr(exit_deals[-1], "comment", "")).lower()
            reason_code = int(getattr(exit_deals[-1], "reason", -1))
            if "force" in comment:
                exit_reason = "force"
            elif reason_code == getattr(mt5, "DEAL_REASON_TP", -999):
                exit_reason = "tp"
            elif reason_code == getattr(mt5, "DEAL_REASON_SL", -999):
                exit_reason = "sl"
        if isinstance(exit_price, float) and trade.atr_at_entry > 0:
            gross_r = trade.direction * (exit_price - trade.actual_entry) / trade.atr_at_entry
            net_r = gross_r - BACKTEST_ROUNDTRIP_SPREAD / trade.atr_at_entry
        else:
            gross_r = ""
            net_r = ""
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
            exit_reason=exit_reason,
            gross_r=gross_r,
            net_r_vs_020_spread=net_r,
            realized_spread_or_slippage=trade.direction * (trade.actual_entry - trade.intended_entry),
            ticket=trade.position_ticket,
            notes="closed position reconciled from MT5 history",
        )

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
            ticket=trade.position_ticket,
            notes="pending range-edge order filled; opposite pending cancelled",
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
