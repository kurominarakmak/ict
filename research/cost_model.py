"""
IUX Standard spread-overlay cost model for Dukascopy XAUUSD research.

Design principle
----------------
Use Dukascopy mid price for signals and price reference:

    mid = (bid + ask) / 2

Then apply a separate IUX Standard spread overlay for trading cost. Dukascopy's
raw bid/ask spread is interbank reference data only and must never be used as
the backtest trading cost for this broker/account.

All spread inputs below are placeholders until replaced with measured values
from the IUX demo spread logger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Iterable, Literal, Optional, Protocol


STD_LOT_OZ: float = 100.0
MICRO_LOT_OZ: float = 1.0


class SpreadMode(str, Enum):
    MEDIAN = "median"
    P90 = "p90"


SessionName = Literal["asian", "london", "ny_overlap", "off_session"]
Side = Literal["long", "short"]


@dataclass(frozen=True)
class SessionSpread:
    """IUX spread overlay in USD/oz for one session."""

    median: float
    p90: float
    source: str = "UNVERIFIED PLACEHOLDER - replace with IUX spread logger output"

    def value(self, mode: SpreadMode = SpreadMode.MEDIAN) -> float:
        if mode == SpreadMode.P90:
            return self.p90
        return self.median


# Replaced with measured IUX Standard spread values supplied 2026-06-27:
# normal spread ~= $0.20/oz; high-vol/news spread ~= $0.40/oz.
# The p90 value is used as stress mode or when a caller explicitly flags a
# high-volatility/news proxy. These values replace the earlier unverified
# $0.35/$0.55 placeholders.
IUX_SPREAD_USD_OZ: dict[SessionName, SessionSpread] = {
    "asian": SessionSpread(median=0.20, p90=0.40, source="Measured IUX Standard spread, 2026-06-27"),
    "london": SessionSpread(median=0.20, p90=0.40, source="Measured IUX Standard spread, 2026-06-27"),
    "ny_overlap": SessionSpread(median=0.20, p90=0.40, source="Measured IUX Standard spread, 2026-06-27"),
    "off_session": SessionSpread(median=0.20, p90=0.40, source="Measured IUX Standard spread, 2026-06-27"),
}

# PLACEHOLDER - pull real values from MT5 > XAUUSD > Specification.
# These are USD per standard lot per rollover night.
SWAP_LONG_USD_PER_LOT_NIGHT: float = -6.80
SWAP_SHORT_USD_PER_LOT_NIGHT: float = -3.20


class NewsCalendar(Protocol):
    """Future economic-calendar hook.

    Implement this later with a real event calendar. The cost model can then
    flag or exclude trades whose entry/exit touches high-impact news windows.
    """

    def is_news_window(self, timestamp: datetime) -> bool:
        ...


@dataclass(frozen=True)
class CostBreakdown:
    timestamp: datetime
    session: SessionName
    lot_size: float
    contract_oz: float
    spread_mode: SpreadMode
    spread_usd_oz: float
    spread_cost_usd: float
    swap_cost_usd: float
    total_round_trip_cost_usd: float
    news_flagged: bool = False


def dukascopy_mid(bid: float, ask: float) -> float:
    """Execution reference price for signals; never a trading-cost estimate."""
    if ask < bid:
        raise ValueError(f"ask must be >= bid, got bid={bid} ask={ask}")
    return (bid + ask) / 2.0


def ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        # Dukascopy CSV timestamps are GMT/UTC by default; treat naive research
        # timestamps as UTC and keep this assumption visible.
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def session_for_timestamp(timestamp: datetime) -> SessionName:
    """Approximate UTC session tag.

    This intentionally mirrors the audit script's simple UTC buckets. Later
    strategy code should replace this with DST-aware London/New York tagging.
    """

    ts = ensure_utc(timestamp)
    t = ts.time()
    if time(0, 0) <= t < time(7, 0):
        return "asian"
    if time(7, 0) <= t < time(12, 0):
        return "london"
    if time(12, 0) <= t < time(17, 0):
        return "ny_overlap"
    return "off_session"


def contract_ounces(lot_size: float) -> float:
    """Convert MT5 lot size to ounces. 1.00 lot = 100 oz; 0.01 lot = 1 oz."""
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    return lot_size * STD_LOT_OZ


def rollover_nights(
    entry_time: datetime,
    exit_time: Optional[datetime],
    rollover_utc: time = time(22, 0),
) -> int:
    """Count approximate overnight rollovers crossed by a position.

    Swap is broker-specific and usually charged around rollover. This helper is
    a configurable placeholder until real IUX swap and rollover behavior are
    pulled from MT5 Specification.
    """

    if exit_time is None:
        return 0
    entry = ensure_utc(entry_time)
    exit_ = ensure_utc(exit_time)
    if exit_ <= entry:
        return 0

    nights = 0
    cursor = datetime.combine(entry.date(), rollover_utc, tzinfo=timezone.utc)
    if cursor <= entry:
        cursor += timedelta(days=1)
    while cursor < exit_:
        # Skip Saturday/Sunday rollover placeholders. Broker triple-swap rules
        # differ and must be configured after checking IUX specifications.
        if cursor.weekday() < 5:
            nights += 1
        cursor += timedelta(days=1)
    return nights


def swap_cost(
    lot_size: float,
    side: Side = "long",
    overnight_nights: int = 0,
    swap_long_usd_per_lot_night: float = SWAP_LONG_USD_PER_LOT_NIGHT,
    swap_short_usd_per_lot_night: float = SWAP_SHORT_USD_PER_LOT_NIGHT,
) -> float:
    """Return signed swap in USD; negative means cost, positive means credit."""

    if overnight_nights <= 0:
        return 0.0
    rate = swap_long_usd_per_lot_night if side == "long" else swap_short_usd_per_lot_night
    return rate * lot_size * overnight_nights


def is_flagged_news_window(
    timestamp: datetime,
    calendar: Optional[NewsCalendar] = None,
    manual_windows: Optional[Iterable[tuple[datetime, datetime]]] = None,
) -> bool:
    """Hook for future news exclusion/flagging.

    TODO: wire this to a real economic calendar. For now callers may pass manual
    UTC windows or an object implementing NewsCalendar.
    """

    ts = ensure_utc(timestamp)
    if calendar is not None and calendar.is_news_window(ts):
        return True
    if manual_windows is None:
        return False
    for start, end in manual_windows:
        if ensure_utc(start) <= ts <= ensure_utc(end):
            return True
    return False


def applied_cost_breakdown(
    timestamp: datetime,
    lot_size: float,
    *,
    spread_mode: SpreadMode | str = SpreadMode.MEDIAN,
    exit_time: Optional[datetime] = None,
    side: Side = "long",
    overnight_nights: Optional[int] = None,
    spread_table: dict[SessionName, SessionSpread] = IUX_SPREAD_USD_OZ,
    news_calendar: Optional[NewsCalendar] = None,
    manual_news_windows: Optional[Iterable[tuple[datetime, datetime]]] = None,
) -> CostBreakdown:
    """Return the round-trip IUX Standard cost overlay for one trade.

    Spread cost is round-trip for a no-commission Standard account: entering at
    ask and exiting at bid costs approximately one full spread versus mid.
    """

    mode = SpreadMode(spread_mode)
    ts = ensure_utc(timestamp)
    session = session_for_timestamp(ts)
    spread = spread_table[session].value(mode)
    ounces = contract_ounces(lot_size)
    spread_cost = spread * ounces

    nights = rollover_nights(ts, exit_time) if overnight_nights is None else overnight_nights
    signed_swap = swap_cost(lot_size=lot_size, side=side, overnight_nights=nights)
    swap_as_cost = -signed_swap

    return CostBreakdown(
        timestamp=ts,
        session=session,
        lot_size=lot_size,
        contract_oz=ounces,
        spread_mode=mode,
        spread_usd_oz=spread,
        spread_cost_usd=spread_cost,
        swap_cost_usd=swap_as_cost,
        total_round_trip_cost_usd=spread_cost + swap_as_cost,
        news_flagged=is_flagged_news_window(ts, news_calendar, manual_news_windows),
    )


def applied_cost(
    timestamp: datetime,
    lot_size: float,
    *,
    spread_mode: SpreadMode | str = SpreadMode.MEDIAN,
    exit_time: Optional[datetime] = None,
    side: Side = "long",
    overnight_nights: Optional[int] = None,
    spread_table: dict[SessionName, SessionSpread] = IUX_SPREAD_USD_OZ,
    news_calendar: Optional[NewsCalendar] = None,
    manual_news_windows: Optional[Iterable[tuple[datetime, datetime]]] = None,
) -> float:
    """Return round-trip cost in USD for the selected timestamp and lot size."""

    return applied_cost_breakdown(
        timestamp,
        lot_size,
        spread_mode=spread_mode,
        exit_time=exit_time,
        side=side,
        overnight_nights=overnight_nights,
        spread_table=spread_table,
        news_calendar=news_calendar,
        manual_news_windows=manual_news_windows,
    ).total_round_trip_cost_usd


def print_assumptions() -> None:
    print("IUX Standard spread-overlay cost model")
    print("=" * 39)
    print("Signals/reference price: Dukascopy mid = (bid + ask) / 2")
    print("Trading cost: IUX Standard spread overlay, not Dukascopy raw spread")
    print(f"Contract: 1.00 lot = {STD_LOT_OZ:.0f} oz; 0.01 lot = {MICRO_LOT_OZ:.0f} oz")
    print("Commission: $0 on IUX Standard")
    print("Spread table: UNVERIFIED PLACEHOLDER values; replace after IUX spread logging")
    print("\nConfigured overlay ($/oz):")
    for session, spread in IUX_SPREAD_USD_OZ.items():
        print(f"  {session:<11} median={spread.median:.2f} p90={spread.p90:.2f}  {spread.source}")
    print("\nSwap: PLACEHOLDER. Pull real Swap Long/Short from MT5 Specification before using overnight results.")
    print("News: hook only. Feed calendar/manual windows later to flag or exclude high-impact releases.")


if __name__ == "__main__":
    print_assumptions()
    example = applied_cost_breakdown(datetime(2024, 4, 10, 12, 30), lot_size=0.01, spread_mode=SpreadMode.P90)
    print("\nExample 0.01-lot p90 cost at 2024-04-10 12:30 UTC:")
    print(f"  session={example.session} spread={example.spread_usd_oz:.2f}/oz cost=${example.total_round_trip_cost_usd:.2f}")
