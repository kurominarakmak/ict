"""
Breaker Block Phase B v2.

Uses existing breaker detections only. No re-detection and no parameter sweep.

Fixes versus v1:
- Tick-ordered SL/TP path after the locked retest trigger.
- Scale-out at +1R/+2R/+3R with breakeven stop after +1R.
- IUX spread overlay on entry and every partial exit.
- Volatility-proxy news exclusion when no calendar is wired.
- 0.01 minimum lot floor for dollar/equity realism.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "strategies"))
sys.path.insert(0, str(ROOT / "research"))

from breaker_block import Breaker, bool_value
from cost_model import IUX_SPREAD_USD_OZ, SpreadMode, session_for_timestamp
from order_block import Bar, default_tick_path, load_bars, parse_timestamp


REACTION_BARS = 20
TP_GRID = (1.0, 1.5, 2.0, 3.0)
START_EQUITY = 100.0
RISK_FRACTION = 0.01
MIN_LOT = 0.01
CONTRACT_OZ_PER_LOT = 100.0
ENTRY_WEIGHT = 1.0
SCALE_PLAN = ((1.0, 0.50), (2.0, 0.25), (3.0, 0.25))


@dataclass(frozen=True)
class TradeSetup:
    breaker: Breaker
    entry_bar: Bar
    horizon_bar: Bar
    atr_bucket: str
    excluded_news_proxy: bool


@dataclass
class TickTrade:
    setup: TradeSetup
    entry_found: bool = False
    entry_time: Optional[datetime] = None
    entry_mid: Optional[float] = None
    open_weight: float = 1.0
    stop_r: float = -1.0
    gross_r: float = 0.0
    net_r_median: float = 0.0
    net_r_p90: float = 0.0
    hit_1r: bool = False
    hit_2r: bool = False
    hit_3r: bool = False
    be_saved: bool = False
    resolved: bool = False
    exit_time: Optional[datetime] = None
    fills: int = 0

    @property
    def direction(self) -> str:
        return self.setup.breaker.flipped_direction

    @property
    def frozen_atr(self) -> float:
        return self.setup.breaker.frozen_atr

    @property
    def year(self) -> int:
        if self.entry_time is not None:
            return self.entry_time.year
        return self.setup.entry_bar.end.year

    @property
    def atr_bucket(self) -> str:
        return self.setup.atr_bucket


def parse_breakers(path: Path) -> list[Breaker]:
    breakers: list[Breaker] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            breakers.append(
                Breaker(
                    breaker_id=int(row["breaker_id"]),
                    source_ob_id=int(row["source_ob_id"]),
                    source_ob_direction=row["source_ob_direction"],
                    flipped_direction=row["flipped_direction"],
                    zone_high=float(row["zone_high"]),
                    zone_low=float(row["zone_low"]),
                    ob_creation_time=parse_timestamp(row["ob_creation_time"]),
                    break_candle_time=parse_timestamp(row["break_candle_time"]),
                    break_candle_index=int(row["break_candle_index"]),
                    break_close_price=float(row["break_close_price"]),
                    frozen_atr=float(row["frozen_atr"]),
                    displacement_atr=float(row["displacement_atr"]),
                    separation_confirm_time=(
                        parse_timestamp(row["separation_confirm_time"])
                        if row["separation_confirm_time"]
                        else None
                    ),
                    separation_confirm_index=(
                        int(row["separation_confirm_index"])
                        if row["separation_confirm_index"]
                        else None
                    ),
                    retest_time=parse_timestamp(row["retest_time"]) if row["retest_time"] else None,
                    retest_index=int(row["retest_index"]) if row["retest_index"] else None,
                    expired=bool_value(row["expired"]),
                    bars_ob_creation_to_break=int(row["bars_ob_creation_to_break"]),
                    bars_break_to_separation=(
                        int(row["bars_break_to_separation"]) if row["bars_break_to_separation"] else None
                    ),
                    bars_break_to_retest=(
                        int(row["bars_break_to_retest"]) if row["bars_break_to_retest"] else None
                    ),
                    session=row["session"],
                    year=int(row["year"]),
                    segment_id=int(row["segment_id"]),
                )
            )
    return breakers


def quantile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def atr_cutoffs(breakers: list[Breaker]) -> tuple[float, float]:
    values = sorted(item.frozen_atr for item in breakers if item.valid_retest)
    return quantile(values, 1 / 3), quantile(values, 2 / 3)


def atr_bucket(value: float, low_cut: float, mid_cut: float) -> str:
    if value <= low_cut:
        return "low"
    if value <= mid_cut:
        return "medium"
    return "high"


def spread_r(ts: datetime, frozen_atr: float, mode: SpreadMode) -> float:
    spread = IUX_SPREAD_USD_OZ[session_for_timestamp(ts)].value(mode)
    return spread / frozen_atr


def half_spread_r(ts: datetime, frozen_atr: float, mode: SpreadMode) -> float:
    return spread_r(ts, frozen_atr, mode) / 2.0


def touches_entry_zone(mid: float, breaker: Breaker) -> bool:
    if breaker.flipped_direction == "bullish":
        return mid <= breaker.zone_high
    return mid >= breaker.zone_low


def r_from_entry(mid: float, trade: TickTrade) -> float:
    assert trade.entry_mid is not None
    if trade.direction == "bullish":
        return (mid - trade.entry_mid) / trade.frozen_atr
    return (trade.entry_mid - mid) / trade.frozen_atr


def apply_exit_cost(trade: TickTrade, ts: datetime, weight: float) -> None:
    trade.net_r_median -= weight * half_spread_r(ts, trade.frozen_atr, SpreadMode.MEDIAN)
    trade.net_r_p90 -= weight * half_spread_r(ts, trade.frozen_atr, SpreadMode.P90)


def enter_trade(trade: TickTrade, ts: datetime, mid: float) -> None:
    trade.entry_found = True
    trade.entry_time = ts
    trade.entry_mid = mid
    trade.net_r_median -= ENTRY_WEIGHT * half_spread_r(ts, trade.frozen_atr, SpreadMode.MEDIAN)
    trade.net_r_p90 -= ENTRY_WEIGHT * half_spread_r(ts, trade.frozen_atr, SpreadMode.P90)
    trade.fills += 1


def close_weight(trade: TickTrade, ts: datetime, weight: float, r_value: float) -> None:
    trade.gross_r += weight * r_value
    trade.net_r_median += weight * r_value
    trade.net_r_p90 += weight * r_value
    apply_exit_cost(trade, ts, weight)
    trade.open_weight = max(0.0, trade.open_weight - weight)
    trade.fills += 1
    if trade.open_weight <= 1e-12:
        trade.resolved = True
        trade.exit_time = ts


def update_trade_on_tick(trade: TickTrade, ts: datetime, mid: float) -> None:
    if trade.resolved:
        return
    if not trade.entry_found:
        if touches_entry_zone(mid, trade.setup.breaker):
            enter_trade(trade, ts, mid)
        return
    r_now = r_from_entry(mid, trade)

    if r_now <= trade.stop_r:
        if trade.stop_r == 0.0 and trade.open_weight > 0:
            trade.be_saved = True
        close_weight(trade, ts, trade.open_weight, trade.stop_r)
        return

    for target_r, weight in SCALE_PLAN:
        if target_r == 1.0 and trade.hit_1r:
            continue
        if target_r == 2.0 and trade.hit_2r:
            continue
        if target_r == 3.0 and trade.hit_3r:
            continue
        if r_now >= target_r:
            close_weight(trade, ts, weight, target_r)
            if target_r == 1.0:
                trade.hit_1r = True
                trade.stop_r = 0.0
            elif target_r == 2.0:
                trade.hit_2r = True
            elif target_r == 3.0:
                trade.hit_3r = True
            if trade.resolved:
                return


def force_horizon_exit(trade: TickTrade, bars: list[Bar]) -> None:
    if trade.resolved or not trade.entry_found:
        return
    exit_bar = trade.setup.horizon_bar
    r_value = r_from_entry(exit_bar.close, trade)
    close_weight(trade, exit_bar.end, trade.open_weight, r_value)


def parse_tick_row(row: list[str], time_idx: int, bid_idx: int, ask_idx: int) -> Optional[tuple[datetime, float]]:
    try:
        ts = parse_timestamp(row[time_idx])
        bid = float(row[bid_idx])
        ask = float(row[ask_idx])
    except (IndexError, ValueError):
        return None
    if ask <= bid or bid <= 0 or ask <= 0:
        return None
    return ts, (bid + ask) / 2.0


def build_setups(bars: list[Bar], breakers: list[Breaker], low_cut: float, mid_cut: float) -> list[TradeSetup]:
    setups: list[TradeSetup] = []
    for breaker in breakers:
        if not breaker.valid_retest or breaker.retest_index is None:
            continue
        if breaker.retest_index + REACTION_BARS >= len(bars):
            continue
        entry_bar = bars[breaker.retest_index]
        horizon_bar = bars[breaker.retest_index + REACTION_BARS]
        future = bars[breaker.retest_index + 1 : breaker.retest_index + REACTION_BARS + 1]
        if any(bar.segment_id != breaker.segment_id for bar in future):
            continue
        proxy_start = max(0, breaker.retest_index - 2)
        proxy_end = min(len(bars) - 1, breaker.retest_index + 2)
        news_proxy = any(
            bars[i].segment_id == breaker.segment_id
            and (bars[i].high - bars[i].low) > 3.0 * breaker.frozen_atr
            for i in range(proxy_start, proxy_end + 1)
        )
        setups.append(
            TradeSetup(
                breaker=breaker,
                entry_bar=entry_bar,
                horizon_bar=horizon_bar,
                atr_bucket=atr_bucket(breaker.frozen_atr, low_cut, mid_cut),
                excluded_news_proxy=news_proxy,
            )
        )
    return setups


def run_tick_model(tick_path: Path, bars: list[Bar], setups: list[TradeSetup]) -> list[TickTrade]:
    trades = [TickTrade(setup=item) for item in setups if not item.excluded_news_proxy]
    trades.sort(key=lambda trade: trade.setup.entry_bar.start)
    pending = list(trades)
    active: list[TickTrade] = []
    completed: list[TickTrade] = []
    pending_idx = 0

    if not trades:
        return []
    first_needed = min(item.setup.entry_bar.start for item in trades)
    last_needed = max(item.setup.horizon_bar.end for item in trades)

    with tick_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        time_idx = header.index("DateTime")
        bid_idx = header.index("Bid")
        ask_idx = header.index("Ask")
        for row in reader:
            parsed = parse_tick_row(row, time_idx, bid_idx, ask_idx)
            if parsed is None:
                continue
            ts, mid = parsed
            if ts < first_needed:
                continue
            if ts > last_needed:
                break

            while pending_idx < len(pending) and pending[pending_idx].setup.entry_bar.start <= ts:
                active.append(pending[pending_idx])
                pending_idx += 1

            still_active: list[TickTrade] = []
            for trade in active:
                if ts > trade.setup.horizon_bar.end:
                    force_horizon_exit(trade, bars)
                    completed.append(trade)
                    continue
                if ts <= trade.setup.horizon_bar.end:
                    update_trade_on_tick(trade, ts, mid)
                if trade.resolved:
                    completed.append(trade)
                else:
                    still_active.append(trade)
            active = still_active

    for trade in active + pending[pending_idx:]:
        force_horizon_exit(trade, bars)
        if trade.entry_found:
            completed.append(trade)
    completed.sort(key=lambda trade: (trade.entry_time or trade.setup.entry_bar.end, trade.setup.breaker.breaker_id))
    return [trade for trade in completed if trade.entry_found]


def old_simple_outcome(setup: TradeSetup, bars: list[Bar], tp: float) -> tuple[float, bool]:
    breaker = setup.breaker
    reference = breaker.zone_high if breaker.flipped_direction == "bullish" else breaker.zone_low
    future = bars[breaker.retest_index + 1 : breaker.retest_index + REACTION_BARS + 1]  # type: ignore[operator]
    for bar in future:
        if breaker.flipped_direction == "bullish":
            if bar.close <= reference - breaker.frozen_atr:
                return -1.0, True
            if bar.close >= reference + tp * breaker.frozen_atr:
                return tp, True
        else:
            if bar.close >= reference + breaker.frozen_atr:
                return -1.0, True
            if bar.close <= reference - tp * breaker.frozen_atr:
                return tp, True
    exit_close = future[-1].close
    if breaker.flipped_direction == "bullish":
        return (exit_close - reference) / breaker.frozen_atr, False
    return (reference - exit_close) / breaker.frozen_atr, False


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": math.nan, "median": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def max_consecutive_losses(values: list[float]) -> int:
    best = 0
    current = 0
    for value in values:
        if value < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def print_distribution(trades: list[TickTrade]) -> None:
    values = [item.net_r_median for item in trades]
    stats = summarize_values(values)
    print("\nREALIZED_R_DISTRIBUTION_NEW_TICK_SCALEOUT")
    print(
        f"n={stats['n']}, mean_net_R={stats['mean']:.4f}, median_net_R={stats['median']:.4f}, "
        f"std_net_R={stats['std']:.4f}, min_net_R={stats['min']:.4f}, max_net_R={stats['max']:.4f}, "
        f"max_consecutive_losses={max_consecutive_losses(values)}"
    )


def print_group_table(title: str, trades: list[TickTrade], groups: list[str], attr: str) -> None:
    print(f"\n{title}")
    print("group,n,gross_exp_R,net_median_exp_R,net_p90_exp_R,median_net_R,std_net_R,be_saved_rate,unresolved_rate,worst_R,max_consec_losses")
    for group in groups:
        rows = [item for item in trades if str(getattr(item, attr)) == str(group)]
        net = [item.net_r_median for item in rows]
        gross = [item.gross_r for item in rows]
        p90 = [item.net_r_p90 for item in rows]
        if not rows:
            print(f"{group},0,n/a,n/a,n/a,n/a,n/a,n/a,n/a,n/a,n/a")
            continue
        unresolved = sum(1 for item in rows if item.exit_time == item.setup.horizon_bar.end and item.open_weight <= 1e-12) / len(rows)
        print(
            f"{group},{len(rows)},{mean(gross):.4f},{mean(net):.4f},{mean(p90):.4f},"
            f"{median(net):.4f},{pstdev(net) if len(net) > 1 else 0.0:.4f},"
            f"{sum(1 for item in rows if item.be_saved) / len(rows):.2%},"
            f"{unresolved:.2%},{min(net):.4f},{max_consecutive_losses(net)}"
        )


def print_old_vs_new(setups: list[TradeSetup], trades: list[TickTrade], bars: list[Bar]) -> None:
    trade_ids = {item.setup.breaker.breaker_id for item in trades}
    comparable_setups = [item for item in setups if item.breaker.breaker_id in trade_ids and not item.excluded_news_proxy]
    print("\nOLD_CLOSE_MODEL_VS_NEW_TICK_SCALEOUT")
    print("model,tp_R,n,gross_exp_R,net_median_exp_R,unresolved_rate")
    for tp in TP_GRID:
        old_gross: list[float] = []
        old_net: list[float] = []
        unresolved = 0
        for setup in comparable_setups:
            value, resolved = old_simple_outcome(setup, bars, tp)
            old_gross.append(value)
            old_net.append(value - spread_r(setup.entry_bar.end, setup.breaker.frozen_atr, SpreadMode.MEDIAN))
            unresolved += 0 if resolved else 1
        print(f"old_close_binary,{tp:.1f},{len(old_gross)},{mean(old_gross):.4f},{mean(old_net):.4f},{unresolved / len(old_gross):.2%}")
    new_gross = [item.gross_r for item in trades]
    new_net = [item.net_r_median for item in trades]
    unresolved_new = sum(1 for item in trades if item.exit_time == item.setup.horizon_bar.end) / len(trades)
    print(f"new_tick_scaleout,scaleout,{len(trades)},{mean(new_gross):.4f},{mean(new_net):.4f},{unresolved_new:.2%}")


def lot_for_equity(equity: float, frozen_atr: float) -> tuple[float, float, bool]:
    desired = equity * RISK_FRACTION / (frozen_atr * CONTRACT_OZ_PER_LOT)
    lot = max(MIN_LOT, desired)
    actual_risk = lot * CONTRACT_OZ_PER_LOT * frozen_atr / equity
    return lot, actual_risk, desired < MIN_LOT


def print_lot_floor_by_year(trades: list[TickTrade]) -> None:
    print("\nLOT_FLOOR_0_01_BY_YEAR_AT_START_EQUITY")
    print("year,n,forced_min_lot_rate,median_actual_risk_pct,mean_actual_risk_pct,max_actual_risk_pct")
    for year in range(2016, 2027):
        rows = [item for item in trades if item.year == year]
        if not rows:
            print(f"{year},0,n/a,n/a,n/a,n/a")
            continue
        risks = []
        forced = 0
        for item in rows:
            _, risk, is_forced = lot_for_equity(START_EQUITY, item.frozen_atr)
            risks.append(risk * 100)
            forced += int(is_forced)
        print(f"{year},{len(rows)},{forced / len(rows):.2%},{median(risks):.2f}%,{mean(risks):.2f}%,{max(risks):.2f}%")


def print_equity_curve(trades: list[TickTrade]) -> None:
    equity = START_EQUITY
    peak = equity
    max_dd = 0.0
    longest_losses = 0
    current_losses = 0
    year_start: dict[int, float] = {}
    year_end: dict[int, float] = {}
    for trade in sorted(trades, key=lambda item: (item.entry_time or item.setup.entry_bar.end, item.setup.breaker.breaker_id)):
        year_start.setdefault(trade.year, equity)
        lot, _, _ = lot_for_equity(equity, trade.frozen_atr)
        risk_dollars = lot * CONTRACT_OZ_PER_LOT * trade.frozen_atr
        pnl = trade.net_r_median * risk_dollars
        equity += pnl
        year_end[trade.year] = equity
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        if pnl < 0:
            current_losses += 1
            longest_losses = max(longest_losses, current_losses)
        else:
            current_losses = 0

    print("\nILLUSTRATIVE_EQUITY_CURVE_NEW_MODEL_IN_SAMPLE")
    print("Uses tick scale-out net outcomes, $100 start, 0.01 lot floor, chronological order. In-sample only.")
    print(f"final_equity=${equity:.2f}, max_drawdown={max_dd:.2%}, longest_losing_streak={longest_losses}")
    print("year,equity_change")
    for year in range(2016, 2027):
        if year in year_start and year in year_end and year_start[year] > 0:
            print(f"{year},{year_end[year] / year_start[year] - 1.0:.2%}")
        else:
            print(f"{year},n/a")


def expired_survivorship(bars: list[Bar], breakers: list[Breaker], low_cut: float, mid_cut: float) -> None:
    moves = []
    for breaker in breakers:
        if not breaker.expired:
            continue
        if breaker.break_candle_index + REACTION_BARS >= len(bars):
            continue
        future = bars[breaker.break_candle_index + 1 : breaker.break_candle_index + REACTION_BARS + 1]
        if any(bar.segment_id != breaker.segment_id for bar in future):
            continue
        if breaker.flipped_direction == "bullish":
            mfe = (max(bar.high for bar in future) - breaker.zone_high) / breaker.frozen_atr
            close_r = (future[-1].close - breaker.zone_high) / breaker.frozen_atr
        else:
            mfe = (breaker.zone_low - min(bar.low for bar in future)) / breaker.frozen_atr
            close_r = (breaker.zone_low - future[-1].close) / breaker.frozen_atr
        moves.append((breaker, mfe, close_r))
    print("\nSURVIVORSHIP_EXPIRED_BREAKERS")
    print("Expired breakers never gave valid retest entry; movement is unentered break-direction movement after break.")
    if not moves:
        print("n=0")
        return
    print(
        f"n={len(moves)}, median_MFE_R={median([m[1] for m in moves]):.3f}, "
        f"mean_MFE_R={mean([m[1] for m in moves]):.3f}, "
        f"median_20bar_close_R={median([m[2] for m in moves]):.3f}, "
        f"mean_20bar_close_R={mean([m[2] for m in moves]):.3f}"
    )
    for bucket in ("low", "medium", "high"):
        rows = [m for m in moves if atr_bucket(m[0].frozen_atr, low_cut, mid_cut) == bucket]
        if rows:
            print(f"{bucket},n={len(rows)},mean_MFE_R={mean([m[1] for m in rows]):.3f},mean_20bar_close_R={mean([m[2] for m in rows]):.3f}")
        else:
            print(f"{bucket},n=0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=None)
    parser.add_argument("--breakers", type=Path, default=Path("research/breaker_block_breakers.csv"))
    parser.add_argument("--gap-minutes", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tick_path = args.ticks or default_tick_path()
    breakers = parse_breakers(args.breakers)
    low_cut, mid_cut = atr_cutoffs(breakers)
    print("Loading gap-aware M15 bars...", flush=True)
    bars = load_bars(tick_path, gap_minutes=args.gap_minutes)
    setups = build_setups(bars, breakers, low_cut, mid_cut)
    excluded = [item for item in setups if item.excluded_news_proxy]
    print("Running tick-level scale-out model...", flush=True)
    trades = run_tick_model(tick_path, bars, setups)

    print("\nBREAKER_PHASE_B_V2_CONTEXT")
    print(f"total_breakers={len(breakers)}")
    print(f"valid_retest_breakers={sum(1 for item in breakers if item.valid_retest)}")
    print(f"full_window_setups={len(setups)}")
    print(f"news_proxy_excluded={len(excluded)}")
    print(f"tick_measured_trades={len(trades)}")
    print(f"atr_terciles: low<= {low_cut:.6f}, medium<= {mid_cut:.6f}, high> {mid_cut:.6f}")
    print("news_filter_note: no NFP/FOMC/CPI calendar is wired; excluded entries within +/-30 minutes of any M15 bar whose range exceeded 3*frozen_ATR.")
    print("cost_note: spread overlay from cost_model.py applied as half-spread at entry and half-spread on each partial exit; median and p90 are identical with current placeholder table.")

    print_distribution(trades)
    print_group_table("NEW_MODEL_BY_ATR_REGIME", trades, ["low", "medium", "high"], "atr_bucket")
    print_group_table("NEW_MODEL_BY_YEAR", trades, [str(year) for year in range(2016, 2027)], "year")
    print_old_vs_new(setups, trades, bars)
    print_lot_floor_by_year(trades)
    print_equity_curve(trades)
    print("\nBREAKEVEN_STOP_SAVED_TRADES")
    print(f"count={sum(1 for item in trades if item.be_saved)}, rate={sum(1 for item in trades if item.be_saved) / len(trades):.2%}")
    print(f"worst_single_trade_realized_R={min(item.net_r_median for item in trades):.4f}")
    expired_survivorship(bars, breakers, low_cut, mid_cut)


if __name__ == "__main__":
    main()
