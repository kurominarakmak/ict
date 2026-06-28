"""
Breaker Block Phase B measurement.

Consumes existing breaker detections only. Does not re-detect or tune breakers.

Measures:
- Close-based RR grid after valid retest.
- Hard-matched deterministic baseline.
- Gross and IUX-spread net expectancy in R.
- ATR-regime and yearly breakdowns.
- Expired breaker survivorship diagnostics.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
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
BASELINE_K = 5
TP_GRID = (1.0, 1.5, 2.0, 3.0)
START_EQUITY = 100.0
RISK_FRACTION = 0.01
CONTRACT_OZ_PER_LOT = 100.0


@dataclass(frozen=True)
class TradeMeasurement:
    key: str
    kind: str
    direction: str
    entry_index: int
    entry_time: datetime
    reference_price: float
    frozen_atr: float
    displacement_atr: float
    session: str
    year: int
    atr_bucket: str
    gross_r: dict[float, float]
    median_net_r: dict[float, float]
    p90_net_r: dict[float, float]
    resolved: dict[float, bool]
    win: dict[float, bool]
    cost_r_median: float
    cost_r_p90: float
    slippage_event: bool
    slippage_extra_loss_r: float
    worst_slippage_r: float
    news_proxy_overlap: bool


@dataclass(frozen=True)
class BaselineCandidate:
    candidate_id: int
    source_bar_index: int
    direction: str
    zone_high: float
    zone_low: float
    frozen_atr: float
    displacement_bucket: str
    session: str
    volatility_bucket: str
    trigger_index: int
    trigger_time: datetime
    owner_breaker_id: int


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


def ranges_overlap(high_a: float, low_a: float, high_b: float, low_b: float) -> bool:
    return low_a <= high_b and low_b <= high_a


def displacement_bucket(value: float) -> str:
    if value < 3.0:
        return "2_to_3"
    if value < 5.0:
        return "3_to_5"
    return "5_plus"


def volatility_bucket(frozen_atr: float, reference_price: float) -> str:
    ratio = frozen_atr / reference_price
    if ratio < 0.0010:
        return "low"
    if ratio < 0.0020:
        return "medium"
    return "high"


def atr_tercile_cutoffs(breakers: list[Breaker]) -> tuple[float, float]:
    values = sorted(item.frozen_atr for item in breakers if item.valid_retest)
    if not values:
        return math.nan, math.nan

    def q(pct: float) -> float:
        index = (len(values) - 1) * pct
        lo = math.floor(index)
        hi = math.ceil(index)
        if lo == hi:
            return values[lo]
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return q(1 / 3), q(2 / 3)


def atr_bucket(value: float, low_cut: float, mid_cut: float) -> str:
    if value <= low_cut:
        return "low"
    if value <= mid_cut:
        return "medium"
    return "high"


def first_touch(bars: list[Bar], high: float, low: float, after_index: int, segment_id: int) -> Optional[int]:
    for i in range(after_index + 1, len(bars)):
        bar = bars[i]
        if bar.segment_id != segment_id:
            return None
        if bar.low <= high and bar.high >= low:
            return i
    return None


def cost_r(timestamp: datetime, frozen_atr: float, mode: SpreadMode) -> float:
    session = session_for_timestamp(timestamp)
    spread = IUX_SPREAD_USD_OZ[session].value(mode)
    return spread / frozen_atr


def measure_trade(
    bars: list[Bar],
    *,
    key: str,
    kind: str,
    direction: str,
    entry_index: int,
    zone_high: float,
    zone_low: float,
    frozen_atr: float,
    displacement_atr: float,
    atr_bucket_name: str,
) -> Optional[TradeMeasurement]:
    if entry_index + REACTION_BARS >= len(bars):
        return None
    entry_bar = bars[entry_index]
    future = bars[entry_index + 1 : entry_index + REACTION_BARS + 1]
    if any(bar.segment_id != entry_bar.segment_id for bar in future):
        return None

    reference = zone_high if direction == "bullish" else zone_low
    med_cost = cost_r(entry_bar.end, frozen_atr, SpreadMode.MEDIAN)
    p90_cost = cost_r(entry_bar.end, frozen_atr, SpreadMode.P90)
    gross: dict[float, float] = {}
    median_net: dict[float, float] = {}
    p90_net: dict[float, float] = {}
    resolved: dict[float, bool] = {}
    win: dict[float, bool] = {}
    news_proxy = False
    slippage_event = False
    slippage_extra_loss = 0.0
    worst_slippage_r = 0.0

    for bar in future:
        if direction == "bullish":
            adverse_r = (reference - bar.low) / frozen_atr
        else:
            adverse_r = (bar.high - reference) / frozen_atr
        if adverse_r > 2.0:
            news_proxy = True
            slippage_event = True
            extra = adverse_r - 1.0
            if extra > slippage_extra_loss:
                slippage_extra_loss = extra
                worst_slippage_r = -adverse_r

    for tp in TP_GRID:
        outcome: Optional[float] = None
        did_win = False
        did_resolve = False
        for bar in future:
            if direction == "bullish":
                tp_hit = bar.close >= reference + tp * frozen_atr
                sl_hit = bar.close <= reference - frozen_atr
            else:
                tp_hit = bar.close <= reference - tp * frozen_atr
                sl_hit = bar.close >= reference + frozen_atr
            if sl_hit and tp_hit:
                # Close cannot realistically be both unless frozen_atr/TP are pathological.
                outcome = -1.0
                did_resolve = True
                break
            if sl_hit:
                outcome = -1.0
                did_resolve = True
                break
            if tp_hit:
                outcome = tp
                did_win = True
                did_resolve = True
                break
        if outcome is None:
            exit_close = future[-1].close
            if direction == "bullish":
                outcome = (exit_close - reference) / frozen_atr
            else:
                outcome = (reference - exit_close) / frozen_atr
        gross[tp] = outcome
        median_net[tp] = outcome - med_cost
        p90_net[tp] = outcome - p90_cost
        resolved[tp] = did_resolve
        win[tp] = did_win

    return TradeMeasurement(
        key=key,
        kind=kind,
        direction=direction,
        entry_index=entry_index,
        entry_time=entry_bar.end,
        reference_price=reference,
        frozen_atr=frozen_atr,
        displacement_atr=displacement_atr,
        session=session_for_timestamp(entry_bar.end),
        year=entry_bar.end.year,
        atr_bucket=atr_bucket_name,
        gross_r=gross,
        median_net_r=median_net,
        p90_net_r=p90_net,
        resolved=resolved,
        win=win,
        cost_r_median=med_cost,
        cost_r_p90=p90_cost,
        slippage_event=slippage_event,
        slippage_extra_loss_r=slippage_extra_loss,
        worst_slippage_r=worst_slippage_r,
        news_proxy_overlap=news_proxy,
    )


def breaker_zones_by_break(breakers: list[Breaker]) -> tuple[list[int], list[list[Breaker]]]:
    known: list[Breaker] = []
    indexes: list[int] = []
    snapshots: list[list[Breaker]] = []
    for item in sorted(breakers, key=lambda b: b.break_candle_index):
        known.append(item)
        indexes.append(item.break_candle_index)
        snapshots.append(list(known))
    return indexes, snapshots


def is_non_breaker_range(bar: Bar, known_breakers: list[Breaker]) -> bool:
    for breaker in known_breakers:
        if ranges_overlap(bar.high, bar.low, breaker.zone_high, breaker.zone_low):
            return False
    return True


def build_baseline_candidates(bars: list[Bar], breakers: list[Breaker]) -> list[BaselineCandidate]:
    candidates: list[BaselineCandidate] = []
    break_indexes, known_snapshots = breaker_zones_by_break(breakers)
    for breaker in breakers:
        known_index = bisect.bisect_right(break_indexes, breaker.break_candle_index) - 1
        known_breakers = known_snapshots[known_index] if known_index >= 0 else []
        disp_bucket = displacement_bucket(breaker.displacement_atr)
        vol_bucket = volatility_bucket(breaker.frozen_atr, breaker.break_close_price)
        for source_idx in range(breaker.break_candle_index - 1, -1, -1):
            source = bars[source_idx]
            if source.segment_id != breaker.segment_id:
                break
            if not is_non_breaker_range(source, known_breakers):
                continue
            trigger_index = first_touch(
                bars=bars,
                high=source.high,
                low=source.low,
                after_index=breaker.break_candle_index,
                segment_id=breaker.segment_id,
            )
            if trigger_index is None:
                continue
            candidates.append(
                BaselineCandidate(
                    candidate_id=len(candidates) + 1,
                    source_bar_index=source.index,
                    direction=breaker.flipped_direction,
                    zone_high=source.high,
                    zone_low=source.low,
                    frozen_atr=breaker.frozen_atr,
                    displacement_bucket=disp_bucket,
                    session=breaker.session,
                    volatility_bucket=vol_bucket,
                    trigger_index=trigger_index,
                    trigger_time=bars[trigger_index].end,
                    owner_breaker_id=breaker.breaker_id,
                )
            )
    candidates.sort(key=lambda item: (item.trigger_index, item.source_bar_index, item.candidate_id))
    return candidates


def select_baselines(
    breaker: Breaker,
    candidates: list[BaselineCandidate],
    bars: list[Bar],
) -> list[BaselineCandidate]:
    if breaker.retest_index is None:
        return []
    disp_bucket = displacement_bucket(breaker.displacement_atr)
    vol_bucket = volatility_bucket(breaker.frozen_atr, breaker.break_close_price)
    qualified = [
        candidate
        for candidate in candidates
        if candidate.trigger_index < breaker.retest_index
        and candidate.direction == breaker.flipped_direction
        and candidate.displacement_bucket == disp_bucket
        and candidate.session == breaker.session
        and candidate.volatility_bucket == vol_bucket
    ]
    qualified.sort(key=lambda item: (-item.trigger_index, item.source_bar_index, item.candidate_id))
    return qualified[:BASELINE_K]


def summarize(samples: list[TradeMeasurement], value_attr: str, tp: float) -> dict[str, float]:
    values = [getattr(item, value_attr)[tp] for item in samples]
    if not values:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "win_rate": math.nan,
            "unresolved": math.nan,
            "max_consec_losses": 0,
        }
    wins = sum(1 for item in samples if item.win[tp])
    unresolved = sum(1 for item in samples if not item.resolved[tp])
    max_losses = max_consecutive_losses(values)
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "win_rate": wins / len(values),
        "unresolved": unresolved / len(values),
        "max_consec_losses": max_losses,
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


def print_rr_table(title: str, ob: list[TradeMeasurement], base: list[TradeMeasurement]) -> None:
    print(f"\n{title}")
    print(
        "kind,tp_R,n,win_rate,unresolved,gross_exp_R,net_median_exp_R,"
        "net_p90_exp_R,std_gross_R,max_consec_losses_gross"
    )
    for kind, samples in (("breaker", ob), ("baseline", base)):
        for tp in TP_GRID:
            gross = summarize(samples, "gross_r", tp)
            med = summarize(samples, "median_net_r", tp)
            p90 = summarize(samples, "p90_net_r", tp)
            print(
                f"{kind},{tp:.1f},{gross['n']},"
                f"{gross['win_rate']:.2%},{gross['unresolved']:.2%},"
                f"{gross['mean']:.4f},{med['mean']:.4f},{p90['mean']:.4f},"
                f"{gross['std']:.4f},{gross['max_consec_losses']}"
            )


def print_breaker_only_table(title: str, samples: list[TradeMeasurement]) -> None:
    print(f"\n{title}")
    print(
        "bucket,tp_R,n,win_rate,unresolved,gross_exp_R,net_median_exp_R,"
        "net_p90_exp_R,std_gross_R,max_consec_losses_gross"
    )
    for bucket, rows in samples:
        for tp in TP_GRID:
            gross = summarize(rows, "gross_r", tp)
            med = summarize(rows, "median_net_r", tp)
            p90 = summarize(rows, "p90_net_r", tp)
            print(
                f"{bucket},{tp:.1f},{gross['n']},"
                f"{gross['win_rate']:.2%},{gross['unresolved']:.2%},"
                f"{gross['mean']:.4f},{med['mean']:.4f},{p90['mean']:.4f},"
                f"{gross['std']:.4f},{gross['max_consec_losses']}"
            )


def print_group_pair_table(
    title: str,
    groups: list[str],
    breaker_samples: list[TradeMeasurement],
    baseline_samples: list[TradeMeasurement],
    attr: str,
) -> None:
    print(f"\n{title}")
    print(
        "group,kind,tp_R,n,win_rate,unresolved,gross_exp_R,net_median_exp_R,"
        "net_p90_exp_R,std_gross_R,max_consec_losses_gross"
    )
    for group in groups:
        for kind, all_rows in (("breaker", breaker_samples), ("baseline", baseline_samples)):
            rows = [item for item in all_rows if str(getattr(item, attr)) == str(group)]
            for tp in TP_GRID:
                gross = summarize(rows, "gross_r", tp)
                med = summarize(rows, "median_net_r", tp)
                p90 = summarize(rows, "p90_net_r", tp)
                print(
                    f"{group},{kind},{tp:.1f},{gross['n']},"
                    f"{gross['win_rate']:.2%},{gross['unresolved']:.2%},"
                    f"{gross['mean']:.4f},{med['mean']:.4f},{p90['mean']:.4f},"
                    f"{gross['std']:.4f},{gross['max_consec_losses']}"
                )


def lot_size(equity: float, frozen_atr: float) -> float:
    risk_dollars = equity * RISK_FRACTION
    return risk_dollars / (frozen_atr * CONTRACT_OZ_PER_LOT)


def print_lot_and_cost_tables(samples: list[TradeMeasurement]) -> None:
    print("\nLOT_SIZE_DISTRIBUTION_BY_YEAR")
    print(f"Assumption: equity=${START_EQUITY:.2f}, risk={RISK_FRACTION:.2%}; lots scale linearly with equity.")
    print("year,n,lot_min,lot_median,lot_mean,lot_p90,lot_max")
    for year in range(2016, 2027):
        rows = [item for item in samples if item.year == year]
        lots = [lot_size(START_EQUITY, item.frozen_atr) for item in rows]
        if not lots:
            print(f"{year},0,n/a,n/a,n/a,n/a,n/a")
            continue
        print(
            f"{year},{len(lots)},{min(lots):.5f},{median(lots):.5f},"
            f"{mean(lots):.5f},{quantile(lots, 0.90):.5f},{max(lots):.5f}"
        )

    print("\nCOST_AS_PERCENT_OF_RISK_BY_YEAR")
    print("year,n,median_cost_pct_risk_mean,median_cost_pct_risk_median,p90_cost_pct_risk_mean,p90_cost_pct_risk_median")
    for year in range(2016, 2027):
        rows = [item for item in samples if item.year == year]
        if not rows:
            print(f"{year},0,n/a,n/a,n/a,n/a")
            continue
        median_costs = [item.cost_r_median * 100 for item in rows]
        p90_costs = [item.cost_r_p90 * 100 for item in rows]
        print(
            f"{year},{len(rows)},{mean(median_costs):.2f}%,{median(median_costs):.2f}%,"
            f"{mean(p90_costs):.2f}%,{median(p90_costs):.2f}%"
        )


def best_tp_by_median_net(samples: list[TradeMeasurement]) -> float:
    scores = {tp: summarize(samples, "median_net_r", tp)["mean"] for tp in TP_GRID}
    return max(TP_GRID, key=lambda tp: scores[tp])


def equity_curve(samples: list[TradeMeasurement], tp: float) -> tuple[float, float, int, dict[int, float]]:
    equity = START_EQUITY
    peak = START_EQUITY
    max_dd = 0.0
    longest_losses = 0
    current_losses = 0
    year_start: dict[int, float] = {}
    year_end: dict[int, float] = {}
    for item in sorted(samples, key=lambda row: (row.entry_index, row.key)):
        year_start.setdefault(item.year, equity)
        r = item.median_net_r[tp]
        equity *= 1.0 + RISK_FRACTION * r
        year_end[item.year] = equity
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        if r < 0:
            current_losses += 1
            longest_losses = max(longest_losses, current_losses)
        else:
            current_losses = 0
    yearly = {
        year: (year_end[year] / year_start[year] - 1.0)
        for year in sorted(year_start)
        if year in year_end and year_start[year] > 0
    }
    return equity, max_dd, longest_losses, yearly


def print_equity_curve(samples: list[TradeMeasurement]) -> None:
    tp = best_tp_by_median_net(samples)
    final_equity, max_dd, longest_losses, yearly = equity_curve(samples, tp)
    print("\nILLUSTRATIVE_IN_SAMPLE_EQUITY_CURVE")
    print("Uses the best-looking overall TP cell by median-net expectancy, in-sample only. Not permission to trade.")
    print(f"tp_R={tp:.1f}, start_equity=${START_EQUITY:.2f}, risk_per_trade={RISK_FRACTION:.2%}")
    print(f"final_equity=${final_equity:.2f}, max_drawdown={max_dd:.2%}, longest_losing_streak={longest_losses}")
    print("year,equity_change")
    for year in range(2016, 2027):
        value = yearly.get(year)
        print(f"{year},{value:.2%}" if value is not None else f"{year},n/a")


def print_tail_tables(samples: list[TradeMeasurement]) -> None:
    print("\nSLIPPAGE_AND_NEWS_PROXY_BY_ATR")
    print("bucket,n,slippage_event_rate,mean_extra_loss_R_if_event,max_extra_loss_R,news_proxy_overlap_rate,worst_single_trade_R")
    for bucket in ("low", "medium", "high"):
        rows = [item for item in samples if item.atr_bucket == bucket]
        print_tail_row(bucket, rows)

    print("\nSLIPPAGE_AND_NEWS_PROXY_BY_YEAR")
    print("year,n,slippage_event_rate,mean_extra_loss_R_if_event,max_extra_loss_R,news_proxy_overlap_rate,worst_single_trade_R,worst_single_trade_dollars_at_1pct")
    for year in range(2016, 2027):
        rows = [item for item in samples if item.year == year]
        print_tail_row(str(year), rows, include_dollars=True)


def print_tail_row(label: str, rows: list[TradeMeasurement], include_dollars: bool = False) -> None:
    if not rows:
        suffix = ",n/a" if include_dollars else ""
        print(f"{label},0,n/a,n/a,n/a,n/a,n/a{suffix}")
        return
    events = [item for item in rows if item.slippage_event]
    extras = [item.slippage_extra_loss_r for item in events]
    worst_r = min([min(item.median_net_r[tp] for tp in TP_GRID) for item in rows] + [item.worst_slippage_r for item in rows])
    event_rate = len(events) / len(rows)
    news_rate = sum(1 for item in rows if item.news_proxy_overlap) / len(rows)
    mean_extra = mean(extras) if extras else 0.0
    max_extra = max(extras) if extras else 0.0
    if include_dollars:
        worst_dollars = START_EQUITY * RISK_FRACTION * worst_r
        print(f"{label},{len(rows)},{event_rate:.2%},{mean_extra:.4f},{max_extra:.4f},{news_rate:.2%},{worst_r:.4f},{worst_dollars:.2f}")
    else:
        print(f"{label},{len(rows)},{event_rate:.2%},{mean_extra:.4f},{max_extra:.4f},{news_rate:.2%},{worst_r:.4f}")


def expired_survivorship(bars: list[Bar], breakers: list[Breaker], low_cut: float, mid_cut: float) -> None:
    rows = [item for item in breakers if item.expired]
    moves: list[tuple[Breaker, float, float]] = []
    for breaker in rows:
        start = breaker.break_candle_index
        if start + REACTION_BARS >= len(bars):
            continue
        future = bars[start + 1 : start + REACTION_BARS + 1]
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
    print("Expired breakers did not get a valid retest entry within 50 bars; values below measure unentered break-direction movement after break.")
    if not moves:
        print("n=0")
        return
    mfes = [item[1] for item in moves]
    closes = [item[2] for item in moves]
    print(
        f"n={len(moves)}, median_MFE_R={median(mfes):.3f}, mean_MFE_R={mean(mfes):.3f}, "
        f"median_20bar_close_R={median(closes):.3f}, mean_20bar_close_R={mean(closes):.3f}"
    )
    counts = Counter(atr_bucket(item[0].frozen_atr, low_cut, mid_cut) for item in moves)
    for bucket in ("low", "medium", "high"):
        bucket_moves = [item for item in moves if atr_bucket(item[0].frozen_atr, low_cut, mid_cut) == bucket]
        if not bucket_moves:
            print(f"{bucket}: n=0")
            continue
        print(
            f"{bucket}: n={counts[bucket]}, mean_MFE_R={mean(item[1] for item in bucket_moves):.3f}, "
            f"mean_20bar_close_R={mean(item[2] for item in bucket_moves):.3f}"
        )


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
    valid = [item for item in breakers if item.valid_retest and item.retest_index is not None]
    low_cut, mid_cut = atr_tercile_cutoffs(breakers)

    print("Loading gap-aware M15 bars...", flush=True)
    bars = load_bars(tick_path, gap_minutes=args.gap_minutes)
    print("Building matched baseline candidates...", flush=True)
    baseline_candidates = build_baseline_candidates(bars, breakers)

    breaker_samples: list[TradeMeasurement] = []
    baseline_samples: list[TradeMeasurement] = []
    under_matched = 0
    for breaker in valid:
        bucket = atr_bucket(breaker.frozen_atr, low_cut, mid_cut)
        measured = measure_trade(
            bars,
            key=str(breaker.breaker_id),
            kind="breaker",
            direction=breaker.flipped_direction,
            entry_index=breaker.retest_index,
            zone_high=breaker.zone_high,
            zone_low=breaker.zone_low,
            frozen_atr=breaker.frozen_atr,
            displacement_atr=breaker.displacement_atr,
            atr_bucket_name=bucket,
        )
        if measured is not None:
            breaker_samples.append(measured)
        selected = select_baselines(breaker, baseline_candidates, bars)
        if len(selected) < BASELINE_K:
            under_matched += 1
        for candidate in selected:
            baseline_measured = measure_trade(
                bars,
                key=f"{breaker.breaker_id}:{candidate.candidate_id}",
                kind="baseline",
                direction=candidate.direction,
                entry_index=candidate.trigger_index,
                zone_high=candidate.zone_high,
                zone_low=candidate.zone_low,
                frozen_atr=candidate.frozen_atr,
                displacement_atr=breaker.displacement_atr,
                atr_bucket_name=bucket,
            )
            if baseline_measured is not None:
                baseline_samples.append(baseline_measured)

    print("\nBREAKER_PHASE_B_CONTEXT")
    print(f"total_breakers={len(breakers)}")
    print(f"valid_retest_breakers={len(valid)}")
    print(f"measured_breaker_trades={len(breaker_samples)}")
    print(f"baseline_candidates={len(baseline_candidates)}")
    print(f"measured_baseline_trades={len(baseline_samples)}")
    print(f"under_matched_breakers={under_matched}")
    print(f"atr_terciles: low<= {low_cut:.6f}, medium<= {mid_cut:.6f}, high> {mid_cut:.6f}")
    print("cost_note: net_R = gross_R - IUX spread_USD_per_oz / frozen_ATR; spread values are cost_model.py placeholders until replaced with measured IUX logger values.")
    print("entry_note: retest trigger bar close timestamp is used for session spread; TP/SL measured on 20 completed bars after retest, excluding retest bar; unresolved exits at 20th-bar close.")

    print_rr_table("OVERALL_RR_GRID_GROSS_AND_NET", breaker_samples, baseline_samples)

    print_group_pair_table(
        "NET_BY_ATR_REGIME_BREAKER_AND_BASELINE",
        ["low", "medium", "high"],
        breaker_samples,
        baseline_samples,
        "atr_bucket",
    )

    print_group_pair_table(
        "NET_BY_YEAR_BREAKER_AND_BASELINE",
        [str(year) for year in range(2016, 2027)],
        breaker_samples,
        baseline_samples,
        "year",
    )

    print_lot_and_cost_tables(breaker_samples)
    print_equity_curve(breaker_samples)
    print_tail_tables(breaker_samples)

    expired_survivorship(bars, breakers, low_cut, mid_cut)


if __name__ == "__main__":
    main()
