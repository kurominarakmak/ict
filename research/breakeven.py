"""
Gate 0: Cost-vs-Edge Breakeven Analysis for XAUUSD
Broker: IUX — Standard account (no commission; spread is the entire visible cost)

PURPOSE
-------
Before writing a single strategy line, prove that your edge can survive the cost
structure. A strategy with a raw edge smaller than round-trip cost has zero chance
of being profitable — no engineering, no ML, no luck fixes that.

This version is calibrated to an IUX Standard account. Two CRITICAL properties of
that account type:
  1. Commission = $0. The spread is the only explicit friction.
  2. Because commission is zero, spread IS everything. A wrong spread estimate is a
     wrong model, full stop. Do not advance to Gate 1 using the PLACEHOLDER numbers
     below — run data/spread_logger.py first to replace them with measured values.

HOW TO USE
----------
    python3 research/breakeven.py

HOW TO UPDATE SPREAD INPUTS
----------------------------
  1. Run data/spread_logger.py for at least 5 full trading days.
  2. Look at the per-session summary it prints (mean / median / P90 / max).
  3. Replace SPREAD_QUIET_USD_OZ with the London/NY-overlap median.
  4. Replace SPREAD_WIDE_USD_OZ with the Asian/off-session median (or the P90
     of the quiet session if you want a conservative estimate).
  5. Re-run this script and check whether the verdict changes.

HOW TO GET THE REAL SWAP RATE
------------------------------
  MT5 → Market Watch → right-click XAUUSD → Specification → scroll to Swap Long
  and Swap Short. They are quoted in currency per lot (USD for XAUUSD).
  Replace SWAP_LONG_PER_NIGHT and SWAP_SHORT_PER_NIGHT below.

XAUUSD MARKET STRUCTURE — ASSUMPTIONS & SOURCES
-------------------------------------------------
All per-lot figures below use a standard lot = 100 troy ounces.
For 0.01 (micro) lots = 1 oz, divide all dollar costs by 100.

1. SPREAD (IUX Standard — markup baked into the quote)
   ⚠ PLACEHOLDER — replace after running spread_logger.py
   - "from 0.2 pip" is IUX's headline for EURUSD at peak London; gold is wider.
   - Retail Standard gold spread typically runs $0.30–$0.60/oz at active sessions,
     wider still in Asian hours and around economic releases.
   - Quiet/active session estimate:  $0.35/oz  ← UNVERIFIED PLACEHOLDER
   - Wide/off-session estimate:      $0.55/oz  ← UNVERIFIED PLACEHOLDER
   - News spikes: $1.00–$3.00+/oz; kept from v1 since they apply to all account types.

2. COMMISSION
   - $0.00 — Standard account has no per-lot commission.

3. SLIPPAGE
   - Standard account is modeled as SPREAD-ONLY for the primary analysis. Slippage
     still exists technically (market orders fill at the ask/bid at fill time, which
     can differ from the ask/bid at signal time by a few cents/oz), but on an ECN-lite
     Standard account during liquid hours it is typically absorbed into the spread quote
     and is not a separately visible line item.
   - Slippage will be measured later via data/spread_logger.py's realized-slippage
     extension once we have real trade fills to compare against quoted prices.
   - For the NEWS scenario, slippage remains a dominant cost and is modeled explicitly.

4. SWAP (overnight financing) — applies only to positions held past rollover (usually
   22:00 broker time / ~17:00 NY time)
   ⚠ PLACEHOLDER — pull real values from MT5 specification page
   - Gold long swap is almost always negative (you pay to hold long gold overnight).
   - Gold short swap varies by broker: can be slightly positive or also negative.
   - Placeholder values below are representative of retail brokers, NOT confirmed IUX.

BREAKEVEN MATH (unchanged from v1)
------------------------------------
  EV = win_rate × avg_win – (1 – win_rate) × avg_loss – RTC = 0
  With avg_win = RR × avg_loss:
    win_rate_BE = (1 + RTC / avg_loss) / (RR + 1)
  Zero-cost baseline: win_rate_0 = 1 / (RR + 1)
  Cost premium:       Δ = RTC / [avg_loss × (RR + 1)]

BROKER COMPARISON (why this matters for account choice)
---------------------------------------------------------
  Standard is cheaper than Raw ECN only when:
    spread_std < spread_raw + (commission_raw + slippage_raw) / LOT_SIZE_OZ
  Using representative Raw ECN numbers ($0.22/oz, $7 comm, $3 slip):
    crossover = $0.22 + $10 / 100 = $0.32/oz
  IUX Standard at $0.35/oz (PLACEHOLDER) is ABOVE the crossover → Raw ECN is
  marginally cheaper in quiet hours by ~$3/lot ($0.03 per 0.01 lot per trade).
  If your measured spread is below $0.32/oz, Standard is the better deal.
  This crossover is printed explicitly in the comparison section below.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# CONTRACT SIZE
# ---------------------------------------------------------------------------

STD_LOT_OZ: float = 100.0      # troy oz per standard lot (exchange convention)
MICRO_LOT_OZ: float = 1.0      # 0.01 lot = 1 oz (user's trading size)
GOLD_PRICE: float = 2350.0     # USD/oz — update this periodically


# ---------------------------------------------------------------------------
# IUX STANDARD ACCOUNT — PRIMARY COST INPUTS
# Replace PLACEHOLDER values after running data/spread_logger.py
# ---------------------------------------------------------------------------

# ⚠ PLACEHOLDER — update from measured spread_logger.py output
SPREAD_QUIET_USD_OZ: float = 0.35   # London / NY-overlap median (UNVERIFIED)
SPREAD_WIDE_USD_OZ: float  = 0.55   # Asian / off-session median (UNVERIFIED)

COMMISSION_PER_SIDE_USD: float = 0.0  # Standard account: NO commission

# ⚠ PLACEHOLDER — pull from MT5 > XAUUSD Specification > Swap Long / Swap Short
SWAP_LONG_PER_NIGHT:  float = -6.80  # USD/lot/night, long  (UNVERIFIED)
SWAP_SHORT_PER_NIGHT: float = -3.20  # USD/lot/night, short (UNVERIFIED)

# Slippage on Standard: modeled as $0 for primary analysis (see module docstring).
# The news scenario overrides this with a realistic execution gap.
SLIPPAGE_STANDARD_RT_USD_LOT: float = 0.0

# News scenario (applies regardless of account type — execution gap dominates)
SPREAD_NEWS_USD_OZ: float          = 1.50   # conservative news spike spread
SLIPPAGE_NEWS_PER_SIDE_USD_OZ: float = 10.0  # $10/oz per side = $1,000/lot per side


# ---------------------------------------------------------------------------
# ANALYSIS GRID
# ---------------------------------------------------------------------------

RR_RATIOS: Sequence[float]       = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
SL_SIZES_USD_LOT: Sequence[float] = [50, 100, 150, 200, 300, 500]
# SL in price terms: $100/lot = $1.00/oz move; $500/lot = $5.00/oz move
TRADE_COUNTS_PER_DAY: Sequence[int] = [1, 2, 4, 8, 16, 48]


# ---------------------------------------------------------------------------
# BROKER PROFILES (for comparison section)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BrokerProfile:
    name: str
    spread_quiet_usd_oz: float
    commission_rt_per_lot: float   # round-trip total
    slippage_rt_per_lot: float     # round-trip total
    note: str = ""

    def rtc_quiet(self) -> float:
        return self.spread_quiet_usd_oz * STD_LOT_OZ + self.commission_rt_per_lot + self.slippage_rt_per_lot

    def rtc_micro(self) -> float:
        """Round-trip cost for a 0.01 (micro) lot."""
        return self.rtc_quiet() * (MICRO_LOT_OZ / STD_LOT_OZ)

    def crossover_spread(self, other: "BrokerProfile") -> float:
        """
        Spread ($/oz) at which self and other are equally expensive.
        Below this: self is cheaper. Above: other is cheaper.
        """
        # self_spread × 100 + self_fixed = other_spread × 100 + other_fixed
        # crossover_spread = other_spread + (other_fixed – self_fixed) / 100
        self_fixed  = self.commission_rt_per_lot  + self.slippage_rt_per_lot
        other_fixed = other.commission_rt_per_lot + other.slippage_rt_per_lot
        return other.spread_quiet_usd_oz + (other_fixed - self_fixed) / STD_LOT_OZ


PROFILE_IUX_STANDARD = BrokerProfile(
    name="IUX Standard (PLACEHOLDER spread)",
    spread_quiet_usd_oz=SPREAD_QUIET_USD_OZ,
    commission_rt_per_lot=0.0,
    slippage_rt_per_lot=0.0,
    note="⚠ UNVERIFIED — replace spread after running spread_logger.py",
)

PROFILE_RAW_ECN = BrokerProfile(
    name="Raw ECN (representative)",
    spread_quiet_usd_oz=0.22,    # midpoint of typical 0.20–0.25 range
    commission_rt_per_lot=7.00,  # $3.50/side, two sides
    slippage_rt_per_lot=3.00,
    note="Composite of Tickmill / IC Markets / FP Markets pricing",
)

PROFILE_ORIGINAL_PLACEHOLDER = BrokerProfile(
    name="Original v1 placeholder",
    spread_quiet_usd_oz=0.20,
    commission_rt_per_lot=7.00,
    slippage_rt_per_lot=3.00,
    note="Gate 0 v1 baseline (generic ECN estimates)",
)

ALL_PROFILES = [PROFILE_IUX_STANDARD, PROFILE_RAW_ECN, PROFILE_ORIGINAL_PLACEHOLDER]


# ---------------------------------------------------------------------------
# COST SCENARIO
# ---------------------------------------------------------------------------

@dataclass
class CostScenario:
    name: str
    spread_usd_oz: float
    commission_rt_usd: float
    slippage_rt_usd: float           # round-trip, standard lot
    swap_per_night_usd: float = 0.0
    overnight_holds: int = 0

    def rtc(self, lot_fraction: float = 1.0) -> float:
        spread_cost = self.spread_usd_oz * STD_LOT_OZ
        swap_cost   = abs(self.swap_per_night_usd) * self.overnight_holds
        total_lot   = spread_cost + self.commission_rt_usd + self.slippage_rt_usd + swap_cost
        return total_lot * lot_fraction

    @property
    def rtc_per_lot(self) -> float:
        return self.rtc(1.0)

    @property
    def rtc_per_micro(self) -> float:
        return self.rtc(MICRO_LOT_OZ / STD_LOT_OZ)

    @property
    def cost_as_pct_notional(self) -> float:
        return self.rtc_per_lot / (GOLD_PRICE * STD_LOT_OZ) * 100


@dataclass
class BreakevenTable:
    scenario: CostScenario
    rows: list[dict] = field(default_factory=list)

    def compute(self, rr_ratios: Sequence[float], sl_sizes_usd_lot: Sequence[float]) -> None:
        rtc = self.scenario.rtc_per_lot
        for sl in sl_sizes_usd_lot:
            for rr in rr_ratios:
                be_with_cost = (1.0 + rtc / sl) / (rr + 1.0)
                be_no_cost   = 1.0 / (rr + 1.0)
                self.rows.append({
                    "sl_usd_lot":        sl,
                    "sl_usd_oz":         sl / STD_LOT_OZ,
                    "sl_usd_micro":      sl * MICRO_LOT_OZ / STD_LOT_OZ,
                    "rr":                rr,
                    "be_no_cost_pct":    be_no_cost   * 100,
                    "be_with_cost_pct":  be_with_cost * 100,
                    "cost_premium_pct": (be_with_cost - be_no_cost) * 100,
                    "cost_pct_of_sl":    rtc / sl * 100,
                    "viable":            be_with_cost < 1.0,
                })


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

SEP = "─" * 84


def _header(title: str) -> str:
    return f"\n{'═'*84}\n  {title}\n{'═'*84}"


def _section(title: str) -> str:
    return f"\n{SEP}\n  {title}\n{SEP}"


def _pct(v: float) -> str:
    return f"{v:6.1f}%"


def _usd(v: float, width: int = 7) -> str:
    return f"${v:{width}.2f}"


def _wrap(text: str) -> str:
    return textwrap.fill(text, width=84, initial_indent="  ", subsequent_indent="  ")


# ---------------------------------------------------------------------------
# Print functions
# ---------------------------------------------------------------------------

def print_cost_breakdown(scenarios: list[CostScenario]) -> None:
    print(_header("COST BREAKDOWN — IUX Standard account, round-trip"))
    print(f"  Reference price: ${GOLD_PRICE:.0f}/oz  |  "
          f"Std lot: ${GOLD_PRICE * STD_LOT_OZ:,.0f} notional  |  "
          f"Micro lot (0.01): ${GOLD_PRICE * MICRO_LOT_OZ:,.0f} notional")
    print()
    print(_wrap(
        "⚠  Spread values marked PLACEHOLDER must be replaced with measured numbers"
        " from spread_logger.py before trusting any of the breakeven figures below."
    ))
    print()

    hdr = (f"  {'Scenario':<32} {'Spread':>8} {'Comm':>6} {'Slip':>7} "
           f"{'Swap':>6}  {'RTC/lot':>9}  {'RTC/0.01lot':>12}  {'%Notl':>7}")
    print(hdr)
    print("  " + "─" * 80)

    for s in scenarios:
        spread_cost = s.spread_usd_oz * STD_LOT_OZ
        swap_cost   = abs(s.swap_per_night_usd) * s.overnight_holds
        print(
            f"  {s.name:<32} "
            f"{_usd(spread_cost, 6):>8} "
            f"{_usd(s.commission_rt_usd, 5):>6} "
            f"{_usd(s.slippage_rt_usd, 6):>7} "
            f"{_usd(swap_cost, 5):>6}  "
            f"{_usd(s.rtc_per_lot, 7):>9}  "
            f"{_usd(s.rtc_per_micro, 8):>12}  "
            f"  {s.cost_as_pct_notional:5.4f}%"
        )
    print()
    print(_wrap(
        "SL column key: $150/lot = $1.50/oz price move = $1.50 actual loss at 0.01 lot."
        " BREAKEVEN WIN RATE is identical regardless of lot size (it is a ratio)."
        " Only the absolute dollar cost per trade scales with lot size."
    ))


def print_breakeven_table(scenario: CostScenario) -> None:
    table = BreakevenTable(scenario=scenario)
    table.compute(RR_RATIOS, SL_SIZES_USD_LOT)

    print(_section(
        f"BREAKEVEN WIN RATE — {scenario.name}  "
        f"(RTC: {_usd(scenario.rtc_per_lot)}/lot | {_usd(scenario.rtc_per_micro, 5)}/0.01lot)"
    ))
    print()
    print("  Reading: 'BE w/cost' is the minimum win rate to not lose money after friction.")
    print("  'Premium' is the extra win rate cost friction demands above zero-cost baseline.")
    print("  'Cost/SL' > 30% means friction is eating a third of your risk — serious.")
    print()

    sl_groups: dict[float, list[dict]] = {}
    for row in table.rows:
        sl_groups.setdefault(row["sl_usd_lot"], []).append(row)

    hdr = (f"  {'SL/lot':>7}  {'$/oz':>5}  {'$0.01lot':>8}  {'RR':>4}  "
           f"{'BE 0-cost':>9}  {'BE w/cost':>9}  {'Premium':>8}  {'Cost/SL':>8}  {'OK?':>5}")
    print(hdr)
    print("  " + "─" * 80)

    for sl_lot, rows in sl_groups.items():
        for row in rows:
            ok_str = "YES" if row["viable"] else "NO ← impossible"
            flag = ""
            c = row["cost_pct_of_sl"]
            if c > 30:
                flag = "  ⚠ cost>30%SL"
            elif c > 15:
                flag = "  △ cost>15%SL"
            print(
                f"  {sl_lot:7.0f}  "
                f"${row['sl_usd_oz']:4.2f}  "
                f"${row['sl_usd_micro']:6.2f}  "
                f"{row['rr']:4.1f}  "
                f"{_pct(row['be_no_cost_pct']):>9}  "
                f"{_pct(row['be_with_cost_pct']):>9}  "
                f"{_pct(row['cost_premium_pct']):>8}  "
                f"{_pct(row['cost_pct_of_sl']):>8}  "
                f"{ok_str}{flag}"
            )
        print()


def print_broker_comparison() -> None:
    print(_section("BROKER COMPARISON — IUX Standard vs Raw ECN vs Original Placeholder"))
    print()
    print(_wrap(
        "The crossover spread is the $/oz value at which IUX Standard becomes"
        " EQUALLY EXPENSIVE to the comparison account. Below the crossover, Standard"
        " is cheaper. Above it, Standard is more expensive. This is the single most"
        " important number from Gate 0 for your account-choice decision."
    ))
    print()

    crossover_vs_raw = PROFILE_IUX_STANDARD.crossover_spread(PROFILE_RAW_ECN)
    crossover_vs_orig = PROFILE_IUX_STANDARD.crossover_spread(PROFILE_ORIGINAL_PLACEHOLDER)

    std_above_raw  = SPREAD_QUIET_USD_OZ > crossover_vs_raw
    std_above_orig = SPREAD_QUIET_USD_OZ > crossover_vs_orig

    print(f"  Crossover vs Raw ECN:            ${crossover_vs_raw:.2f}/oz")
    print(f"  Crossover vs Original v1:        ${crossover_vs_orig:.2f}/oz")
    print(f"  IUX Standard quiet spread (est): ${SPREAD_QUIET_USD_OZ:.2f}/oz  "
          f"[PLACEHOLDER — measure before trusting]")
    print()
    if std_above_raw:
        print(f"  VERDICT (vs Raw): Standard PLACEHOLDER is ${SPREAD_QUIET_USD_OZ - crossover_vs_raw:.2f}/oz "
              f"ABOVE crossover → Raw ECN is cheaper in quiet hours (PLACEHOLDER).")
    else:
        print(f"  VERDICT (vs Raw): Standard PLACEHOLDER is ${crossover_vs_raw - SPREAD_QUIET_USD_OZ:.2f}/oz "
              f"BELOW crossover → Standard is cheaper than Raw ECN (PLACEHOLDER).")
    print()

    # Cost table per account
    print(f"  {'Account':<38} {'Spread/oz':>10} {'Fixed/lot':>10} {'RTC/lot':>9} {'RTC/0.01lot':>12}")
    print("  " + "─" * 80)
    for p in ALL_PROFILES:
        fixed = p.commission_rt_per_lot + p.slippage_rt_per_lot
        print(
            f"  {p.name:<38} "
            f"${p.spread_quiet_usd_oz:8.2f}  "
            f"{_usd(fixed, 7):>10}  "
            f"{_usd(p.rtc_quiet(), 6):>9}  "
            f"{_usd(p.rtc_micro(), 5):>12}"
        )
        if p.note:
            print(f"  {'':38}  ↳ {p.note}")

    print()
    diff_std_raw  = PROFILE_IUX_STANDARD.rtc_quiet() - PROFILE_RAW_ECN.rtc_quiet()
    diff_std_orig = PROFILE_IUX_STANDARD.rtc_quiet() - PROFILE_ORIGINAL_PLACEHOLDER.rtc_quiet()
    micro_diff    = PROFILE_IUX_STANDARD.rtc_micro() - PROFILE_RAW_ECN.rtc_micro()

    print(_wrap(
        f"At these PLACEHOLDER values: Standard costs ${abs(diff_std_raw):.2f}/lot "
        f"({'more' if diff_std_raw > 0 else 'less'}) than Raw ECN, which is "
        f"${abs(micro_diff):.2f} per 0.01-lot trade. Over 200 trades that is "
        f"${abs(micro_diff * 200):.2f} total — {'material' if abs(micro_diff * 200) > 20 else 'negligible'}."
        f" The spread simplicity of Standard (no separate commission) is a real"
        f" advantage for bookkeeping. The measured spread will settle this."
    ))
    print()
    print(_wrap(
        "ACCOUNT CHOICE RULE: if measured Standard spread (London median) < "
        f"${crossover_vs_raw:.2f}/oz, prefer Standard. If it is ≥ ${crossover_vs_raw:.2f}/oz,"
        " compare with the Raw account's actual spread + commission at that moment."
        " Do not switch accounts based on PLACEHOLDER numbers alone."
    ))


def print_frequency_erosion(quiet_scenario: CostScenario) -> None:
    print(_section("FREQUENCY EROSION — cost vs a 1%/day target, $10,000 account"))
    print()
    print("  RTC costs shown for BOTH lot sizes. The win rate hurdle is the same.")
    print("  The dollar erosion differs — which is why lot sizing matters for frequency.")
    print()

    account          = 10_000.0
    target_pct       = 1.0
    target_usd       = account * target_pct / 100
    rtc_lot          = quiet_scenario.rtc_per_lot
    rtc_micro        = quiet_scenario.rtc_per_micro

    print(f"  Account: ${account:,.0f}  |  Target: {target_pct}% = ${target_usd:.2f}/day  |  "
          f"RTC/lot: ${rtc_lot:.2f}  |  RTC/0.01lot: ${rtc_micro:.2f}")
    print()

    intervals = {1: "1/day", 2: "every 12h", 4: "every 6h",
                 8: "every 3h", 16: "every 90min", 48: "every 30min"}

    print(f"  {'Trades':>6}  {'Interval':>12}  {'Cost/lot':>9}  {'Cost/0.01lot':>12}  "
          f"{'% target':>9}  {'Verdict'}")
    print("  " + "─" * 75)

    for n in TRADE_COUNTS_PER_DAY:
        cost_lot   = n * rtc_lot
        cost_micro = n * rtc_micro
        ratio      = cost_micro / target_usd * 100  # use micro since that is trading size
        interval   = intervals.get(n, f"every {1440 // n}min")
        if ratio > 200:
            v = "STOP — cost > 2× target"
        elif ratio > 100:
            v = "STOP — cost > target"
        elif ratio > 50:
            v = "WARNING — cost > 50% target"
        elif ratio > 25:
            v = "Marginal"
        else:
            v = "Manageable"
        print(f"  {n:6d}  {interval:>12}  ${cost_lot:7.2f}  ${cost_micro:10.4f}  "
              f"{ratio:8.1f}%  {v}")

    print()
    print(_wrap(
        "NOTE: The ratio uses 0.01-lot cost vs the account target. At micro-lot sizes,"
        " friction per trade is tiny in absolute dollars ($0.35 per trade), so even"
        " 16 trades/day costs only $5.60 — 5.6% of the 1% target. BUT: the win-rate"
        " hurdle (the ratio column in the breakeven tables) doesn't shrink. You still"
        " need the same edge; you are just burning less money per losing trade."
    ))
    print()
    print(_wrap(
        "POLLING FREQUENCY RULING: unchanged from v1. ICT strategies (OB/FVG/Sweep)"
        " fire 1–5 setups per session. Poll every 60–300 s; drop to 30 s only"
        " inside an active setup window where entry timing matters."
    ))


def print_news_warning(news_scenario: CostScenario) -> None:
    print(_section("NEWS TRADE WARNING — unchanged from v1 (slippage dominates, not spread)"))
    print()
    rtc_lot   = news_scenario.rtc_per_lot
    rtc_micro = news_scenario.rtc_per_micro
    print(f"  RTC during high-impact event: {_usd(rtc_lot)}/lot | {_usd(rtc_micro, 5)}/0.01lot")
    print()
    print(_wrap(
        "The news cost is the same on Standard and Raw because execution slippage"
        " ($10/oz per side = $1,000/lot per side) dominates; commission ($0 vs $7)"
        " is noise at that scale. At 0.01-lot size the absolute loss is smaller"
        " ($20 per trade round-trip), but $20 on a micro-lot with a $5 SL is still"
        " 4× the stop distance — you are stopped before the fill is processed."
    ))
    print()
    print(f"  {'SL/lot':>8}  {'$/0.01lot SL':>13}  {'News RTC/lot':>13}  {'News as % SL':>13}  {'Verdict'}")
    print("  " + "─" * 75)
    for sl in SL_SIZES_USD_LOT:
        pct     = rtc_lot / sl * 100
        sl_micro = sl * MICRO_LOT_OZ / STD_LOT_OZ
        if pct > 100:
            v = "IMPOSSIBLE — cost > SL"
        elif pct > 50:
            v = "Cost > 50% risk — avoid"
        elif pct > 25:
            v = "Marginal at best"
        else:
            v = "Theoretically viable"
        print(f"  {sl:8.0f}  ${sl_micro:11.2f}  {_usd(rtc_lot):>13}  {pct:12.1f}%  {v}")


def print_gate0_summary(iux_quiet: CostScenario, iux_wide: CostScenario) -> None:
    print(_header("GATE 0 SUMMARY — IUX Standard, minimum edge required"))
    print()

    sl_ref = 150.0
    rr_pairs = [(1.0, "1:1"), (2.0, "2:1"), (3.0, "3:1")]
    sl_ref_oz    = sl_ref / STD_LOT_OZ
    sl_ref_micro = sl_ref * MICRO_LOT_OZ / STD_LOT_OZ

    print(f"  Reference SL: ${sl_ref:.0f}/lot = ${sl_ref_oz:.2f}/oz price move "
          f"= ${sl_ref_micro:.2f} actual loss at 0.01 lot")
    print()
    print(f"  {'Scenario':<35} {'1:1 RR':>9} {'2:1 RR':>9} {'3:1 RR':>9}  Note")
    print("  " + "─" * 75)

    all_s = [iux_quiet, iux_wide]
    for s in all_s:
        rtc = s.rtc_per_lot
        vals = []
        for rr, _ in rr_pairs:
            be = min((1.0 + rtc / sl_ref) / (rr + 1.0), 1.0)
            vals.append(f"{be*100:8.1f}%")
        note = "USE THIS" if "quiet" in s.name.lower() else "stress test"
        print(f"  {s.name:<35} {vals[0]:>9} {vals[1]:>9} {vals[2]:>9}  ← {note}")

    print()
    print(_wrap(
        "DECISION RULE (Gate 0): your strategy's OUT-OF-SAMPLE win rate must be at"
        " least 5 percentage points above the IUX Standard quiet-session breakeven"
        " line. Below that margin, sampling noise from hundreds of trades will"
        " routinely erase the apparent edge."
    ))
    print()
    print(_wrap(
        "⚠ IMPORTANT: these breakeven numbers are ONLY as good as the spread"
        " input. The spread is PLACEHOLDER. Run spread_logger.py, collect 5+ days of"
        " session-tagged bid/ask samples, then replace SPREAD_QUIET_USD_OZ and"
        " SPREAD_WIDE_USD_OZ at the top of this file and re-run Gate 0."
        " Gate 1 starts after that — not before."
    ))
    print()
    print("  " + "═" * 80)
    print("  Gate 0 complete (with PLACEHOLDER spread). DO NOT proceed to Gate 1 yet.")
    print("  Next step: run  data/spread_logger.py  and measure the real spread.")
    print("  " + "═" * 80)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def build_iux_scenarios() -> list[CostScenario]:
    commission_rt = COMMISSION_PER_SIDE_USD * 2  # = $0 for Standard

    return [
        CostScenario(
            name="IUX Std — quiet session [PLACEHOLDER]",
            spread_usd_oz=SPREAD_QUIET_USD_OZ,
            commission_rt_usd=commission_rt,
            slippage_rt_usd=SLIPPAGE_STANDARD_RT_USD_LOT,
        ),
        CostScenario(
            name="IUX Std — wide/off-session [PLACEHOLDER]",
            spread_usd_oz=SPREAD_WIDE_USD_OZ,
            commission_rt_usd=commission_rt,
            slippage_rt_usd=SLIPPAGE_STANDARD_RT_USD_LOT,
        ),
        CostScenario(
            name="IUX Std — overnight long [PLACEHOLDER swap]",
            spread_usd_oz=SPREAD_QUIET_USD_OZ,
            commission_rt_usd=commission_rt,
            slippage_rt_usd=SLIPPAGE_STANDARD_RT_USD_LOT,
            swap_per_night_usd=SWAP_LONG_PER_NIGHT,
            overnight_holds=1,
        ),
        CostScenario(
            name="NEWS trade (spans event)",
            spread_usd_oz=SPREAD_NEWS_USD_OZ,
            commission_rt_usd=commission_rt,
            slippage_rt_usd=SLIPPAGE_NEWS_PER_SIDE_USD_OZ * 2 * STD_LOT_OZ,
        ),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_gate_0() -> None:
    scenarios = build_iux_scenarios()
    iux_quiet = scenarios[0]
    iux_wide  = scenarios[1]

    print(_header("GATE 0 — COST-vs-EDGE BREAKEVEN  |  IUX Standard  |  XAUUSD  |  MT5"))
    print()
    print(_wrap(
        "Primary trading size: 0.01 lots (1 oz). All breakeven WIN RATES are"
        " identical across lot sizes (they are ratios). Absolute dollar costs scale"
        " proportionally. Both are shown throughout."
    ))
    print()
    print(_wrap(
        "⚠  SPREAD VALUES ARE PLACEHOLDERS — all breakeven figures below are"
        " provisional until spread_logger.py replaces them with real measurements."
    ))

    print_cost_breakdown(scenarios)
    print_broker_comparison()

    # Run breakeven tables only for IUX scenarios (not news — handled separately)
    for s in scenarios[:3]:  # quiet, wide, overnight
        print_breakeven_table(s)

    print_frequency_erosion(iux_quiet)
    print_news_warning(scenarios[3])
    print_gate0_summary(iux_quiet, iux_wide)


if __name__ == "__main__":
    run_gate_0()
