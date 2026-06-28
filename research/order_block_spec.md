# Order Block Candidate Strategy Specification

Status: pre-registered rule specification only. No backtest code.

This document defines the bare Order Block (OB) test as a falsifiable candidate
strategy. It strips the SMC concept down to a zone created by displacement plus
break of structure (BOS), then measures the later reaction. The later reaction is
never used to define the zone.

## Locked Bare-Test Parameters

- Timeframe: M15, resampled from Dukascopy tick data.
- ATR: ATR(14).
- Displacement threshold: impulse move >= 2.0 \* ATR(14).
- Swing definition for BOS: fractal high/low with 10 bars on each side.
- Reaction window: 20 completed M15 bars after the touch bar.
- Success threshold: close moves >= 1.0 \* frozen ATR(14) in the OB direction.
- HTF bias: off.
- FVG: recorded as a flag only, not a required filter.

Changing any parameter above is a new trial for Deflated Sharpe / multiple-testing
accounting. Do not sweep parameters to improve results.

All thresholds are ATR-relative, never fixed dollars.

## Anti-Hindsight Rule

A zone is never defined by the reversal that follows it. Define zones only from
displacement plus already-known structure. The subsequent reaction is the outcome
being measured, never an input to the definition.

Reject any rule that can identify a zone only after seeing the later bounce,
rejection, reversal, target hit, or failure.

## Swing Confirmation Timing

A fractal swing high/low with 10 bars on each side is confirmed only after the
10 right-side bars have fully closed.

At impulse/BOS time, the BOS check may reference only swings that are already
confirmed as of the impulse bar. The system must never use a swing whose right-side
10 bars have not completed.

Concrete no-look-ahead check:

- For a candidate pivot at index `p`, the swing is eligible only at bars with
  index `t >= p + 10`.
- A BOS occurring on impulse bar `i` may use that swing only if `p + 10 <= i`.
- The swing-high/low decision may inspect bars `[p - 10, p + 10]`, but only after
  bar `p + 10` is closed.
- If `p + 10 > i`, that swing is unavailable and must not be used for the BOS at
  bar `i`.

## Zone Creation

Bullish OB:

1. Use only completed M15 candles available at the impulse bar.
2. Find the most recent confirmed fractal swing high available at that time.
3. Detect an impulse that breaks above that swing high.
4. The impulse displacement must be >= 2.0 \* ATR(14), using the frozen ATR snapshot
   described below.
5. The OB candle is the last bearish candle immediately before the impulse.
6. Zone = that candle's full high-low range.
7. Zone is created only when displacement plus BOS is complete.

Bearish OB:

1. Use only completed M15 candles available at the impulse bar.
2. Find the most recent confirmed fractal swing low available at that time.
3. Detect an impulse that breaks below that swing low.
4. The impulse displacement must be >= 2.0 \* ATR(14), using the frozen ATR snapshot.
5. The OB candle is the last bullish candle immediately before the impulse.
6. Zone = that candle's full high-low range.
7. Zone is created only when displacement plus BOS is complete.

At zone creation, record:

- direction
- zone_high
- zone_low
- zone_creation_time
- frozen ATR(14)
- displacement in ATR units
- BOS swing level
- FVG_present flag

## ATR Snapshot

ATR(14) is snapshotted at `zone_creation_time` and frozen.

The same frozen ATR value is used for both:

1. the displacement test: `displacement >= 2.0 * frozen_ATR`
2. the success threshold after touch: `close move >= 1.0 * frozen_ATR`

Do not recompute ATR at touch time. Do not use later volatility to resize the
target or reclassify the setup.

## Mitigation Tracking

The zone starts as fresh / unmitigated.

The zone becomes mitigated once later price first touches or enters the zone.
The main bare test uses first-touch setups only. Already-touched retests must be
reported separately if measured.

Touch condition:

```text
bar.low <= zone_high and bar.high >= zone_low
```

## Success Measurement

The touch bar starts the event, but is not counted in the reaction window.

Reaction window:

```text
20 completed M15 bars after the touch bar, exclusive of the touch bar itself.
```

Conservative success rule uses bar closes, not intrabar high/low wicks.

Bullish OB:

- Reference price: `zone_high`, the upper boundary touched on re-entry.
- Success if any of the next 20 completed M15 bars closes at or above:

```text
zone_high + 1.0 * frozen_ATR
```

Bearish OB:

- Reference price: `zone_low`, the lower boundary touched on re-entry.
- Success if any of the next 20 completed M15 bars closes at or below:

```text
zone_low - 1.0 * frozen_ATR
```

If the threshold is not reached by close within those 20 completed bars, the setup
is a failure.

## Baseline

The baseline must match the OB context. It cannot be a generic random candle range
that price merely drifts through, because that would compare a post-displacement OB
touch against unrelated market conditions and inflate measured edge.

For each OB setup, sample baseline zones from prior non-OB candle ranges that meet
the following criteria:

- The baseline candle existed strictly before the OB trigger time.
- The baseline candle is strictly non-OB: it is not itself an identified OB candle
  and its high-low range does not overlap any detected OB zone known before the OB
  trigger time.
- Price later returns to that baseline candle's range after a comparable impulse
  move.
- Comparable impulse means same direction and same displacement-ATR bucket as the
  OB setup.
- Use the same M15 timeframe.
- Match the same session bucket as the OB setup.
- Match the same volatility bucket as the OB setup, using the frozen ATR regime at
  baseline zone creation.
- Apply the same first-touch logic and the same 20-bar close-based outcome rule.

All matching criteria are mandatory. If a candidate baseline sample fails any
criterion, discard it. Never relax session, volatility, direction, displacement
bucket, timing, or non-OB requirements to fill the quota.

Displacement-ATR buckets are pre-registered as:

```text
2.0-3.0 ATR
3.0-4.0 ATR
4.0+ ATR
```

Baseline event definition:

1. Identify a qualifying impulse plus BOS-like displacement context.
2. Select a prior non-OB candle range from before that impulse.
3. Wait for price to return and first-touch that range.
4. Measure the same directional 1.0 \* frozen_ATR close-based outcome within 20
   completed bars after touch.

Baseline samples per OB:

```text
K = 5 matched baseline samples per OB setup.
```

Baseline selection is deterministic and reproducible:

1. Build the full qualified baseline pool using the mandatory criteria above.
2. Sort qualified candidates by baseline trigger time descending.
3. Select the 5 nearest qualified baseline triggers before the OB trigger time.
4. If two candidates have identical trigger time, tie-break by earlier baseline
   candle timestamp, then by lower source row/bar index.

Two runs over the same data must produce identical baseline selections. No random
sampling is used.

If fewer than 5 qualified matches exist for a given OB, use however many qualify
and record the actual baseline count for that OB. Flag any OB with fewer than 5
valid baselines as under-matched. Do not loosen matching rules after seeing results.

Measured edge:

```text
P(success | OB first touch) - P(success | matched random non-OB range first touch)
```

## HTF Bias Layer

HTF bias is an optional overlay, not part of the bare OB test.

Default:

```text
require_htf_alignment = False
```

Future pluggable interface:

```text
htf_trend(timestamp) -> bullish | bearish | neutral
```

If enabled later:

- Bullish OB eligible only if `htf_trend(zone_creation_time) == bullish`.
- Bearish OB eligible only if `htf_trend(zone_creation_time) == bearish`.
- Neutral HTF means no trade unless separately pre-registered.

HTF trend must use only higher-timeframe candles closed before `zone_creation_time`.
It must not use the later touch, reaction, or outcome.

The first measurement keeps `require_htf_alignment = False`. HTF alignment can be
A/B tested only after the bare OB result is known.

Each added filter shrinks setup count and raises the Deflated Sharpe / trial-count
penalty. HTF must improve edge enough to justify the lost sample size and extra
trial.

## Overlap With Other Candidates

Expected overlap:

- OB + FVG: high.
- OB + Liquidity Sweep: moderate.
- FVG + Liquidity Sweep: moderate to high.
- OB + FVG + Sweep: smaller but important confluence subset.

Gate 1 tests each candidate standalone. Gate 2 quantifies overlap with a trigger
co-occurrence / correlation matrix to determine whether these are independent
signals or the same event under different labels.

## Required Reporting

Report only the locked values:

- raw setup count
- fresh first-touch count
- already-touched retest count, if measured
- OB success rate
- matched baseline success rate
- edge over baseline
- confidence interval / uncertainty
- session breakdown
- FVG-present vs no-FVG stratification, descriptive only

Minimum usefulness target: hundreds of setups, not dozens. If sample count is too
low or results are weak, report that honestly. Do not tune parameters.
