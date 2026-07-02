# Hypothesis Registry

- 2026-07-02: Track B compression quality audit. Hypothesis: longer compression duration and tighter range/ATR predict higher MFE and net R. Pre-registered tercile buckets; no threshold optimization; analysis-only for FDR accounting.
- 2026-07-02: H-2026-EXIT-01 live shadow validation phase started; decision on exit switch deferred until live gate (50-100 trades) with realized-A vs shadow-C comparison.
- 2026-07-02: H-2026-REV-01 registered. Failed-breakout reversal hypothesis: validated compression trades that hit 1R stop become opposite-direction next-bar-open signals with 1R SL, 1.5R TP, 10-bar force close, $0.20 spread; no parameter tuning.
- 2026-07-02: H-2026-REV-01 result: FAIL under pre-registered gates; reversal does not clear train/test net-of-cost CI and/or does not beat required controls. Hypothesis closed unless re-registered with new data.
- 2026-07-02: H-2026-SESS-01 registered. Session-conditional compression quality hypothesis: ASIA 23:00-06:59 UTC should have lower net R than LONDON_ONLY/OVERLAP; pre-registered four UTC buckets; no boundary tuning; future filter candidate only if a bucket is significantly negative in both train and test.
- 2026-07-02: H-2026-SESS-01 result: FAIL_FILTER_RULE no session has significantly negative net R in both train and test; trade all sessions under current evidence.
