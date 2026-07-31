st.markdown("""
⭐ Full Intraday Market Microstructure (Extended to 4 PM + After‑Hours)
This is the actual institutional flow pattern your SRC logic is built around.

⭐ Phase 1 — The Retail Rush (9:30 AM – 9:35 AM)
This is the most chaotic part of the day.

What happens:
Retail traders react emotionally to overnight news.
Options traders chase pre‑market gaps.
Market orders flood the book.
Liquidity is thin → spreads wide.
Price jumps $2–$4 easily.

Why it matters:
Not reliable for structural signals.
SRC should never activate here.
PCA1_slope is unstable.
EMA9/EMA20 crosses are meaningless.

This is pure noise.

⭐ Phase 2 — The Profit‑Taking Shakeout (9:35 AM – 9:55 AM)
This is where the early spike unwinds.

What happens:
Pre‑market traders take profits.
Early short sellers attack the overextension.
Retail stop‑loss cascades trigger.
Price often retraces 30–70% of the opening spike.

Why it matters:
This is the dip that SRC later evaluates.
But recovery signals here are unreliable.
Institutions are still waiting.

This is the “shakeout” phase.

⭐ Phase 3 — Institutional Accumulation (10:00 AM – 11:30 AM)
This is the prime time window — the heart of SRC.

What happens:
Institutions begin executing large buy programs.
Algorithms accumulate quietly near VWAP.
Short sellers get trapped.
Weak retail hands are gone.
Price stabilizes and forms higher‑lows.
EMA9/EMA20 reclaim becomes meaningful.
OR15 reclaim becomes meaningful.

Why SRC activates only here:
Volume is real.
Trend is real.
Recovery is real.
Insitutional flow is real.
Historical recovery probability becomes predictive.

This is the structural recovery window.

⭐Phase 4 — Midday Drift (11:30 AM – 1:30 PM)
This is where SRC should not activate.

What happens:
Volume collapses.
Volatility collapses.
Institutions pause.
Retail scalpers dominate.
Breakouts fail.
Recoveries stall.

Why SRC = NO:
Recovery probability drops sharply.
EMA9/EMA20 crosses lose meaning.
OR15 becomes irrelevant.
False signals increase.

This is why your SRC classification is time‑locked.

⭐Phase 5 — Afternoon Trend Continuation (1:30 PM – 2:45 PM)
This is where SRC_PM could exist (optional future feature).

What happens:
Institutions resume execution.
Afternoon buy/sell programs activate.
If the morning recovery was real, continuation happens here.
If the morning trend was bearish, afternoon flushes happen here.

Why SRC does not activate:
Afternoon recoveries are event‑driven, not structural.
They require different logic (SRC_PM).

But this phase is still important for Page5 momentum.

⭐Phase 6 — Closing Auction Positioning (2:45 PM – 3:55 PM)
This is where the market prepares for the closing auction.

What happens:
Institutions position for the 4 PM auction.
VWAP becomes extremely important.
Liquidity increases.
Large block trades appear.
Price becomes more stable.

Why SRC does not activate:
This is not recovery behavior.
This is auction positioning.

But Page5_v3 momentum signals often fire here.

⭐Phase 7 — The Closing Auction (4:00 PM)
This is the official closing price — the anchor for your model.

What happens:
All Market‑On‑Close (MOC) orders execute.
All Limit‑On‑Close (LOC) orders execute.
ETF hedging flows finalize.
Index rebalancing occurs.

Mutual funds adjust positions.

Why this matters:
Your model uses the closing price as the anchor for next‑day orders.
You cannot buy the closing price after hours.
This is the most important price of the day.

⭐Phase 8 — After‑Hours (4:00 PM – 8:00 PM)
This is a completely different market.

What happens:
Liquidity is extremely thin.
Spreads widen dramatically.
Only professional participants trade.
Price moves are unreliable.
News reactions are exaggerated.

Why SRC does not apply:
No institutional accumulation.
No OR15.
No meaningful EMA9/EMA20 reclaim.
No recovery probability relevance.

After‑hours is not part of structural recovery.

# Structural Day vs Market Day

structural day — today was a macro day

⭐ 1. A +1% or +2% market day does NOT guarantee Page1 candidates
Today:
DOW +1.22%
S&P +1.72%
NASDAQ +2.82%

But this strength was index-driven, not structural-driven.

Why?
Because Microsoft +15% distorted the entire NASDAQ.
When a mega-cap explodes:
NASDAQ looks strong
SP500 looks strong
but individual stocks do NOT show structural setups
dispersion collapses
correlation spikes
anomalies disappear

This is exactly what Page1 is designed to detect.

And today it correctly said:

“Strong market, but weak structural opportunities.”

That’s the right call.

⭐ 2. Yesterday was a structural day — today was a macro day
Yesterday:
QRVO dipped
QRVO recovered
EMA9 crossed EMA20
PCA1 anomaly was strong
execution was aligned
SRC conditions were met

So Page1 showed strong candidates.

Today:
no dip
no recovery
no structural compression
no EMA alignment
no PCA1 anomaly
no SRC
late-day data
index trend = Unknown

So Page1 showed weak candidates.

This is exactly how a structural engine should behave.

⭐ 3. You learned the most important structural trading rule
You said it perfectly:

“It is good to know the model can give opportunities when the market is not so strong.”

Yes — that’s the entire point of structural trading.

✔ Strong market days → fewer structural opportunities
✔ Weak or mixed market days → more structural opportunities
Because structural setups come from:
mispricing
compression
dip → recovery
anomaly
institutional flow
EMA alignment

These happen more often when the market is not trending strongly.
""")