📘 Page 1 — Premium Structural Tracking Engine

Version: 2026-08-01
Purpose: Specialized macro scanning and sorting engine that screens the unified US equity exchanges to isolate high-probability structural setups.
Page1 serves as the master entry pipeline for the entire Intraday Ranker application, feeding clean data into Pages 2 through 7.

This engine focuses on:
- Multi-exchange directory integration (NYSE + NASDAQ)
- Eliminating cross-index duplication bottlenecks
- Dimensionality reduction via primary momentum vectors (PCA1)
- Temporal rate-of-change momentum acceleration (PCA1_slope)
- Structural coiling detection ("Crossing Soon")
- Time-locked statistical capitulation hunting (SRC Flags)

It is designed to track 100% of the premium liquid market playground without missing mid-cap or value-sector leaders.

==============================================================================
1. Executive Summary — Why Page1 Was Re-Engineered
==============================================================================
Traditional trading architectures build universes using arbitrary market index filters like the S&P 500 or the NASDAQ 100. This design introduces severe flaws:
- Index Bloat & Duplication: Mega-cap technology assets (e.g., AAPL, MSFT, NVDA) occupy both indexes simultaneously, causing scripts to waste computing power processing duplicates.
- Mid-Cap Blind Spots: Highly liquid, volatile, fast-moving mid-cap companies ($2B to $15B market cap) listed on the NYSE are completely skipped because they are too small for the S&P 500 and do not belong to the tech-centric NASDAQ index.

Page1 bypasses index boundaries by filtering the underlying physical exchanges directly:
- Pool: Unified NYSE + NASDAQ full listings [url=https://fool.com].
- Filter Window: Strictly $40.00 to $110.00 close prices.
- Duplication: Completely eliminated at the data-ingestion layer.

This expansion doubles your operational trading pool from 349 tickers to 729 verified tokens, unlocking financial, industrial, energy, and consumer staple giants that were previously missing.

==============================================================================
2. Active UI Matrix — Column Dictionary
==============================================================================
The table below defines every single output parameter generated on screen by a Page1 system scan:

| Column Name | Metric Family | Mathematical Derivation | End-User Trading Interpretation |
| :--- | :--- | :--- | :--- |
| **Ticker** | Asset Identity | Standard market ticker symbol appended with exchange tags: `(NYSE)` or `(NASDAQ)`. | Identifies the asset vehicle and assigns the appropriate global index compass. |
| **Close** | Baseline Value | Current real-time closing or last-sale spot price of the asset. | Confirms entry bounds remain securely within the premium $40–$110 parameters. |
| **ATR%** | Volatility | `(Trailing 14-day Average True Range / Close Price) * 100` | **Range Availability.** Values > 2.0% indicate wide trading ranges. Values < 1.0% signal friction risks. |
| **RVOL** | Institutional Flow | `Current Intraday Volume / Trailing 20-day Mean Volume` | **Participation Strength.** Values > 1.5 signal institutional presence. Values < 1.0 indicate weak retail interest. |
| **Gap%** | Market Structure | `((Session Open - Yesterday Close) / Yesterday Close) * 100` | **Overnight Displacement.** Measures opening structural risk. Over-gapping (> 1.5%) invalidates entries. |
| **Trend** | Trend Stacking | Directional evaluation of Daily Exponential Moving Averages (`EMA9 > EMA20 > EMA50`). | **Structural Alignment.** Tracks if structure is `UP` (Bullish), `DOWN` (Bearish), or `FLAT` (Choppy compression). |
| **Execution** | Action Triggers | Classification matrix based on price-to-EMA proximity curves and slope vectors. | **Immediate Task Status.** Labels tokens into actionable tiers (`Watch List`, `Crossing Soon`, etc.). |
| **PCA1** | Machine Learning | Vector 1 extracted from your Scikit-Learn Principal Component Analysis backend. | **Daily Momentum Cluster.** Values > 0.00 confirm institutional velocity backing the main trend. |
| **Avg_Volume_20d**| Liquidity Baseline| The arithmetic mean of traded shares over the last 20 market sessions. | **Capacity Gate.** Confirms if an asset can absorb large entry order lots without slippage. |
| **Price_vs_Close** | Intraday Location | Real-time tracking location evaluated directly relative to yesterday's closing boundary. | Flags immediate relative strength lines (`Above Close`, `Below Close`, or `Equal`). |

==============================================================================
3. The Heart of the Engine — PCA1 and PCA1_slope Matrix
==============================================================================
Raw technical indicators are noisy, highly redundant, and collinear. If an asset surges rapidly, its RSI, Rate of Change (ROC), and Stochastic %K will all blast upward simultaneously, distorting probability calculations. 

To solve this, Page1 connects directly to `intraday_ranker_v3.py`, which strips out indicator noise using a Scikit-Learn Principal Component Analysis (PCA) framework. PCA projects 7 correlated variables down onto a single orthogonal axis to extract **PCA1: The Dominant Daily Momentum Cluster**.

Input Indicators Compressed:
- RSI (Strength of gains vs. losses)
- ROC (Rate of Change velocity)
- Stochastic %K (Location within range)
- EMA Curvature (Trend acceleration)
- VWAP Deviation (Distance from institutional fair value)
- Volume Delta (Net buying vs. selling pressure)
- Bollinger Width (Volatility expansion)

------------------------------------------------------------------------------
The Role of PCA1_slope (The Speedometer)
------------------------------------------------------------------------------
While PCA1 captures the absolute directional weight of daily momentum, **PCA1_slope** provides the real-time velocity profile. It functions as the numerical first derivative of PCA1 across trailing intraday windows ($\Delta \text{PCA1} / \Delta t$).

- When PCA1_slope > 0: The daily momentum cluster is actively accelerating intraday.
- When PCA1_slope < 0: Intraday momentum is fading out, signaling a structural exhaust peak.

------------------------------------------------------------------------------
Why PCA2 and PCA3 Are Omitted From Page1
------------------------------------------------------------------------------
Mathematically, a 7-indicator PCA produces 7 individual vectors (PCA1 through PCA7). In financial markets, only the first three contain meaningful variance:
- PCA1: Momentum Factor (Dominant Cluster) — *Active on Page1*
- PCA2: Volatility Factor (Breakout/Compression Curves) — *Reserved for Phase 2*
- PCA3: Participation Factor (Order Book/Liquidity Depth) — *Reserved for Phase 2*

Page1 intentionally uses PCA1 and PCA1_slope only. This represents correct quantitative design. By isolating primary momentum and its first derivative first, Page1 provides an ultra-fast, easily maintainable core screen before secondary volatility variables are checked on later app pages.

==============================================================================
4. Production Execution Labels & Compression
==============================================================================
Assets are assigned an **Execution Label** based on how their structural layers line up:

1. Watch List (Institutional Greenlight)
   - Trend stack is cleanly aligned `UP` (`EMA9 > EMA20 > EMA50`)
   - Both `EMA9` and `EMA20` tracking slopes are moving positive
   - `PCA1 > 0` and accelerating intraday

2. Crossing Soon (The Compression Zone)
   - Fast daily moving averages move into a tight coiling pattern.
   - The absolute spread between `EMA9` and `EMA20` shrinks within a tight **0.3% band**:
     $$\frac{|\text{EMA9} - \text{EMA20}|}{\text{EMA20}} < 0.003$$
   - Indicates stored structural energy ready for an imminent expansion breakout.

3. Not Watch List (Intraday Fade)
   - Trend stack is technically aligned `UP`, but intraday acceleration has failed.
   - `EMA9_slope` or `EMA20_slope` has rolled negative; momentum is actively draining.

4. Setup Only (Inactive Noise)
   - No trend alignment exists, and PCA1 elements are weak. The asset is ignored.

==============================================================================
5. SRC Flag & Recovery Probability Math
==============================================================================
The **SRC (Structural Recovery Candidate)** flag detects high-value institutional capitulation events. 

To maximize the signal-to-noise ratio, the output column uses a clean interface design:
- If ALL conditions are met → Output reads **`YES`**
- If ANY condition fails → Output returns an **empty string (`""`)**

This prevents your screen from being flooded with hundreds of distracting "NO" labels, ensuring instant visual recognition of rare recovery opportunities.

------------------------------------------------------------------------------
Statistical Frequency Math behind `recovery_prob`
------------------------------------------------------------------------------
The `recovery_prob` metric is **not a technical indicator**. It is a pure historical frequency estimator that acts as a behavioral tracking tool for individual tickers. It calculates the statistical frequency that an asset will reverse major intraday sell-offs based on its daily candles over a trailing 30-to-60 day window.

For each session in the window, the engine runs this exact calculation:
1. Find the Absolute Intraday Dip:
   $$\text{Dip} = \text{High} - \text{Low}$$
2. Find the Post-Low Rebound Window:
   $$\text{Recovered} = \text{Close} - \text{Low}$$
3. Evaluate the Threshold:

If the close successfully reclaims 50% or more of that day's high-to-low range, it counts as a successful recovery day:$$\text{Recovered} \ge 0.50 \times \text{Dip}$$4. Establish the Probability Output:$$\text{Recovery_Probability} = \frac{\text{Total Successful Recovery Days}}{\text{Total Sessions Scanned}}$$Example Validation:Out of 60 trading days scanned, a stock flushes intraday but reclaims over 50% of its range by the closing bell on 38 of those days.$$\text{Recovery_Probability} = \frac{38}{60} = 0.6333 \longrightarrow \mathbf{63.33%}$$This confirms the ticker has a reliable behavioral habit of reversing dips, validating it as a safe structural candidate for morning trading operations.

==============================================================================
6. Technical Glossary
==============================================================================
- MultiIndex Flattening: A thread-safe data copying protocol that cleans yfinance column headers without corrupting global application caches during parallel scans.
- PCA1: The primary momentum component vector; strips out technical noise across 7 indicators to find the true institutional trend.
- PCA1_slope: The first derivative of PCA1; functions as an intraday speedometer tracking momentum acceleration.
- EMA Compression: Coiling of moving averages within a 0.3% band, indicating stored energy preceding a breakout.
- Recovery Probability: Historical frequency estimator calculating how often an asset recovers 50% of its intraday drop by the close.
- RVOL: Relative Volume; evaluates current volume against a 20-day baseline to verify active institutional block interest.

==============================================================================
7. Production Execution Labels & Compression
==============================================================================
intraday_ranker_v3 is a hybrid engine that uses daily data for its core structural filtering and intraday data to check real-time execution entry.

Here is exactly how the script breaks down your historical windows and candle intervals:

1. Where it uses Daily Data (Last 90 Days)When rank_universe calls fetch_daily(ticker), it downloads 3 months (90 days) of 1-day candles. The code uses this daily data to calculate:
-  The Trend Stack: Checks if EMA9 > EMA20 > EMA50 on the daily chart.
-  The Slopes: Checks the 5-day trajectory of the daily EMA9 and EMA20.
-  Volatility Metrics: Calculates the 14-day ATR% and 20-day Relative Volume (RVOL).
-  The PCA1 Daily Cluster: Computes your primary machine learning momentum score using daily indicator metrics [url=github.com].
-  Recovery Probability: Scans the daily candles to see how often the asset reversed its high-to-low dips.2. 

2. Where it uses Intraday Data (1-Minute and 5-Minute Candles)
When the model checks if a stock is ready to trade right now, it calls fetch_intraday(ticker) to pull the current session's 1-minute candles. The code uses this intraday data to calculate:
-  Current Price Mapping: It pulls the absolute last 1-minute close to determine the exact price.Price vs. Close Line: Compares the last 1-minute spot price directly against yesterday's daily closing boundary to tag the stock as Above Close or Below Close.
-  The Velocity Overlay: Links with your Page 1 UI to evaluate real-time intraday metrics like PCA1_slope and VWAP Deviation across the immediate market session.

Summary Strategy
- The Daily Data determines the Structure (Is this a strong company aligned in a clean macro trend?).
-  The Intraday Data determines the Execution (Is the stock breaking out or crossing a key boundary right now?).

intraday_ranker_v3 acts as a multi-stage funnel that uses daily structural metrics to screen the market before running intraday analysis:

 [729 Raw Multi-Exchange Tickers]
                 │
                 ▼  STAGE 1: Daily Structural Filter
 [Passed Tickers: e.g., 726 Tickers] ───► Must meet Price Gates ($40-$110) & 40+ daily rows
                 │
                 ▼  STAGE 2: Daily Indicator Engine
 [Calculated Metrics for Pool] ──────────► Computes Daily Trend, RVOL, ATR%, PCA1 Cluster
                 │
                 ▼  STAGE 3: Intraday Execution Scan
 [Final Output Rows: e.g., 18 Tickers] ──► Passes 1-Min Spot Price vs. Yesterday Close,
                                          PCA1_slope, and "Crossing Soon" criteria

✔️ End of Page1 Documentation
