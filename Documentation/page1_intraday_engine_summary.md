# 📘 Page 1 — Premium Structural Tracking Engine

**Version:** 2026-08-05 (Production-Hardened)  
**Purpose:** Specialized high-speed macro scanning, unpivoting, and sorting engine that screens the unified US equity exchanges to isolate high-probability structural breakout setups. Page 1 serves as the master vectorized entry pipeline for the entire Intraday Ranker application, feeding clean data into Pages 2 through 7.

### Key Technological Focus Areas:
* 🌐 **Multi-Exchange Directory Integration:** Directly queries full listings across physical exchanges (NYSE + NASDAQ).
* ⚡ **Vectorized In-Memory Batching:** Eliminates cross-index duplication and network multi-threading bottleneck failures via unified batch requests.
* 📈 **Structural Foundation Remapping:** Anchors market trend stacking to multi-week intermediate baselines (`EMA20 > EMA50`) to capture setups during healthy pullbacks.
* 🎯 **Predictive Low-Lag Leaderboard Sorting:** Prioritizes compressed coiling energy over lagging, extended historical momentum.
* 🛡️ **Capacity Gate Shielding:** Completely insulates calculations from illiquid ghost-printing ticker distortions.

---

## 1. Executive Summary — Why Page 1 Was Re-Engineered

Traditional trading architectures build asset universes using arbitrary market index components like the S&P 500 or the NASDAQ 100. This structural constraint introduces severe mathematical flaws:
* **Index Bloat & Duplication:** Mega-cap technology assets (e.g., AAPL, MSFT, NVDA) occupy both indexes simultaneously. This layout forces standard single-threaded scripts to waste critical processing power evaluating identical assets multiple times.
* **Mid-Cap Blind Spots:** High-velocity, highly liquid mid-cap companies (\$2B to \$15B market cap) listed on the NYSE are completely skipped because they are too small for the S&P 500 and do not belong to the tech-centric NASDAQ index.

Page 1 completely bypasses traditional index boundaries by filtering the underlying physical exchange directories directly at the data-ingestion layer:
* **Asset Pool:** Unified NYSE + NASDAQ full listings.
* **Price Filter Window:** Strictly \$40.00 to \$110.00 close boundaries.
* **Duplication:** 100% eliminated before calculation streams materialize.
* **Liquidity Gate:** Strictly enforces an immediate daily volume baseline cutoff.

This structural expansion doubles your operational trading pool from 349 tickers to over 570+ verified tokens, unlocking financial, industrial, energy, and consumer staple giants that were previously missing from your radar.

---

## 2. Active UI Matrix — Column Dictionary

The table below defines every single output parameter generated on screen by a Page 1 system scan:

| Column Name | Metric Family | Mathematical Derivation | End-User Trading Interpretation |
| :--- | :--- | :--- | :--- |
| **Ticker** | Asset Identity | Standard market ticker symbol appended with dynamic exchange tags: `(NYSE)` or `(NASDAQ)`. | Identifies the asset vehicle and assigns the appropriate global index compass. |
| **Close** | Baseline Value | Current real-time closing or last-sale spot price of the asset. | Confirms entry bounds remain securely within the premium \$40–\$110 parameters. |
| **ATR%** | Volatility | `(Trailing 14-day Average True Range / Close Price) * 100` | **Range Availability.** Values > 2.0% indicate wide trading ranges. Values < 1.0% signal friction risks. |
| **RVOL** | Institutional Flow | `Current Intraday Volume / Trailing 20-day Mean Volume` | **Participation Strength.** Values > 1.5 signal institutional presence. Values < 1.0 indicate weak retail interest. |
| **Gap%** | Market Structure | `((Session Open - Yesterday Close) / Yesterday Close) * 100` | **Overnight Displacement.** Measures opening structural risk. Over-gapping (> 1.5%) invalidates entries. |
| **Trend** | Structural Foundation | Directional evaluation of Daily Exponential Moving Averages (`EMA20 > EMA50`). | **Macro Core Alignment.** Identifies multi-week structure as `UP` (Bullish), `DOWN` (Bearish), or `FLAT` (Compression). |
| **Execution** | Action Triggers | Real-time classification based on price-to-EMA proximity curves and intraday slope vectors. | **Immediate Task Status.** Labels tokens into actionable tiers (`Crossing Soon`, `Watch List`, etc.). |
| **PCA1** | Machine Learning | Vector 1 extracted from your Scikit-Learn Principal Component Analysis backend. | **Daily Momentum Cluster.** Values > 0.00 confirm institutional velocity backing the main trend. |
| **Avg_Volume_20d**| Liquidity Baseline| The arithmetic mean of traded shares over the last 20 market sessions. | **Capacity Gate.** Discards ghost-printing tickers. Assets must maintain > 250,000 shares to pass. |
| **Price_vs_Close** | Intraday Location | Real-time tracking location evaluated directly relative to yesterday's closing boundary. | Flags immediate relative strength lines (`Above Close`, `Below Close`, or `Equal`). |

---

## 3. The Heart of the Engine — PCA1 and PCA1_slope Matrix

Raw technical indicators are highly redundant, noisy, and collinear. If an asset surges rapidly, its RSI, Rate of Change (ROC), and Stochastic %K all blast upward simultaneously, distorting probability metrics. 

To solve this, Page 1 connects directly to `intraday_ranker_v3.py`, which strips out indicator noise using a Scikit-Learn Principal Component Analysis (PCA) framework. PCA projects 7 correlated variables down onto a single orthogonal axis to extract **PCA1: The Dominant Daily Momentum Factor**.

### Input Indicators Compressed:
* **RSI:** Strength of gains vs. losses (14-period lookback).
* **ROC:** Rate of Change velocity parameters (10-period interval).
* **Stochastic %K:** Location of close relative to high/low range boundaries.
* **EMA Curvature:** Moving average trend acceleration trajectory profiles.
* **VWAP Deviation:** Current distance from institutional fair value bounds.
* **Volume Delta:** Net buying vs. net selling tick pressure.
* **Bollinger Width:** Structural volatility compression and expansion curves.

### The Role of PCA1_slope (The Speedometer)
While PCA1 captures the absolute directional weight of daily momentum, **PCA1_slope** provides the real-time velocity profile. It functions as the numerical first derivative of PCA1 across trailing intraday windows ($\Delta \text{PCA1} / \Delta t$).
* **When PCA1_slope > 0:** The dominant daily momentum cluster is actively accelerating intraday.
* **When PCA1_slope < 0:** Intraday momentum is fading out, signaling a structural exhaust peak or near-term reversal.

### Why PCA2 and PCA3 Are Omitted From Page 1
Mathematically, a 7-indicator PCA produces 7 individual vectors (PCA1 through PCA7). In financial markets, only the first three contain meaningful variance:
* **PCA1:** Momentum Factor (Dominant Cluster) — *Active on Page 1 Leaderboard*
* **PCA2:** Volatility Factor (Breakout/Compression Curves) — *Reserved for Phase 2 Engine Maps*
* **PCA3:** Participation Factor (Order Book/Liquidity Depth) — *Reserved for Phase 2 Engine Maps*

By isolating primary momentum and its first derivative first, Page 1 provides an ultra-fast core screen before secondary volatility variables are checked on later application pages.

---

## 4. Production Execution Labels & Compression Logic

Assets are assigned an **Execution Label** based on how their localized structural layers line up:

### 1. Crossing Soon (The True Predictive Energy Coil)
* Fast daily moving averages move into a tight coiling pattern.
* The absolute spread between `EMA9` and `EMA20` shrinks within a tight **0.3% band**:
  $$\frac{|\text{EMA9} - \text{EMA20}|}{\text{EMA20}} < 0.003$$
* **Trading Interpretation:** This represents a period of extreme volatility compression and stored structural energy. Because the stock has not broken out yet, your **2-to-3 minute chart verification delay is completely neutralized**. The radar highlights these candidates, giving you ample time to bring up the ticker on E*TRADE and prepare for execution before the breakout candlestick triggers.

### 2. Watch List (Institutional Momentum Extension)
* The daily trend stack foundation is securely established `UP` (`EMA20 > EMA50`).
* Both `EMA9` and `EMA20` tracking slopes are moving positive.
* **Trading Interpretation:** These are steady, active trend-following vehicles. However, because they are already moving, they possess higher extension risks than the compressed "Crossing Soon" entries.

### 3. Not Watch List (Intraday Slopes Exhaustion)
* Fast moving averages are trading above slower baselines, but the tracking slopes (`EMA9_slope` or `EMA20_slope`) have curled negative.
* **Trading Interpretation:** Intraday momentum is actively fading out or experiencing short-term distribution.

### 4. Setup Only (Passive Background Layer)
* Assets that pass the basic exchange filter gates but do not satisfy either the "Crossing Soon" compression band or active "Watch List" slope trajectories.

## 5. Advanced Visual Price-Action Guards (The Shield Protocols)

To completely protect trading capital from false breakout traps, morning fades, and sudden trend collapse structures (e.g., historical cases like SR), the engine programmatically emulates professional manual chart checks. If a ticker violates any single parameter below, it is purged from memory before it can touch the active leaderboard.

### 🛡️ Guard 1: Open-Drive Symmetry (Intraday Bullish Structural Health)
* **The Manual Logic:** Checking that today's candlestick profile is green and holding structural strength from the opening bell.
* **The Vector Filter:** `current_price >= session_open_price`
* **Trading Impact:** Instantly drops assets that break out early but fill down red below their open. This blocks high-risk intraday distribution spikes.

🛡️ Guard 2: Multi-Day Resistance Shield (Clear Room to Run)The Manual Logic: Visually scanning the daily chart back 5 sessions to guarantee price isn't smashing straight into a wall of resting sell orders.The Vector Filter: current_price > max_5day_overhead_resistanceTrading Impact: Confirms the ticker has crossed over the highest wicks of the trailing calendar week. It prevents entry into overhead distribution ceilings.

🛡️ Guard 3: The 10:30 AM Breakout Retest Gate (Sustained Momentum Hold)The Manual Logic: Verifying that after the initial 30 minutes of opening chaos settles, the asset holds above its early morning resistance marker.The Vector Filter: if current_bars > 45: current_price > morning_high_930_1000Trading Impact: Selects only the setups where institutions actively absorb afternoon pullbacks. If an asset spikes at 09:45 AM but slips deep back into yesterday's range later in the session, this gate drops it from the terminal radar.

---

## 6. Statistical Frequency Math Behind `recovery_prob`

The `recovery_prob` metric is a pure historical frequency estimator that acts as a behavioral tracking tool for individual tickers. It calculates the statistical frequency that an asset will reverse major intraday sell-offs based on its daily candles over a trailing **60-day window**.

For each session in the historical window, the engine runs this exact calculation:
1. **Find the Absolute Intraday Dip:**
   $$\text{Dip} = \text{High} - \text{Low}$$
2. **Find the Post-Low Rebound Window:**
   $$\text{Recovered} = \text{Close} - \text{Low}$$
3. **Evaluate the Threshold:** If the close successfully reclaims 50% or more of that day's high-to-low range, it counts as a successful recovery day:
   $$\text{Recovered} \ge 0.50 \times \text{Dip}$$
4. **Establish the Probability Output:**
   $$\text{Recovery\_Probability} = \frac{\text{Total Successful Recovery Days}}{\text{Total Sessions Scanned (60)}}$$

### Example Validation Baseline:
Out of 60 trading days scanned, a stock flushes intraday but reclaims over 50% of its range by the closing bell on 38 of those days:$$\text{Recovery_Probability} = \frac{38}{60} = 0.6333 \longrightarrow \mathbf{63.33%}$$This confirms the ticker has a reliable behavioral habit of reversing dips, validating it as a safe structural candidate for morning capitulation trading operations.

7. The Predictive Low-Lag Scoring Engine

To neutralize the 2-to-3 minute delay factor associated with opening chart layouts manually on E*TRADE, the system utilizes an Un-Clipped, Real-Time Scoring Formula.

By removing artificial clipping constraints (.clip(lower=0)), any asset experiencing active selling pressure, low volume, or negative velocity is immediately penalized, allowing true, compressed breakout springs to ascend the board.

$$\text{Score} = (\text{Trend == UP} \times 2.0) + (\text{Execution == Watch List} \times 1.0) + (\text{Execution == Crossing Soon} \times 4.0) + \text{RVOL} + \text{PCA1} + (\text{PCA1_slope} \times 3.5)$$

Score Priority Framework:
🎯 The Predictive Alpha Core (Scores ≥ 8.5): Tickers with a long-term macro trend, ultra-tight price coiling (Crossing Soon), an active volume catalyst (RVOL > 1.0), and sharp real-time acceleration (PCA1_slope).
🐢 The Lagging Extension Core (Scores 5.0 to 8.4): Assets with solid historical trends (Watch List) that are currently extended away from their moving averages or lack immediate intraday volume support.
⚠️ The Penetration Penalty (Scores < 5.0): Inactive or bleeding assets. Pullbacks and negative slope angles drag down the overall ranking, pushing zombie stocks off the front page.

8. High-Speed Vectorized Funnel Processing ArchitectureTo bypass individual network connection overhead and eliminate Yahoo Finance API rate limiting blocks (HTTP 429), Page 1 uses a single-pass vectorized data ingestion layer:

                  [ load_universe() Ticker SANDBOX Pool ]
                                     │
                                     ▼  
              STAGE 1: Vectorized Batch Download Layer
     yf.download() executes 2 unified hits for all assets simultaneously
              (1 Contract for Daily, 1 Contract for 1-Min)
                                     │
                                     ▼  
                 STAGE 2: RAM Column Unpivoting Layer
      Swaps levels en masse so Ticker symbols occupy Level 0
        Bypasses expensive pandas cross-section loops (.xs)
                                     │
                                     ▼  
                 STAGE 3: Core Capacity Funnel Gate
     Drops assets with Avg_Volume_20d < 250k or Close Price outside $40-$110
                                     │
                                     ▼  
                 STAGE 4: Vectorized Indicators Math
    Calculates moving averages, ATR%, RVOL, and un-clipped Scores in RAM
                                     │
                                     ▼  
              [ Final Output: Elite Low-Lag Trading Table ]


8. High-Speed Vectorized Funnel Processing ArchitectureTo bypass individual network connection overhead and eliminate Yahoo Finance API rate limiting blocks (HTTP 429), Page 1 uses a single-pass vectorized data ingestion layer:

                  [ load_universe() Ticker SANDBOX Pool ]
                                     │
                                     ▼  
              STAGE 1: Vectorized Batch Download Layer
     yf.download() executes 2 unified hits for all assets simultaneously
              (1 Contract for Daily, 1 Contract for 1-Min)
                                     │
                                     ▼  
                 STAGE 2: RAM Column Unpivoting Layer
      Swaps levels en masse so Ticker symbols occupy Level 0
        Bypasses expensive pandas cross-section loops (.xs)
                                     │
                                     ▼  
                 STAGE 3: Core Capacity Funnel Gate
     Drops assets with Avg_Volume_20d < 250k or Close Price outside $40-$110
                                     │
                                     ▼  
              STAGE 4: Hardened Visual Checklist Protections
      Filters out files trading below Open, under 5-day Highs, or below early highs
                                     │
                                     ▼  
                 STAGE 5: Vectorized Indicators Math
    Calculates moving averages, ATR%, RVOL, and un-clipped Scores in RAM
                                     │
                                     ▼  
              [ Final Output: Elite Low-Lag Trading Table ]


## 9. Trade Execution Process Detail

1. The Multi-Day Structure Checklist (The Daily Chart)

Before looking at today’s action, professional traders analyze the immediate historical boundaries to ensure they aren't buying directly into a brick wall of resistance.

                  [ DAILY CHART VISUAL INSPECTION ]
                                  │
      ┌───────────────────────────┴───────────────────────────┐
      ▼                                                       ▼
CRITERION 1: Clear Room to Run              CRITERION 2: The Multi-Day Base
Is price clear of the last 5 days' highs?    Has the asset consolidated for 3-5 days?

🛡️Criterion 1: "Room to Run" (Immediate Resistance Check)
The Visual: Look back at the daily candlesticks for the last 5 to 10 sessions.

The Rule: Ensure the current intraday price is breaking above the highest wick of those previous days. If the current price is right under a previous daily high, do not enter. Institutional supply sits there, and the stock is highly likely to hit that level and reverse.

🛡️Criterion 2: The Multi-Day Base (Catalyst Check)
The Visual: Look for a horizontal cluster of daily candles moving sideways in a tight, narrow channel over the last 3 to 5 days.

The Rule: A sustainable breakout requires a base. If a stock has already surged green for 3 days straight without pausing, today's intraday breakout is exhausted. You want to buy the first expansion day out of a multi-day tight consolidation base.

2. Today's Valuation Checklist (The Intraday Anchor)

Once the daily chart is cleared, a trader evaluates today’s opening footprint to confirm the asset is experiencing controlled accumulation rather than erratic retail chasing.

                 [ INTRADAY CHART VISUAL INSPECTION ]
                                  │
      ┌───────────────────────────┴───────────────────────────┐
      ▼                                                       ▼
CRITERION 3: Open-Drive Symmetry            CRITERION 4: The 10:30 AM Retest
Did the asset open near its low?            Does price hold above the morning high?

🛡️Criterion 3: Open-Drive Symmetry (Price vs. Open)

The Visual: Check where today's absolute Low is relative to today's Open price.

The Rule: The ideal bullish intraday setup opens, dips down just a fraction (less than 0.2%), and immediately drives upward. If today's low is far below the open (meaning it flushed hard off the open before recovering), it indicates highly erratic, emotional retail supply. You want clean "Open-Drive" setups where buyers took control from the opening print.

🛡️Criterion 4: The 10:30 AM Pullback Retest (Sustained Hold)

The Visual: Look at the high print established during the first 15–30 minutes of the morning session (09:30–10:00 AM).The Rule: Never buy the initial breakout candle blindly. Wait for the stock to experience its first inevitable pullback (usually between 10:00 AM and 10:30 AM). If the stock pulls back but holds strictly above that early morning high line, it proves institutions are actively absorbing the selling pressure to defend the move. This retest is your optimal, low-risk entry point.



3. The Institutional Tape Checklist (E*TRADE Order Book)

Professional traders monitor the raw momentum of the current candle to separate sustainable institutional block buying from weak retail chasing

                    [ LIVE ORDER BOOK INSPECTION ]
                                  │
      ┌───────────────────────────┴───────────────────────────┐
      ▼                                                       ▼
CRITERION 5: Thick Ask Clearing              CRITERION 6: The 3-Bar Volume Steps
Are heavy block orders being absorbed?      Are volume bars stepping up consecutively?

🛡️Criterion 5: Thick Ask Clearing (Level 2 Book)

The Visual: Open your E*TRADE Level 2 order book and look at the sizes resting on the Ask side.

The Rule: Counter-intuitively, you want to see large sell orders stacked closely together on the Ask. If the Ask side is completely thin and empty, it takes very little buying power to warp the price higher, creating a fake breakout. When you see large institutional blocks sitting on the Ask and the live tape shows buyers aggressively chewing right through those thick walls, it confirms authentic, institutional demand.

🛡️Criterion 6: The 3-Bar Volume EscalationThe Visual: Look at your 1-minute volume histogram bars at the bottom of the intraday chart.The Rule: Avoid entry if the breakout is driven by a single, solitary giant volume spike (this is often a block trade print cross or an exhaustion climax). A sustainable move exhibits consecutive ascending stair-steps across 3 or more 1-minute volume bars. This visual pattern proves new buyers are continuously entering the arena to sustain the price appreciation

## 10. Manual Charts
For professional momentum traders using manual charts (like your E*TRADE setup) alongside a real-time scanner, the absolute best practice is to use a Dual-Timeframe Display Layout:

Chart 1 (The Tactical Executive): 1-Minute Interval | 1-Day View

Chart 2 (The Compass Matrix): 5-Minute Interval | 3-Day ViewHere is exactly how these two lookups interact with your rules to eliminate false signals and give you a low-risk entry

🗺️ Chart 2: The 5-Minute Chart (3-Day View) — The Compass Matrix

You open this chart first to confirm the stock has an authentic macro structural engine. It acts as a bridge between the Daily chart and the immediate 1-Minute entry.

How it applies to your rules:

Criterion 1 (Room to Run): A 3-day view on a 5-minute chart perfectly exposes the exact intraday high wicks from yesterday and two days ago. You can see instantly if today's price is genuinely breaking out into "clear air" or if it is running straight into a wall of resting sell orders from yesterday's session.

Criterion 2 (The Multi-Day Base): It shows you if the stock has been consolidating sideways over the last 48–72 hours or if it is already vertically exhausted.

The Blueprint: If the 5-minute chart shows the price breaking out above yesterday's high wick after a long sideways consolidation, the macro compass is Green.


🎯 Chart 1: The 1-Minute Chart (1-Day View) — The Tactical Executive

Once the 5-minute chart grants a macro green light, you switch your eyes exclusively to today's 1-minute chart to hunt for your precise execution trigger.

How it applies to your rules:

Criterion 3 (Open-Drive Symmetry): You can look at the very first few candles of the morning (09:30–09:35 AM) to confirm the stock opened, held near its low, and drove aggressively upward.

Criterion 4 (The 10:30 AM Retest): This is where you draw your horizontal support line right at the peak of the 09:30–10:00 AM morning high. You watch the 1-minute candles pull back toward that line between 10:00 AM and 10:30 AM.

The Blueprint: You trigger your buy order on E*TRADE when a 1-minute candle bounces off that morning high line, accompanied by Criterion 6 (Ascending 3-Bar Volume stair-steps).

💻 Recommended Workspace Screen LayoutTo make your 2-to-3 minute lookup loop as fast as possible, arrange your E*TRADE terminal monitor layout using a Split-Window view:

┌───────────────────────────────────────┬───────────────────────────────────────┐
│              CHART 1                  │              CHART 2                  │
│       1-Minute / 1-Day View           │       5-Minute / 3-Day View           │
│                                       │                                       │
│   - Tracks the 10:30 AM Retest Line   │   - Verifies Yesterday's High Wicks   │
│   - Monitors 3-Bar Volume Steps       │   - Confirms Multi-Day Baseline Coils │
│   - Confirms Open-Drive Symmetry      │   - Screens out Overhead Supply Walls │
└───────────────────────────────────────┴───────────────────────────────────────┘

By keeping these two exact views open side-by-side for your candidates, you instantly see the macro strength (5-min) and the microscopic execution trigger (1-min) simultaneously, completely neutralizing the market delay factor.

✔️ End of Page 1 Technical Documentation Document

