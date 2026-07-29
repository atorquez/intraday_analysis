📈 Page 5 — Exceptions & Outlier Engine
Version: 2026‑07‑28
Purpose: Identify intraday anomalies across the full universe (SP500, NASDAQ1000, NYSE100, IPO, OTHER) regardless of market regime.

Page5 is the intraday anomaly radar. It does not depend on Page1, Page2, or Page3. It runs in all market conditions, including bearish, choppy, and low‑signal days. Its goal is to surface unusual tickers that may deserve deeper inspection.

🔍 1. Summary
Page5 scans the entire universe and computes:
RS Divergence — relative strength anomaly
RVOL — volume participation anomaly
ROC — short‑term momentum anomaly
Volume Intensity — last‑bar volume anomaly
OR15 Breakout Score — opening range breakout anomaly
Structural Readiness — compression signal from Page4
Intraday Actionability — alignment signal from Page4
Price — latest intraday close

These metrics are combined into a Composite Outlier Score, producing a ranked list of the strongest intraday anomalies.

Page5 does not generate trade signals. It identifies exceptions — tickers doing something unusual.

🧮 2. Calculations
Price
Latest intraday close:
Price = intraday_df['Close'].iloc[-1]

RS Divergence
Deviation between intraday strength and recent daily trend:
RS_Divergence = (current_price / previous_close)
               - (daily_close[-1] / daily_close[-5])

RVOL (Relative Volume)
Intraday volume vs 20‑day average daily volume:
RVOL = intraday_volume / avg_daily_volume_20

ROC (Rate of Change)
Short‑term intraday momentum:
ROC = (Close[-1] / Close[-10]) - 1

Volume Intensity
Last intraday volume bar vs average intraday volume:
Vol_Intensity = last_volume / avg_intraday_volume

OR15 Breakout Score
if last_price > OR15_High:        OR15_Score = 3.0
elif last_price > OR15_Close:     OR15_Score = 1.5
elif last_price < OR15_Low:       OR15_Score = -2.0
else:                             OR15_Score = 0.0

Structural Readiness
From Page4 EMA compression engine:
Structural_Ready = df_struct['Structural_Ready'].iloc[-1]

Intraday Actionability
From Page4 intraday alignment engine:
Actionable = check_intraday_actionability(current_price,
                                          previous_close,
                                          intraday_df)

Composite Score
Score = RS_Divergence
      + ROC
      + log1p(RVOL)
      + log1p(Vol_Intensity)
      + OR15_Score

📘 3. Definitions
Outlier / Exception
A ticker showing abnormal intraday behavior such as:
RVOL spike
ROC burst
OR15 breakout
unusual volume intensity
structural compression
intraday alignment

OR15 (Opening Range 15)
The first 15 minutes of the trading day:
OR15 High
OR15 Low
OR15 Open
OR15 Close

Used as a reference map for breakout/breakdown detection.

RS Divergence
Measures deviation between intraday strength and recent daily trend.

RVOL
Relative Volume: intraday volume vs historical average daily volume.

ROC
Rate of Change: short‑term momentum indicator.

Volume Intensity
Measures whether the latest volume bar is unusually large.

Structural Readiness
Indicates compression and readiness for movement (from Page4).

Intraday Actionability
Indicates whether intraday conditions support potential execution (from Page4).

✔️ End of Page5 Documentation (.md)