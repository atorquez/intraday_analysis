# 📘 Page 3 — Regime‑Adaptive Intraday Model  
**Version:** 2026‑07‑21  
**Purpose:** A lightweight, regime‑aware momentum engine that complements Page1’s structural scanner.

Page3 does **not** scan the full universe.  
It uses **only the filtered tickers selected by Page1**, then applies:

- Market regime detection (SPY/QQQ)  
- Regime‑specific filtering  
- Intraday PCA trigger  
- Momentum continuation logic  
- Long/short routing  
- Multi‑timeframe guardrails  

This creates a **two‑layer system**:

- **Page1 → Structural Value Engine**  
- **Page3 → Regime‑Aligned Momentum Engine**

---

## # 1. Executive Summary — How Page3 Works with Page1

### ✔ Page1 = Structural Value Engine  
Page1 scans the entire universe (S&P 500 + Nasdaq 100) and computes:

- Trend Stack  
- BuyZone  
- VMAS  
- PCA(7) Daily Momentum Cluster  
- Execution Readiness  
- ATR%  
- RVOL  
- Structural Score  

Page1 selects the **best 80–120 tickers**.

### ✔ Page3 = Regime Momentum Engine  
Page3 reads Page1’s selected tickers and uses:

- Trend  
- Execution  
- Gap%  
- Market regime (SPY/QQQ)  
- Intraday PCA trigger  

Page3 finds **momentum continuation** opportunities aligned with the current market environment.

---

## # 2. Why Page3 Exists

### ✔ Page1 is strongest in bullish markets  
It finds:

- Deep value dips  
- Mid‑value dips  
- Structural alignment + intraday readiness  

### ✔ Page3 is strongest in bearish or mixed markets  
It finds:

- Breakdown momentum  
- Relative strength longs  
- Waterfall shorts  
- Regime‑aligned intraday triggers  

### ✔ Page3 activates when Page1 has no candidates  
This prevents forcing trades in bad market conditions.

---

## # 3. Stage 1 — Automatic Market Regime Assessment

At the open, Page3 evaluates SPY and QQQ using:

- Opening gap  
- Intraday return  
- Early volatility footprint  

This produces one of three regimes:

---

### 🟢 **BULLISH_MARKET**  
Strong upward opening momentum.

Page3 targets:

- Breakout continuation  
- UP + Ready tickers from Page1  
- High‑momentum intraday triggers  

---

### 🔴 **BEARISH_MARKET**  
Downward gap or morning flush.

Page3 switches to:

- Relative strength longs (rare green stocks)  
- Waterfall shorts (DOWN + Ready / Crossing Soon)  
- Breakdown continuation  

---

### 🟡 **CHOPPY_MARKET**  
Indecisive, mean‑reverting environment.

Page3 enforces:

- Caution  
- Tight risk  
- Reduced position sizing  
- Avoiding trend continuation setups  

---

## # 4. Stage 2 — How Page3 Builds Its Watchlist

Page3 does **not** scan 600–700 tickers.

It uses **only the tickers Page1 already selected**.

Example:

If Page1 selected **100 tickers**, Page3 applies regime‑specific filters:

---

### ✔ Bullish Regime  
Select tickers where:

- Trend = UP  
- Execution = Ready  
- Gap% positive  
- Strong intraday alignment  

---

### ✔ Bearish Regime — Relative Strength  
Select tickers where:

- Trend = UP  
- Gap% positive  
- Strong intraday alignment  
- Rare green stocks in a red market  

---

### ✔ Bearish Regime — Short Candidates  
Select tickers where:

- Trend = DOWN  
- Execution ∈ {Ready, Crossing Soon}  
- Gap% negative  
- Breakdown continuation potential  

---

This ensures Page3 **never forces trades** in the wrong market environment.

---

## # 5. Stage 3 — Intraday PCA Trigger (1‑Minute)

Once the watchlist is built, Page3 runs a **fast intraday trigger** using:

- EMA9/EMA20 cross  
- EMA stack  
- Residual return (beta‑neutral)  
- PCA_Alpha (momentum PCA)

This is **not** Page1’s PCA engine.

### ✔ Page1 PCA  
Uses **7 PCA features** (RSI, ROC, StochK, curvature, VWAP, delta, BBW).

### ✔ Page3 PCA  
Uses **2 PCA features**:

- `ema_spread`  
- `residual_ret`  

This makes Page3:

- extremely fast  
- regime‑sensitive  
- ideal for continuation setups  

---

## # 6. Intraday Trigger Logic

### ✔ Long Trigger  
- Price crosses above EMA9  
- EMA9 > EMA20  
- PCA_Alpha rising (independent buying)  
- Residual return positive  

### ✔ Short Trigger  
- Price crosses below EMA9  
- EMA9 < EMA20  
- PCA_Alpha falling (independent selling)  
- Residual return negative  

These triggers are **regime‑aligned**, meaning Page3 only fires them when the market regime supports the direction.

---

## # 7. Multi‑Timeframe Guardrail

To prevent data leakage:

- Daily indicators (Page1)  
- Intraday indicators (Page3)  

are computed **separately**.

This avoids:

- drift errors  
- mixed‑timeframe contamination  
- false alignment signals  

---

## # 8. Glossary — Core Module Definitions

### **Market Index Beta (β)**  
Measures how much a stock moves relative to SPY/QQQ.  
Beta 1.0 = moves in sync with the market.

---

### **Residual Return**  
Independent price movement after removing market influence:



\[
\text{Residual} = \text{Stock Return} - (\beta \times \text{Market Return})
\]



Used to detect **true alpha**.

---

### **PCA_Alpha**  
A single‑component PCA capturing pure intraday alpha  
(independent buying or selling pressure).

---

### **EMA Stack**  
EMA9 > EMA20 → bullish micro‑trend  
EMA9 < EMA20 → bearish micro‑trend

---

### **Gap%**  


\[
\text{Gap\%} = \frac{\text{Open} - \text{Prev Close}}{\text{Prev Close}} \times 100
\]



Used for regime classification.

---

### **Regime**  
Market environment classification:

- Bullish  
- Bearish  
- Choppy  

---

## # 9. Relationship Between Page1 and Page3

### **Page1 = Structural Engine**
- Daily momentum  
- Trend stack  
- PCA1  
- Execution readiness  
- Regime classification  
- Structural value  

### **Page3 = Regime Momentum Engine**
- Intraday continuation  
- Regime‑aligned triggers  
- Residual return  
- PCA_Alpha  
- Fast EMA stack logic  

Together they form a **complete trading system**:

- Page1 → *What is structurally strong?*  
- Page3 → *What is intraday‑ready in this regime?*  

---

## # 10. Future Enhancements (Optional)

- PCA_Alpha_slope  
- Multi‑regime volatility filters  
- Intraday regime transitions  
- Relative strength scoring  
- Short‑side optimization  
- Regime‑specific position sizing  

---

# ✔️ End of Page3 Documentation

