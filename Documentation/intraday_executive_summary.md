# Executive Summary — SPC Intraday Readiness Model (2026 Edition)

The Intraday Readiness Model is a multi‑factor, PCA‑enhanced intraday scanning system designed to identify statistically favorable trading locations during the trading session. It integrates daily historical structure with real‑time intraday momentum, volatility, and participation signals to produce a ranked universe of tickers and classify each into two orthogonal dimensions:

1. **Buy_Signal** — value‑momentum alignment  
2. **Execution_Status** — intraday actionability  

The model is intentionally **location‑based**, not predictive. It identifies *where* opportunity exists, while the trader determines *when* to act through visual inspection and intraday monitoring.

---

## Two‑Stage Scientific Framework

### **Stage 1 — Daily Structure (Value + Trend Foundation)**  
Using 3 months of daily OHLCV data, the model computes:

- Trend alignment (EMA9, EMA20, EMA50)  
- Volatility (ATR%)  
- Participation (RVOL)  
- Opening pressure (Gap%)  
- Value zones (BuyZone10, BuyZone5 via rolling percentiles)  

This establishes the macro context and identifies statistically favorable value areas.

---

### **Stage 2 — Intraday PCA Engine (Momentum + Volatility + Participation)**  
Using intraday‑sensitive indicators, the model applies StandardScaler + PCA to extract three orthogonal “super‑signals”:

- **PCA1 — Momentum cluster**  
- **PCA2 — Volatility cluster**  
- **PCA3 — Participation cluster**  

These components update intraday, giving the model real‑time sensitivity to market shifts.

---

## Decision Layer — BuyZone + PCA + Trend Integration

The model combines:

- Value proximity (BuyZone10/5)  
- Trend alignment  
- PCA1 momentum  
- VMAS (value–momentum alignment)  
- Distance from value  
- Heatmap classification  

To produce the **Buy_Signal** classification:

- **Strong Buy Zone**  
- **Buy Zone**  
- **Neutral**  
- **Avoid**  
- **Extended**

This dimension answers:  
**“Is the ticker in a statistically favorable location?”**

---

## NEW (2026) — Execution Readiness Status

Execution readiness evaluates whether a ticker is **actionable intraday**, combining trend structure, VWAP behavior, RVOL participation, volatility state, and PCA momentum slope.

The model classifies each ticker into:

- **Ready** — Trend aligned, VWAP reclaimed, momentum supportive  
- **Crossing Soon** — Alignment forming, momentum improving  
- **Intraday False Ready** — Appears ready but fails VWAP/RVOL/volatility confirmation  
- **Setup Only** — Structure forming but missing momentum or participation  
- **UNKNOWN** — Insufficient or conflicting data  

This dimension answers:  
**“Is the ticker actionable *right now*?”**

---

## Trader Workflow

The trader runs the model periodically (e.g., every 15–30 minutes), identifies **Strong Buy Zone + Ready** candidates, and then uses visual inspection to confirm timing based on:

- EMA9 reclaim  
- RSI > 50  
- PCA1 slope  
- RVOL trend  
- Higher‑low formation  
- VWAP behavior  

This creates a disciplined intraday workflow where the model finds **location**, and the trader confirms **timing**.

---

## Summary

The Intraday Readiness Model is a multi‑factor, PCA‑enhanced, value‑momentum classifier that identifies statistically favorable intraday trading locations using daily historical structure and real‑time intraday signals. It produces two orthogonal classifications — **Buy_Signal** and **Execution_Status** — enabling traders to focus only on high‑quality, actionable intraday opportunities. The model is location‑based, not predictive: it finds *where* opportunity exists, while the trader confirms *when* to act.



