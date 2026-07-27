# 📈 Page 2 — Intraday Engine  
**Version:** 2026‑07‑21  
**Purpose:** Pure intraday scanner based on structural tickers filtered by Page1.

Page2 takes the **structural universe** from Page1 and applies a **real‑time intraday engine** to determine:

- Intraday momentum  
- Intraday acceleration  
- Volume participation  
- Volatility availability  
- Execution readiness  
- Position status  
- Exit triggers  
- Risk efficiency  
- Fundamental warnings  

It is designed to support **live trading**, **position management**, and **intraday decision‑making**.

---

## 🔍 1. Structural Universe Dependency  
Page2 **requires Page1** to run first.

Page1 provides:

- Trend Stack  
- PCA1 (Daily Momentum Cluster)  
- PCA1_slope (Daily Acceleration)  
- Regime Classification  
- Execution Labels  

Page2 then applies **intraday logic** on top of Page1’s structural universe.

If Page1 has not been run, Page2 will stop and warn the user.

---

## 🧠 2. Intraday Engine Overview  
The intraday engine computes:

### **Intraday Indicators**
- EMA9 / EMA20  
- VWAP  
- ATR%  
- RVOL  
- Intraday returns  
- Intraday volatility  
- Intraday volume patterns  

### **Intraday PCA Engine**
PCA1 is recomputed intraday using:

- EMA spread (EMA9 − EMA20)  
- VWAP deviation  
- Intraday returns  
- ATR%  
- RVOL  

This produces:

- **PCA1 (Intraday Momentum Cluster)**  
- **PCA1_slope (Intraday Acceleration)**  

These are the core of Page2.

---

## 📘 3. Intraday PCA Definitions

### **PCA1 — Intraday Momentum Cluster**
Captures the dominant intraday factor across:

- EMA spread  
- VWAP deviation  
- pct_return  
- ATR%  
- RVOL  

High PCA1 → strong intraday momentum  
Low PCA1 → weak intraday momentum  
Negative PCA1 → intraday reversal pressure

---

### **PCA1_slope — Intraday Acceleration**
Measures the **rate of change** of PCA1.

- Positive → momentum accelerating  
- Negative → momentum fading  
- Strong negative → reversal risk  

This is one of the most important signals in Page2.

---

## 🎯 4. Execution Readiness (Intraday)

Execution readiness is classified using:

- EMA alignment  
- EMA slopes  
- PCA1  
- PCA1_slope  
- EMA9/EMA20 compression  

### **Labels**
- **Watch List** — trend aligned, momentum accelerating  
- **Crossing Soon** — EMA9 ≈ EMA20 compression  
- **Setup Only** — partial alignment  
- **Not Watch List** — trend aligned but momentum fading  

This is the intraday equivalent of Page1’s execution labels.

---

## 📊 5. Intraday Buy Signal

Based on PCA1 and VWAP:

| Condition | Signal |
|----------|--------|
| PCA1 > 0 and price > VWAP | **Strong Intraday** |
| PCA1 > 0 | **Intraday Buy** |
| PCA1 < 0 | **Avoid** |
| Otherwise | **Neutral** |

This is the simplest and most direct intraday signal.

---

## ⚡ 6. Risk Efficiency Score  
Risk efficiency measures:



\[
\text{Risk Efficiency} = \frac{|PCA1\_slope| \times RVOL}{ATR\%}
\]



High score → strong acceleration with volume  
Low score → weak acceleration or high volatility

---

## 📉 7. Position Status (Intraday)

| Condition | Status |
|----------|--------|
| PCA1 < 0 or PCA1_slope < −0.5 | 🛑 EXIT — Momentum Reversed |
| Price < VWAP and EMA9 > EMA20 | ⚠️ TIGHTEN — Below VWAP |
| RVOL < 1.5 or PCA1_slope < 0 | 📉 FADE — Volume Drying |
| Otherwise | ✅ HOLD — Structure Intact |

This is used for **active position management**.

---

## 🎯 8. Exit Trigger Engine  
Exit triggers consider:

- P&L %  
- EMA9 breach  
- EMA20 breach  
- PCA1 reversal  
- PCA1_slope collapse  
- RVOL drying  
- Session extension (parabolic moves)

### **Examples**
- **TAKE PROFIT** — EMA9 breach with +5% or +10%  
- **EXTREME EXIT** — +15% extension + EMA9 break  
- **REVERSAL EXIT** — PCA1 negative + EMA9 break  
- **TREND FAILURE** — price below EMA20 + PCA1 negative  

Each exit trigger produces:

- Exit Action  
- Exit Color  
- Exit Message  

---

## 📈 9. Price Extension Classifier

| Extension % | Label |
|-------------|--------|
| ≥ 10% | 🚨 Extended |
| ≥ 5% | ⚠️ Chasing |
| ≥ 2% | ✅ Good Entry |
| < 2% | 💤 Baseline |

Used to avoid chasing extended moves.

---

## 🛡️ 10. Stop Recommendation Engine  
Stops and targets are computed using ATR dollars:



\[
ATR\$ = ATR\% \times \text{Price}
\]



Outputs:

- Suggested Stop  
- Suggested Target  
- Risk Dollars  
- Distance to Stop %  
- P&L % (if in position)

---

## 📚 11. Fundamental Warning System  
Page2 integrates Page1’s fundamental scoring:

| Fund Score | Warning |
|------------|---------|
| ≥ Strong Threshold | **Strong** — normal size |
| Weak–Strong | **Moderate** — reduce 25% |
| < Weak Threshold | **Weak** — reduce 50% or avoid |

This helps prevent overnight risk in weak companies.

---

## 🧮 12. Intraday Score  
Composite score:

- EMA alignment  
- RVOL  
- ATR%  
- VWAP reclaim  
- PCA1  
- PCA1_slope  

Used for ranking the intraday universe.

---

## 📂 13. Manual Position Tracker  
Page2 includes a built‑in position tracker:

- Log entries  
- Track time in trade  
- Track P&L  
- Track stop/target  
- Track exit signals  
- Track position status  

This makes Page2 a **live trading dashboard**.

---

## 🚀 14. Live Intraday Universe Matrix  
Final output includes:

- Price  
- EMA alignment  
- VWAP status  
- RVOL  
- ATR%  
- PCA1  
- PCA1_slope  
- Execution readiness  
- Intraday score  
- Buy signal  
- Risk efficiency  
- Position status  
- Price extension  
- Fundamental warnings  
- Stops/targets  
- Exit signals  

Sorted by:

- P&L % (if in position)  
- Intraday Score  
- Risk Efficiency  

---

## 🧩 15. Relationship Between Page1 and Page2

### **Page1 = Structural Engine**
- Trend  
- Momentum cluster (PCA1)  
- Daily acceleration (PCA1_slope)  
- Regime  
- Execution labels  

### **Page2 = Intraday Engine**
- Intraday momentum cluster  
- Intraday acceleration  
- Volume participation  
- Volatility availability  
- Position management  
- Exit triggers  

Together they form a **full trading system**:

- Page1 → *What to watch*  
- Page2 → *How to trade it intraday*  

---

## 📘 16. Future Enhancements (Optional)
- PCA2 (Volatility Factor)  
- PCA3 (Participation Factor)  
- Intraday regime detection  
- Intraday PCA visualization  
- Multi‑timeframe PCA  
- Intraday structural map  

---

# ✔️ End of Page2 Documentation
