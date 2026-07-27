# 📘 Page 4 — Choppy Market Opportunity Engine  
**Version:** 2026‑07‑21  
**Purpose:** A specialized scanning model designed for **choppy market regimes**, where traditional trend‑based signals lose reliability.

Page6 activates **only** when Page3 (Regime Engine) identifies:

### **CHOPPY_MARKET**

This engine focuses on:

- EMA compression  
- Independent momentum (Residual PCA1)  
- Sector participation strength  
- Noise‑filtered intraday triggers  
- Compression → expansion setups  

It is designed to extract high‑quality opportunities in otherwise noisy, rotational, low‑conviction environments.

---

## # 1. Executive Summary — Why Page6 Exists

Choppy markets are defined by:

- Weak or inconsistent trend structure  
- Low directional conviction  
- High intraday noise  
- Sector rotation  
- Frequent failed breakouts  

In these conditions, **Page1 and Page2 lose reliability** because trend‑based signals break down.

Page6 provides a **structure‑first, noise‑filtered** approach to identify:

- localized compression  
- independent momentum  
- sector‑aligned setups  
- intraday expansion triggers  

This makes Page6 the **specialized engine** for choppy regimes.

---

## # 2. Stage 1 — Structural Compression Engine (EMA Compression)

Trend stacking is unreliable in choppy markets.  
Instead, Page6 uses **EMA Compression**, detecting when EMAs coil tightly.

### **EMA Compression Definition**
- EMA9, EMA20, EMA50 converge within a narrow band  
- EMA_Spread ≤ 1.5%  
- Price above EMA50  
- Indicates **stored energy** and **potential expansion**

Compression setups are statistically favorable in choppy regimes because they represent **localized structure** independent of broad market trend.

---

## # 3. Stage 2 — Residual PCA1 (Independent Momentum)

Choppy markets are dominated by index noise.  
Page6 isolates **ticker‑specific momentum** by removing SPY’s influence.

### **Residual PCA1 Calculation**
1. Compute PCA1 for ticker  
2. Compute PCA1 for SPY  
3. Regress ticker PCA1 against SPY PCA1  
4. Extract residuals (momentum not explained by SPY)

### **Residual PCA1 > 0 and expanding** indicates:
- independent strength  
- sector rotation  
- institutional accumulation  
- non‑index‑driven movement  

Residual PCA1 is the **core signal** of Page6.

---

## # 4. Stage 3 — Sector Participation Strength (ATR% + RVOL)

Choppy markets often rotate between sectors.  
Page6 identifies the **3 strongest sectors** using:

### **Sector Filters**
- **ATR%** → volatility availability  
- **RVOL** → participation strength  
- **Compression_Spread** → structural readiness  
- **Residual PCA1** → independent momentum  

Only tickers inside the **top 3 sectors** are considered actionable.

---

## # 5. Stage 4 — Intraday Actionability (Noise‑Filtered Triggers)

Page6 applies strict intraday filters to avoid false signals:

### **Intraday Filters**
- Opening gap < 1.5%  
- VWAP reclaim  
- RVOL spike  
- PCA1_slope > 0  
- Higher‑low formation (clean intraday structure)  

These ensure Page6 only triggers when **real intraday expansion** is forming.

---

## # 6. Execution Readiness — Page6 Classification

Tickers passing all Page6 filters are classified as:

### **Actionable (Choppy Model)**
- EMA compression  
- Residual PCA1 expanding  
- Sector strength confirmed  
- Intraday acceleration present  

### **Structural Only**
- Compression present  
- Residual PCA1 present  
- Sector strength present  
- Intraday confirmation missing  

### **Not Actionable**
- No compression  
- No independent momentum  
- Weak sector  
- Intraday noise  

---

## # 7. Trader Workflow — Page6

The trader uses Page6 in a **three‑stage workflow**:

---

### **Stage A — Structural Compression Scan**
Identify:
- EMA compression  
- Residual PCA1 expansion  
- Sector strength  

---

### **Stage B — Intraday Confirmation**
Confirm:
1. VWAP reclaim  
2. RVOL spike  
3. PCA1_slope > 0  
4. Opening gap < 1.5%  
5. Clean intraday structure  

---

### **Stage C — Execution**
Act only when:
- structural compression  
- independent momentum  
- sector strength  
- intraday acceleration  
all align.

---

## # 8. Technical Glossary

### **EMA Compression**  
Tight coiling of EMAs indicating stored energy.

### **Residual PCA1**  
Momentum not explained by SPY; measures independent strength.

### **Sector Strength**  
ATR% + RVOL + compression + residual PCA1.

### **VWAP Reclaim**  
Price moves above institutional cost basis.

### **RVOL Spike**  
Volume expansion indicating institutional activity.

### **PCA1_slope**  
Real‑time acceleration of intraday momentum.

---

# ✔️ End of Page6 Documentation
