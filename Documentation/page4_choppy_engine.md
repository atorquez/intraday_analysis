📘 Page 4 — Choppy Market Opportunity Engine

Version: 2026‑07‑28Purpose: Specialized scanning model for choppy market regimes, where trend‑based signals lose reliability and noise dominates intraday movement.
Page4 activates only when Page3 (Regime Engine) identifies:

CHOPPY_MARKET

This engine focuses on:
EMA compression
Independent momentum (Residual PCA1)
Intraday actionability
Noise‑filtered triggers
Compression → expansion setups

It is designed to extract high‑quality opportunities in noisy, rotational, low‑conviction environments.

1. Executive Summary — Why Page4 Exists

Choppy markets are defined by:
Weak or inconsistent trend structure
Low directional conviction
High intraday noise
Sector rotation
Frequent failed breakouts

In these conditions, Page1 and Page2 lose reliability because trend‑based signals break down.

Page4 provides a structure‑first, noise‑filtered approach to identify:
Localized compression
Independent momentum
Intraday alignment
Early expansion triggers

This makes Page4 the specialized engine for choppy regimes.

2. Stage 1 — EMA Compression Engine

Trend stacking is unreliable in choppy markets. Page4 uses EMA Compression to detect when EMAs coil tightly.

EMA Compression Definition

Compression occurs when:
EMA9, EMA20, EMA50 converge within a narrow band
EMA_Spread ≤ 1.5%
Price above EMA50

Compression indicates stored energy and potential expansion, independent of broad market trend.

3. Stage 2 — Residual PCA1 (Independent Momentum)

Choppy markets are dominated by index noise. Page4 isolates ticker‑specific momentum by removing SPY’s influence.

Residual PCA1 Calculation
Compute PCA1 for ticker
Compute PCA1 for SPY
Regress ticker PCA1 against SPY PCA1
Extract residuals (momentum not explained by SPY)

Interpretation

Residual PCA1 > 0 and expanding indicates:
Independent strength
Sector rotation
Institutional accumulation
Non‑index‑driven movement

Residual PCA1 is a core signal of Page4.

4. Stage 3 — Intraday Actionability (Noise‑Filtered Triggers)

Page4 applies strict intraday filters to avoid false signals.

Intraday Filters
Opening gap < 1.5%
VWAP reclaim
RVOL spike
PCA1_slope > 0
Higher‑low formation (clean intraday structure)

These ensure Page4 only triggers when real intraday expansion is forming.

5. Stage 4 — Execution Readiness Classification

Tickers passing Page4 filters are classified as:
Actionable (Choppy Model)
EMA compression
Residual PCA1 expanding
Intraday acceleration present

Structural Only
Compression present
Residual PCA1 present
Intraday confirmation missing

Not Actionable
No compression
No independent momentum
Intraday noise

6. Trader Workflow — Page4

The trader uses Page4 in a three‑stage workflow:

Stage A — Structural Compression Scan

Identify:
EMA compression
Residual PCA1 expansion

Stage B — Intraday Confirmation

Confirm:
VWAP reclaim
RVOL spike
PCA1_slope > 0
Opening gap < 1.5%
Clean intraday structure

Stage C — Execution

Act only when:
Compression
Independent momentum
Intraday acceleration all align.

7. Technical Glossary

EMA Compression

Tight coiling of EMAs indicating stored energy.

Residual PCA1

Momentum not explained by SPY; measures independent strength.

VWAP Reclaim

Price moves above institutional cost basis.

RVOL Spike

Volume expansion indicating institutional activity.

PCA1_slope

Real‑time acceleration of intraday momentum.

✔️ End of Page4 Documentation
