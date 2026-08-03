Developer-Focused PCA Guide (Math + Code)

1. Mathematical Foundations

Principal Component Analysis (PCA) decomposes a normalized feature matrix (X) into orthogonal components.

1.1 Covariance Matrix

Given standardized features: [ X \in \mathbb{R}^{n \times d} ] Compute covariance: [ \Sigma = \frac{1}{n-1} X^T X ]

1.2 Eigen Decomposition

Solve: [ \Sigma v_i = \lambda_i v_i ] Where:

(v_i) = eigenvector (principal component)

(\lambda_i) = eigenvalue (variance explained)

1.3 PCA Projection

Project data onto principal components: [ Z = X V ] Where:

(Z) = PCA-transformed matrix

(V) = matrix of eigenvectors

2. Feature Engineering for Trading Models

Your engine uses a momentum‑centric feature matrix:

features = np.column_stack([
    rsi,
    roc,
    stoch_k,
    ema_curvature,
    vwap_deviation,
    volume_delta,
    bollinger_width
])

2.1 Normalization

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(features)

2.2 PCA Extraction

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
components = pca.fit_transform(X)

Outputs:

components[:,0] → PCA1 (momentum)

components[:,1] → PCA2 (volatility)

components[:,2] → PCA3 (participation)

3. PCA1 — Momentum Factor

PCA1 captures the dominant direction of variance across momentum indicators.

3.1 Mathematical Interpretation

If (v_1) is the first eigenvector: [ \text{PCA1} = X v_1 ]

High PCA1 → strong daily momentum. Low PCA1 → weakening momentum.

3.2 Code Snippet

pca1 = components[:, 0]

4. PCA1_slope — Intraday Acceleration

Compute slope across intraday windows:

4.1 Math

Given PCA1 values over time (t): [ \text{PCA1_slope} = \frac{d(\text{PCA1})}{dt} ]

4.2 Code

pca1_slope = np.gradient(pca1)

Positive slope → acceleration. Negative slope → fading.

5. PCA2 — Volatility Factor

Captures variance in volatility‑related features.

5.1 Math

[ \text{PCA2} = X v_2 ]

5.2 Code

pca2 = components[:, 1]

High PCA2 → volatility expansion. Low PCA2 → compression.

6. PCA3 — Participation Factor

Captures institutional participation.

6.1 Math

[ \text{PCA3} = X v_3 ]

6.2 Code

pca3 = components[:, 2]

High PCA3 → strong participation. Low PCA3 → weak liquidity.

7. Regime Behavior (Developer View)

Trending

PCA1 ↑

PCA1_slope ↑

PCA2 ↔

PCA3 ↑

Mixed

PCA1 ↔

PCA1_slope oscillates

PCA2 ↑

PCA3 inconsistent

Choppy

PCA1 ↓

PCA1_slope oscillates

PCA2 ↓

PCA3 ↓

Bearish

PCA1 ↓

PCA1_slope ↓

PCA2 ↑

PCA3 ↑

8. Execution Logic Integration

Execution labels combine PCA factors + trend stack + volatility + participation.

Example

if trend_up and pca1 > 0 and pca1_slope > 0:
    label = "Watch List"
elif pca2 < threshold:
    label = "Crossing Soon"
else:
    label = "Not Watch List"

9. Full PCA Pipeline (Developer Template)

# 1. Build feature matrix
features = np.column_stack([
    rsi,
    roc,
    stoch_k,
    ema_curvature,
    vwap_deviation,
    volume_delta,
    bollinger_width
])

# 2. Normalize
X = StandardScaler().fit_transform(features)

# 3. PCA
pca = PCA(n_components=3)
components = pca.fit_transform(X)

# 4. Extract factors
pca1 = components[:, 0]
pca2 = components[:, 1]
pca3 = components[:, 2]

# 5. Compute intraday acceleration
pca1_slope = np.gradient(pca1)

10. Summary

This developer‑focused PCA guide provides:

Mathematical foundations

Clean code templates

Factor interpretations

Regime behavior

Execution logic integration

Perfect for quant developers extending your structural engine.