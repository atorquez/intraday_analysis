# ==============================================================================
# 📊 PREDICTIVE SNIPER MODEL EXECUTION MAP — HARDENED VISUALIZER
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np

# Setup structural grid vectors
time_bars = np.arange(1, 61)
base_price = 81.10
np.random.seed(42)  # Set seed for identical reproducible prints
noise = np.random.normal(0, 0.02, 60)

# Simulate initial horizontal compression coil (Bars 1 to 35)
price_profile = np.full(35, base_price) + noise[:35]
# Simulate institutional breakout expansion leg (Bars 36 to 60)
breakout_leg = np.linspace(81.10, 81.35, 25) + noise[35:]
full_price_array = np.concatenate([price_profile, breakout_leg])

# Build the dual axis canvas with corrected explicit height ratios
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
plt.style.use('dark_background')

# Plot Tactical Chart 1 (Price Arrays)
ax1.plot(time_bars, full_price_array, color='#00E6FF', lw=2, label='Live 1-Min Spot Price Line')
ax1.axhline(81.22, color='#FF3B30', linestyle='--', alpha=0.8, label="Yesterday's Close Axis ($81.22)")
ax1.axhline(81.29, color='#FFCC00', linestyle=':', alpha=0.9, label='5-Day Daily Overhead Resistance Wick ($81.29)')
ax1.axhline(83.65, color='#FF9500', linestyle='-.', alpha=0.5, label='3.0% Maximum Exhaustion Cap ($83.65)')

# Highlight the entry trigger zone
ax1.fill_between(time_bars[35:45], 81.22, 81.35, color='#4CD964', alpha=0.15, label='Optimal Entry Trigger Target Window')
ax1.scatter(38, full_price_array[37], color='#4CD964', s=120, edgecolors='white', zorder=5, label='Zero-Line Cross Multiplier Signal Trigger')

ax1.set_title("📈 Predictive Sniper Model Execution Map (Theoretical Token Setup Profile)", color='white', fontsize=12, pad=10)
ax1.set_ylabel("Asset Spot Price ($)", color='white')
ax1.legend(loc='upper left', fontsize=8, facecolor='#1C1C1E', edgecolor='none')
ax1.grid(True, alpha=0.1)

# Plot Tactical Chart 2 (Volume Bars with corrected explicit 3-Bar Stair Steps)
volume_bars = np.random.randint(400, 1800, 60)
volume_bars[35] = 2500  # Bar 1 Step
volume_bars[36] = 4100  # Bar 2 Step
volume_bars[37] = 6800  # Bar 3 Step (Ignition Print Cross)

colors = ['#FF3B30' if full_price_array[i] < full_price_array[i-1] else '#4CD964' for i in range(1, 60)]
colors.insert(0, '#4CD964')

ax2.bar(time_bars, volume_bars, color=colors, alpha=0.8, width=0.8, label='1-Min Share Volume Delta Prints')
ax2.set_ylabel("Volume Delta", color='white')
ax2.set_xlabel("Session Timeline Horizon (Trailing 1-Minute Candlestick Intervals)", color='white')
ax2.grid(True, alpha=0.1)

plt.tight_layout()
plt.show()
