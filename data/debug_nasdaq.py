import pandas as pd

df = pd.read_csv("data/nasdaq_full_list.csv")

print("\n=== COLUMN NAMES ===")
print(df.columns)

print("\n=== FIRST 10 ROWS ===")
print(df.head(10))

print("\n=== ROW COUNT ===")
print(len(df))

# Check Market Cap column variations
possible_caps = ["Market Cap", "MarketCap", "Market_Cap", "marketcap"]
for col in possible_caps:
    if col in df.columns:
        print(f"\nFound market cap column: {col}")
        print(df[col].head())
