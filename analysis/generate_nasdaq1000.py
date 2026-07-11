import pandas as pd

# Load your NASDAQ full list
df = pd.read_csv("data/nasdaq_full_list.csv", sep="|")

# --- FILTERING RULES ---

# Remove ETFs
df = df[df["ETF"] != "Y"]

# Remove NextShares
df = df[df["NextShares"] != "Y"]

# Remove warrants, rights, units, preferred, notes, bonds
blocklist_keywords = [
    "Warrant", "Right", "Unit", "Preferred", "Depositary",
    "Note", "Bond", "Senior", "Series", "Trust Preferred",
    "Closed End", "Fund"
]

pattern = "|".join(blocklist_keywords)
df = df[~df["Security Name"].str.contains(pattern, case=False, na=False)]

# Keep only STOCKS (Common, Ordinary, ADS/ADR)
keep_keywords = ["Common", "Ordinary", "ADS", "Depositary", "American Depositary"]
pattern_keep = "|".join(keep_keywords)
df = df[df["Security Name"].str.contains(pattern_keep, case=False, na=False)]

# Sort alphabetically
df = df.sort_values("Symbol")

# Select first 1000 tickers
nasdaq1000 = df["Symbol"].head(1000).tolist()

# Write to Python file
with open("nasdaq1000_list.py", "w") as f:
    f.write("# NASDAQ-1000 Universe (Stocks only, ADRs included)\n")
    f.write("# Generated automatically from nasdaq_full_list.txt\n\n")
    f.write("NASDAQ1000 = [\n")
    for t in nasdaq1000:
        f.write(f'    "{t}",\n')
    f.write("]\n")

print("NASDAQ-1000 list generated successfully.")

