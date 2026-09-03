import pandas as pd
import os

# ---------------------------------------------------------
# CLEAN + PRICE FILTER
# ---------------------------------------------------------
def load_and_filter_nasdaq(csv_path, min_price=5, max_price=500):
    """
    Loads NASDAQ full list CSV and filters:
    - common stocks only
    - removes ETFs, preferreds, warrants, notes, REITs, LPs, units
    - selects tickers within a price range
    """

    df = pd.read_csv(csv_path)

    # Normalize text columns
    for col in ["Name", "Industry", "Sector"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
        else:
            df[col] = ""

    # Remove non-company tickers
    blacklist_keywords = [
        "PREFERRED", "SERIES", "NOTE", "NOTES", "SENIOR", "DEPOSITARY",
        "WARRANT", "UNIT", "UNITS", "FUND", "ETF", "TRUST", "LP",
        "BOND", "REIT", "INCOME", "CONVERTIBLE", "FLOATING",
        "FIXED", "CUMULATIVE", "REDEEMABLE"
    ]
    pattern = "|".join(blacklist_keywords)
    df = df[~df["Name"].str.contains(pattern, regex=True)]

    # Remove REIT preferreds
    df = df[~df["Industry"].str.contains("REAL ESTATE INVESTMENT TRUST", regex=False)]

    # ---------------------------------------------------------
    # DETECT PRICE COLUMN
    # ---------------------------------------------------------
    price_col = None
    for candidate in ["Last Sale", "Last", "Close", "Price"]:
        if candidate in df.columns:
            price_col = candidate
            break

    if price_col is None:
        print("ERROR: No usable price column found in CSV.")
        print("Columns available:", df.columns.tolist())
        return pd.DataFrame()

    print(f"Using price column: {price_col}")

    # Clean price column
    df["Price"] = (
        df[price_col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["Price"])

    # ---------------------------------------------------------
    # FILTER BY PRICE RANGE
    # ---------------------------------------------------------
    df_filtered = df[(df["Price"] >= min_price) & (df["Price"] <= max_price)]

    print(f"Tickers in price range {min_price}–{max_price}: {len(df_filtered)}")

    return df_filtered.reset_index(drop=True)

# ---------------------------------------------------------
# WRITE PYTHON UNIVERSE FILE
# ---------------------------------------------------------
def write_universe_file(tickers, output_path, min_price, max_price):
    with open(output_path, "w") as f:
        f.write(f"# NASDAQ1000 Universe (Price {min_price}-{max_price})\n")
        f.write("# Auto-generated from nasdaq_full_list.csv\n\n")
        f.write("nasdaq1000 = [\n")
        for t in tickers:
            f.write(f'    "{t}",\n')
        f.write("]\n")

    print(f"Universe file written → {output_path}")
    print(f"Total tickers: {len(tickers)}")

# ---------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------
def generate_nasdaq_price_universe(min_price=5, max_price=500):
    csv_path = "data/nasdaq_full_list.csv"
    output_path = "data/nasdaq_list.py"

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found → {csv_path}")
        return

    print("Loading NASDAQ full list...")
    df_filtered = load_and_filter_nasdaq(csv_path, min_price=min_price, max_price=max_price)

    if df_filtered.empty:
        print("ERROR: No tickers found in the specified price range.")
        return

    tickers = df_filtered["Symbol"].tolist()

    write_universe_file(tickers, output_path, min_price, max_price)


if __name__ == "__main__":
    generate_nasdaq_price_universe(min_price=5, max_price=500)
