import pandas as pd
import os

# ---------------------------------------------------------
# FILTER + TOP 1000 SELECTOR
# ---------------------------------------------------------
def filter_and_select_top1000(df):
    """
    Cleans a raw Nasdaq universe and selects the top 1000 common-stock companies.
    """

    # Normalize text columns
    for col in ["Name", "Industry", "Sector"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
        else:
            df[col] = ""

    # Keywords indicating NON-company tickers
    blacklist_keywords = [
        "PREFERRED", "SERIES", "NOTE", "NOTES", "SENIOR", "DEPOSITARY",
        "WARRANT", "UNIT", "UNITS", "FUND", "ETF", "TRUST", "LP",
        "BOND", "REIT", "INCOME", "CONVERTIBLE", "FLOATING",
        "FIXED", "CUMULATIVE", "REDEEMABLE"
    ]

    pattern = "|".join(blacklist_keywords)

    # Remove non-company tickers
    df_clean = df[~df["Name"].str.contains(pattern, regex=True)]

    # Remove REIT preferreds
    df_clean = df_clean[
        ~df_clean["Industry"].str.contains("REAL ESTATE INVESTMENT TRUST", regex=False)
    ]

    # ---------------------------------------------------------
    # FIX MARKET CAP (critical)
    # ---------------------------------------------------------
    df_clean["Market Cap"] = (
        df_clean["Market Cap"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df_clean["Market Cap"] = pd.to_numeric(df_clean["Market Cap"], errors="coerce")

    # Drop rows with missing market cap
    df_clean = df_clean.dropna(subset=["Market Cap"])

    # Sort by Market Cap descending
    df_sorted = df_clean.sort_values(by="Market Cap", ascending=False)

    # Select top 1000
    df_top1000 = df_sorted.head(1000).reset_index(drop=True)

    return df_top1000


# ---------------------------------------------------------
# MAIN SCRIPT: READ CSV → FILTER → WRITE PY FILE
# ---------------------------------------------------------
def generate_nasdaq1000_list():
    csv_path = "data/nasdaq_full_list.csv"   # Correct path

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found → {csv_path}")
        return

    print("Loading raw Nasdaq universe...")
    df_raw = pd.read_csv(csv_path)

    print("Filtering and selecting top 1000...")
    df_top1000 = filter_and_select_top1000(df_raw)

    tickers = df_top1000["Symbol"].tolist()

    output_path = "data/nasdaq1000_list.py"

    print(f"Writing Python universe file → {output_path}")

    with open(output_path, "w") as f:
        f.write("# NASDAQ-1000 Universe (Common Stocks Only)\n")
        f.write("# Auto-generated from nasdaq_full_list.csv\n\n")
        f.write("nasdaq1000 = [\n")   # <-- FIXED HERE
        for t in tickers:
            f.write(f'    "{t}",\n')
        f.write("]\n")


    print("NASDAQ-1000 list generated successfully.")
    print(f"Total tickers: {len(tickers)}")


if __name__ == "__main__":
    generate_nasdaq1000_list()


