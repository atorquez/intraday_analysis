import pandas as pd
import os

def load_and_filter_all_sources(csv_paths, min_price=40, max_price=120):
    """
    Loops through multiple CSV data sources (NYSE, NASDAQ, etc.),
    cleans them, and combines them into one unique dataframe.
    """
    combined_dfs = []
    
    blacklist_keywords = [
        "PREFERRED", "SERIES", "NOTE", "NOTES", "SENIOR", "DEPOSITARY",
        "WARRANT", "UNIT", "UNITS", "FUND", "ETF", "TRUST", "LP",
        "BOND", "REIT", "INCOME", "CONVERTIBLE", "FLOATING",
        "FIXED", "CUMULATIVE", "REDEEMABLE"
    ]
    pattern = "|".join(blacklist_keywords)

    for path in csv_paths:
        if not os.path.exists(path):
            print(f"WARNING: File skipped (not found) → {path}")
            continue
            
        print(f"Processing source file: {path}")
        df = pd.read_csv(path)

        # Normalize text columns
        for col in ["Name", "Industry", "Sector"]:
            df[col] = df[col].astype(str).str.upper() if col in df.columns else ""

        # Apply common stock filters
        df = df[~df["Name"].str.contains(pattern, regex=True)]
        df = df[~df["Industry"].str.contains("REAL ESTATE INVESTMENT TRUST", regex=False)]

        # Find price column dynamically
        price_col = next((c for c in ["Last Sale", "Last", "Close", "Price"] if c in df.columns), None)
        if price_col is None:
            print(f"Skipping {path}: No usable price column found.")
            continue

        # Clean price data
        df["Price"] = df[price_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=["Price"])

        # Filter by price range
        df_filtered = df[(df["Price"] >= min_price) & (df["Price"] <= max_price)]
        combined_dfs.append(df_filtered)

    if not combined_dfs:
        return pd.DataFrame()

    # Merge all sources together
    return pd.concat(combined_dfs, ignore_index=True)


def generate_master_universe(min_price=40, max_price=12):
    # INSTRUCTION: Simply add your source data file paths to this list
    source_files = [
        "data/nasdaq_full_list.csv",
        "data/nyse_full_list.csv" 
    ]
    output_path = "data/us_universe_list.py"

    print("Building master stock universe...")
    df_master = load_and_filter_all_sources(source_files, min_price, max_price)

    if df_master.empty:
        print("ERROR: No tickers found across any data sources.")
        return

    # Clean ticker symbols and drop duplicates across files
    df_master["Symbol"] = df_master["Symbol"].astype(str).str.strip().str.upper()
    unique_tickers = sorted(list(set(df_master["Symbol"].tolist())))

    # Write out the single unified python file
    with open(output_path, "w") as f:
        f.write(f"# Comprehensive US Market Universe (Price {min_price}-{max_price})\n")
        f.write("# Automatically combined and deduplicated across exchanges\n\n")
        f.write("us_universe = [\n")
        for t in unique_tickers:
            f.write(f'    "{t}",\n')
        f.write("]\n")

    print(f"\nSUCCESS → Master list written to {output_path}")
    print(f"Total comprehensive tickers: {len(unique_tickers)}")


if __name__ == "__main__":
    generate_master_universe(min_price=40, max_price=120)
