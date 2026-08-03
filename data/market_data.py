import pandas as pd
import os

# ---------------------------------------------------------
# OFFLINE DATA PARSER & FILTER
# ---------------------------------------------------------
def load_and_filter_all_sources(csv_paths, min_price=40, max_price=110):
    combined_dfs = []
    
    blacklist_keywords = [
        "PREFERRED", "SERIES", "NOTE", "NOTES", "SENIOR", "DEPOSITARY",
        "WARRANT", "UNIT", "UNITS", "FUND", "ETF", "TRUST", "LP",
        "BOND", "REIT", "INCOME", "CONVERTIBLE", "FLOATING",
        "FIXED", "CUMULATIVE", "REDEEMABLE"
    ]
    pattern = "|".join(blacklist_keywords)

    for path in csv_paths:
        # Step 1: Ensure file exists and is not an empty 0-byte file
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"WARNING: File skipped (Missing or Empty) → {path}")
            continue
            
        print(f"Processing offline source file: {path}")
        
        # Step 2: Flexibly parse file engine to handle multiple formats automatically
        try:
            df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
        except Exception as e:
            print(f"Skipping corrupt file {path}: {e}")
            continue

        if df.empty:
            print(f"Skipping {path}: File structure is empty.")
            continue

        # Step 3: Map data column headers dynamically
        df.columns = df.columns.str.strip()
        
        symbol_col = next((c for c in ["Symbol", "ACT Symbol", "Ticker", "symbol", "UnifiedSymbol"] if c in df.columns), None)
        name_col = next((c for c in ["Name", "Security Name", "Company Name", "name", "UnifiedName"] if c in df.columns), None)
        price_col = next((c for c in ["Last Sale", "Last", "Close", "Price", "lastsale"] if c in df.columns), None)

        if not symbol_col or not name_col:
            print(f"Skipping {path}: Missing standard Symbol/Name headers. Available: {df.columns.tolist()}")
            continue

        # Map to our standard variable format
        df = df.rename(columns={symbol_col: "UnifiedSymbol", name_col: "UnifiedName"})
        
        # Step 4: Apply common stock structural filters
        df["UnifiedName"] = df["UnifiedName"].astype(str).str.upper()
        df = df[~df["UnifiedName"].str.contains(pattern, regex=True)]

        # Step 5: Clean and check the price values if present
        if price_col:
            df["Price"] = df[price_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
            df = df.dropna(subset=["Price"])
            df_filtered = df[(df["Price"] >= min_price) & (df["Price"] <= max_price)]
            print(f" Found {len(df_filtered)} valid tickers matching range in {path}")
        else:
            print(f" Note: No raw pricing data found in {path}. Forwarding all tickers for validation.")
            df_filtered = df

        combined_dfs.append(df_filtered)

    if not combined_dfs:
        return pd.DataFrame()

    return pd.concat(combined_dfs, ignore_index=True)


# ---------------------------------------------------------
# MASTER UNIVERSE BUILDER
# ---------------------------------------------------------
def generate_master_universe(min_price=40, max_price=110):
    # Core local paths (Manually update/save your CSV files here)
    source_files = [
        "data/nasdaq_full_list.csv",
        "data/nyse_full_list.csv" 
    ]
    output_path = "data/us_universe_list.py"

    print(f"Building consolidated universe from local directory (Target: ${min_price} - ${max_price})...")
    df_master = load_and_filter_all_sources(source_files, min_price, max_price)

    if df_master.empty:
        print("ERROR: No valid data found in your data folder.")
        return

    # Standardize and deduplicate the ticker array output
    df_master["UnifiedSymbol"] = df_master["UnifiedSymbol"].astype(str).str.strip().str.upper()
    unique_tickers = sorted(list(set(df_master["UnifiedSymbol"].tolist())))

    # Write out the clean Python module file
    with open(output_path, "w") as f:
        f.write(f"# Comprehensive US Market Universe (Price {min_price}-{max_price})\n")
        f.write("# Generated cleanly using local workspace files\n\n")
        f.write("us_universe = [\n")
        for t in unique_tickers:
            f.write(f'    "{t}",\n')
        f.write("]\n")

    print(f"\nSUCCESS → Master list written to: {output_path}")
    print(f"Total verified tickers: {len(unique_tickers)}")


if __name__ == "__main__":
    generate_master_universe(min_price=40, max_price=110)

