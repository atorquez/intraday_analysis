import pandas as pd
import os
import re

# ==============================================================================
# UNIVERSE BUILDER — price-filtered, exchange-tagged
#
# CHANGE FROM PREVIOUS VERSION:
#   Price filtering is back, but explicit and on-demand rather than baked
#   in silently. Pass min_price/max_price to generate_master_universe()
#   (or edit the __main__ call below) and re-run whenever you want to
#   switch which range this universe file covers — e.g. 1-10 for the
#   penny model, 40-120 for momentum. Each run overwrites
#   data/us_universe_list.py with a universe scoped to that range.
#
#   This is a deliberate, known trade-off: filtering by price HERE uses
#   whatever "Last Sale" value sat in the NASDAQ/NYSE CSV snapshot at
#   build time, which is not live intraday data. A ticker that gaps
#   dramatically between when you last generated this file and when you
#   run the model won't be reflected until you regenerate. That's the
#   cost you're accepting in exchange for a fast, small, scannable
#   universe — appropriate given the model's ≤15-minute opportunity
#   window makes intraday-fetch speed the priority.
#
#   Both models STILL also re-check the exact price against LIVE intraday
#   data inside their own engines — this build-time filter only decides
#   which tickers are worth including in the universe file at all; it's
#   not the final word on price.
#
# ALSO KEPT FROM PREVIOUS VERSION:
#   Exchange is carried through from which source file a ticker came from
#   (NASDAQ CSV vs NYSE CSV), not guessed from ticker symbol length.
# ==============================================================================

BLACKLIST_KEYWORDS = [
    "PREFERRED", "SERIES", "NOTE", "NOTES", "SENIOR", "DEPOSITARY",
    "WARRANT", "UNIT", "UNITS", "FUND", "ETF", "TRUST", "LP",
    "BOND", "REIT", "INCOME", "CONVERTIBLE", "FLOATING",
    "FIXED", "CUMULATIVE", "REDEEMABLE"
]
BLACKLIST_PATTERN = "|".join(BLACKLIST_KEYWORDS)

# Common column names for last-traded price across different NASDAQ/NYSE
# screener CSV exports. Checked in order; first match wins.
PRICE_COLUMN_CANDIDATES = ["Last Sale", "LastSale", "Last Price", "Price", "Close"]


def _parse_price_column(series):
    """Strip $ / commas / whitespace and coerce to float. Unparseable
    values become NaN (and get dropped by the price-range filter, same as
    any other missing price — not treated as a pass)."""
    cleaned = series.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def load_and_filter_source(path, exchange_label, min_price=None, max_price=None):
    """
    Load one exchange's CSV, apply security-type filtering, and — if
    min_price/max_price are given — filter by that file's Last Sale price
    at build time. Tags every surviving row with its source exchange.
    """
    if not os.path.exists(path):
        print(f"WARNING: File skipped (not found) → {path}")
        return pd.DataFrame()

    print(f"Processing source file: {path} → tagging as {exchange_label}")
    df = pd.read_csv(path)

    if "Symbol" not in df.columns:
        print(f"Skipping {path}: No Symbol column found.")
        return pd.DataFrame()

    for col in ["Name", "Industry", "Sector"]:
        df[col] = df[col].astype(str).str.upper() if col in df.columns else ""

    df = df[~df["Name"].str.contains(BLACKLIST_PATTERN, regex=True, na=False)]
    df = df[~df["Industry"].str.contains("REAL ESTATE INVESTMENT TRUST", regex=False, na=False)]

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df = df[df["Symbol"] != ""]

    if min_price is not None or max_price is not None:
        price_col = next((c for c in PRICE_COLUMN_CANDIDATES if c in df.columns), None)
        if price_col is None:
            print(f"WARNING: No recognizable price column found in {path} "
                  f"(looked for {PRICE_COLUMN_CANDIDATES}) — price filter "
                  f"SKIPPED for this file, all structurally-valid tickers kept.")
        else:
            before = len(df)
            df["_price"] = _parse_price_column(df[price_col])
            if min_price is not None:
                df = df[df["_price"] >= min_price]
            if max_price is not None:
                df = df[df["_price"] <= max_price]
            df = df.drop(columns=["_price"])
            print(f"  Price filter [{min_price}, {max_price}] on '{price_col}': "
                  f"{before} → {len(df)} tickers")

    df["Exchange"] = exchange_label

    return df[["Symbol", "Exchange"]]


def generate_master_universe(min_price=None, max_price=None, output_path="data/us_universe_list.py"):
    """
    Build a price-filtered (if min_price/max_price given), exchange-tagged
    universe file. Re-run with different min_price/max_price whenever you
    want to switch which range this file covers — e.g.:

        generate_master_universe(1, 10)     # penny model
        generate_master_universe(40, 120)   # momentum model
        generate_master_universe()          # no price filter (old 3700-wide behavior)

    Each call overwrites output_path, so only one range is "active" in
    data/us_universe_list.py at a time unless you pass a different
    output_path per range.
    """
    sources = [
        ("data/nasdaq_full_list.csv", "NASDAQ"),
        ("data/nyse_full_list.csv", "NYSE"),
    ]

    print(f"Building universe (price range: {min_price}-{max_price})...")

    combined = []
    for path, exchange_label in sources:
        part = load_and_filter_source(path, exchange_label, min_price, max_price)
        if not part.empty:
            combined.append(part)

    if not combined:
        print("ERROR: No tickers found across any data sources.")
        return

    df_master = pd.concat(combined, ignore_index=True)
    df_master = df_master.drop_duplicates(subset="Symbol", keep="first")
    df_master = df_master.sort_values("Symbol")

    tickers = df_master["Symbol"].tolist()
    exchange_map = dict(zip(df_master["Symbol"], df_master["Exchange"]))

    with open(output_path, "w") as f:
        f.write("# US common-stock universe (NASDAQ + NYSE)\n")
        f.write(f"# Price filter at build time: min={min_price}, max={max_price}\n")
        f.write("# This is a SNAPSHOT filter (Last Sale at generation time), not\n")
        f.write("# live. Re-run generate_universe.py with new min/max whenever you\n")
        f.write("# want this file to cover a different range. The model still\n")
        f.write("# checks exact price against LIVE intraday data on top of this.\n\n")

        f.write("us_universe = [\n")
        for t in tickers:
            f.write(f'    "{t}",\n')
        f.write("]\n\n")

        f.write("us_universe_exchange = {\n")
        for t in tickers:
            f.write(f'    "{t}": "{exchange_map[t]}",\n')
        f.write("}\n")

    print(f"\nSUCCESS → Universe written to {output_path}")
    print(f"Total tickers: {len(tickers)}")
    for label in sorted(set(exchange_map.values())):
        count = sum(1 for v in exchange_map.values() if v == label)
        print(f"  {label}: {count}")


if __name__ == "__main__":
    # EDIT THESE two lines each time you want to switch which range
    # data/us_universe_list.py covers, then re-run this script.
    MIN_PRICE = 1
    MAX_PRICE = 200
    generate_master_universe(MIN_PRICE, MAX_PRICE)