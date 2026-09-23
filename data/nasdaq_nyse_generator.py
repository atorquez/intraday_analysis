import pandas as pd
import os

# ==============================================================================
# UNIVERSE BUILDER — price-agnostic, exchange-tagged
#
# CHANGE FROM PREVIOUS VERSION:
#   Price is NO LONGER filtered here. The old min_price/max_price args baked
#   a price range into the generated file using whatever "Last Sale" value
#   sat in the NASDAQ/NYSE CSV snapshot at build time — which goes stale the
#   moment the market moves, and forces you to regenerate the whole universe
#   file every time you switch between the penny model (1-10) and the
#   momentum model (40-120).
#
#   Both models already re-check price against LIVE intraday data inside
#   their own engines (that's the correct, non-stale place to do it), so
#   filtering here was redundant and, being based on older data, actually
#   less accurate than the live check each model already performs.
#
#   This builder now only does what genuinely doesn't go stale: excluding
#   non-common-stock security types (preferreds, warrants, REITs, etc.) via
#   the blacklist below. Run it once (or on a periodic schedule, e.g.
#   nightly) rather than per-model, per-session.
#
# ALSO FIXED:
#   Exchange is now carried through from which source file a ticker came
#   from (NASDAQ CSV vs NYSE CSV), instead of being guessed after the fact
#   from ticker symbol length — that heuristic is unreliable in both
#   directions (plenty of NASDAQ tickers are 1-3 letters; NYSE has some
#   4-letter tickers too).
# ==============================================================================

BLACKLIST_KEYWORDS = [
    "PREFERRED", "SERIES", "NOTE", "NOTES", "SENIOR", "DEPOSITARY",
    "WARRANT", "UNIT", "UNITS", "FUND", "ETF", "TRUST", "LP",
    "BOND", "REIT", "INCOME", "CONVERTIBLE", "FLOATING",
    "FIXED", "CUMULATIVE", "REDEEMABLE"
]
BLACKLIST_PATTERN = "|".join(BLACKLIST_KEYWORDS)


def load_and_filter_source(path, exchange_label):
    """
    Load one exchange's CSV, apply security-type filtering only (no price
    filtering), and tag every surviving row with its source exchange.
    """
    if not os.path.exists(path):
        print(f"WARNING: File skipped (not found) → {path}")
        return pd.DataFrame()

    print(f"Processing source file: {path} → tagging as {exchange_label}")
    df = pd.read_csv(path)

    if "Symbol" not in df.columns:
        print(f"Skipping {path}: No Symbol column found.")
        return pd.DataFrame()

    # Normalize text columns used for security-type filtering.
    for col in ["Name", "Industry", "Sector"]:
        df[col] = df[col].astype(str).str.upper() if col in df.columns else ""

    # Structural filtering only — excludes security TYPES, not price levels.
    # This doesn't go stale the way a price snapshot does.
    df = df[~df["Name"].str.contains(BLACKLIST_PATTERN, regex=True, na=False)]
    df = df[~df["Industry"].str.contains("REAL ESTATE INVESTMENT TRUST", regex=False, na=False)]

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df = df[df["Symbol"] != ""]

    df["Exchange"] = exchange_label

    return df[["Symbol", "Exchange"]]


def generate_master_universe():
    """
    Build one price-agnostic, exchange-tagged universe file shared by every
    model (penny, momentum, or anything added later). Each model's own
    min_price/max_price widget filters live at scan time — this file just
    defines "what's a real, scannable common-stock ticker."
    """
    # INSTRUCTION: add your source data file paths + exchange labels here.
    sources = [
        ("data/nasdaq_full_list.csv", "NASDAQ"),
        ("data/nyse_full_list.csv", "NYSE"),
    ]
    output_path = "data/us_universe_list.py"

    print("Building master stock universe (no price filter, exchange-tagged)...")

    combined = []
    for path, exchange_label in sources:
        part = load_and_filter_source(path, exchange_label)
        if not part.empty:
            combined.append(part)

    if not combined:
        print("ERROR: No tickers found across any data sources.")
        return

    df_master = pd.concat(combined, ignore_index=True)

    # De-dupe symbols that appear in more than one source file. Keep the
    # first exchange tag seen (sources list order = priority order) rather
    # than silently dropping the exchange info.
    df_master = df_master.drop_duplicates(subset="Symbol", keep="first")
    df_master = df_master.sort_values("Symbol")

    tickers = df_master["Symbol"].tolist()
    exchange_map = dict(zip(df_master["Symbol"], df_master["Exchange"]))

    with open(output_path, "w") as f:
        f.write("# Comprehensive US common-stock universe (NASDAQ + NYSE)\n")
        f.write("# Price-agnostic by design — each model filters price live,\n")
        f.write("# at scan time, against current data. Regenerate this file\n")
        f.write("# periodically (e.g. nightly) to pick up new/delisted tickers,\n")
        f.write("# not to switch price ranges — no price range is baked in here.\n\n")

        f.write("us_universe = [\n")
        for t in tickers:
            f.write(f'    "{t}",\n')
        f.write("]\n\n")

        f.write("# Symbol -> exchange, taken from which source file it came\n")
        f.write("# from (not guessed from ticker length).\n")
        f.write("us_universe_exchange = {\n")
        for t in tickers:
            f.write(f'    "{t}": "{exchange_map[t]}",\n')
        f.write("}\n")

    print(f"\nSUCCESS → Master list written to {output_path}")
    print(f"Total tickers: {len(tickers)}")
    for label in sorted(set(exchange_map.values())):
        count = sum(1 for v in exchange_map.values() if v == label)
        print(f"  {label}: {count}")


if __name__ == "__main__":
    generate_master_universe()
