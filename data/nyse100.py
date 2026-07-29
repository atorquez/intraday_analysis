def load_nyse_top100():
    """
    Load the NYSE Top 100 institutional tickers.
    These are mega-cap and large-cap NYSE stocks with clean intraday behavior,
    compatible with EMA/VWAP/PCA1 logic.
    """

    nyse_top100 = [
        # Industrials
        "CAT","DE","GE","BA","HON","MMM","UPS","UNP","LMT","RTX","GD","EMR",
        "ETN","ITW","CMI","DHR","ROP","PH","JCI",

        # Energy
        "XOM","CVX","COP","EOG","SLB","HAL","MPC","VLO","PSX",

        # Materials
        "LIN","APD","SHW","NEM","NUE","FCX","ECL","PPG",

        # Financials
        "JPM","BAC","WFC","GS","MS","BLK","CME","ICE","AXP","COF","USB","PNC","TFC",

        # Consumer (Staples + Discretionary)
        "WMT","HD","MCD","KO","PEP","PG","COST","TGT","LOW","SBUX","TJX","NKE","CL","KMB",

        # Healthcare
        "JNJ","UNH","MRK","ABBV","PFE","TMO","MDT","BMY","AMGN","CVS","CI",

        # Telecom
        "VZ","T",

        # Utilities (only the cleanest)
        "NEE","DUK","SO",

        # Real Estate (mega-caps only)
        "PLD","AMT","EQIX",

        # Industrials / Tech hybrids
        "IBM","ORCL","CRM","SAP",

        # Specialty / Logistics / Other
        "WCC","BKR","FDX"
    ]

    return sorted(list(set(nyse_top100)))
