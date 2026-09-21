# ==============================================================================
# 🔥 MOVERS LIST — curated watchlist of known pre-market / after-hours movers
# ==============================================================================
# Edit this list before each session with the tickers you've identified as
# significant movers (e.g. from pre-market/after-hours scanners, news,
# gap screeners). The penny model reloads this file fresh on every run, so
# you can edit it mid-session and the tags will update on your next click
# of "Run Model" without restarting the app.
#
# Place this file at: data/movers_list.py  (same folder as us_universe_list.py)
#
# The model does NOT restrict its scan to this list — it still scans the
# full universe. This list only adds a "Movers List" / "Not Movers List"
# tag to every result, plus a scorecard showing exactly what happened to
# each of these tickers this run (qualified / rejected + reason / not in
# the scanned universe at all).

def load_movers():
    return [
        # "GLND",
        # "GRML",
        # "NCPL",
    ]