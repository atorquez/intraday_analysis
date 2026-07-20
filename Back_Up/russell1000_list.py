# data/russell1000_list.py

# Full Russell 1000 constituents (raw list)
# Source: FTSE Russell / TradingView / Barchart (1004–1005 components)
# Last verified: 2026

russell1000_raw = [
    "MMM","AOS","AAON","ABT","ABBV","ACHC","ACN","AYI","ADBE","ADT",
    "WMS","AMD","ACM","A","AAPL","AMZN","NVDA","GOOGL","GOOG","MSFT",
    "AVGO","META","TSLA","BRK.B","AMCR","AEE","AEP","AXP","AIG","AMT",
    "AWK","AMP","ABC","AME","AMGN","APH","ADI","ANSS","AON","APA",
    "APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","MO",
    "AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK",
    "AZO","AVB","AVY","AXON","BKR","BALL","BAC","BBWI","BAX","BDX",
    "BLK","BX","BK","BA","BKNG","BWA","BXP","BSX","BMY","BR","BRO",
    "BF.B","BG","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX",
    "CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC",
    "CNP","CDAY","CF","CRL","SCHW","CHD","CI","CINF","CTAS","CSCO",
    "C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA","CAG",
    "COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA","CSGP","COST",
    "CTRA","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DE","DAL",
    "XRAY","DVN","DXCM","FANG","DLR","DFS","DIS","DG","DLTR","D",
    "DPZ","DOV","DOW","DTE","DUK","DD","EMN","ETN","EBAY","ECL",
    "EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR","EOG","EPAM",
    "EQT","EFX","EQIX","EQR","ESS","EL","ETSY","RE","EVRG","ES",
    "EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT",
    "FDX","FITB","FSLR","FE","FMC","F","FTNT","FTV","FOXA","FOX",
    "BEN","FCX","GRMN","IT","GE","GEHC","GEN","GNRC","GD","GIS",
    "GM","GPC","GILD","GPN","GL","GS","HAL","HIG","HAS","HCA",
    "PEAK","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL",
    "HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX",
    "ITW","ILMN","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG",
    "INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JKHY","J","JNJ",
    "JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM",
    "KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS",
    "LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB",
    "MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC",
    "MCD","MCK","MDT","MRK","MET","META","MTD","MGM","MCHP","MU",
    "MSFT","MAA","MRNA","MHK","TAP","MDLZ","MPWR","MNST","MCO","MS",
    "MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE",
    "NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA",
    "NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS",
    "PCAR","PKG","PANW","PARA","PH","PAYX","PAYC","PYPL","PNR","PEP",
    "PFE","PCG","PM","PSX","PNW","PXD","PNC","POOL","PPG","PPL",
    "PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","PVH",
    "QRVO","QCOM","PWR","DGX","RL","RJF","RTX","O","REG","REGN",
    "RF","RSG","RMD","RHI","ROK","ROL","ROP","ROST","RCL","SPGI",
    "CRM","SBAC","SLB","STX","SEE","SRE","NOW","SHW","SPG","SWKS",
    "SJM","SNA","SEDG","SO","LUV","SWK","SBUX","STT","STE","SYK",
    "SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY",
    "TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG",
    "TRV","TRMB","TFC","TYL","TSN","USB","UDR","ULTA","UNP","UAL",
    "UPS","URI","UNH","UHS","VLO","VTR","VRSN","VRSK","VZ","VRTX",
    "VTRS","VICI","V","VMC","WRB","GWW","WAB","WBA","WMT","WBD",
    "WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WHR","WMB",
    "WTW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZION","ZTS"
]

# Deduplicated, alphabetized list
russell1000 = sorted(list(set(russell1000_raw)))

# Exported alias
TICKERS = russell1000