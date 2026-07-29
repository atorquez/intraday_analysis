from data.sp500_list import sp500
from data.nasdaq1000 import nasdaq1000

def load_sp500():
    return sp500

def load_nasdaq1000():
    return nasdaq1000

def load_nyse_top100():
    return sorted(list(set([
        "CAT","DE","GE","BA","HON","MMM","UPS","UNP","LMT","RTX","GD","EMR",
        "ETN","ITW","CMI","DHR","ROP","PH","JCI",
        "XOM","CVX","COP","EOG","SLB","HAL","MPC","VLO","PSX",
        "LIN","APD","SHW","NEM","NUE","FCX","ECL","PPG",
        "JPM","BAC","WFC","GS","MS","BLK","CME","ICE","AXP","COF","USB","PNC","TFC",
        "WMT","HD","MCD","KO","PEP","PG","COST","TGT","LOW","SBUX","TJX","NKE","CL","KMB",
        "JNJ","UNH","MRK","ABBV","PFE","TMO","MDT","BMY","AMGN","CVS","CI",
        "VZ","T",
        "NEE","DUK","SO",
        "PLD","AMT","EQIX",
        "IBM","ORCL","CRM","SAP",
        "WCC","BKR","FDX"
    ])))
