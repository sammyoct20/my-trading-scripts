import os
import sys
from datetime import datetime, time, timedelta
import numpy as np
import pandas as pd
import ta
import yfinance as yf

# ==========================================
# CONFIGURATION AND CLEAN WATCHLIST
# ==========================================
RUN_MODE = "SCAN"  # Set to "BACKTEST" to execute the 10-year simulation matrix.

def fetch_nifty_100_tickers():
    return [
        "ABB.NS", "ACC.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", "ATGL.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASHOKLEY.NS",
        "ASIANPAINT.NS", "AUBANK.NS", "DMART.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJHLDNG.NS", "BALKRISIND.NS", "BANDHANBNK.NS",
        "BANKBARODA.NS", "BANKINDIA.NS", "BHEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BIOCON.NS", "BOSCHLTD.NS", "BRITANNIA.NS", "CGPOWER.NS", "CANBK.NS",
        "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "CUMMINSIND.NS", "DLF.NS", "DABUR.NS", "DIVISLAB.NS",
        "DRREDDY.NS", "EICHERMOT.NS", "GAIL.NS", "GODREJCP.NS", "GODREJPROP.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
        "HAVELLS.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HAL.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDFCFIRSTB.NS", "ITC.NS",
        "INDIANB.NS", "INDHOTEL.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS", "IGL.NS", "INDUSTOWER.NS", "INDUSINDBK.NS", "INFY.NS", "JINDALSTEL.NS",
        "JIOFIN.NS", "JSWSTEEL.NS", "JUBLFOOD.NS", "KOTAKBANK.NS", "LT.NS", "LTTS.NS", "LICHSGFIN.NS", "LUPIN.NS", "M&M.NS",
        "M&MFIN.NS", "MARUTI.NS", "MAXHEALTH.NS", "MPHASIS.NS", "NHPC.NS", "NTPC.NS", "NESTLEIND.NS", "OBEROIRLTY.NS", "ONGC.NS", "OIL.NS",
        "PIIND.NS", "PFC.NS", "POLYMED.NS", "POLYCAB.NS", "POWERGRID.NS", "PNB.NS", "RELIANCE.NS", "SBICARD.NS", "SBILIFE.NS", "SHRIRAMFIN.NS"
    ]

# ==========================================
# MATH CALCULATION CORE ENGINE
# ==========================================
def calculate_indicators(df, is_intraday=True):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.index = pd.to_datetime(df.index)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    
    if is_intraday:
        df["DG"] = df.index.date
        df["VWAP"] = (tp * df["Volume"]).groupby(df["DG"]).cumsum() / (df["Volume"].groupby(df["DG"]).cumsum() + 1e-5)
    else:
        df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"], window=14)
        
    df["EMA"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_S"] = ta.trend.macd_signal(df["Close"])
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    df["DP"] = ta.trend.adx_pos(df["High"], df["Low"], df["Close"], window=14)
    df["DM"] = ta.trend.adx_neg(df["High"], df["Low"], df["Close"], window=14)
    df["CMF"] = ta.volume.chaikin_money_flow(df["High"], df["Low"], df["Close"], df["Volume"], window=20)
    return df

# ==========================================
# MODE 1: DYNAMIC LIVE ACTION SCANNER
# ==========================================
def run_live_scanner(ticker):
    try:
        df = yf.download(ticker, period="15d", interval="15m", progress=False, auto_adjust=False)
        if df.empty or len(df) < 50: return None
        df = calculate_indicators(df, is_intraday=True)
        
        ud = sorted(list(set(df["DG"])))
        c_df, h_df = df[df["DG"] == ud[-1]], df[df["DG"] < ud[-1]]
        if c_df.empty or h_df.empty: return None
        
        vol_m = c_df["Volume"].mean() / (h_df["Volume"].mean() + 1e-5)
        last = df.iloc[-1]
        
        cp, op, rsi, ema, vwap, macd, macd_s, atr, adx, dp, dm, cmf = (
            float(last["Close"]), float(last["Open"]), float(last["RSI"]), float(last["EMA"]),
            float(last["VWAP"]), float(last["MACD"]), float(last["MACD_S"]), float(last["ATR"]),
            float(last["ADX"]), float(last["DP"]), float(last["DM"]), float(last["CMF"])
        )
        dh, dl = float(c_df["High"].max()), float(c_df["Low"].min())
        dr = dh - dl
        
        if cp < ema: return None  # Trend must be positive
        
        score = 10
        reasons = ["Bullish Local Baseline"]
        if cp > op: score += 15; reasons.append("Green Session Close")
        if cp > vwap: score += 30; reasons.append("Above VWAP")
        if dp > dm: score += 20; reasons.append("Bullish DI")
        if adx > 15: score += 20; reasons.append("Trend Active")
        if cmf > 0: score += 20; reasons.append("Inflowing CMF")
        if cp >= float(h_df["High"].max()): score += 35; reasons.append("Resistance Breakout")
        if vol_m >= 1.05: score += 30; reasons.append("Volume Surge")
        
        return {
            "Ticker": ticker.replace(".NS", ""), "Entry Price": round(cp, 2),
            "Target": round(cp + (2.0 * atr), 2), "Stop Loss": round(cp - (1.25 * atr), 2),
            "Score": score, "Vol_Mult": round(vol_m, 2), "Signals": ", ".join(reasons)
        }
    except Exception as e:
        print(f"\n❌ Pipeline runtime exception on {ticker}: {str(e)}")
        return None

# ==========================================
# MASTER DRIVER INTERACTION LAYER
# ==========================================
if __name__ == "__main__":
    watchlist = fetch_nifty_100_tickers()
    if RUN_MODE == "SCAN":
        print("🚀 LIVE BTST SCAN ENGINE INITIALIZED\n" + "="*48)
        signals = []
        for idx, stock in enumerate(watchlist, 1):
            sys.stdout.write(f"\rSweeping Components: [{idx}/{len(watchlist)}] Parsing Ticker: {stock}")
            sys.stdout.flush()
            res = run_live_scanner(stock)
            if res: signals.append(res)
        print("\n\n📊 LIVE SCAN OPERATION COMPLETE")
        if signals:
            picks = pd.DataFrame(signals).sort_values(by="Score", ascending=False).head(5)
            print("\n🔥 ACTIONABLE BTST CALL ALERTS COMPILATION\n" + "="*48)
            for r, (_, row) in enumerate(picks.iterrows(), 1):
                print(f"📢 CALL #{r} | TICKER: {row['Ticker']} (Score: {row['Score']})\n   💰 ENTRY: {row['Entry Price']} | TARGET: {row['Target']} | STOP: {row['Stop Loss']}\n   ⚙️ Triggers: {row['Signals']}\n" + "-"*48)
            
            # Save directly inside GitHub Repository root directory
            path = "btst_live_calls.xlsx"
            picks.insert(0, "Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            try:
                out = pd.concat([pd.read_excel(path), picks], ignore_index=True) if os.path.exists(path) else picks
                out.to_excel(path, index=False)
                print(f"💾 Live calls logged cleanly to {path}")
            except Exception as e:
                alt_path = f"btst_calls_alt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                picks.to_excel(alt_path, index=False)
                print(f"💾 Saved alternative file {alt_path} due to write lock.")
        else:
            print("🛡️ Scan Complete: 0 stocks cleared momentum filters during this session run.")