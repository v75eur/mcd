import time, requests, io, pytz, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd

# ===== CONFIGURATION NTFY =====
NTFY_XAU = os.getenv("NTFY_XAU", "https://ntfy.sh/rick-xau-macd-secret-2026")
NTFY_EUR = os.getenv("NTFY_EUR", "https://ntfy.sh/rick-eur-macd-secret-2026")
NTFY_GBP = os.getenv("NTFY_GBP", "https://ntfy.sh/rick-gbp-macd-secret-2026")
NTFY_V75 = os.getenv("NTFY_V75", "https://ntfy.sh/rick-v75-macd-secret-2026")

PAIRS = {
    "XAUUSD": {"symbol": "GC=F", "ntfy": NTFY_XAU, "dec": 2, "name": "XAUUSD (Or)"},
    "EURUSD": {"symbol": "EURUSD=X", "ntfy": NTFY_EUR, "dec": 5, "name": "EURUSD"},
    "GBPUSD": {"symbol": "GBPUSD=X", "ntfy": NTFY_GBP, "dec": 5, "name": "GBPUSD"},
    "V75": {"symbol": "R_75", "ntfy": NTFY_V75, "dec": 2, "name": "Volatility 75", "source": "deriv"},
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def send(url, title, msg, img=None):
    for i in range(5):
        try:
            h = {"Title": title}
            if img:
                h["Filename"] = "chart.png"
                h["X-Message"] = msg
                r = requests.post(url, data=img, headers={k: v.encode() for k, v in h.items()}, timeout=30)
            else:
                r = requests.post(url, data=msg.encode(), headers=h, timeout=15)
            if r.status_code == 200:
                log(f"✅ {title}")
                return True
            log(f"⚠️ Tentative {i+1}: status {r.status_code}")
            time.sleep(2**i)
        except Exception as e:
            log(f"❌ Erreur: {e}")
            time.sleep(2**i)
    return False

def get_candles_deriv(sym, granularity=60):
    try:
        import websocket as ws_client
        log(f"🔌 Connexion Deriv pour {sym}...")
        ws = ws_client.create_connection('wss://ws.binaryws.com/websockets/v3?app_id=1089', timeout=10)
        ws.send(json.dumps({"ticks_history": sym, "count": 100, "end": "latest", "start": 1, "style": "candles", "granularity": granularity}))
        r = json.loads(ws.recv())
        ws.close()
        if "candles" in r:
            candles = [{"t": c["epoch"], "o": float(c["open"]), "h": float(c["high"]), "l": float(c["low"]), "c": float(c["close"])} for c in r["candles"]]
            log(f"✅ Deriv: {len(candles)} bougies pour {sym}")
            return candles
        else:
            log(f"⚠️ Deriv: pas de bougies pour {sym}")
            return []
    except Exception as e:
        log(f"❌ Deriv {sym}: {e}")
        return []

def get_candles(sym, interval="1h", range_days="60d"):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval={interval}&range={range_days}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = r.json()["chart"]["result"][0]
        closes = data["indicators"]["quote"][0]["close"]
        opens = data["indicators"]["quote"][0]["open"]
        highs = data["indicators"]["quote"][0]["high"]
        lows = data["indicators"]["quote"][0]["low"]
        timestamps = data["timestamp"]
        candles = []
        for i in range(len(closes)):
            if closes[i] and opens[i] and highs[i] and lows[i]:
                candles.append({"t": timestamps[i], "o": opens[i], "h": highs[i], "l": lows[i], "c": closes[i]})
        candles = candles[-100:] if len(candles) > 100 else candles
        log(f"✅ Yahoo: {len(candles)} bougies pour {sym} en {interval}")
        return candles
    except Exception as e:
        log(f"❌ Yahoo {sym}: {e}")
        return []

def calc_macd(candles, fast=3, slow=100, signal=3):
    closes = np.array([c["c"] for c in candles])
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).rolling(window=signal).mean().values
    histo = macd_line - signal_line
    return macd_line, signal_line, histo

def detect_cross_zero(macd, signal, histo):
    if len(macd) < 2:
        return None, False
    if macd[-1] > 0 and signal[-1] > 0 and histo[-1] > 0:
        if macd[-2] <= 0 and signal[-2] <= 0 and histo[-2] <= 0:
            return "HAUT", True
    elif macd[-1] < 0 and signal[-1] < 0 and histo[-1] < 0:
        if macd[-2] >= 0 and signal[-2] >= 0 and histo[-2] >= 0:
            return "BAS", True
    return None, False

def chart_macd_mt5(candles, macd_line, signal_line, histo, title, price, dec=2):
    if len(candles) < 20:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#0a0a0a')
    
    ax1.set_facecolor('#0d1117')
    for i, c in enumerate(candles):
        col = '#26a69a' if c["c"] >= c["o"] else '#ef5350'
        ax1.plot([i, i], [c["l"], c["h"]], color=col, lw=1.2)
        ax1.add_patch(plt.Rectangle((i-0.35, min(c["o"], c["c"])), 0.7, abs(c["c"]-c["o"]) or 0.0001, facecolor=col, edgecolor=col))
    
    ax1.axhline(price, color='#f59e0b', ls='-.', lw=2, alpha=0.7)
    ax1.text(len(candles)-1, price, f'  {price:.{dec}f}', color='#f59e0b', fontsize=12, fontweight='bold')
    ax1.set_title(f"{title} - Chandeliers", color='white', fontsize=14, fontweight='bold')
    ax1.set_ylabel("Prix", color='white', fontsize=12)
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='gray')
    
    ax2.set_facecolor('#0d1117')
    colors = ['#26a69a' if h >= 0 else '#ef5350' for h in histo]
    ax2.bar(range(len(histo)), histo, color=colors, alpha=0.7, width=0.8, label='Histogramme')
    ax2.plot(range(len(macd_line)), macd_line, color='#00b4d8', lw=2, label='MACD (3)')
    ax2.plot(range(len(signal_line)), signal_line, color='#ff6b6b', lw=2, label='Signal (3 SMA)')
    ax2.axhline(0, color='white', lw=1, linestyle='--', alpha=0.5)
    ax2.set_title("MACD (3, 100, 3)", color='white', fontsize=14, fontweight='bold')
    ax2.set_xlabel("Bougies", color='white', fontsize=12)
    ax2.set_ylabel("MACD", color='white', fontsize=12)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.15, color='gray')
    ax2.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#555', labelcolor='white', framealpha=0.9)
    
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor='#0a0a0a')
    buf.seek(0)
    plt.close()
    return buf.getvalue()

def analyze_macd(key, info):
    log(f"🔍 Analyse MACD {key}...")
    
    src = info.get("source", "yahoo")
    if src == "deriv":
        candles_h4 = get_candles_deriv(info["symbol"], granularity=14400)
        candles_m10 = get_candles_deriv(info["symbol"], granularity=600)
    else:
        candles_h4 = get_candles(info["symbol"], interval="1h", range_days="60d")
        if len(candles_h4) >= 4:
            df = pd.DataFrame(candles_h4)
            df['t'] = pd.to_datetime(df['t'], unit='s')
            df.set_index('t', inplace=True)
            resampled = df.resample('4H').agg({
                'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'
            }).dropna()
            candles_h4 = [{"t": int(ts.timestamp()), "o": row['o'], "h": row['h'], "l": row['l'], "c": row['c']} for ts, row in resampled.iterrows()]
            candles_h4 = candles_h4[-50:]
        candles_m10 = get_candles(info["symbol"], interval="5m", range_days="7d")
    
    if not candles_h4 or not candles_m10:
        log(f"⚠️ Pas de données pour {key}")
        return
    
    macd_h4, signal_h4, histo_h4 = calc_macd(candles_h4)
    macd_m10, signal_m10, histo_m10 = calc_macd(candles_m10)
    
    direction_h4, cross_h4 = detect_cross_zero(macd_h4, signal_h4, histo_h4)
    direction_m10, cross_m10 = detect_cross_zero(macd_m10, signal_m10, histo_m10)
    
    if macd_h4[-1] > 0 and signal_h4[-1] > 0 and histo_h4[-1] > 0:
        tendance_h4 = "HAUSSIERE"
    elif macd_h4[-1] < 0 and signal_h4[-1] < 0 and histo_h4[-1] < 0:
        tendance_h4 = "BAISSIERE"
    else:
        tendance_h4 = "NEUTRE"
    
    dec = info["dec"]
    cp = candles_m10[-1]["c"]
    h = datetime.now(pytz.timezone('Africa/Porto-Novo')).hour
    
    msg = ""
    conseil = ""
    condition_remplie = False
    
    if cross_h4:
        if direction_h4 == "HAUT":
            msg = f"📊 Tendance H4 DEVIENT HAUSSIERE\n✅ MACD + Signal + Histo croisent 0 vers le HAUT"
            conseil = "🔍 Tendance HAUSSIERE confirmée (H4)"
        elif direction_h4 == "BAS":
            msg = f"📊 Tendance H4 DEVIENT BAISSIERE\n✅ MACD + Signal + Histo croisent 0 vers le BAS"
            conseil = "🔍 Tendance BAISSIERE confirmée (H4)"
        condition_remplie = True
    elif cross_m10 and direction_m10 == "HAUT" and tendance_h4 == "HAUSSIERE":
        msg = f"📈 SIGNAL ACHAT MACD\n✅ Tendance HAUSSIERE (H4) confirmée\n✅ Croisement HAUT sur M10"
        conseil = "📈 ACHAT (H4 HAUSSIER + M10 HAUT)"
        condition_remplie = True
    elif cross_m10 and direction_m10 == "BAS" and tendance_h4 == "BAISSIERE":
        msg = f"📉 SIGNAL VENTE MACD\n✅ Tendance BAISSIERE (H4) confirmée\n✅ Croisement BAS sur M10"
        conseil = "📉 VENTE (H4 BAISSIER + M10 BAS)"
        condition_remplie = True
    
    if condition_remplie:
        full_msg = f"{msg}\n\n💰 Prix: {cp:.{dec}f}\n🕒 {h}H Bénin\n🤖 MACD Bot"
        
        log(f"📤 ENVOI ÉVÉNEMENT {key}...")
        send(info["ntfy"], f"🚨 {key} - {conseil}", full_msg)
        
        time.sleep(1)
        
        log(f"📤 Graphique H4 pour {key}...")
        img_h4 = chart_macd_mt5(candles_h4, macd_h4, signal_h4, histo_h4, f"{info['name']} H4", cp, dec)
        if img_h4:
            send(info["ntfy"], f"{key} H4 - {conseil}", "MACD H4 (3,100,3)", img_h4)
        
        time.sleep(1)
        
        log(f"📤 Graphique M10 pour {key}...")
        img_m10 = chart_macd_mt5(candles_m10, macd_m10, signal_m10, histo_m10, f"{info['name']} M10", cp, dec)
        if img_m10:
            send(info["ntfy"], f"{key} M10 - {conseil}", "MACD M10 (3,100,3)", img_m10)
    else:
        log(f"⏭️ RIEN NE SE PASSE POUR {key} - SILENCE")

if __name__ == "__main__":
    log("🚀 MACD BOT - Stratégie H4 + M10")
    now = datetime.now(pytz.timezone('Africa/Porto-Novo'))
    h, j = now.hour, now.weekday()
    
    log("→ V75 (7j/7)")
    analyze_macd("V75", PAIRS["V75"])
    
    if j < 5:
        log(f"📊 Analyse Forex {h}H")
        for key in ["XAUUSD", "EURUSD", "GBPUSD"]:
            log(f"→ {key}")
            analyze_macd(key, PAIRS[key])
    else:
        log(f"💤 Forex ferme week-end")
    
    log("✅ Termine")
