"""
================================================================================
SCREENER CORE — Logika inti screening saham IHSG
================================================================================
Modul ini SENGAJA tidak bergantung pada Streamlit, supaya bisa dipakai oleh:
  1. app.py            -> UI interaktif untuk dijalankan on-demand di laptop
  2. notify_telegram.py -> script headless untuk dijalankan otomatis via
                            GitHub Actions terjadwal (kirim notifikasi Telegram)

Memisahkan logika inti dari UI adalah praktik umum (separation of concerns)
supaya kedua "wajah" aplikasi ini selalu memakai aturan screening yang identik
dan tidak saling drift ketika salah satunya diubah.
================================================================================
"""

from typing import Optional

import numpy as np

# --- PATCH KOMPATIBILITAS PANDAS_TA vs NUMPY BARU ---
# pandas_ta (versi 0.3.14b) masih memanggil np.NaN (huruf besar), yang sudah
# dihapus di numpy >= 1.24. Patch ini mencegah AttributeError saat import.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import yfinance as yf
import pandas_ta as ta

# ==============================================================================
# KONSTANTA
# ==============================================================================
CHUNK_SIZE = 25             # jumlah ticker per batch download
BACKTEST_FORWARD_DAYS = 20  # window hari ke depan untuk cek TP/SL saat backtest

# ==============================================================================
# DAFTAR STATIS KODE SAHAM IHSG + PEMETAAN SEKTOR (± 160 saham)
#    Catatan: Daftar ini statis, perlu di-update manual berkala (IPO/delisting).
# ==============================================================================
SECTOR_GROUPS = {
    "Perbankan": [
        "BBCA", "BBRI", "BBNI", "BMRI", "BBTN", "BRIS", "ARTO", "BJBR", "BJTM",
        "BNGA", "PNBN", "NISP", "MEGA", "BDMN", "BNLI", "BABP", "BVIC", "BSIM",
        "BEKS", "BGTG", "AGRO", "BKSW", "MAYA", "INPC", "SDRA",
    ],
    "Konsumer / FMCG": [
        "UNVR", "ICBP", "INDF", "MYOR", "GGRM", "HMSP", "KLBF", "SIDO", "CPIN",
        "JPFA", "ULTJ", "ROTI", "MLBI", "CAMP", "TCID", "HOKI", "KAEF", "TSPC",
        "PYFA", "DVLA", "MERK", "SCPI", "MBTO", "WIIM", "STTP",
    ],
    "Telekomunikasi & Menara": ["TLKM", "ISAT", "EXCL", "FREN", "TBIG", "TOWR", "MTEL", "LINK"],
    "Energi & Pertambangan": [
        "ADRO", "PTBA", "ITMG", "INDY", "HRUM", "BUMI", "BYAN", "MEDC", "PGAS",
        "ELSA", "AKRA", "ANTM", "INCO", "TINS", "MDKA", "TPIA", "ESSA", "PSAB",
        "SMRU", "DEWA", "DOID", "GEMS", "TOBA", "ABMM", "MBAP", "BSSR", "PTRO",
        "RUIS", "WINS", "ENRG",
    ],
    "Properti & Real Estate": [
        "BSDE", "CTRA", "PWON", "SMRA", "ASRI", "APLN", "LPKR", "DMAS", "PANI",
        "CBDK", "KIJA", "MTLA",
    ],
    "Infrastruktur & Konstruksi": ["WIKA", "WSKT", "PTPP", "ADHI", "JSMR", "WSBP", "TOTL", "ACST"],
    "Otomotif & Komponen": ["ASII", "AUTO", "IMAS", "SMSM", "GJTL", "GDYR"],
    "Ritel": ["MAPI", "ACES", "LPPF", "RALS", "MIDI", "AMRT", "ERAA", "MAPA"],
    "Semen": ["SMGR", "INTP", "SMBR"],
    "Teknologi & Digital": ["GOTO", "BUKA", "EMTK", "DCII", "MTDL", "WIFI"],
    "Kesehatan / Rumah Sakit": ["MIKA", "HEAL", "SILO", "PRDA"],
    "Agrikultur / CPO": ["AALI", "LSIP", "SIMP", "SGRO", "TBLA", "DSNG"],
    "Kertas & Kimia": ["INKP", "TKIM", "SMPL", "BRPT"],
    "Media": ["SCMA", "MNCN", "VIVA", "MDIA"],
    "Lain-lain": ["UNTR", "SRTG", "BRMS", "HEXA", "PGEO", "BREN", "AMMN", "CUAN", "RATU", "ARCI"],
}

# Bangun daftar flat + mapping ticker -> sektor
SECTOR_MAP = {}
IHSG_TICKERS = []
for _sektor, _tickers in SECTOR_GROUPS.items():
    for _t in _tickers:
        if _t not in SECTOR_MAP:
            SECTOR_MAP[_t] = _sektor
            IHSG_TICKERS.append(_t)


# ==============================================================================
# DOWNLOAD DATA (BATCH)
# ==============================================================================
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_batch_data(tickers_yf: tuple, period: str = "1y"):
    """
    Download data OHLCV untuk sekumpulan ticker sekaligus (jauh lebih cepat
    daripada satu-satu).

    Catatan: fungsi ini SENGAJA tidak memakai caching apapun di sini.
    - app.py (Streamlit) membungkus fungsi ini dengan @st.cache_data.
    - notify_telegram.py memanggilnya langsung tanpa cache (sekali jalan lalu selesai).
    """
    raw = yf.download(
        tickers=list(tickers_yf),
        period=period,
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=True,
    )
    return raw


def extract_ticker_df(raw_data: pd.DataFrame, ticker_yf: str) -> Optional[pd.DataFrame]:
    """Ambil dataframe OHLCV satu ticker dari hasil batch download."""
    if raw_data is None or raw_data.empty:
        return None
    if isinstance(raw_data.columns, pd.MultiIndex):
        if ticker_yf not in raw_data.columns.get_level_values(0):
            return None
        df = raw_data[ticker_yf].copy()
    else:
        # Chunk hanya berisi 1 ticker -> yfinance kadang tidak multi-index
        df = raw_data.copy()
    df = df.dropna(how="all")
    return df if not df.empty else None


# ==============================================================================
# INDIKATOR TEKNIKAL
# ==============================================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung MA50, MA200, RSI(14), dan ATR(14) menggunakan pandas_ta."""
    df = df.copy()
    df["MA50"] = ta.sma(df["Close"], length=50)
    df["MA200"] = ta.sma(df["Close"], length=200)
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    return df


def weekly_trend_ok(df: pd.DataFrame) -> Optional[bool]:
    """
    Konfirmasi trend mingguan: harga close mingguan > MA-10 mingguan
    (setara ~50 hari trading). Return None jika data mingguan belum cukup
    (tidak dianggap gagal, cukup diabaikan sebagai syarat).
    """
    weekly_close = df["Close"].resample("W-FRI").last().dropna()
    if len(weekly_close) < 12:
        return None
    weekly_ma = weekly_close.rolling(10).mean()
    if pd.isna(weekly_ma.iloc[-1]):
        return None
    return bool(weekly_close.iloc[-1] > weekly_ma.iloc[-1])


# ==============================================================================
# QUICK BACKTEST (INFORMASIONAL, BUKAN JAMINAN HASIL KE DEPAN)
# ==============================================================================
def quick_backtest(df: pd.DataFrame, forward_days: int = BACKTEST_FORWARD_DAYS):
    """
    Backtest sederhana pada data yang sudah ditarik (tanpa request tambahan):
    cari semua kejadian historis di mana kondisi trend+RSI yang sama terpenuhi,
    lalu cek apakah TP (3x ATR) atau SL (1.5x ATR) tersentuh lebih dulu dalam
    N hari trading ke depan.

    KETERBATASAN (disengaja dibuat sederhana):
    - Tidak mempertimbangkan compounding / overlapping trade
    - Tidak mempertimbangkan slippage & fee
    - Sinyal yang belum hit TP/SL dalam window diabaikan (bukan dihitung loss)
    Gunakan hanya sebagai indikasi kasar, bukan jaminan performa ke depan.

    Return: (jumlah_sinyal_historis, win_rate_persen atau None jika tidak ada sinyal)
    """
    wins = 0
    total = 0
    n = len(df)
    for i in range(200, n - forward_days):
        row = df.iloc[i]
        if any(pd.isna(row[c]) for c in ["MA50", "MA200", "RSI14", "ATR14"]):
            continue
        harga = row["Close"]
        if harga > row["MA50"] and harga > row["MA200"] and 40 <= row["RSI14"] <= 55:
            sl = harga - 1.5 * row["ATR14"]
            tp = harga + 3.0 * row["ATR14"]
            future = df.iloc[i + 1: i + 1 + forward_days]
            idx_tp = future.index[future["High"] >= tp]
            idx_sl = future.index[future["Low"] <= sl]
            hit_tp = idx_tp[0] if len(idx_tp) > 0 else None
            hit_sl = idx_sl[0] if len(idx_sl) > 0 else None

            if hit_tp is not None and (hit_sl is None or hit_tp <= hit_sl):
                wins += 1
                total += 1
            elif hit_sl is not None:
                total += 1
            # jika keduanya None -> belum hit apapun dalam window, diabaikan

    if total == 0:
        return 0, None
    return total, round(wins / total * 100, 1)


# ==============================================================================
# ANALISIS PER SAHAM (memakai data yang sudah didownload batch)
# ==============================================================================
def analyze_stock(
    kode_saham: str,
    df_raw: pd.DataFrame,
    harga_maksimal: float,
    min_volume: float,
    fee_buy_pct: float,
    fee_sell_pct: float,
    require_weekly: bool,
):
    """
    Mengolah dataframe OHLCV satu saham (yang sudah didapat dari batch download)
    menjadi hasil analisis. Return None jika tidak lolos filter harga (dibuang
    dari hasil), atau dict berisi 'error' jika data bermasalah.
    """
    if df_raw is None or df_raw.empty:
        return {"error": "Data kosong (kemungkinan delisted/simbol salah)", "ticker": kode_saham}

    if len(df_raw) < 210:
        return {"error": f"Histori data tidak cukup ({len(df_raw)} baris)", "ticker": kode_saham}

    df = calculate_indicators(df_raw)
    last = df.iloc[-1]

    harga_now = float(last["Close"])
    ma50 = float(last["MA50"])
    ma200 = float(last["MA200"])
    rsi = float(last["RSI14"])
    atr = float(last["ATR14"])

    if any(pd.isna(x) for x in [ma50, ma200, rsi, atr]):
        return {"error": "Indikator belum lengkap (NaN)", "ticker": kode_saham}

    # --- Filter #1: Harga harus terjangkau modal ---
    if harga_now > harga_maksimal:
        return None

    # --- Filter #2: Likuiditas (volume rata-rata 20 hari) ---
    avg_vol_20 = float(df["Volume"].tail(20).mean())
    if avg_vol_20 < min_volume:
        return {"error": f"Likuiditas rendah (avg vol {avg_vol_20:,.0f} < batas {min_volume:,.0f})",
                "ticker": kode_saham, "_low_liquidity": True}

    # --- Filter #3: Uptrend harian ---
    syarat_trend = (harga_now > ma50) and (harga_now > ma200)

    # --- Filter #4: RSI Buy on Weakness ---
    syarat_rsi = 40 <= rsi <= 55

    # --- Filter #5 (opsional): Konfirmasi trend mingguan ---
    wk_ok = weekly_trend_ok(df)
    syarat_weekly = True if (wk_ok is None or not require_weekly) else wk_ok

    sinyal_buy = syarat_trend and syarat_rsi and syarat_weekly
    status = "✅ BUY" if sinyal_buy else "⏸️ WATCHLIST"

    sl = harga_now - (1.5 * atr)
    tp = harga_now + (3.0 * atr)

    total_fee_pct = fee_buy_pct + fee_sell_pct
    gross_profit_pct = (tp - harga_now) / harga_now * 100
    gross_loss_pct = (harga_now - sl) / harga_now * 100
    net_profit_pct = gross_profit_pct - total_fee_pct
    net_loss_pct = gross_loss_pct + total_fee_pct

    return {
        "Ticker": kode_saham,
        "Sektor": SECTOR_MAP.get(kode_saham, "Lainnya"),
        "Harga Saat Ini": round(harga_now, 0),
        "MA50": round(ma50, 0),
        "MA200": round(ma200, 0),
        "RSI(14)": round(rsi, 2),
        "ATR(14)": round(atr, 2),
        "Avg Vol 20D": round(avg_vol_20, 0),
        "Target TP": round(tp, 0),
        "Batas SL": round(sl, 0),
        "Est. Profit Net (%)": round(net_profit_pct, 2),
        "Est. Rugi Net (%)": round(net_loss_pct, 2),
        "Trend Mingguan": "✔️" if (wk_ok is True) else ("✖️" if wk_ok is False else "N/A"),
        "Status": status,
        "_prioritas": 0 if sinyal_buy else 1,
        "_df": df,  # disimpan sementara untuk chart & backtest, dibuang sebelum ditampilkan
    }


def run_full_screening(
    tickers: list,
    harga_maksimal: float,
    min_volume: float = 500_000,
    fee_buy_pct: float = 0.15,
    fee_sell_pct: float = 0.25,
    require_weekly: bool = False,
    run_backtest: bool = True,
    chunk_size: int = CHUNK_SIZE,
    jeda_batch: float = 1.0,
    on_progress=None,
):
    """
    Menjalankan seluruh pipeline screening (download batch -> analisis ->
    backtest) untuk daftar ticker yang diberikan. Dipakai bersama oleh
    app.py maupun notify_telegram.py supaya alurnya identik.

    on_progress: callback opsional dipanggil sebagai on_progress(pesan: str)
                 untuk melaporkan progres (dipakai untuk print() di script
                 headless, atau update UI di Streamlit).

    Return: dict berisi:
        hasil_list        -> list of dict hasil analisis (BUY & WATCHLIST)
        error_list         -> list of dict error/likuiditas rendah
        low_liquidity_count -> jumlah saham yang disaring karena likuiditas
        chart_data_cache   -> dict {ticker: df_dengan_indikator}
    """
    import time as _time

    def _log(msg):
        if on_progress:
            on_progress(msg)

    hasil_list = []
    error_list = []
    low_liquidity_count = 0
    chart_data_cache = {}

    chunks = list(chunk_list(tickers, chunk_size))
    for ci, chunk in enumerate(chunks):
        tickers_yf = tuple(f"{k}.JK" for k in chunk)
        _log(f"⬇️ Mengunduh batch {ci + 1}/{len(chunks)} ({len(chunk)} saham)...")

        try:
            raw_batch = fetch_batch_data(tickers_yf, period="1y")
        except Exception as e:
            for kode in chunk:
                error_list.append({"ticker": kode, "error": f"Gagal unduh batch: {e}"})
            continue

        for kode in chunk:
            ticker_yf = f"{kode}.JK"
            df_ticker = extract_ticker_df(raw_batch, ticker_yf)

            hasil = analyze_stock(
                kode, df_ticker, harga_maksimal, min_volume,
                fee_buy_pct, fee_sell_pct, require_weekly,
            )

            if hasil is not None:
                if "error" in hasil:
                    if hasil.get("_low_liquidity"):
                        low_liquidity_count += 1
                    else:
                        error_list.append(hasil)
                else:
                    chart_data_cache[kode] = hasil.pop("_df")
                    hasil_list.append(hasil)

        if jeda_batch > 0 and ci < len(chunks) - 1:
            _time.sleep(jeda_batch)

    if run_backtest:
        buy_tickers = [h["Ticker"] for h in hasil_list if h["Status"] == "✅ BUY"]
        for kode in buy_tickers:
            _log(f"🧪 Backtest historis {kode}.JK ...")
            n_signal, win_rate = quick_backtest(chart_data_cache[kode])
            for h in hasil_list:
                if h["Ticker"] == kode:
                    h["Sinyal Historis (1Y)"] = n_signal
                    h["Win Rate Historis (%)"] = win_rate if win_rate is not None else "N/A"
                    break

    for h in hasil_list:
        h.setdefault("Sinyal Historis (1Y)", 0)
        h.setdefault("Win Rate Historis (%)", "N/A")

    return {
        "hasil_list": hasil_list,
        "error_list": error_list,
        "low_liquidity_count": low_liquidity_count,
        "chart_data_cache": chart_data_cache,
    }
