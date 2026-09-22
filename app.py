"""
================================================================================
SCREENER SAHAM IHSG LOKAL (v3)
Aplikasi berbasis Streamlit untuk screening saham IHSG. Logika inti (indikator,
download data, backtest) ada di screener_core.py — file ini murni UI.

Didesain untuk dijalankan on-demand di laptop lokal (bukan server 24 jam).
Untuk notifikasi otomatis tiap pagi tanpa buka app, lihat notify_telegram.py
(dijalankan terjadwal via GitHub Actions, gratis).
================================================================================
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import screener_core as core

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Screener Saham IHSG Lokal",
    page_icon="📈",
    layout="wide",
)

HISTORY_FILE = Path(__file__).parent / "riwayat_screening.csv"
CACHE_TTL = 3600  # detik, cache download bertahan 1 jam dalam 1 sesi


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_fetch_batch_data(tickers_yf: tuple, period: str = "1y"):
    """Wrapper cache Streamlit di atas core.fetch_batch_data (core-nya sendiri
    tidak bergantung pada Streamlit supaya bisa dipakai juga oleh script headless)."""
    return core.fetch_batch_data(tickers_yf, period)


# ==============================================================================
# CHART CANDLESTICK
# ==============================================================================
def render_chart(kode: str, df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=kode,
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50",
                              line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200",
                              line=dict(color="blue", width=1)))
    fig.update_layout(
        height=480,
        xaxis_rangeslider_visible=False,
        title=f"{kode}.JK — 1 Tahun Terakhir",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# RIWAYAT SCREENING LOKAL (CSV)
# ==============================================================================
def save_to_history(df_hasil: pd.DataFrame, modal: float):
    df_save = df_hasil.copy()
    df_save.insert(0, "Timestamp", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    df_save.insert(1, "Modal", modal)
    header_needed = not HISTORY_FILE.exists()
    df_save.to_csv(HISTORY_FILE, mode="a", index=False, header=header_needed)


def load_history() -> Optional[pd.DataFrame]:
    if HISTORY_FILE.exists():
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return None
    return None


# ==============================================================================
# UI STREAMLIT
# ==============================================================================
def main():
    st.title("📈 Screener Saham IHSG Lokal")
    st.caption(
        "Analisis teknikal otomatis (MA50/MA200/RSI/ATR) untuk saham IHSG "
        "berdasarkan modal Anda. Data via Yahoo Finance (yfinance)."
    )

    # --- Sidebar: Pengaturan ---
    with st.sidebar:
        st.header("⚙️ Pengaturan Screening")

        sektor_pilihan = st.multiselect(
            "Filter Sektor (kosongkan = semua sektor)",
            options=sorted(core.SECTOR_GROUPS.keys()),
        )

        min_volume = st.number_input(
            "Minimal Volume Rata-rata 20 Hari (lembar)",
            min_value=0, value=500_000, step=100_000,
            help="Saham dengan volume di bawah ini dianggap kurang likuid dan disaring.",
        )

        require_weekly = st.checkbox(
            "Wajib Konfirmasi Trend Mingguan",
            value=False,
            help="Jika aktif, sinyal BUY hanya valid jika trend mingguan juga naik "
                 "(mengurangi false-signal, tapi sinyal jadi lebih sedikit).",
        )

        st.markdown("**Estimasi Biaya Transaksi (Broker)**")
        c_fee1, c_fee2 = st.columns(2)
        fee_buy_pct = c_fee1.number_input("Fee Beli (%)", min_value=0.0, value=0.15, step=0.01, format="%.2f")
        fee_sell_pct = c_fee2.number_input("Fee Jual (%)", min_value=0.0, value=0.25, step=0.01, format="%.2f")

        run_backtest = st.checkbox(
            "Jalankan Quick Backtest untuk sinyal BUY",
            value=True,
            help="Mengecek seberapa sering pola sinyal ini historically kena TP duluan "
                 "vs SL duluan pada saham yang sama (1 tahun terakhir). Simulasi sederhana, "
                 "bukan jaminan hasil ke depan.",
        )

        with st.expander("Advanced"):
            jeda_batch = st.number_input(
                "Jeda antar batch unduhan (detik)", min_value=0.0, value=1.0, step=0.5,
                help="Mencegah rate-limit dari Yahoo Finance saat mengunduh banyak batch.",
            )
            chunk_size = st.slider("Ukuran batch (jumlah saham per unduhan)", 10, 50, core.CHUNK_SIZE)

        st.markdown("---")
        st.markdown(f"**Total saham dalam daftar:** {len(core.IHSG_TICKERS)}")
        st.markdown(
            "💡 Mau dapat sinyal BUY otomatis tiap pagi tanpa buka app ini? "
            "Lihat `notify_telegram.py` + `.github/workflows/screener-daily.yml` "
            "(dijalankan gratis via GitHub Actions)."
        )
        st.markdown(
            "⚠️ Ini bukan rekomendasi investasi. Selalu lakukan riset mandiri "
            "(DYOR) sebelum mengambil keputusan trading."
        )

    # --- Input Modal ---
    col1, col2 = st.columns([2, 1])
    with col1:
        modal = st.number_input(
            "💰 Modal Awal Anda (Rp)", min_value=10_000, value=150_000, step=10_000, format="%d",
        )
    with col2:
        harga_maksimal = modal / 100  # 1 lot = 100 lembar
        st.metric("Harga Maksimal / Lembar", f"Rp {harga_maksimal:,.0f}")

    tickers_to_scan = (
        [t for t in core.IHSG_TICKERS if core.SECTOR_MAP.get(t) in sektor_pilihan]
        if sektor_pilihan else core.IHSG_TICKERS
    )

    mulai = st.button("🚀 Mulai Analisis", type="primary", use_container_width=True)

    if mulai:
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(tickers_to_scan)
        state = {"processed": 0}

        def on_progress(msg: str):
            status_text.text(msg)

        # NB: run_full_screening (di screener_core) melakukan batch download +
        # analisis + backtest sekaligus, dipakai identik oleh notify_telegram.py.
        # Progress bar di sini kita gerakkan per-chunk secara kasar karena
        # granularitas asli ada di dalam fungsi tsb.
        chunks = list(core.chunk_list(tickers_to_scan, chunk_size))
        hasil_list, error_list = [], []
        low_liquidity_count = 0
        chart_data_cache = {}

        for ci, chunk in enumerate(chunks):
            on_progress(f"⬇️ Mengunduh batch {ci + 1}/{len(chunks)} ({len(chunk)} saham)...")
            tickers_yf = tuple(f"{k}.JK" for k in chunk)
            try:
                raw_batch = cached_fetch_batch_data(tickers_yf, period="1y")
            except Exception as e:
                for kode in chunk:
                    error_list.append({"ticker": kode, "error": f"Gagal unduh batch: {e}"})
                state["processed"] += len(chunk)
                progress_bar.progress(min(state["processed"] / total, 1.0))
                continue

            for kode in chunk:
                on_progress(f"🔍 Memproses {kode}.JK ...")
                df_ticker = core.extract_ticker_df(raw_batch, f"{kode}.JK")
                hasil = core.analyze_stock(
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
                state["processed"] += 1
                progress_bar.progress(min(state["processed"] / total, 1.0))

            if jeda_batch > 0 and ci < len(chunks) - 1:
                import time
                time.sleep(jeda_batch)

        status_text.text("✅ Analisis selesai!")
        progress_bar.empty()

        # --- Quick Backtest untuk sinyal BUY ---
        if run_backtest:
            buy_tickers = [h["Ticker"] for h in hasil_list if h["Status"] == "✅ BUY"]
            if buy_tickers:
                bt_progress = st.progress(0)
                bt_status = st.empty()
                for i, kode in enumerate(buy_tickers):
                    bt_status.text(f"🧪 Backtest historis {kode}.JK ...")
                    n_signal, win_rate = core.quick_backtest(chart_data_cache[kode])
                    for h in hasil_list:
                        if h["Ticker"] == kode:
                            h["Sinyal Historis (1Y)"] = n_signal
                            h["Win Rate Historis (%)"] = win_rate if win_rate is not None else "N/A"
                            break
                    bt_progress.progress((i + 1) / len(buy_tickers))
                bt_status.empty()
                bt_progress.empty()
        for h in hasil_list:
            h.setdefault("Sinyal Historis (1Y)", 0)
            h.setdefault("Win Rate Historis (%)", "N/A")

        st.session_state["chart_data_cache"] = chart_data_cache

        # --- Ringkasan ---
        jumlah_buy = sum(1 for h in hasil_list if h["Status"] == "✅ BUY")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Dianalisis", total)
        c2.metric("Terjangkau Modal", len(hasil_list))
        c3.metric("Sinyal BUY", jumlah_buy)
        c4.metric("Likuiditas Rendah", low_liquidity_count)
        c5.metric("Gagal Ditarik", len(error_list))

        st.markdown("---")

        if len(hasil_list) == 0:
            st.warning(
                "Tidak ada saham yang terjangkau modal & memenuhi filter likuiditas. "
                "Coba naikkan modal atau turunkan batas volume minimum."
            )
        else:
            df_hasil = pd.DataFrame(hasil_list)
            df_hasil = df_hasil.sort_values(
                by=["_prioritas", "RSI(14)"], ascending=[True, True]
            ).drop(columns=["_prioritas"]).reset_index(drop=True)

            st.subheader("📋 Hasil Screening")

            def highlight_buy(row):
                if row["Status"] == "✅ BUY":
                    return ["background-color: #d4f8d4"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_hasil.style.apply(highlight_buy, axis=1).format({
                    "Harga Saat Ini": "Rp {:,.0f}",
                    "MA50": "Rp {:,.0f}",
                    "MA200": "Rp {:,.0f}",
                    "Avg Vol 20D": "{:,.0f}",
                    "Target TP": "Rp {:,.0f}",
                    "Batas SL": "Rp {:,.0f}",
                    "Est. Profit Net (%)": "{:.2f}%",
                    "Est. Rugi Net (%)": "{:.2f}%",
                }),
                use_container_width=True,
                height=500,
            )

            csv = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil (CSV)", data=csv,
                file_name="hasil_screener_ihsg.csv", mime="text/csv",
            )

            save_to_history(df_hasil, modal)

            # --- Chart Viewer ---
            st.markdown("---")
            st.subheader("📊 Chart Saham")
            pilihan_chart = st.selectbox("Pilih saham untuk melihat chart:", df_hasil["Ticker"].tolist())
            if pilihan_chart and pilihan_chart in chart_data_cache:
                render_chart(pilihan_chart, chart_data_cache[pilihan_chart])

            if run_backtest:
                st.caption(
                    "ℹ️ **Tentang Win Rate Historis**: dihitung dari kejadian pola sinyal yang sama "
                    "(trend + RSI) pada saham yang sama selama 1 tahun terakhir, lalu dicek apakah "
                    "TP atau SL tersentuh lebih dulu. Simulasi sederhana — tidak memperhitungkan fee, "
                    "slippage, atau overlapping trade. Sinyal historis yang sedikit (<5) kurang "
                    "signifikan secara statistik."
                )

        # --- Detail Error ---
        if error_list:
            with st.expander(f"⚠️ Lihat {len(error_list)} saham yang gagal ditarik/diproses"):
                st.dataframe(pd.DataFrame(error_list), use_container_width=True)

    # --- Riwayat Screening ---
    st.markdown("---")
    with st.expander("📜 Riwayat Screening Sebelumnya"):
        df_hist = load_history()
        if df_hist is None or df_hist.empty:
            st.info("Belum ada riwayat screening tersimpan.")
        else:
            st.dataframe(df_hist.tail(300).iloc[::-1], use_container_width=True, height=300)
            hist_csv = df_hist.to_csv(index=False).encode("utf-8")
            col_a, col_b = st.columns(2)
            col_a.download_button(
                "⬇️ Download Seluruh Riwayat", data=hist_csv,
                file_name="riwayat_screening_full.csv", mime="text/csv",
            )
            if col_b.button("🗑️ Hapus Riwayat"):
                HISTORY_FILE.unlink(missing_ok=True)
                st.success("Riwayat dihapus. Refresh halaman untuk melihat perubahan.")


if __name__ == "__main__":
    main()
