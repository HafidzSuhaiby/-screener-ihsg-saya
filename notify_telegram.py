"""
================================================================================
NOTIFY TELEGRAM — Screener IHSG otomatis via GitHub Actions
================================================================================
Script headless (tanpa UI) yang menjalankan screening penuh lalu mengirim
hasilnya (khusus sinyal BUY) ke Telegram. Didesain untuk dijalankan otomatis
tiap pagi via GitHub Actions terjadwal — lihat
.github/workflows/screener-daily.yml

KONFIGURASI (lewat environment variable, diisi sebagai GitHub Secrets):
  TELEGRAM_BOT_TOKEN   (wajib) - token bot dari @BotFather
  TELEGRAM_CHAT_ID     (wajib) - chat id tujuan notifikasi
  MODAL                (opsional, default 150000)
  MIN_VOLUME           (opsional, default 500000)
  FEE_BUY_PCT          (opsional, default 0.15)
  FEE_SELL_PCT         (opsional, default 0.25)
  REQUIRE_WEEKLY       (opsional, "true"/"false", default "false")
  RUN_BACKTEST         (opsional, "true"/"false", default "true")
  SEKTOR_FILTER        (opsional, nama sektor dipisah koma; kosong = semua)

Cara test lokal (tanpa GitHub Actions):
  export TELEGRAM_BOT_TOKEN="123456:ABC..."
  export TELEGRAM_CHAT_ID="123456789"
  python notify_telegram.py
================================================================================
"""

import os
import sys
from datetime import datetime

import requests

import screener_core as core

TELEGRAM_API_LIMIT = 4000  # batas aman per pesan Telegram (max asli 4096)
OUTPUT_CSV = "hasil_screening_terbaru.csv"  # diunggah sebagai artifact GH Actions


def env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    try:
        return float(val) if val else default
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "ya")


def send_telegram_message(token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, data=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()


def send_long_message(token: str, chat_id: str, full_text: str):
    """Telegram membatasi ~4096 karakter per pesan; pecah jadi beberapa pesan jika perlu."""
    if len(full_text) <= TELEGRAM_API_LIMIT:
        send_telegram_message(token, chat_id, full_text)
        return

    parts = []
    current = ""
    for line in full_text.split("\n"):
        if len(current) + len(line) + 1 > TELEGRAM_API_LIMIT:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)

    for part in parts:
        send_telegram_message(token, chat_id, part)


def format_message(hasil_list, total_dianalisis, modal, harga_maksimal, run_backtest):
    tanggal = datetime.now().strftime("%A, %d %B %Y")
    buy_signals = [h for h in hasil_list if h["Status"] == "✅ BUY"]

    lines = [
        f"📈 *Screener Saham IHSG* — {tanggal}",
        f"Modal: Rp {modal:,.0f} | Harga Maks/Lembar: Rp {harga_maksimal:,.0f}",
        f"Total dianalisis: {total_dianalisis} | Sinyal BUY: {len(buy_signals)}",
        "",
    ]

    if not buy_signals:
        lines.append("Tidak ada sinyal BUY hari ini yang memenuhi semua syarat.")
    else:
        for h in buy_signals:
            block = [
                f"*{h['Ticker']}* — {h['Sektor']}",
                f"Harga: Rp {h['Harga Saat Ini']:,.0f} | RSI: {h['RSI(14)']}",
                f"TP: Rp {h['Target TP']:,.0f} (net +{h['Est. Profit Net (%)']}%) | "
                f"SL: Rp {h['Batas SL']:,.0f} (net -{h['Est. Rugi Net (%)']}%)",
            ]
            if run_backtest:
                wr = h.get("Win Rate Historis (%)", "N/A")
                ns = h.get("Sinyal Historis (1Y)", 0)
                block.append(f"Win-rate historis 1Y: {wr}% ({ns} sinyal)")
            lines.append("\n".join(block))
            lines.append("")

    lines.append("⚠️ Bukan rekomendasi investasi. Selalu DYOR sebelum eksekusi.")
    return "\n".join(lines)


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID wajib di-set sebagai "
              "environment variable (GitHub Secrets).", file=sys.stderr)
        sys.exit(1)

    modal = env_float("MODAL", 150_000)
    min_volume = env_float("MIN_VOLUME", 500_000)
    fee_buy_pct = env_float("FEE_BUY_PCT", 0.15)
    fee_sell_pct = env_float("FEE_SELL_PCT", 0.25)
    require_weekly = env_bool("REQUIRE_WEEKLY", False)
    run_backtest = env_bool("RUN_BACKTEST", True)

    sektor_filter_raw = os.environ.get("SEKTOR_FILTER", "").strip()
    if sektor_filter_raw:
        sektor_filter = [s.strip() for s in sektor_filter_raw.split(",")]
        tickers = [t for t in core.IHSG_TICKERS if core.SECTOR_MAP.get(t) in sektor_filter]
    else:
        tickers = core.IHSG_TICKERS

    harga_maksimal = modal / 100

    print(f"Mulai screening {len(tickers)} saham | Modal Rp{modal:,.0f} "
          f"| Harga maks Rp{harga_maksimal:,.0f}")

    try:
        result = core.run_full_screening(
            tickers=tickers,
            harga_maksimal=harga_maksimal,
            min_volume=min_volume,
            fee_buy_pct=fee_buy_pct,
            fee_sell_pct=fee_sell_pct,
            require_weekly=require_weekly,
            run_backtest=run_backtest,
            on_progress=lambda msg: print(msg),
        )
    except Exception as e:
        # Kalau screening gagal total, tetap coba kabari lewat Telegram
        # supaya kegagalan tidak senyap.
        error_text = f"⚠️ *Screener IHSG gagal dijalankan hari ini.*\nError: `{e}`"
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        try:
            send_telegram_message(token, chat_id, error_text)
        except Exception as send_err:
            print(f"Gagal juga mengirim notifikasi error: {send_err}", file=sys.stderr)
        sys.exit(1)

    hasil_list = result["hasil_list"]
    total_dianalisis = len(tickers)

    # Simpan CSV lokal (di GitHub Actions ini akan diunggah sebagai artifact)
    if hasil_list:
        import pandas as pd
        df_out = pd.DataFrame(hasil_list).drop(columns=["_prioritas"], errors="ignore")
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"Hasil disimpan ke {OUTPUT_CSV} ({len(df_out)} baris)")

    message = format_message(hasil_list, total_dianalisis, modal, harga_maksimal, run_backtest)
    print("--- Pesan yang akan dikirim ---")
    print(message)
    print("-------------------------------")

    send_long_message(token, chat_id, message)
    print("Notifikasi Telegram terkirim.")


if __name__ == "__main__":
    main()
