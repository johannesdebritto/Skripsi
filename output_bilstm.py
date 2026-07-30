import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Pastikan impor berasal dari file spesifik Bi-LSTM yang sudah kita perbarui
from spk_logic_bilstm import evaluasi_spk
from ai_engine_bilstm import hitung_sekuens_prediksi
from grafik_bilstm import buat_grafik_multimodal_bilstm

# Mengambil fungsi Gemini dari file gemini_helper.py
from gemini_helper import ambil_analisis_gemini 

def tampilkan_hasil_bilstm(hari_target, model, scalers, df):
    # ---> PERBAIKAN DI SINI: Mencegah unpack error karena scalers adalah objek tunggal
    scaler_fitur = scaler_target = scalers

    # AMBIL TANGGAL AKURAT
    try:
        if 'Tanggal' in df.columns:
            tanggal_terakhir = pd.to_datetime(df['Tanggal'].iloc[-1])
        elif 'Date' in df.columns:
            tanggal_terakhir = pd.to_datetime(df['Date'].iloc[-1])
        else:
            tanggal_terakhir = pd.to_datetime(df.index[-1])
    except Exception:
        tanggal_terakhir = datetime.now()

    # Hitung tanggal target yang sebenarnya untuk ditampilkan di UI
    tanggal_target = (tanggal_terakhir + timedelta(days=int(hari_target))).strftime("%d/%m/%Y")
    tanggal_sekarang_str = tanggal_terakhir.strftime("%d %B %Y").upper()

    with st.spinner(f"Bi-LSTM engine is calculating projections for {tanggal_target}..."):
        
        skor, keputusan, warna, indikator, h_sekarang, h_prediksi, selisih, margin = evaluasi_spk(
            hari_target, model, scaler_fitur, scaler_target, df
        )

        # 1. Judul trik pakai <div> agar ikon link 100% hilang
        st.markdown(f"<div style='text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 10px;'>📊 BI-LSTM DECISION RESULT ({tanggal_target})</div>", unsafe_allow_html=True)
        st.divider()

        # ---------------------------------------------------------
        # TAMPILAN KEPUTUSAN UTAMA
        # ---------------------------------------------------------
        if warna == "success":
            bg_color = "#28a745"  # Hijau untuk BUY
        elif warna == "error":
            bg_color = "#dc3545"  # Merah untuk SELL
        else:
            bg_color = "#ffb300"  # Kuning/Orange gelap untuk HOLD
        
        st.markdown(
            f"<div style='text-align: center; border-radius: 10px; padding: 20px; background-color: {bg_color};'>"
            f"<p style='color: white; font-weight: 900; margin: 0px; font-size: 48px; text-transform: uppercase; font-family: sans-serif; letter-spacing: 1px;'>{keputusan}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.markdown(
            "<p style='text-align: center; color: #888888; font-size: 14px; margin-top: 10px;'>"
            "⚠️ <b>Note:</b> Transaction decisions are entirely at your own risk. "
            "This AI is a probabilistic forecasting tool, not an absolute guarantee or financial advice."
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # ANGKA METRIK UTAMA 
        # ---------------------------------------------------------
        str_h_sekarang = f"IDR {int(h_sekarang):,}".replace(',', '.')
        str_h_prediksi = f"IDR {int(h_prediksi):,}".replace(',', '.')
        str_margin = f"IDR {int(margin):,}".replace(',', '.')
        
        if selisih > 0:
            warna_selisih = "#28a745" # Hijau
            tanda = "+"
        elif selisih < 0:
            warna_selisih = "#dc3545" # Merah
            tanda = ""
        else:
            warna_selisih = "#888888" # Abu-abu
            tanda = ""
            
        str_selisih = f"{tanda}{int(selisih):,} IDR".replace(',', '.')

        html_metrik = f"""
        <div style="display: flex; justify-content: space-around; align-items: center; background: linear-gradient(135deg, rgba(240, 242, 246, 0.7), rgba(255, 255, 255, 0.4)); padding: 25px 15px; border-radius: 20px; margin-bottom: 20px; border: 1px solid rgba(0,0,0,0.05);">
            <div style="text-align: center; flex: 1; border-right: 1px solid rgba(0,0,0,0.1);">
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;"> PRICE ON {tanggal_sekarang_str}</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 26px;">{str_h_sekarang}</h3>
            </div>
            <div style="text-align: center; flex: 1; border-right: 1px solid rgba(0,0,0,0.1);">
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;"> BI-LSTM PROJECTION</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 26px;">{str_h_prediksi}</h3>
                <p style="margin: 0; font-size: 14px; color: {warna_selisih}; font-weight: bold;">{str_selisih}</p>
            </div>
            <div style="text-align: center; flex: 1;">
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;"> MARGIN LIMIT</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 26px;">{str_margin}</h3>
            </div>
        </div>
        """
        st.markdown(html_metrik, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # GRID 2x2: BUKTI INDIKATOR
        # ---------------------------------------------------------
        st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>🔍 Decision Foundation Indicators</div>", unsafe_allow_html=True)
        
        st.caption("Multivariate data (Last 30 days) underlying the expert system analysis.")

        col1, col2 = st.columns(2)

        # Kotak Kiri Atas (AI Bi-LSTM)
        with col1:
            with st.container(border=True):
                st.markdown(f"**{indikator['ai']['icon']} Bi-LSTM Prediction Trend**")
                
                ai_delta_val = int(indikator['ai']['delta'])
                ai_sentimen = str(indikator['ai']['sentimen']).strip()
                
                if ai_delta_val < 0:
                    val_str = f"-IDR {abs(ai_delta_val):,}".replace(',', '.')
                    # Memaksa string diawali '-' kalau belum ada
                    if not ai_sentimen.startswith("-"):
                        ai_sentimen = f"- {ai_sentimen}"
                else:
                    val_str = f"IDR {ai_delta_val:,}".replace(',', '.')
                    # Menghapus karakter '-' jika ternyata positif
                    ai_sentimen = ai_sentimen.lstrip("- ")

                st.metric("Est. Profit/Loss", val_str, ai_sentimen)

        # Kotak Kanan Atas (KURS USD)
        with col2:
            with st.container(border=True):
                st.markdown(f"**{indikator['kurs']['icon']} Macroeconomy: USD/IDR Rate**")
                st.metric("Actual Exchange Rate", f"IDR {int(indikator['kurs']['sekarang']):,}".replace(',', '.'), f"{int(indikator['kurs']['delta']):,} IDR".replace(',', '.'), delta_color="inverse")

        col3, col4 = st.columns(2)

        # Kotak Kiri Bawah (MINYAK DUNIA)10
        with col3:
            with st.container(border=True):
                st.markdown(f"**{indikator['minyak']['icon']} Commodity: Crude Oil (IDR)**")
                st.metric("Actual Value per Barrel", f"IDR {int(indikator['minyak']['sekarang']):,}".replace(',', '.'), f"{int(indikator['minyak']['delta']):,} IDR".replace(',', '.'))

        # Kotak Kanan Bawah (FED RATE)
        with col4:
            with st.container(border=True):
                st.markdown(f"**{indikator['fed']['icon']} Policy: US Fed Rate**")
                st.metric("Actual Fed Rate", f"{indikator['fed']['sekarang']:.2f}%", f"{indikator['fed']['delta']:.2f}%", delta_color="inverse")

        # ---------------------------------------------------------
        # BAGIAN ANALISIS LLM: ALASAN MULTIVARIAT
        # ---------------------------------------------------------
        st.divider()
        st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 15px;'>🤖 Comprehensive Expert System Analysis</div>", unsafe_allow_html=True)

        with st.spinner("Generating decision rationale based on multivariate data..."):
            sentimen = "BULLISH (Strongly Positive)" if skor >= 2 else "BEARISH (Strongly Negative)" if skor <= -2 else "SIDEWAYS (Neutral Consolidation)"
            
            hasil_analisis = ambil_analisis_gemini(
                "Bi-LSTM", hari_target, h_sekarang, h_prediksi, selisih, keputusan, sentimen, indikator, skor
            )
            
            st.info(hasil_analisis, icon="💡")
            
        # ---------------------------------------------------------
        # TAMPILKAN GRAFIK MULTIMODAL
        # ---------------------------------------------------------
        st.divider()
        st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>📈 Trend & Price Projection Visualization</div>", unsafe_allow_html=True)
        
        with st.spinner("Rendering historical and projection charts..."):
            list_prediksi = hitung_sekuens_prediksi(hari_target, model, scalers, df)
            fig = buat_grafik_multimodal_bilstm(df, list_prediksi, hari_target, bg_color)
            st.plotly_chart(fig, use_container_width=True)