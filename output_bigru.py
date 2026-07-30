import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from spk_logic_bigru import evaluasi_spk_bigru
from gemini_helper import ambil_analisis_gemini
from ai_engine_bigru import hitung_sekuens_prediksi_bigru
from grafik_bigru import buat_grafik_bigru_multimodal

def tampilkan_hasil_bigru(hari_target, model, scaler, df):
    # 1. LOGIKA TANGGAL
    try:
        if 'Tanggal' in df.columns:
            tanggal_terakhir = pd.to_datetime(df['Tanggal'].iloc[-1])
        elif 'Date' in df.columns:
            tanggal_terakhir = pd.to_datetime(df['Date'].iloc[-1])
        else:
            tanggal_terakhir = pd.to_datetime(df.index[-1])
    except Exception:
        tanggal_terakhir = datetime.now()

    tanggal_target = (tanggal_terakhir + timedelta(days=int(hari_target))).strftime("%d/%m/%Y")
    str_tanggal_terakhir = tanggal_terakhir.strftime("%d %b %Y").upper()

    with st.spinner(f"The BiGRU engine is calculating projections for {tanggal_target}..."):
        skor, keputusan, warna, indikator, h_sekarang, h_prediksi, selisih, margin = evaluasi_spk_bigru(
            hari_target, model, scaler, df
        )

        # 2. JUDUL (Desain Clean tanpa ikon link)
        st.markdown(f"<div style='text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 10px;'>📊 BiGRU DECISION RESULT ({tanggal_target})</div>", unsafe_allow_html=True)
        st.divider()

        # 3. BANNER KEPUTUSAN UTAMA (Warna dinamis sesuai hasil)
        if warna == "success":
            bg_color = "#28a745"  # Hijau untuk BUY
        elif warna == "error":
            bg_color = "#dc3545"  # Merah untuk SELL
        else:
            bg_color = "#ffb300"  # Kuning/Orange untuk HOLD
        
        st.markdown(
            f"<div style='text-align: center; border-radius: 10px; padding: 20px; background-color: {bg_color};'>"
            f"<p style='color: white; font-weight: 900; margin: 0px; font-size: 48px; text-transform: uppercase; font-family: sans-serif; letter-spacing: 1px;'>{keputusan}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Disclaimer Profesional
        st.markdown(
            "<p style='text-align: center; color: #888888; font-size: 14px; margin-top: 10px;'>"
            "⚠️ <b>Disclaimer:</b> Trading decisions are entirely in your hands. "
            "This AI is a probabilistic forecasting tool, not an absolute guarantee or financial advice."
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # 4. BANNER METRIK ELEGAN
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
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;"> PRICE {str_tanggal_terakhir}</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 26px;">{str_h_sekarang}</h3>
            </div>
            <div style="text-align: center; flex: 1; border-right: 1px solid rgba(0,0,0,0.1);">
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;"> BiGRU AI PROJECTION</p>
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

        # 5. GRID 2x2: BUKTI INDIKATOR
        st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>🔍 Decision Foundation Indicators</div>", unsafe_allow_html=True)
        st.caption("Multivariate data (last 30 days) underlying the expert system analysis.")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown(f"**{indikator['ai']['icon']} BiGRU AI Prediction Trend**")
                # Menyesuaikan sentimen ke bahasa Inggris secara dinamis di level UI
                sentimen_en = indikator['ai']['sentimen'].replace("Tren Positif", "Positive Trend").replace("Tren Negatif", "Negative Trend").replace("Tren Netral", "Neutral Trend")
                st.metric("Estimated Profit/Loss", f"IDR {int(indikator['ai']['delta']):,}".replace(',', '.'), sentimen_en)

        with col2:
            with st.container(border=True):
                st.markdown(f"**{indikator['kurs']['icon']} Macroeconomics: USD/IDR Rate**")
                st.metric("Actual Exchange Rate", f"IDR {int(indikator['kurs']['sekarang']):,}".replace(',', '.'), f"{int(indikator['kurs']['delta']):,} IDR".replace(',', '.'), delta_color="inverse")

        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.markdown(f"**{indikator['minyak']['icon']} Commodity: Global Oil (IDR)**")
                st.metric("Actual Price per Barrel", f"IDR {int(indikator['minyak']['sekarang']):,}".replace(',', '.'), f"{int(indikator['minyak']['delta']):,} IDR".replace(',', '.'))

        with col4:
            with st.container(border=True):
                st.markdown(f"**{indikator['fed']['icon']} Policy: US Fed Rate**")
                st.metric("Actual Rate Value", f"{indikator['fed']['sekarang']:.2f}%", f"{indikator['fed']['delta']:.2f}%", delta_color="inverse")

        # 6. BAGIAN ANALISIS LLM
        st.divider()
        st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 15px;'>🤖 Comprehensive Expert System Analysis</div>", unsafe_allow_html=True)

        with st.spinner("Generating decision reasoning based on multivariate data..."):
            sentimen_market = "BULLISH (Very Positive)" if skor >= 2 else "BEARISH (Very Negative)" if skor <= -2 else "SIDEWAYS (Neutral Consolidation)"
            
            # Memasukkan argumen 'BiGRU' agar prompt ke Gemini disesuaikan
            hasil_analisis = ambil_analisis_gemini(
                "BiGRU", hari_target, h_sekarang, h_prediksi, selisih, keputusan, sentimen_market, indikator, skor
            )
            
            # Tampilan LLM bersih menggunakan st.info
            st.info(hasil_analisis, icon="💡")
            
    # 7. BAGIAN GRAFIK 
    st.divider()
    st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>📈 BiGRU Projection & Multimodal Foundation Visualization</div>", unsafe_allow_html=True)

    with st.spinner("Rendering BiGRU sequential chart..."):
        list_prediksi_bigru = hitung_sekuens_prediksi_bigru(hari_target, model, scaler, df)
        
        fig_bigru = buat_grafik_bigru_multimodal(df, list_prediksi_bigru, hari_target, bg_color)
        
        st.plotly_chart(fig_bigru, use_container_width=True)