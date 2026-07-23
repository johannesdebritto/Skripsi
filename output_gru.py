import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from spk_logic_gru import evaluasi_spk_gru
from gemini_helper import ambil_analisis_gemini
from ai_engine_gru import hitung_sekuens_prediksi_gru
from grafik_gru import buat_grafik_gru_multimodal

def tampilkan_hasil_gru(hari_target, model, scaler, df):
    # 1. LOGIKA TANGGAL (Agar sama persis dengan desain LSTM)
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

    with st.spinner(f"Mesin GRU sedang mengkalkulasi proyeksi untuk tanggal {tanggal_target}..."):
        skor, keputusan, warna, indikator, h_sekarang, h_prediksi, selisih, margin = evaluasi_spk_gru(
            hari_target, model, scaler, df
        )

        # 2. JUDUL (Desain Clean tanpa ikon link)
        st.markdown(f"<div style='text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 10px;'>📊 HASIL KEPUTUSAN GRU ({tanggal_target})</div>", unsafe_allow_html=True)
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
            "⚠️ <b>Catatan:</b> Keputusan transaksi sepenuhnya berada di tangan Anda. "
            "AI ini merupakan alat bantu prediksi probabilistik, bukan jaminan mutlak atau nasihat keuangan."
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # 4. BANNER METRIK ELEGAN (Sama dengan desain LSTM)
        str_h_sekarang = f"Rp {int(h_sekarang):,}".replace(',', '.')
        str_h_prediksi = f"Rp {int(h_prediksi):,}".replace(',', '.')
        str_margin = f"Rp {int(margin):,}".replace(',', '.')
        
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
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;"> HARGA 24 JUNI 2026</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 26px;">{str_h_sekarang}</h3>
            </div>
            <div style="text-align: center; flex: 1; border-right: 1px solid rgba(0,0,0,0.1);">
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;"> PROYEKSI AI GRU</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 26px;">{str_h_prediksi}</h3>
                <p style="margin: 0; font-size: 14px; color: {warna_selisih}; font-weight: bold;">{str_selisih}</p>
            </div>
            <div style="text-align: center; flex: 1;">
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;"> BATAS MARGIN</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 26px;">{str_margin}</h3>
            </div>
        </div>
        """
        st.markdown(html_metrik, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # 5. GRID 2x2: BUKTI INDIKATOR
        st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>🔍 Bukti Indikator Fondasi Keputusan</div>", unsafe_allow_html=True)
        st.caption("Data multivariat (30 hari terakhir) yang mendasari analisis sistem pakar.")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown(f"**{indikator['ai']['icon']} Tren Prediksi AI GRU**")
                st.metric("Estimasi Profit/Loss", f"Rp {int(indikator['ai']['delta']):,}".replace(',', '.'), indikator['ai']['sentimen'])

        with col2:
            with st.container(border=True):
                st.markdown(f"**{indikator['kurs']['icon']} Makroekonomi: Kurs USD/IDR**")
                st.metric("Nilai Kurs Aktual", f"Rp {int(indikator['kurs']['sekarang']):,}".replace(',', '.'), f"{int(indikator['kurs']['delta']):,} IDR".replace(',', '.'), delta_color="inverse")

        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.markdown(f"**{indikator['minyak']['icon']} Komoditas: Minyak Dunia (IDR)**")
                st.metric("Nilai Aktual per Barel", f"Rp {int(indikator['minyak']['sekarang']):,}".replace(',', '.'), f"{int(indikator['minyak']['delta']):,} IDR".replace(',', '.'))

        with col4:
            with st.container(border=True):
                st.markdown(f"**{indikator['fed']['icon']} Kebijakan: Suku Bunga AS (Fed)**")
                st.metric("Nilai Rate Aktual", f"{indikator['fed']['sekarang']:.2f}%", f"{indikator['fed']['delta']:.2f}%", delta_color="inverse")

        # 6. BAGIAN ANALISIS LLM
        st.divider()
        st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 15px;'>🤖 Analisis Komprehensif Sistem Pakar</div>", unsafe_allow_html=True)

        with st.spinner("Menyusun alasan keputusan berdasarkan data multivariat..."):
            sentimen = "BULLISH (Sangat Positif)" if skor >= 2 else "BEARISH (Sangat Negatif)" if skor <= -2 else "SIDEWAYS (Konsolidasi Netral)"
            
            hasil_analisis = ambil_analisis_gemini(
                "GRU", hari_target, h_sekarang, h_prediksi, selisih, keputusan, sentimen, indikator, skor
            )
            
            # Tampilan LLM bersih menggunakan st.info sesuai desain LSTM
            st.info(hasil_analisis, icon="💡")
            # --- KODE GRAFIK HARUS DI SINI (SEJAJAR DENGAN ST.INFO) ---
    st.divider()
    st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>📈 Visualisasi Proyeksi & Fondasi Multimodal (GRU)</div>", unsafe_allow_html=True)

    with st.spinner("Merender grafik sekuensial GRU..."):
        # PENTING: Baris di bawah ini harus menjorok ke dalam (pakai Tab atau 4 Spasi)
        list_prediksi_gru = hitung_sekuens_prediksi_gru(hari_target, model, scaler, df)
        
        fig_gru = buat_grafik_gru_multimodal(df, list_prediksi_gru, hari_target, bg_color)
        
        st.plotly_chart(fig_gru, use_container_width=True)