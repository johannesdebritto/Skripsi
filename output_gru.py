import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Import modul internal SPK/AI Anda
from spk_logic_gru import evaluasi_spk_gru
from gemini_helper import ambil_analisis_gemini
from ai_engine_gru import hitung_sekuens_prediksi_gru
from grafik_gru import buat_grafik_gru_multimodal


def format_rupiah(nilai: float, tanda_plus: bool = False) -> str:
    """Helper untuk memformat nilai angka ke format Rupiah standar IDR."""
    prefix = "+" if tanda_plus and nilai > 0 else ""
    return f"{prefix}Rp {int(nilai):,}".replace(",", ".")


def tampilkan_hasil_gru(hari_target: int, model, scaler, df: pd.DataFrame):
    # -------------------------------------------------------------------------
    # 1. PARSING TANGGAL & LOGIKA TARGET
    # -------------------------------------------------------------------------
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
    tanggal_sekarang_str = tanggal_terakhir.strftime("%d %B %Y").upper()

    # -------------------------------------------------------------------------
    # 2. EVALUASI LOGIKA SPK GRU
    # -------------------------------------------------------------------------
    with st.spinner(f"GRU Engine sedang menghitung proyeksi untuk {tanggal_target}..."):
        skor, keputusan, warna, indikator, h_sekarang, h_prediksi, selisih, margin = evaluasi_spk_gru(
            hari_target, model, scaler, df
        )

        # Map warna keputusan
        if warna == "success":
            bg_color = "#28a745"  # Hijau (BUY)
        elif warna == "error":
            bg_color = "#dc3545"  # Merah (SELL)
        else:
            bg_color = "#ffb300"  # Kuning/Oranye (HOLD)

        # -------------------------------------------------------------------------
        # 3. HEADER & BANNER KEPUTUSAN UTAMA
        # -------------------------------------------------------------------------
        st.markdown(
            f"<h2 style='text-align: center; margin-bottom: 10px;'>"
            f"📊 HASIL KEPUTUSAN GRU ({tanggal_target})"
            f"</h2>", 
            unsafe_allow_html=True
        )
        st.divider()

        # Banner Keputusan
        st.markdown(
            f"""
            <div style='text-align: center; border-radius: 12px; padding: 20px; background-color: {bg_color}; margin-bottom: 10px;'>
                <p style='color: white; font-weight: 900; margin: 0px; font-size: 42px; text-transform: uppercase; font-family: sans-serif; letter-spacing: 1.5px;'>
                    {keputusan}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Disclaimer
        st.caption(
            "⚠️ **Catatan:** Keputusan transaksi sepenuhnya menjadi tanggung jawab pengguna. "
            "Sistem AI ini berfungsi sebagai alat bantu estimasi probabilitas, bukan jaminan atau saran keuangan mutlak."
        )
        st.write("")

        # -------------------------------------------------------------------------
        # 4. METRIC BANNER (HARGA & ESTIMASI)
        # -------------------------------------------------------------------------
        str_h_sekarang = format_rupiah(h_sekarang)
        str_h_prediksi = format_rupiah(h_prediksi)
        str_margin = format_rupiah(margin)

        if selisih > 0:
            warna_selisih = "#28a745"
            str_selisih = format_rupiah(selisih, tanda_plus=True)
        elif selisih < 0:
            warna_selisih = "#dc3545"
            str_selisih = format_rupiah(selisih)
        else:
            warna_selisih = "#888888"
            str_selisih = "Rp 0"

        html_metrik = f"""
        <div style="display: flex; justify-content: space-around; align-items: center; background: linear-gradient(135deg, rgba(240, 242, 246, 0.7), rgba(255, 255, 255, 0.4)); padding: 20px 15px; border-radius: 16px; margin-bottom: 20px; border: 1px solid rgba(0,0,0,0.08);">
            <div style="text-align: center; flex: 1; border-right: 1px solid rgba(0,0,0,0.1);">
                <p style="margin:0; font-size: 13px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;">HARGA TERAKHIR ({tanggal_sekarang_str})</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 24px;">{str_h_sekarang}</h3>
            </div>
            <div style="text-align: center; flex: 1; border-right: 1px solid rgba(0,0,0,0.1);">
                <p style="margin:0; font-size: 13px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;">PROYEKSI AI (GRU)</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 24px;">{str_h_prediksi}</h3>
                <p style="margin: 0; font-size: 13px; color: {warna_selisih}; font-weight: bold;">{str_selisih}</p>
            </div>
            <div style="text-align: center; flex: 1;">
                <p style="margin:0; font-size: 13px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;">BATAS MARGIN</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 24px;">{str_margin}</h3>
            </div>
        </div>
        """
        st.markdown(html_metrik, unsafe_allow_html=True)

        # -------------------------------------------------------------------------
        # 5. INDIKATOR UTAMA (GRID 2x2)
        # -------------------------------------------------------------------------
        st.subheader("🔍 Indikator Keputusan Utama")
        st.caption("Data multivariat (30 hari terakhir) yang menjadi basis analisis sistem pakar.")

        # Baris 1: Trend Prediksi AI & Nilai Tukar USD/IDR
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown(f"**{indikator['ai']['icon']} Tren Prediksi GRU AI**")
                st.metric(
                    label="Estimasi Profit/Loss",
                    value=format_rupiah(indikator['ai']['delta']),
                    delta=indikator['ai']['sentimen']
                )

        with col2:
            with st.container(border=True):
                st.markdown(f"**{indikator['kurs']['icon']} Makroekonomi: Kurs USD/IDR**")
                st.metric(
                    label="Nilai Tukar Saat Ini",
                    value=format_rupiah(indikator['kurs']['sekarang']),
                    delta=format_rupiah(indikator['kurs']['delta']),
                    delta_color="inverse"
                )

        # Baris 2: Minyak Mentah & Suku Bunga The Fed
        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.markdown(f"**{indikator['minyak']['icon']} Komoditas: Minyak Mentah (IDR)**")
                st.metric(
                    label="Harga Real per Barel",
                    value=format_rupiah(indikator['minyak']['sekarang']),
                    delta=format_rupiah(indikator['minyak']['delta'])
                )

        with col4:
            with st.container(border=True):
                st.markdown(f"**{indikator['fed']['icon']} Kebijakan: Suku Bunga Fed**")
                st.metric(
                    label="Suku Bunga Saat Ini",
                    value=f"{indikator['fed']['sekarang']:.2f}%",
                    delta=f"{indikator['fed']['delta']:.2f}%",
                    delta_color="inverse"
                )

        # -------------------------------------------------------------------------
        # 6. ANALISIS EXPERT SYSTEM (LLM / GEMINI)
        # -------------------------------------------------------------------------
        st.divider()
        st.subheader("🤖 Analisis Komprehensif Sistem Pakar")

        with st.spinner("Menyusun rincian analisis berdasarkan data multivariat..."):
            if skor >= 2:
                sentimen = "BULLISH (Sangat Positif)"
            elif skor <= -2:
                sentimen = "BEARISH (Sangat Negatif)"
            else:
                sentimen = "SIDEWAYS (Konsolidasi Netral)"

            hasil_analisis = ambil_analisis_gemini(
                "GRU", hari_target, h_sekarang, h_prediksi, selisih, keputusan, sentimen, indikator, skor
            )

            st.info(hasil_analisis, icon="💡")

    # -------------------------------------------------------------------------
    # 7. VISUALISASI GRAFIK MULTIMODAL
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("📈 Visualisasi Proyeksi & Basis Multimodal (GRU)")

    with st.spinner("Membuat grafik sekuensial GRU..."):
        list_prediksi_gru = hitung_sekuens_prediksi_gru(hari_target, model, scaler, df)
        fig_gru = buat_grafik_gru_multimodal(df, list_prediksi_gru, hari_target, bg_color)
        st.plotly_chart(fig_gru, use_container_width=True)