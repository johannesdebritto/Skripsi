import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from spk_logic_xgb import evaluasi_spk_xgb
from gemini_helper import ambil_analisis_gemini

def tampilkan_hasil_xgb(hari_target, model, scaler, df):
    # ACCURATE DATE LOGIC: Extracted from the last row of the dataset (df)
    try:
        if 'Tanggal' in df.columns:
            tanggal_terakhir = pd.to_datetime(df['Tanggal'].iloc[-1])
        elif 'Date' in df.columns:
            tanggal_terakhir = pd.to_datetime(df['Date'].iloc[-1])
        else:
            tanggal_terakhir = pd.to_datetime(df.index[-1])
    except Exception:
        tanggal_terakhir = datetime.now()

    # Calculate actual target date & current date string
    tanggal_target = (tanggal_terakhir + timedelta(days=int(hari_target))).strftime("%d/%m/%Y")
    tanggal_sekarang_str = tanggal_terakhir.strftime("%d %B %Y").upper()

    with st.spinner(f"XGBoost Engine is calculating projections for {tanggal_target}..."):
        skor, keputusan, warna, indikator, h_sekarang, h_prediksi, selisih, margin = evaluasi_spk_xgb(
            hari_target, model, scaler, df
        )

        # 1. TITLE (Clean <div> trick to remove anchor link icon)
        st.markdown(f"<div style='text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 10px;'>📊 XGBOOST DECISION RESULTS ({tanggal_target})</div>", unsafe_allow_html=True)
        st.divider()

        # ---------------------------------------------------------
        # MAIN DECISION BANNER (Dynamic Background Color)
        # ---------------------------------------------------------
        if warna == "success":
            bg_color = "#28a745"  # Green for BUY
        elif warna == "error":
            bg_color = "#dc3545"  # Red for SELL
        else:
            bg_color = "#ffb300"  # Yellow/Dark Orange for HOLD
        
        # Uses large <p> tag to prevent Streamlit heading auto-copy popup
        st.markdown(
            f"<div style='text-align: center; border-radius: 10px; padding: 20px; background-color: {bg_color};'>"
            f"<p style='color: white; font-weight: 900; margin: 0px; font-size: 48px; text-transform: uppercase; font-family: sans-serif; letter-spacing: 1px;'>{keputusan}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Professional Disclaimer below decision box
        st.markdown(
            "<p style='text-align: center; color: #888888; font-size: 14px; margin-top: 10px;'>"
            "⚠️ <b>Note:</b> Transaction decisions are entirely at your own discretion. "
            "This AI serves as a probabilistic forecasting tool, not a guarantee or financial advice."
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # KEY METRICS BANNER (Elegant Unified Layout)
        # ---------------------------------------------------------
        # 1. Format numbers with Indonesian period separators
        str_h_sekarang = f"Rp {int(h_sekarang):,}".replace(',', '.')
        str_h_prediksi = f"Rp {int(h_prediksi):,}".replace(',', '.')
        str_margin = f"Rp {int(margin):,}".replace(',', '.')
        
        # 2. Configure colors and sign for Delta
        if selisih > 0:
            warna_selisih = "#28a745" # Green
            tanda = "+"
        elif selisih < 0:
            warna_selisih = "#dc3545" # Red
            tanda = ""
        else:
            warna_selisih = "#888888" # Grey
            tanda = ""
            
        str_selisih = f"{tanda}{int(selisih):,} IDR".replace(',', '.')

        # 3. Dynamic layout with current date from dataset
        html_metrik = f"""
        <div style="display: flex; justify-content: space-around; align-items: center; background: linear-gradient(135deg, rgba(240, 242, 246, 0.7), rgba(255, 255, 255, 0.4)); padding: 25px 15px; border-radius: 20px; margin-bottom: 20px; border: 1px solid rgba(0,0,0,0.05);">
            <div style="text-align: center; flex: 1; border-right: 1px solid rgba(0,0,0,0.1);">
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;">PRICE AS OF {tanggal_sekarang_str}</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 26px;">{str_h_sekarang}</h3>
            </div>
            <div style="text-align: center; flex: 1; border-right: 1px solid rgba(0,0,0,0.1);">
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;">XGBOOST AI PROJECTION</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 26px;">{str_h_prediksi}</h3>
                <p style="margin: 0; font-size: 14px; color: {warna_selisih}; font-weight: bold;">{str_selisih}</p>
            </div>
            <div style="text-align: center; flex: 1;">
                <p style="margin:0; font-size: 14px; color: #6b7280; font-weight: 600; letter-spacing: 0.5px;">MARGIN LIMIT</p>
                <h3 style="margin: 5px 0 0 0; color: #1f2937; font-size: 26px;">{str_margin}</h3>
            </div>
        </div>
        """
        st.markdown(html_metrik, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 2x2 GRID: CORE INDICATOR EVIDENCE
        # ---------------------------------------------------------
        st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>🔍 Core Decision Indicators</div>", unsafe_allow_html=True)
        st.caption("Multivariate data (last 30 days) underlying the expert system analysis.")

        col1, col2 = st.columns(2)

        # Top Left (AI XGBOOST)
        with col1:
            with st.container(border=True):
                st.markdown(f"**{indikator['ai']['icon']} XGBoost AI Prediction Trend**")
                st.metric("Estimated Profit/Loss", f"Rp {int(indikator['ai']['delta']):,}".replace(',', '.'), indikator['ai']['sentimen'])

        # Top Right (USD RATE)
        with col2:
            with st.container(border=True):
                st.markdown(f"**{indikator['kurs']['icon']} Macroeconomics: USD/IDR Rate**")
                st.metric("Actual Exchange Rate", f"Rp {int(indikator['kurs']['sekarang']):,}".replace(',', '.'), f"{int(indikator['kurs']['delta']):,} IDR".replace(',', '.'), delta_color="inverse")

        col3, col4 = st.columns(2)

        # Bottom Left (CRUDE OIL)
        with col3:
            with st.container(border=True):
                st.markdown(f"**{indikator['minyak']['icon']} Commodity: Crude Oil (IDR)**")
                st.metric("Actual Price per Barrel", f"Rp {int(indikator['minyak']['sekarang']):,}".replace(',', '.'), f"{int(indikator['minyak']['delta']):,} IDR".replace(',', '.'))

        # Bottom Right (FED RATE)
        with col4:
            with st.container(border=True):
                st.markdown(f"**{indikator['fed']['icon']} Policy: Fed Interest Rate**")
                st.metric("Actual Rate Value", f"{indikator['fed']['sekarang']:.2f}%", f"{indikator['fed']['delta']:.2f}%", delta_color="inverse")

        # ---------------------------------------------------------
        # LLM ANALYSIS SECTION
        # ---------------------------------------------------------
        st.divider()
        st.markdown("<div style='font-size: 22px; font-weight: bold; margin-top: 15px; margin-bottom: 15px;'>🤖 Comprehensive Expert System Analysis</div>", unsafe_allow_html=True)

        with st.spinner("Synthesizing decision rationale based on multivariate data..."):
            sentimen = "BULLISH (Strongly Positive)" if skor >= 2 else "BEARISH (Strongly Negative)" if skor <= -2 else "SIDEWAYS (Neutral Consolidation)"
            
            hasil_analisis = ambil_analisis_gemini(
                "XGBOOST", hari_target, h_sekarang, h_prediksi, selisih, keputusan, sentimen, indikator, skor
            )
            
            # Clean LLM display using st.info
            st.info(hasil_analisis, icon="💡")

        # ---------------------------------------------------------
        # DISPLAY MULTIMODAL CHART
        # ---------------------------------------------------------
        st.divider()
        st.markdown("<div style='font-size: 22px; font-weight: bold;'>📈 XGBoost Projection Visualization</div>", unsafe_allow_html=True)
        with st.spinner("Processing XGBoost multimodal chart..."):
            from ai_engine_xgb import hitung_sekuens_prediksi_xgb
            from grafik_xgb import buat_grafik_xgb_multimodal
            list_p = hitung_sekuens_prediksi_xgb(hari_target, model, scaler, df)
            fig = buat_grafik_xgb_multimodal(df, list_p, hari_target, bg_color)
            st.plotly_chart(fig, use_container_width=True)