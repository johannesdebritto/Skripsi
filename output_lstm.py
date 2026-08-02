import datetime
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from ai_engine_lstm import hitung_sekuens_prediksi
from gemini_helper import ambil_analisis_gemini
from grafik_lstm import buat_grafik_multimodal
from spk_logic_lstm import evaluasi_spk


def tampilkan_hasil_lstm(hari_target, model, scaler, df):
  # ACCURATE DATE LOGIC
  try:
    if "Tanggal" in df.columns:
      tanggal_terakhir = pd.to_datetime(df["Tanggal"].iloc[-1])
    elif "Date" in df.columns:
      tanggal_terakhir = pd.to_datetime(df["Date"].iloc[-1])
    else:
      tanggal_terakhir = pd.to_datetime(df.index[-1])
  except Exception:
    tanggal_terakhir = datetime.now()

  tanggal_target = (
      tanggal_terakhir + timedelta(days=int(hari_target))
  ).strftime("%d/%m/%Y")
  tanggal_sekarang_str = tanggal_terakhir.strftime("%d %B %Y").upper()

  with st.spinner(
      f"LSTM Engine is calculating projections for {tanggal_target}..."
  ):
    (
        skor,
        keputusan,
        warna,
        indikator,
        h_sekarang,
        h_prediksi,
        selisih,
        margin,
    ) = evaluasi_spk(hari_target, model, scaler, df)

    # 1. TITLE (Responsive Font Size)
    st.markdown(
        f"<div style='text-align: center; font-size: clamp(20px, 4vw, 28px);"
        " font-weight: bold; margin-bottom: 10px;'>📊 LSTM DECISION RESULTS"
        f" ({tanggal_target})</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ---------------------------------------------------------
    # MAIN DECISION BANNER (Responsive Padding & Font)
    # ---------------------------------------------------------
    if warna == "success":
      bg_color = "#28a745"  # Green
    elif warna == "error":
      bg_color = "#dc3545"  # Red
    else:
      bg_color = "#ffb300"  # Yellow/Orange

    st.markdown(
        f"<div style='text-align: center; border-radius: 12px; padding: 15px"
        f" 10px; background-color: {bg_color};'>"
        "<p style='color: white; font-weight: 900; margin: 0px; font-size:"
        " clamp(32px, 8vw, 48px); text-transform: uppercase; font-family:"
        f" sans-serif; letter-spacing: 1px;'>{keputusan}</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align: center; color: #888888; font-size: 12px;"
        " margin-top: 8px;'>⚠️ <b>Note:</b> Transaction decisions are"
        " entirely at your own discretion. This AI serves as a probabilistic"
        " forecasting tool.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # KEY METRICS BANNER (Mobile-Friendly Flex Layout)
    # ---------------------------------------------------------
    str_h_sekarang = f"Rp {int(h_sekarang):,}".replace(",", ".")
    str_h_prediksi = f"Rp {int(h_prediksi):,}".replace(",", ".")
    str_margin = f"Rp {int(margin):,}".replace(",", ".")

    if selisih > 0:
      warna_selisih = "#28a745"
      tanda = "+"
    elif selisih < 0:
      warna_selisih = "#dc3545"
      tanda = ""
    else:
      warna_selisih = "#888888"
      tanda = ""

    str_selisih = f"{tanda}{int(selisih):,} IDR".replace(",", ".")

    # HTML Metrik dengan CSS Responsive (flex-wrap + auto min-width)
    html_metrik = f"""
        <div style="
            display: flex; 
            flex-wrap: wrap; 
            justify-content: space-around; 
            align-items: center; 
            gap: 15px;
            background: linear-gradient(135deg, rgba(240, 242, 246, 0.8), rgba(255, 255, 255, 0.5)); 
            padding: 15px; 
            border-radius: 16px; 
            margin-bottom: 20px; 
            border: 1px solid rgba(0,0,0,0.08);
        ">
            <div style="text-align: center; flex: 1 1 250px; padding: 8px 0;">
                <p style="margin:0; font-size: 12px; color: #6b7280; font-weight: 700; letter-spacing: 0.5px;">PRICE AS OF {tanggal_sekarang_str}</p>
                <h3 style="margin: 4px 0 0 0; color: #1f2937; font-size: clamp(20px, 4vw, 24px);">{str_h_sekarang}</h3>
            </div>
            <div style="text-align: center; flex: 1 1 250px; padding: 8px 0;">
                <p style="margin:0; font-size: 12px; color: #6b7280; font-weight: 700; letter-spacing: 0.5px;">LSTM AI PROJECTION</p>
                <h3 style="margin: 4px 0 0 0; color: #1f2937; font-size: clamp(20px, 4vw, 24px);">{str_h_prediksi}</h3>
                <p style="margin: 2px 0 0 0; font-size: 13px; color: {warna_selisih}; font-weight: bold;">{str_selisih}</p>
            </div>
            <div style="text-align: center; flex: 1 1 250px; padding: 8px 0;">
                <p style="margin:0; font-size: 12px; color: #6b7280; font-weight: 700; letter-spacing: 0.5px;">MARGIN LIMIT</p>
                <h3 style="margin: 4px 0 0 0; color: #1f2937; font-size: clamp(20px, 4vw, 24px);">{str_margin}</h3>
            </div>
        </div>
        """
    st.markdown(html_metrik, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2x2 GRID: CORE INDICATOR EVIDENCE
    # ---------------------------------------------------------
    st.markdown(
        "<div style='font-size: 18px; font-weight: bold; margin-top: 10px;'"
        ">🔍 Core Decision Indicators</div>",
        unsafe_allow_html=True,
    )
    st.caption("Multivariate data underlying the analysis.")

    col1, col2 = st.columns(2)

    with col1:
      with st.container(border=True):
        st.markdown(
            f"**{indikator['ai']['icon']} LSTM AI Trend**",
            help="Prediksi pergerakan harga oleh model LSTM",
        )
        st.metric(
            "Estimated Profit/Loss",
            f"Rp {int(indikator['ai']['delta']):,}".replace(",", "."),
            indikator["ai"]["sentimen"],
        )

    with col2:
      with st.container(border=True):
        st.markdown(f"**{indikator['kurs']['icon']} USD/IDR Rate**")
        st.metric(
            "Exchange Rate",
            f"Rp {int(indikator['kurs']['sekarang']):,}".replace(",", "."),
            f"{int(indikator['kurs']['delta']):,} IDR".replace(",", "."),
            delta_color="inverse",
        )

    col3, col4 = st.columns(2)

    with col3:
      with st.container(border=True):
        st.markdown(f"**{indikator['minyak']['icon']} Crude Oil**")
        st.metric(
            "Price per Barrel",
            f"Rp {int(indikator['minyak']['sekarang']):,}".replace(",", "."),
            f"{int(indikator['minyak']['delta']):,} IDR".replace(",", "."),
        )

    with col4:
      with st.container(border=True):
        st.markdown(f"**{indikator['fed']['icon']} Fed Interest Rate**")
        st.metric(
            "Rate Value",
            f"{indikator['fed']['sekarang']:.2f}%",
            f"{indikator['fed']['delta']:.2f}%",
            delta_color="inverse",
        )

    # ---------------------------------------------------------
    # LLM ANALYSIS SECTION
    # ---------------------------------------------------------
    st.divider()
    st.markdown(
        "<div style='font-size: 18px; font-weight: bold;'>🤖 Expert System"
        " Rationale</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Synthesizing decision rationale..."):
      sentimen = (
          "BULLISH (Strongly Positive)"
          if skor >= 2
          else "BEARISH (Strongly Negative)"
          if skor <= -2
          else "SIDEWAYS (Neutral Consolidation)"
      )

      hasil_analisis = ambil_analisis_gemini(
          "LSTM",
          hari_target,
          h_sekarang,
          h_prediksi,
          selisih,
          keputusan,
          sentimen,
          indikator,
          skor,
      )

      st.info(hasil_analisis, icon="💡")

    # ---------------------------------------------------------
    # DISPLAY MULTIMODAL CHART
    # ---------------------------------------------------------
    st.divider()
    st.markdown(
        "<div style='font-size: 18px; font-weight: bold;'>📈 Trend"
        " Visualization</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Rendering charts..."):
      list_prediksi = hitung_sekuens_prediksi(hari_target, model, scaler, df)
      fig = buat_grafik_multimodal(df, list_prediksi, hari_target, bg_color)

      # use_container_width=True memastikan grafik Plotly menyesuaikan lebar layar HP
      st.plotly_chart(fig, use_container_width=True)