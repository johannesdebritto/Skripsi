import datetime
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from ai_engine_bigru import hitung_sekuens_prediksi_bigru
from gemini_helper import ambil_analisis_gemini
from grafik_bigru import buat_grafik_bigru_multimodal
from spk_logic_bigru import evaluasi_spk_bigru


def tampilkan_hasil_bigru(hari_target, model, scaler, df):
  # ACCURATE DATE LOGIC
  col_tanggal = next((c for c in ["Tanggal", "Date"] if c in df.columns), None)
  tanggal_terakhir = (
      pd.to_datetime(df[col_tanggal].iloc[-1])
      if col_tanggal
      else datetime.now()
  )

  tanggal_target = (
      tanggal_terakhir + timedelta(days=int(hari_target))
  ).strftime("%d/%m/%Y")
  tanggal_sekarang_str = tanggal_terakhir.strftime("%d %B %Y").upper()

  with st.spinner(
      f"The BiGRU engine is calculating projections for {tanggal_target}..."
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
    ) = evaluasi_spk_bigru(hari_target, model, scaler, df)

    # 1. TITLE (Responsive Font Size)
    st.markdown(
        f"<div style='text-align: center; font-size: clamp(20px, 4vw, 28px);"
        " font-weight: bold; margin-bottom: 10px;'>📊 BiGRU DECISION RESULTS"
        f" ({tanggal_target})</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ---------------------------------------------------------
    # MAIN DECISION BANNER (Responsive Padding & Font)
    # ---------------------------------------------------------
    warna_map = {"success": "#28a745", "error": "#dc3545"}
    bg_color = warna_map.get(warna, "#ffb300")

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

    warna_selisih = (
        "#28a745" if selisih > 0 else "#dc3545" if selisih < 0 else "#888888"
    )
    tanda = "+" if selisih > 0 else ""
    str_selisih = f"{tanda}{int(selisih):,} IDR".replace(",", ".")

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
                <p style="margin:0; font-size: 12px; color: #6b7280; font-weight: 700; letter-spacing: 0.5px;">BIGRU AI PROJECTION</p>
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
            f"**{indikator['ai']['icon']} BiGRU AI Trend**",
            help="Prediksi pergerakan harga oleh model BiGRU",
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

    with st.spinner("Generating decision reasoning..."):
      sentimen_market = (
          "BULLISH (Very Positive)"
          if skor >= 2
          else "BEARISH (Very Negative)"
          if skor <= -2
          else "SIDEWAYS (Neutral Consolidation)"
      )

      hasil_analisis = ambil_analisis_gemini(
          "BiGRU",
          hari_target,
          h_sekarang,
          h_prediksi,
          selisih,
          keputusan,
          sentimen_market,
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

    with st.spinner("Rendering BiGRU sequential chart..."):
      list_prediksi_bigru = hitung_sekuens_prediksi_bigru(
          hari_target, model, scaler, df
      )
      fig_bigru = buat_grafik_bigru_multimodal(
          df, list_prediksi_bigru, hari_target, bg_color
      )

      st.plotly_chart(fig_bigru, use_container_width=True)