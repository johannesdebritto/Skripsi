from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from ai_engine_bigru import hitung_sekuens_prediksi_bigru
from gemini_helper import ambil_analisis_gemini
from grafik_bigru import buat_grafik_bigru_multimodal
from spk_logic_bigru import evaluasi_spk_bigru

# ==============================================================================
# HELPER FUNCTIONS (Clean Code Architecture)
# ==============================================================================

def _get_target_and_current_date(df: pd.DataFrame, hari_target: int):
    """Mendapatkan tanggal acuan terakhir dan tanggal target yang valid."""
    col_tanggal = next((c for c in ["Tanggal", "Date"] if c in df.columns), None)
    tanggal_terakhir = (
        pd.to_datetime(df[col_tanggal].iloc[-1])
        if col_tanggal
        else (pd.to_datetime(df.index[-1]) if len(df.index) > 0 else datetime.now())
    )

    tanggal_target = (tanggal_terakhir + timedelta(days=int(hari_target))).strftime("%d/%m/%Y")
    tanggal_sekarang_str = tanggal_terakhir.strftime("%d %B %Y").upper()
    return tanggal_terakhir, tanggal_target, tanggal_sekarang_str


def _format_currency(value: float, prefix: str = "Rp ") -> str:
    """Format angka ke mata uang IDR standar Indonesia."""
    return f"{prefix}{int(value):,}".replace(",", ".")


def _render_decision_banner(keputusan: str, warna: str):
    """Komponen Banner Keputusan Utama BiGRU."""
    warna_map = {"success": "#28a745", "error": "#dc3545"}
    bg_color = warna_map.get(warna, "#ffb300")

    st.markdown(
        f"""
        <div style="text-align: center; border-radius: 12px; padding: 16px; background-color: {bg_color}; margin-bottom: 8px;">
            <p style="color: white; font-weight: 900; margin: 0; font-size: clamp(28px, 6vw, 44px); text-transform: uppercase; font-family: sans-serif; letter-spacing: 1px;">
                {keputusan}
            </p>
        </div>
        <p style="text-align: center; color: #888888; font-size: 12px; margin-bottom: 20px;">
            ⚠️ <b>Note:</b> Transaction decisions are entirely at your own discretion. This AI serves as a probabilistic forecasting tool.
        </p>
        """,
        unsafe_allow_html=True,
    )
    return bg_color


def _render_metrics_banner(tanggal_sekarang_str: str, h_sekarang: float, h_prediksi: float, selisih: float, margin: float):
    """Metrik Ringkasan Utama menggunakan CSS Grid agar Simetris di Desktop & Mobile."""
    str_h_sekarang = _format_currency(h_sekarang)
    str_h_prediksi = _format_currency(h_prediksi)
    str_margin = _format_currency(margin)

    # Logika Warna dan Tanda Selisih
    tanda = "+" if selisih > 0 else ""
    warna_selisih = "#28a745" if selisih > 0 else ("#dc3545" if selisih < 0 else "#888888")
    str_selisih = f"{tanda}{int(selisih):,} IDR".replace(",", ".")

    html_metrik = f"""
    <div style="
        display: grid; 
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
        gap: 16px; 
        background: linear-gradient(135deg, rgba(240, 242, 246, 0.9), rgba(255, 255, 255, 0.7)); 
        padding: 20px; 
        border-radius: 16px; 
        border: 1px solid rgba(0,0,0,0.08); 
        margin-bottom: 24px;
    ">
        <div style="text-align: center; padding: 4px;">
            <span style="font-size: 11px; color: #6b7280; font-weight: 700; letter-spacing: 0.5px; display: block;">PRICE AS OF {tanggal_sekarang_str}</span>
            <h3 style="margin: 6px 0 0 0; color: #1f2937; font-size: clamp(18px, 2.5vw, 22px); font-weight: 700;">{str_h_sekarang}</h3>
        </div>
        <div style="text-align: center; padding: 4px; border-left: 1px solid rgba(0,0,0,0.05); border-right: 1px solid rgba(0,0,0,0.05);">
            <span style="font-size: 11px; color: #6b7280; font-weight: 700; letter-spacing: 0.5px; display: block;">BIGRU AI PROJECTION</span>
            <h3 style="margin: 6px 0 0 0; color: #1f2937; font-size: clamp(18px, 2.5vw, 22px); font-weight: 700;">{str_h_prediksi}</h3>
            <span style="font-size: 12px; color: {warna_selisih}; font-weight: 700; margin-top: 2px; display: block;">{str_selisih}</span>
        </div>
        <div style="text-align: center; padding: 4px;">
            <span style="font-size: 11px; color: #6b7280; font-weight: 700; letter-spacing: 0.5px; display: block;">MARGIN LIMIT</span>
            <h3 style="margin: 6px 0 0 0; color: #1f2937; font-size: clamp(18px, 2.5vw, 22px); font-weight: 700;">{str_margin}</h3>
        </div>
    </div>
    """
    st.markdown(html_metrik, unsafe_allow_html=True)


def _render_indicator_cards(indikator: dict):
    """Menampilkan 4 Indikator Utama menggunakan Layout Grid 2x2 Streamlit."""
    st.markdown("<h4 style='margin-bottom: 0px;'>🔍 Core Decision Indicators</h4>", unsafe_allow_html=True)
    st.caption("Multivariate data underlying the analysis.")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown(f"**{indikator['ai']['icon']} BiGRU AI Trend**", help="Prediksi pergerakan harga oleh model BiGRU")
            st.metric("Estimated Profit/Loss", _format_currency(indikator['ai']['delta']), indikator["ai"]["sentimen"])

    with c2:
        with st.container(border=True):
            st.markdown(f"**{indikator['kurs']['icon']} USD/IDR Rate**")
            st.metric("Exchange Rate", _format_currency(indikator['kurs']['sekarang']), _format_currency(indikator['kurs']['delta'], prefix=""), delta_color="inverse")

    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.markdown(f"**{indikator['minyak']['icon']} Crude Oil**")
            st.metric("Price per Barrel", _format_currency(indikator['minyak']['sekarang']), _format_currency(indikator['minyak']['delta'], prefix=""))

    with c4:
        with st.container(border=True):
            st.markdown(f"**{indikator['fed']['icon']} Fed Interest Rate**")
            st.metric("Rate Value", f"{indikator['fed']['sekarang']:.2f}%", f"{indikator['fed']['delta']:.2f}%", delta_color="inverse")


# ==============================================================================
# MAIN DISPLAY FUNCTION
# ==============================================================================

def tampilkan_hasil_bigru(hari_target, model, scaler, df):
    _, tanggal_target, tanggal_sekarang_str = _get_target_and_current_date(df, hari_target)

    with st.spinner(f"The BiGRU engine is calculating projections for {tanggal_target}..."):
        skor, keputusan, warna, indikator, h_sekarang, h_prediksi, selisih, margin = evaluasi_spk_bigru(
            hari_target, model, scaler, df
        )

        # Header Title
        st.markdown(f"<h3 style='text-align: center;'>📊 BiGRU DECISION RESULTS ({tanggal_target})</h3>", unsafe_allow_html=True)
        st.divider()

        # Decision Banner & Key Metrics
        bg_color = _render_decision_banner(keputusan, warna)
        _render_metrics_banner(tanggal_sekarang_str, h_sekarang, h_prediksi, selisih, margin)

        # 2x2 Indicators Grid
        _render_indicator_cards(indikator)

        # LLM Analysis
        st.divider()
        st.markdown("#### 🤖 Expert System Rationale")

        with st.spinner("Generating decision reasoning..."):
            sentimen_market = (
                "BULLISH (Very Positive)" if skor >= 2
                else "BEARISH (Very Negative)" if skor <= -2
                else "SIDEWAYS (Neutral Consolidation)"
            )
            hasil_analisis = ambil_analisis_gemini(
                "BiGRU", hari_target, h_sekarang, h_prediksi, selisih, keputusan, sentimen_market, indikator, skor
            )
            st.info(hasil_analisis, icon="💡")

        # Visualization
        st.divider()
        st.markdown("#### 📈 Trend Visualization")

        with st.spinner("Rendering BiGRU sequential chart..."):
            list_prediksi_bigru = hitung_sekuens_prediksi_bigru(hari_target, model, scaler, df)
            fig_bigru = buat_grafik_bigru_multimodal(df, list_prediksi_bigru, hari_target, bg_color)
            st.plotly_chart(fig_bigru, use_container_width=True)