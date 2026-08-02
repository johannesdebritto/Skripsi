from datetime import datetime, timedelta
from ai_engine_bigru import hitung_sekuens_prediksi_bigru
from gemini_helper import ambil_analisis_gemini
from grafik_bigru import buat_grafik_bigru_multimodal
from spk_logic_bigru import evaluasi_spk_bigru
import pandas as pd
import streamlit as st


def tampilkan_hasil_bigru(hari_target, model, scaler, df):
  # 1. LOGIKA TANGGAL
  col_tanggal = next((c for c in ['Tanggal', 'Date'] if c in df.columns), None)
  tanggal_terakhir = (
      pd.to_datetime(df[col_tanggal].iloc[-1])
      if col_tanggal
      else datetime.now()
  )

  tanggal_target = (tanggal_terakhir + timedelta(days=int(hari_target))).strftime(
      '%d/%m/%Y'
  )
  str_tanggal_terakhir = tanggal_terakhir.strftime('%d %b %Y').upper()

  with st.spinner(
      f'The BiGRU engine is calculating projections for {tanggal_target}...'
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

    # 2. JUDUL & BANNER KEPUTUSAN
    st.markdown(
        f"<h3 style='text-align: center; margin-bottom: 5px;'>📊 BiGRU"
        f' DECISION RESULT ({tanggal_target})</h3>',
        unsafe_allow_html=True,
    )

    warna_map = {'success': '#28a745', 'error': '#dc3545'}
    bg_color = warna_map.get(warna, '#ffb300')

    st.markdown(
        f"<div style='text-align: center; border-radius: 10px; padding: 12px"
        f" 20px; background-color: {bg_color}; margin-bottom: 10px;'>"
        f"<h1 style='color: white; margin: 0; font-size: 36px;"
        f" letter-spacing: 1px;'>{keputusan}</h1>"
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        '⚠️ **Disclaimer:** Trading decisions are entirely in your hands.'
        ' Probabilistic forecasting tool.'
    )

    # 3. METRIK UTAMA (Menggunakan st.columns agar responsif di HP)
    st.markdown('---')
    m1, m2, m3 = st.columns(3)
    m1.metric(
        f'PRICE ({str_tanggal_terakhir})', f'Rp {int(h_sekarang):,}'
    )
    m2.metric(
        'BiGRU PROJECTION',
        f'Rp {int(h_prediksi):,}',
        f'{int(selisih):,}',
    )
    m3.metric('MARGIN LIMIT', f'Rp {int(margin):,}')

    # 4. GRID INDIKATOR (Efisien dengan Loop)
    st.markdown('### 🔍 Decision Foundation Indicators')

    item_indikator = [
        ('ai', 'BiGRU Prediction Trend', 'profit'),
        ('kurs', 'Macro: USD/IDR Rate', 'inverse'),
        ('minyak', 'Commodity: Global Oil', 'normal'),
        ('fed', 'Policy: US Fed Rate', 'inverse'),
    ]

    cols = st.columns(2)
    for idx, (key, label, delta_mode) in enumerate(item_indikator):
        data = indikator[key]
        with cols[idx % 2]:
          with st.container(border=True):
            st.markdown(f"**{data['icon']} {label}**")

            if key == 'ai':
              sentimen_en = (
                  data['sentimen']
                  .replace('Tren Positif', 'Positive Trend')
                  .replace('Tren Negatif', 'Negative Trend')
                  .replace('Tren Netral', 'Neutral Trend')
              )
              st.metric(
                  'Estimated P/L', f"Rp {int(data['delta']):,}", sentimen_en
              )
            elif key == 'fed':
              st.metric(
                  'Actual Rate',
                  f"{data['sekarang']:.2f}%",
                  f"{data['delta']:.2f}%",
                  delta_color=delta_mode,
              )
            else:
              st.metric(
                  'Actual Price',
                  f"Rp {int(data['sekarang']):,}",
                  f"{int(data['delta']):,}",
                  delta_color=delta_mode,
              )

    # 5. BAGIAN ANALISIS LLM
    st.markdown('---')
    st.markdown('### 🤖 Comprehensive Expert System Analysis')

    with st.spinner('Generating decision reasoning...'):
      sentimen_market = (
          'BULLISH (Very Positive)'
          if skor >= 2
          else 'BEARISH (Very Negative)'
          if skor <= -2
          else 'SIDEWAYS (Neutral Consolidation)'
      )

      hasil_analisis = ambil_analisis_gemini(
          'BiGRU',
          hari_target,
          h_sekarang,
          h_prediksi,
          selisih,
          keputusan,
          sentimen_market,
          indikator,
          skor,
      )
      st.info(hasil_analisis, icon='💡')

  # 6. BAGIAN GRAFIK
  st.markdown('---')
  st.markdown('### 📈 BiGRU Projection & Multimodal Chart')

  with st.spinner('Rendering BiGRU sequential chart...'):
    list_prediksi_bigru = hitung_sekuens_prediksi_bigru(
        hari_target, model, scaler, df
    )
    fig_bigru = buat_grafik_bigru_multimodal(
        df, list_prediksi_bigru, hari_target, bg_color
    )
    st.plotly_chart(fig_bigru, use_container_width=True)