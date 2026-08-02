from datetime import datetime, timedelta
from ai_engine_bilstm import hitung_sekuens_prediksi
from gemini_helper import ambil_analisis_gemini
from grafik_bilstm import buat_grafik_multimodal_bilstm
from spk_logic_bilstm import evaluasi_spk
import pandas as pd
import streamlit as st


def tampilkan_hasil_bilstm(hari_target, model, scalers, df):
  scaler_fitur = scaler_target = scalers

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
  tanggal_sekarang_str = tanggal_terakhir.strftime('%d %B %Y').upper()

  with st.spinner(
      f'Bi-LSTM engine is calculating projections for {tanggal_target}...'
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
    ) = evaluasi_spk(hari_target, model, scaler_fitur, scaler_target, df)

    # 2. JUDUL & BANNER KEPUTUSAN UTAMA
    st.markdown(
        f"<h3 style='text-align: center; margin-bottom: 5px;'>📊 BI-LSTM"
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
        '⚠️ **Note:** Transaction decisions are entirely at your own risk. '
        'This AI is a probabilistic forecasting tool, not an absolute guarantee'
        ' or financial advice.'
    )

    # 3. METRIK UTAMA (Layout Native Responsif)
    st.markdown('---')
    m1, m2, m3 = st.columns(3)
    m1.metric(
        f'PRICE ON {tanggal_sekarang_str}', f'IDR {int(h_sekarang):,}'
    )
    m2.metric(
        'BI-LSTM PROJECTION',
        f'IDR {int(h_prediksi):,}',
        f'{int(selisih):,} IDR',
    )
    m3.metric('MARGIN LIMIT', f'IDR {int(margin):,}')

    # 4. GRID INDIKATOR (Efisien dengan Loop)
    st.markdown('### 🔍 Decision Foundation Indicators')
    st.caption(
        'Multivariate data (Last 30 days) underlying the expert system'
        ' analysis.'
    )

    item_indikator = [
        ('ai', 'Bi-LSTM Prediction Trend', 'profit'),
        ('kurs', 'Macroeconomy: USD/IDR Rate', 'inverse'),
        ('minyak', 'Commodity: Crude Oil (IDR)', 'normal'),
        ('fed', 'Policy: US Fed Rate', 'inverse'),
    ]

    cols = st.columns(2)
    for idx, (key, label, delta_mode) in enumerate(item_indikator):
      data = indikator[key]
      with cols[idx % 2]:
        with st.container(border=True):
          st.markdown(f"**{data['icon']} {label}**")

          if key == 'ai':
            ai_delta_val = int(data['delta'])
            ai_sentimen = str(data['sentimen']).strip()

            if ai_delta_val < 0:
              val_str = f'-IDR {abs(ai_delta_val):,}'
              if not ai_sentimen.startswith('-'):
                ai_sentimen = f'- {ai_sentimen}'
            else:
              val_str = f'IDR {ai_delta_val:,}'
              ai_sentimen = ai_sentimen.lstrip('- ')

            st.metric('Est. Profit/Loss', val_str, ai_sentimen)
          elif key == 'fed':
            st.metric(
                'Actual Fed Rate',
                f"{data['sekarang']:.2f}%",
                f"{data['delta']:.2f}%",
                delta_color=delta_mode,
            )
          else:
            st.metric(
                'Actual Value',
                f"IDR {int(data['sekarang']):,}",
                f"{int(data['delta']):,} IDR",
                delta_color=delta_mode,
            )

    # 5. BAGIAN ANALISIS LLM
    st.markdown('---')
    st.markdown('### 🤖 Comprehensive Expert System Analysis')

    with st.spinner(
        'Generating decision rationale based on multivariate data...'
    ):
      sentimen = (
          'BULLISH (Strongly Positive)'
          if skor >= 2
          else 'BEARISH (Strongly Negative)'
          if skor <= -2
          else 'SIDEWAYS (Neutral Consolidation)'
      )

      hasil_analisis = ambil_analisis_gemini(
          'Bi-LSTM',
          hari_target,
          h_sekarang,
          h_prediksi,
          selisih,
          keputusan,
          sentimen,
          indikator,
          skor,
      )
      st.info(hasil_analisis, icon='💡')

  # 6. BAGIAN GRAFIK MULTIMODAL
  st.markdown('---')
  st.markdown('### 📈 Trend & Price Projection Visualization')

  with st.spinner('Rendering historical and projection charts...'):
    list_prediksi = hitung_sekuens_prediksi(hari_target, model, scalers, df)
    fig = buat_grafik_multimodal_bilstm(
        df, list_prediksi, hari_target, bg_color
    )
    st.plotly_chart(fig, use_container_width=True)