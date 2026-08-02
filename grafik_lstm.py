from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go


def buat_grafik_multimodal(df, list_prediksi, hari_target, warna_garis):
  # 1. Siapkan Data Historis (60 hari ke belakang)
  df_hist = df.tail(60).copy()
  if 'Tanggal' not in df_hist.columns:
    df_hist['Tanggal'] = df_hist.index

  last_date = pd.to_datetime(df_hist['Tanggal'].iloc[-1])
  dates_pred = [last_date + timedelta(days=i + 1) for i in range(hari_target)]

  # 2. Gabung data Emas Historis + Prediksi
  emas_hist = df_hist['Harga_Emas_IDR_per_Gram'].tolist()
  emas_full = emas_hist + list_prediksi
  tanggal_full = df_hist['Tanggal'].tolist() + dates_pred

  # --- TEKNIK BASE 100 INDEXING ---
  def jadikan_indeks(data_list):
    nilai_awal = data_list[0]
    return [(x / nilai_awal) * 100 for x in data_list]

  emas_idx = jadikan_indeks(emas_full)
  kurs_idx = jadikan_indeks(df_hist['Kurs_USD_IDR'].tolist())
  minyak_idx = jadikan_indeks(df_hist['Harga_Minyak_IDR_per_Barel'].tolist())
  fed_idx = jadikan_indeks(df_hist['Fed_Rate_Percent'].tolist())

  # 3. Buat Kanvas Grafik
  fig = go.Figure()

  # --- A. GARIS EMAS (Historis & Prediksi) ---
  len_hist = len(emas_hist)

  fig.add_trace(
      go.Scatter(
          x=tanggal_full[:len_hist],
          y=emas_idx[:len_hist],
          mode='lines',
          name='Emas Hist.',
          line=dict(color='#1f2937', width=3),
          customdata=emas_hist,
          hovertemplate='Emas: Rp %{customdata:,.0f}<extra></extra>',
      )
  )

  fig.add_trace(
      go.Scatter(
          x=tanggal_full[len_hist - 1 :],
          y=emas_idx[len_hist - 1 :],
          mode='lines+markers',
          name='Emas Proyeksi',
          line=dict(color=warna_garis, width=3, dash='dot'),
          marker=dict(size=5),
          customdata=emas_full[len_hist - 1 :],
          hovertemplate=(
              'Proyeksi Emas: Rp %{customdata:,.0f}<extra></extra>'
          ),
      )
  )

  # --- B. GARIS MAKROEKONOMI ---
  fig.add_trace(
      go.Scatter(
          x=tanggal_full[:len_hist],
          y=kurs_idx,
          mode='lines',
          name='USD/IDR',
          line=dict(color='#28a745', width=1.8, dash='dash'),
          customdata=df_hist['Kurs_USD_IDR'].tolist(),
          hovertemplate='Kurs: Rp %{customdata:,.0f}<extra></extra>',
      )
  )

  fig.add_trace(
      go.Scatter(
          x=tanggal_full[:len_hist],
          y=minyak_idx,
          mode='lines',
          name='Minyak',
          line=dict(color='#dc3545', width=1.8, dash='dash'),
          customdata=df_hist['Harga_Minyak_IDR_per_Barel'].tolist(),
          hovertemplate='Minyak: Rp %{customdata:,.0f}<extra></extra>',
      )
  )

  fig.add_trace(
      go.Scatter(
          x=tanggal_full[:len_hist],
          y=fed_idx,
          mode='lines',
          name='Fed Rate',
          line=dict(color='#6f42c1', width=1.8, dash='dash'),
          customdata=df_hist['Fed_Rate_Percent'].tolist(),
          hovertemplate='Fed Rate: %{customdata:.2f}%<extra></extra>',
      )
  )

  # 4. Layout Responsif (Optimasi Mobile & Desktop)
  fig.update_layout(
      title={
          'text': '<b>Grafik Multimodal: Evaluasi vs Proyeksi</b>',
          'y': 0.96,
          'x': 0.5,
          'xanchor': 'center',
          'yanchor': 'top',
          'font': dict(size=14),  # Font judul mengecil agar pas di layar HP
      },
      yaxis_title='Indeks (Base=100)',
      height=450,  # Disesuaikan dari 600 ke 450 agar pas di layar HP
      hovermode='x unified',
      template='plotly_white',
      margin=dict(
          l=10, r=10, t=90, b=10
      ),  # Padding lebih presisi untuk layar kecil
      # Legend ditaruh di bawah grafik agar tidak menutupi area plot di HP
      legend=dict(
          orientation='h',
          yanchor='bottom',
          y=-0.35,  # Pindah ke bawah grafik
          xanchor='center',
          x=0.5,
          bgcolor='rgba(248, 249, 250, 0.8)',
          bordercolor='rgba(0, 0, 0, 0.15)',
          borderwidth=1,
          font=dict(size=10),  # Ukuran font legenda diperkecil
      ),
  )

  # Mengatur sumbu X & Y agar ringkas di layar kecil
  fig.update_xaxes(
      tickfont=dict(size=10),
      showgrid=True,
      gridcolor='rgba(0,0,0,0.05)',
  )

  fig.update_yaxes(
      tickfont=dict(size=10),
      showgrid=True,
      gridcolor='rgba(0,0,0,0.05)',
  )

  return fig