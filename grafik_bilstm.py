import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go

def buat_grafik_multimodal_bilstm(df, list_prediksi, hari_target, warna_garis):
    # 1. Siapkan Data Historis (60 hari ke belakang)
    df_hist = df.tail(60).copy()
    if 'Tanggal' not in df_hist.columns:
        df_hist['Tanggal'] = df_hist.index
        
    last_date = pd.to_datetime(df_hist['Tanggal'].iloc[-1])
    dates_pred = [last_date + timedelta(days=i+1) for i in range(hari_target)]
    
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
    
    # 3. Buat Kanvas (Satu Grafik Gabungan)
    fig = go.Figure()
    
    # --- A. GARIS EMAS (Historis & Prediksi) ---
    len_hist = len(emas_hist)
    
    fig.add_trace(go.Scatter(x=tanggal_full[:len_hist], y=emas_idx[:len_hist],
                             mode='lines', name='Gold (Historical)', 
                             line=dict(color='#1f2937', width=4),
                             customdata=emas_hist, 
                             hovertemplate="Gold: IDR %{customdata:,.0f}<extra></extra>"))
                             
    fig.add_trace(go.Scatter(x=tanggal_full[len_hist-1:], y=emas_idx[len_hist-1:],
                             mode='lines+markers', name='Gold (Bi-LSTM Proj.)', 
                             line=dict(color=warna_garis, width=4, dash='dot'),
                             customdata=emas_full[len_hist-1:],
                             hovertemplate="Bi-LSTM Proj: IDR %{customdata:,.0f}<extra></extra>"))

    # --- B. GARIS MAKROEKONOMI ---
    fig.add_trace(go.Scatter(x=tanggal_full[:len_hist], y=kurs_idx,
                             mode='lines', name='USD/IDR Rate', 
                             line=dict(color='#28a745', width=2, dash='dash'),
                             customdata=df_hist['Kurs_USD_IDR'].tolist(),
                             hovertemplate="USD/IDR: IDR %{customdata:,.0f}<extra></extra>"))
                             
    fig.add_trace(go.Scatter(x=tanggal_full[:len_hist], y=minyak_idx,
                             mode='lines', name='Crude Oil', 
                             line=dict(color='#dc3545', width=2, dash='dash'),
                             customdata=df_hist['Harga_Minyak_IDR_per_Barel'].tolist(),
                             hovertemplate="Crude Oil: IDR %{customdata:,.0f}<extra></extra>"))
                             
    fig.add_trace(go.Scatter(x=tanggal_full[:len_hist], y=fed_idx,
                             mode='lines', name='Fed Rate', 
                             line=dict(color='#6f42c1', width=2, dash='dash'),
                             customdata=df_hist['Fed_Rate_Percent'].tolist(),
                             hovertemplate="Fed Rate: %{customdata:.2f}%<extra></extra>"))

    # 4. Rapikan Layout & Bikin Kotak Legend Tersendiri
    fig.update_layout(
        title={
            'text': "<b>Multimodal Chart: Historical vs Bi-LSTM Projection</b>",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        yaxis_title="Trend Index (Base = 100)",
        height=600,
        hovermode="x unified",
        template="plotly_white",
        # Margin atas (t) dibesarkan jadi 130 biar legend punya lapak sendiri dan gak numpuk
        margin=dict(l=20, r=20, t=130, b=20), 
        legend=dict(
            orientation="h",         # Bikin legend menyamping (horizontal)
            yanchor="bottom",
            y=1.05,                  # Posisi Y ditarik ke atas keluar dari area garis grafik
            xanchor="center",
            x=0.5,                   # Ditaruh pas di tengah
            bgcolor="rgba(248, 249, 250, 0.9)", # Kasih warna background abu-abu sangat muda
            bordercolor="rgba(0, 0, 0, 0.2)",   # Kasih garis pinggir (border)
            borderwidth=1,           # Ketebalan border
            font=dict(size=12)       # Ukuran font legend dirapikan
        )
    )
    
    return fig