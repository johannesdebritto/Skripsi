import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go

def buat_grafik_gru_multimodal(df, list_prediksi, hari_target, warna_garis):
    # 1. Persiapan Data Historis
    df_hist = df.tail(60).copy()
    if 'Tanggal' not in df_hist.columns:
        df_hist['Tanggal'] = df_hist.index
        
    last_date = pd.to_datetime(df_hist['Tanggal'].iloc[-1])
    dates_pred = [last_date + timedelta(days=i+1) for i in range(hari_target)]
    
    emas_hist = df_hist['Harga_Emas_IDR_per_Gram'].tolist()
    emas_full = emas_hist + list_prediksi
    tanggal_full = df_hist['Tanggal'].tolist() + dates_pred
    
    # 2. Teknik Base 100 Indexing
    def jadikan_indeks(data_list):
        nilai_awal = data_list[0]
        return [(x / nilai_awal) * 100 for x in data_list]

    emas_idx = jadikan_indeks(emas_full)
    kurs_idx = jadikan_indeks(df_hist['Kurs_USD_IDR'].tolist())
    minyak_idx = jadikan_indeks(df_hist['Harga_Minyak_IDR_per_Barel'].tolist())
    fed_idx = jadikan_indeks(df_hist['Fed_Rate_Percent'].tolist())
    
    # 3. Plotting
    fig = go.Figure()
    len_hist = len(emas_hist)

    # Garis Emas
    fig.add_trace(go.Scatter(x=tanggal_full[:len_hist], y=emas_idx[:len_hist], 
                             mode='lines', name='Emas (Historis)', line=dict(color='#1f2937', width=4), 
                             customdata=emas_hist, hovertemplate="Emas: Rp %{customdata:,.0f}<extra></extra>"))
    
    fig.add_trace(go.Scatter(x=tanggal_full[len_hist-1:], y=emas_idx[len_hist-1:], 
                             mode='lines+markers', name='GRU Proyeksi', line=dict(color=warna_garis, width=4, dash='dot'), 
                             customdata=emas_full[len_hist-1:], hovertemplate="Prediksi GRU: Rp %{customdata:,.0f}<extra></extra>"))

    # Garis Indikator Multimodal
    fig.add_trace(go.Scatter(x=tanggal_full[:len_hist], y=kurs_idx, mode='lines', name='Kurs USD', line=dict(color='#28a745', width=2, dash='dash'), customdata=df_hist['Kurs_USD_IDR'].tolist(), hovertemplate="Kurs: Rp %{customdata:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tanggal_full[:len_hist], y=minyak_idx, mode='lines', name='Minyak Dunia', line=dict(color='#dc3545', width=2, dash='dash'), customdata=df_hist['Harga_Minyak_IDR_per_Barel'].tolist(), hovertemplate="Minyak: Rp %{customdata:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tanggal_full[:len_hist], y=fed_idx, mode='lines', name='Fed Rate', line=dict(color='#6f42c1', width=2, dash='dash'), customdata=df_hist['Fed_Rate_Percent'].tolist(), hovertemplate="Fed Rate: %{customdata:.2f}%<extra></extra>"))

    # 4. Layout & Legend Rapi
    fig.update_layout(
        title={'text': "<b>Analisis Multimodal: Perbandingan Tren Indikator & Proyeksi GRU</b>", 'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis_title="Indeks Pergerakan (Awal = 100)", height=600, hovermode="x unified", template="plotly_white",
        margin=dict(l=20, r=20, t=130, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, bgcolor="rgba(248, 249, 250, 0.9)", bordercolor="rgba(0, 0, 0, 0.1)", borderwidth=1)
    )
    return fig