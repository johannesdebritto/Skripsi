import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go

def buat_grafik_xgb_multimodal(df, list_prediksi, hari_target, warna_garis):
    df_hist = df.tail(60).copy()
    if 'Tanggal' not in df_hist.columns: 
        df_hist['Tanggal'] = df_hist.index
        
    last_date = pd.to_datetime(df_hist['Tanggal'].iloc[-1])
    dates_pred = [last_date + timedelta(days=i+1) for i in range(hari_target)]
    
    emas_hist = df_hist['Harga_Emas_IDR_per_Gram'].tolist()
    emas_full = emas_hist + list_prediksi
    tanggal_full = df_hist['Tanggal'].tolist() + dates_pred
    
    def jadikan_indeks(data_list):
        nilai_awal = data_list[0]
        return [(x / nilai_awal) * 100 for x in data_list]

    fig = go.Figure()
    len_h = len(emas_hist)
    
    # Gold (Historical)
    fig.add_trace(go.Scatter(
        x=tanggal_full[:len_h], 
        y=jadikan_indeks(emas_hist), 
        mode='lines', 
        name='Gold (Hist)', 
        line=dict(color='#1f2937', width=4), 
        customdata=emas_hist, 
        hovertemplate="IDR %{customdata:,.0f}"
    ))
    
    # XGBoost Prediction
    fig.add_trace(go.Scatter(
        x=tanggal_full[len_h-1:], 
        y=jadikan_indeks(emas_full)[len_h-1:], 
        mode='lines+markers', 
        name='XGB Prediction', 
        line=dict(color=warna_garis, width=4, dash='dot'), 
        customdata=emas_full[len_h-1:], 
        hovertemplate="IDR %{customdata:,.0f}"
    ))
    
    # Exchange Rate
    fig.add_trace(go.Scatter(
        x=tanggal_full[:len_h], 
        y=jadikan_indeks(df_hist['Kurs_USD_IDR'].tolist()), 
        mode='lines', 
        name='USD/IDR Rate', 
        line=dict(color='#28a745', dash='dash'), 
        customdata=df_hist['Kurs_USD_IDR'].tolist(), 
        hovertemplate="IDR %{customdata:,.0f}"
    ))
    
    # Oil Price
    fig.add_trace(go.Scatter(
        x=tanggal_full[:len_h], 
        y=jadikan_indeks(df_hist['Harga_Minyak_IDR_per_Barel'].tolist()), 
        mode='lines', 
        name='Oil Price (IDR)', 
        line=dict(color='#dc3545', dash='dash'), 
        customdata=df_hist['Harga_Minyak_IDR_per_Barel'].tolist(), 
        hovertemplate="IDR %{customdata:,.0f}"
    ))
    
    # Fed Rate
    fig.add_trace(go.Scatter(
        x=tanggal_full[:len_h], 
        y=jadikan_indeks(df_hist['Fed_Rate_Percent'].tolist()), 
        mode='lines', 
        name='Fed Rate', 
        line=dict(color='#6f42c1', dash='dash'), 
        customdata=df_hist['Fed_Rate_Percent'].tolist(), 
        hovertemplate="%{customdata:.2f}%"
    ))

    fig.update_layout(
        title={'text': "<b>Chart: Indicator Trends & XGBoost Projection</b>", 'y':0.98, 'x':0.5, 'xanchor':'center'},
        xaxis_title="Date",
        yaxis_title="Indexed Value (Base 100)",
        height=600, 
        hovermode="x unified", 
        template="plotly_white", 
        margin=dict(t=130), 
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center", borderwidth=1)
    )
    
    return fig