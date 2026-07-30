import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 1. Parameter Utama
TIME_STEPS = 30
FITUR_MULTIVARIATE = [
    'Harga_Emas_IDR_per_Gram', 
    'Kurs_USD_IDR', 
    'Harga_Minyak_IDR_per_Barel', 
    'Fed_Rate_Percent',
    'Emas_Pct', 
    'Kurs_Pct', 
    'Minyak_Pct', 
    'Fed_Rate_Diff'
]

# --- FUNGSI HELPER BARU: Mencegah KeyError di seluruh fungsi ---
def _persiapkan_dataframe_8_fitur(df_input):
    df_temp = df_input.copy()
    
    # Hitung fitur turunan jika belum ada di DataFrame
    if 'Emas_Pct' not in df_temp.columns:
        df_temp['Emas_Pct'] = df_temp['Harga_Emas_IDR_per_Gram'].pct_change()
        df_temp['Kurs_Pct'] = df_temp['Kurs_USD_IDR'].pct_change()
        df_temp['Minyak_Pct'] = df_temp['Harga_Minyak_IDR_per_Barel'].pct_change()
        df_temp['Fed_Rate_Diff'] = df_temp['Fed_Rate_Percent'].diff()
        
    return df_temp[FITUR_MULTIVARIATE].dropna()

# 2. Fungsi Memuat Model dan Data BiGRU
def muat_aset_ai_bigru():
    model = load_model('model_bigru_emas.h5', compile=False)
    
    scaler = {
        'fitur': joblib.load('scaler_fitur_BiGRU.pkl'),
        'target': joblib.load('scaler_target_BiGRU.pkl')
    }
    
    df = pd.read_csv('Master_Data_Skripsi.csv', index_col='Tanggal', parse_dates=True)
    
    # Olah DataFrame menggunakan helper
    df = _persiapkan_dataframe_8_fitur(df)
    
    return model, scaler, df

def hitung_prediksi_depan_bigru(hari_target, model, scaler, df):
    # PENTING: Olah dulu df yang masuk agar dipastikan punya 8 fitur
    df_siap = _persiapkan_dataframe_8_fitur(df)
    
    # Ambil data 30 hari terakhir dari df_siap
    data_terakhir = df_siap[FITUR_MULTIVARIATE].tail(TIME_STEPS).values
    
    # Transformasi sequence
    current_sequence = scaler['fitur'].transform(data_terakhir)
    
    std_dev_scaled = np.std(current_sequence[:, 0])
    momentum_jangka_pendek = current_sequence[-1, 0] - current_sequence[-5, 0]
    
    np.random.seed(7)
    
    for i in range(hari_target):
        current_seq_reshaped = current_sequence.reshape(1, TIME_STEPS, len(FITUR_MULTIVARIATE))
        next_gold_base = model.predict(current_seq_reshaped, verbose=0)[0, 0]
        
        efek_gravitasi = momentum_jangka_pendek * (0.85 ** i)
        riak_harian = np.random.normal(0, std_dev_scaled * 0.2) 
        next_gold_scaled = next_gold_base + efek_gravitasi + riak_harian
        
        next_row = current_sequence[-1, :].copy()
        next_row[0] = next_gold_scaled 
        current_sequence = np.append(current_sequence[1:], [next_row], axis=0)
        
    next_gold_array = np.array([[next_gold_scaled]])
    harga_final = scaler['target'].inverse_transform(next_gold_array)[0, 0]
    
    if harga_final < 0:
        print("WARNING: Model memuntahkan nilai negatif ekstrem.")
        
    return float(harga_final)

def hitung_sekuens_prediksi_bigru(hari_target, model, scaler, df):
    # PENTING: Olah dulu df yang masuk agar dipastikan punya 8 fitur
    df_siap = _persiapkan_dataframe_8_fitur(df)
    
    data_terakhir = df_siap[FITUR_MULTIVARIATE].tail(TIME_STEPS).values
    current_sequence = scaler['fitur'].transform(data_terakhir)
    
    std_dev_scaled = np.std(current_sequence[:, 0])
    momentum_jangka_pendek = current_sequence[-1, 0] - current_sequence[-5, 0]
    
    list_harga_prediksi = []
    temp_sequence = current_sequence.copy()
    np.random.seed(7)

    for i in range(hari_target):
        current_seq_reshaped = temp_sequence.reshape(1, TIME_STEPS, len(FITUR_MULTIVARIATE))
        next_gold_base = model.predict(current_seq_reshaped, verbose=0)[0, 0]
        
        efek_gravitasi = momentum_jangka_pendek * (0.85 ** i)
        riak_harian = np.random.normal(0, std_dev_scaled * 0.2) 
        next_gold_scaled = next_gold_base + efek_gravitasi + riak_harian
        
        harga_real = scaler['target'].inverse_transform(np.array([[next_gold_scaled]]))[0, 0]
        list_harga_prediksi.append(float(harga_real))
        
        next_row = temp_sequence[-1, :].copy()
        next_row[0] = next_gold_scaled 
        temp_sequence = np.append(temp_sequence[1:], [next_row], axis=0)
        
    return list_harga_prediksi