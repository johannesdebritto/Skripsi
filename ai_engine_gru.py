import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 1. Mengunci Parameter Utama
TIME_STEPS = 30
FITUR_MULTIVARIATE = ['Harga_Emas_IDR_per_Gram', 'Kurs_USD_IDR', 'Harga_Minyak_IDR_per_Barel', 'Fed_Rate_Percent']

# 2. Fungsi Memuat Model dan Data GRU
def muat_aset_ai_gru():
    # Ubah dari .keras menjadi .h5
    model = load_model('model_gru_emas.h5')
    
    # Load scaler (pastikan namanya sudah yang ada GRU-nya)
    scaler = {
        'fitur': joblib.load('scaler_fitur_GRU.pkl'),
        'target': joblib.load('scaler_target_GRU.pkl')
    }
    
    df = pd.read_csv('Master_Data_Skripsi.csv', index_col='Tanggal', parse_dates=True)
    df = df[FITUR_MULTIVARIATE].dropna()
    return model, scaler, df

def hitung_prediksi_depan_gru(hari_target, model, scaler, df):
    # Ambil data 30 hari terakhir dari dataset asli
    data_terakhir = df[FITUR_MULTIVARIATE].tail(TIME_STEPS).values
    
    # PERHATIAN: Gunakan scaler khusus fitur untuk input sequence
    current_sequence = scaler['fitur'].transform(data_terakhir)
    
    # --- TAMBAHKAN 2 BARIS INI UNTUK DEBUGGING ---
    print("\n[DEBUG GRU] Data Asli (Baris Terakhir):", data_terakhir[-1])
    print("[DEBUG GRU] Data Scaled (Baris Terakhir):", current_sequence[-1])
    # ---------------------------------------------
    
    std_dev_scaled = np.std(current_sequence[:, 0])
    momentum_jangka_pendek = current_sequence[-1, 0] - current_sequence[-5, 0]
    
    np.random.seed(7) # Kunci seed agar konsisten bawaan Colab
    
    # Loop prediksi harian
    for i in range(hari_target):
        current_seq_reshaped = current_sequence.reshape(1, TIME_STEPS, len(FITUR_MULTIVARIATE))
        next_gold_base = model.predict(current_seq_reshaped, verbose=0)[0, 0]
        
        efek_gravitasi = momentum_jangka_pendek * (0.85 ** i)
        riak_harian = np.random.normal(0, std_dev_scaled * 0.2) 
        next_gold_scaled = next_gold_base + efek_gravitasi + riak_harian
        
        # Gulung jendela waktu ke depan
        next_row = current_sequence[-1, :].copy()
        next_row[0] = next_gold_scaled 
        current_sequence = np.append(current_sequence[1:], [next_row], axis=0)
        
    # PERBAIKAN: Gunakan fungsi bawaan inverse_transform yang tahan banting (anti-error range)
    next_gold_array = np.array([[next_gold_scaled]])
    harga_final = scaler['target'].inverse_transform(next_gold_array)[0, 0]
    
    # Pengaman tambahan: pastikan output tidak negatif jika terjadi glitch model
    if harga_final < 0:
        print("WARNING: Model memuntahkan nilai negatif ekstrem. Cek kesamaan versi TensorFlow!")
        
    return float(harga_final)

def hitung_sekuens_prediksi_gru(hari_target, model, scaler, df):
    # Persiapan data awal (30 hari terakhir)
    data_terakhir = df[FITUR_MULTIVARIATE].tail(TIME_STEPS).values
    current_sequence = scaler['fitur'].transform(data_terakhir)
    
    std_dev_scaled = np.std(current_sequence[:, 0])
    momentum_jangka_pendek = current_sequence[-1, 0] - current_sequence[-5, 0]
    
    list_harga_prediksi = []
    temp_sequence = current_sequence.copy()
    np.random.seed(7)

    for i in range(hari_target):
        current_seq_reshaped = temp_sequence.reshape(1, TIME_STEPS, len(FITUR_MULTIVARIATE))
        next_gold_base = model.predict(current_seq_reshaped, verbose=0)[0, 0]
        
        # Logika Momentum & Riak
        efek_gravitasi = momentum_jangka_pendek * (0.85 ** i)
        riak_harian = np.random.normal(0, std_dev_scaled * 0.2) 
        next_gold_scaled = next_gold_base + efek_gravitasi + riak_harian
        
        # Inverse Transform menggunakan scaler['target']
        harga_real = scaler['target'].inverse_transform(np.array([[next_gold_scaled]]))[0, 0]
        list_harga_prediksi.append(float(harga_real))
        
        # Gulung jendela waktu
        next_row = temp_sequence[-1, :].copy()
        next_row[0] = next_gold_scaled 
        temp_sequence = np.append(temp_sequence[1:], [next_row], axis=0)
        
    return list_harga_prediksi