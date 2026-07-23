import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# 1. Mengunci Parameter Utama
TIME_STEPS = 30
FITUR_MULTIVARIATE = ['Harga_Emas_IDR_per_Gram', 'Kurs_USD_IDR', 'Harga_Minyak_IDR_per_Barel', 'Fed_Rate_Percent']

# 2. Fungsi Memuat Model dan Data
def muat_aset_ai():
    model = load_model('model_lstm_emas.h5')
    with open('scaler_lstm_emas.pkl', 'rb') as f:
        scaler = pickle.load(f)
    df = pd.read_csv('Master_Data_Skripsi.csv', index_col='Tanggal', parse_dates=True)
    df = df[FITUR_MULTIVARIATE].dropna()
    return model, scaler, df

# 3. Fungsi Prediksi Jangka Panjang (Momentum Decay)
def hitung_prediksi_depan(hari_target, model, scaler, df):
    # Ambil data 30 hari terakhir dari dataset asli
    data_terakhir = df[FITUR_MULTIVARIATE].tail(TIME_STEPS).values
    current_sequence = scaler.transform(data_terakhir)
    
    # --- TAMBAHAN DEBUGGING ---
    print("\n[DEBUG LSTM] Data Asli (Baris Terakhir):", data_terakhir[-1])
    print("[DEBUG LSTM] Data Scaled (Baris Terakhir):", current_sequence[-1])
    # --------------------------
    
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
        
    # Kembalikan hasil ke Rupiah asli
    min_gold = scaler.data_min_[0]
    scale_gold = scaler.data_range_[0]
    harga_final = (next_gold_scaled * scale_gold) + min_gold
    
    return float(harga_final)

def hitung_sekuens_prediksi(hari_target, model, scaler, df):
    # Mengambil data 30 hari terakhir buat awalan
    data_terakhir = df[FITUR_MULTIVARIATE].tail(30).values
    current_sequence = scaler.transform(data_terakhir)
    
    std_dev_scaled = np.std(current_sequence[:, 0])
    momentum_jangka_pendek = current_sequence[-1, 0] - current_sequence[-5, 0]
    
    list_harga_prediksi = []
    temp_sequence = current_sequence.copy()

    for i in range(hari_target):
        current_seq_reshaped = temp_sequence.reshape(1, 30, len(FITUR_MULTIVARIATE))
        next_gold_base = model.predict(current_seq_reshaped, verbose=0)[0, 0]
        
        efek_gravitasi = momentum_jangka_pendek * (0.85 ** i)
        riak_harian = np.random.normal(0, std_dev_scaled * 0.2) 
        next_gold_scaled = next_gold_base + efek_gravitasi + riak_harian
        
        # Balikkan ke format Rupiah asli dan simpan ke list
        min_gold = scaler.data_min_[0]
        scale_gold = scaler.data_range_[0]
        harga_real = (next_gold_scaled * scale_gold) + min_gold
        list_harga_prediksi.append(float(harga_real))
        
        # Gulung waktu ke depan untuk prediksi besoknya
        next_row = temp_sequence[-1, :].copy()
        next_row[0] = next_gold_scaled 
        temp_sequence = np.append(temp_sequence[1:], [next_row], axis=0)
        
    return list_harga_prediksi