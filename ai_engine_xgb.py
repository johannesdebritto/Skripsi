import numpy as np
import pandas as pd
import xgboost as xgb

# ==============================================================================
# 1. DAFTAR FITUR (Harus 100% sama dengan urutan di Colab)
# ==============================================================================
FITUR_XGB = [
    'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag7', 'Return_Lag30',
    'Deviasi_Pct_MA_7Hari', 'Deviasi_Pct_MA_30Hari', 'Volatilitas_Return_7Hari', 'Volatilitas_Return_30Hari',
    'Kurs_Return_Lag1', 'Minyak_Return_Lag1', 'Fed_Rate_Diff_Lag1'
]

# ==============================================================================
# 2. FUNGSI MUAT MODEL & PABRIK FITUR
# ==============================================================================
def muat_aset_ai_xgb():
    # --- A. LOAD MODEL JSON DENGAN AMAN ---
    try:
        model = xgb.XGBRegressor()
        model.load_model('model_xgb_emas.json')
        print("-> [SUCCESS] Model XGBoost JSON aman berhasil dimuat!")
    except Exception as e:
        print(f"\n[WARNING XGBOOST] Gagal load model: {e}")
        model = None

    # --- B. BACA DATA CSV & FEATURE ENGINEERING (DISINKRONKAN DENGAN COLAB) ---
    try:
        df = pd.read_csv('Master_Data_Skripsi.csv')
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df = df.sort_values('Tanggal').reset_index(drop=True)
        
        # 1. Target Rekonstruksi
        df['Target_Harga'] = df['Harga_Emas_IDR_per_Gram']
        df['Harga_Lag1'] = df['Target_Harga'].shift(1)
        df['Target_Return'] = df['Target_Harga'].pct_change()

        # 2. Fitur Sejarah Return (Persentase)
        df['Return_Lag1'] = df['Target_Return'].shift(1)
        df['Return_Lag2'] = df['Target_Return'].shift(2)
        df['Return_Lag3'] = df['Target_Return'].shift(3)
        df['Return_Lag7'] = df['Target_Return'].shift(7)
        df['Return_Lag30'] = df['Target_Return'].shift(30)
        
        # 3. Fitur Deviasi Moving Average
        df['Harga_MA_7Hari'] = df['Harga_Lag1'].rolling(window=7).mean()
        df['Harga_MA_30Hari'] = df['Harga_Lag1'].rolling(window=30).mean()
        df['Deviasi_Pct_MA_7Hari'] = (df['Harga_Lag1'] - df['Harga_MA_7Hari']) / df['Harga_MA_7Hari']
        df['Deviasi_Pct_MA_30Hari'] = (df['Harga_Lag1'] - df['Harga_MA_30Hari']) / df['Harga_MA_30Hari']

        # 4. Fitur Volatilitas
        df['Volatilitas_Return_7Hari'] = df['Return_Lag1'].rolling(window=7).std()
        df['Volatilitas_Return_30Hari'] = df['Return_Lag1'].rolling(window=30).std()

        # 5. Fitur Eksternal
        df['Kurs_Return_Lag1'] = df['Kurs_USD_IDR'].pct_change().shift(1)
        df['Minyak_Return_Lag1'] = df['Harga_Minyak_IDR_per_Barel'].pct_change().shift(1)
        df['Fed_Rate_Diff_Lag1'] = df['Fed_Rate_Percent'].diff().shift(1)
        
        # Hapus baris NaN dan set index kembali
        df = df.dropna()
        df.set_index('Tanggal', inplace=True)

    except Exception as e:
        print(f"[ERROR CSV] Gagal memproses Master_Data_Skripsi.csv: {e}")
        df = None

    scaler = None 
    return model, scaler, df

# ==============================================================================
# 3. FUNGSI PREDIKSI SEKUENS (UNTUK GRAFIK)
# ==============================================================================
def hitung_sekuens_prediksi_xgb(hari_target, model, scaler, df):
    if model is None or df is None:
        print("[ERROR] Prediksi dibatalkan karena model atau dataframe kosong.")
        return [0] * hari_target

    harga_berjalan = df['Target_Harga'].iloc[-1]
    input_features = df[FITUR_XGB].iloc[-1].values.copy()
    
    # Ambil history 30 hari terakhir untuk perhitungan dinamis (Return dan Harga)
    history_return = list(df['Target_Return'].iloc[-30:].values)
    history_harga = list(df['Target_Harga'].iloc[-30:].values)
    
    list_harga_prediksi = []
    np.random.seed(7)
    
    # pandas default std menggunakan ddof=1, numpy default ddof=0, kita pakai 1 agar sinkron
    std_dev_return = np.std(history_return, ddof=1)

    for i in range(hari_target):
        # 1. Prediksi return hari esok
        prediksi_return_base = model.predict(input_features.reshape(1, -1))[0]
        
        # 2. Efek gravitasi dan riak harian
        efek_gravitasi = prediksi_return_base * (0.85 ** i)
        riak_harian = np.random.normal(0, std_dev_return * 0.2)
        prediksi_return_final = efek_gravitasi + riak_harian
        
        # 3. Update harga berjalan
        harga_berjalan = harga_berjalan * (1 + prediksi_return_final)
        list_harga_prediksi.append(float(harga_berjalan))
        
        # --- RECURSIVE UPDATE (Siapkan fitur untuk diprediksi di putaran besoknya) ---
        history_return.append(prediksi_return_final)
        history_harga.append(harga_berjalan)
        
        # Update fitur Lags Return (Indeks 0-4)
        input_features[0] = history_return[-1]  # Return_Lag1
        input_features[1] = history_return[-2]  # Return_Lag2
        input_features[2] = history_return[-3]  # Return_Lag3
        input_features[3] = history_return[-7]  # Return_Lag7
        input_features[4] = history_return[-30] # Return_Lag30
        
        # Update fitur Deviasi MA (Indeks 5-6)
        ma_7_baru = np.mean(history_harga[-7:])
        ma_30_baru = np.mean(history_harga[-30:])
        
        # history_harga[-1] adalah harga_berjalan (Harga_Lag1 untuk prediksi besok)
        input_features[5] = (history_harga[-1] - ma_7_baru) / ma_7_baru     # Deviasi_Pct_MA_7Hari
        input_features[6] = (history_harga[-1] - ma_30_baru) / ma_30_baru   # Deviasi_Pct_MA_30Hari
        
        # Update fitur Volatilitas (Indeks 7-8)
        input_features[7] = np.std(history_return[-7:], ddof=1)             # Volatilitas_Return_7Hari
        input_features[8] = np.std(history_return[-30:], ddof=1)            # Volatilitas_Return_30Hari
        
        # Untuk indeks 9, 10, 11 (Faktor Eksternal), kita biarkan nilainya menggunakan 
        # data hari terakhir karena kita tidak bisa memprediksi masa depan Kurs/Minyak/The Fed.
        
    return list_harga_prediksi

# ==============================================================================
# 4. FUNGSI PREDIKSI FINAL (UNTUK LOGIKA KEPUTUSAN SPK)
# ==============================================================================
def hitung_prediksi_depan_xgb(hari_target, model, scaler, df):
    list_harga = hitung_sekuens_prediksi_xgb(hari_target, model, scaler, df)
    return list_harga[-1]