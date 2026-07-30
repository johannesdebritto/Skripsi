from ai_engine_bilstm import hitung_prediksi_depan

def evaluasi_spk(hari_target, model, scaler_fitur, scaler_target, df):
    margin_rp = hari_target * 1500 
    
    harga_sekarang = df.iloc[-1]['Harga_Emas_IDR_per_Gram']
    
    # --- PERBAIKAN UTAMA: Hapus kurung tuple (scaler_fitur, scaler_target) ---
    # Cukup kirim scaler_fitur saja karena objeknya sama dengan scaler_target
    harga_prediksi = hitung_prediksi_depan(hari_target, model, scaler_fitur, df)
    
    selisih_prediksi = harga_prediksi - harga_sekarang
    
    data_terakhir = df.iloc[-1]
    
    # PERBAIKAN: Disamakan dengan TIME_STEPS = 30 (Melihat data sebulan lalu)
    data_30_hari_lalu = df.iloc[-31] 
    
    skor_total = 0
    indikator_detail = {} 

    # --- 1. AI BiLSTM ---
    if selisih_prediksi > margin_rp:
        skor_total += 2
        indikator_detail['ai'] = {"sentimen": "Tren Positif", "icon": "🤖", "delta": selisih_prediksi, "teks": f"Harga diproyeksikan **naik Rp {int(selisih_prediksi):,}**, menembus batas aman margin spread sebesar Rp {int(margin_rp):,}."}
    elif selisih_prediksi < -margin_rp:
        skor_total -= 2
        indikator_detail['ai'] = {"sentimen": "Tren Negatif", "icon": "🤖", "delta": selisih_prediksi, "teks": f"Harga diproyeksikan **turun tajam (Rp {int(abs(selisih_prediksi)):,})** melebihi batas toleransi risiko."}
    else:
        indikator_detail['ai'] = {"sentimen": "Tren Netral", "icon": "🤖", "delta": selisih_prediksi, "teks": f"Proyeksi kenaikan/penurunan masih di bawah batas margin spread (Rp {int(margin_rp):,}). Disarankan menahan aset."}

    # --- 2. KURS USD (Momentum 30 Hari) ---
    kurs_sekarang = data_terakhir['Kurs_USD_IDR']
    kurs_30 = data_30_hari_lalu['Kurs_USD_IDR']  
    selisih_kurs = kurs_sekarang - kurs_30       
    if selisih_kurs > 0:
        skor_total += 1
        indikator_detail['kurs'] = {"sekarang": kurs_sekarang, "delta": selisih_kurs, "icon": "💵", "teks": "Dalam 30 hari terakhir, Rupiah melemah terhadap Dolar AS. Kondisi ini secara historis memicu dorongan naik pada harga emas lokal."}
    else:
        skor_total -= 1
        indikator_detail['kurs'] = {"sekarang": kurs_sekarang, "delta": selisih_kurs, "icon": "💵", "teks": "Dalam 30 hari terakhir, Rupiah menguat terhadap Dolar AS. Kondisi ini menekan nilai konversi harga emas ke Rupiah."}

    # --- 3. MINYAK (Momentum 30 Hari) ---
    minyak_sekarang = data_terakhir['Harga_Minyak_IDR_per_Barel']
    minyak_30 = data_30_hari_lalu['Harga_Minyak_IDR_per_Barel']  
    selisih_minyak = minyak_sekarang - minyak_30                 
    if selisih_minyak > 0:
        skor_total += 1
        indikator_detail['minyak'] = {"sekarang": minyak_sekarang, "delta": selisih_minyak, "icon": "🛢️", "teks": "Harga energi dalam 30 hari melonjak, memicu risiko peningkatan inflasi. Daya tarik emas sebagai aset pelindung (safe haven) meningkat."}
    else:
        skor_total -= 1
        indikator_detail['minyak'] = {"sekarang": minyak_sekarang, "delta": selisih_minyak, "icon": "🛢️", "teks": "Tren harga energi sebulan terakhir melandai, meredakan tekanan inflasi pasar dan ikut meredupkan pamor emas."}

    # --- 4. FED RATE (Momentum 30 Hari + Filter Statis) ---
    fed_sekarang = data_terakhir['Fed_Rate_Percent']
    fed_30 = data_30_hari_lalu['Fed_Rate_Percent']  
    selisih_fed = fed_sekarang - fed_30             
    
    if selisih_fed < 0:
        skor_total += 1
        indikator_detail['fed'] = {"sekarang": fed_sekarang, "delta": selisih_fed, "icon": "🏦", "teks": "Suku bunga AS dipangkas bulan ini. Memegang aset emas menjadi jauh lebih menarik ketimbang Dolar AS."}
    elif selisih_fed > 0:
        skor_total -= 1
        indikator_detail['fed'] = {"sekarang": fed_sekarang, "delta": selisih_fed, "icon": "🏦", "teks": "Suku bunga AS naik. Investor cenderung memindahkan dananya dari emas kembali ke mata uang Dolar atau obligasi."}
    else:
        # Jika selisih 0 (Suku bunga tidak berubah bulan ini), kita nilai dari posisinya
        if fed_sekarang > 5.0:
            skor_total -= 1
            indikator_detail['fed'] = {"sekarang": fed_sekarang, "delta": selisih_fed, "icon": "🏦", "teks": "Suku bunga AS tertahan di level tinggi (>5%). Ini memberikan tekanan pada laju pertumbuhan harga emas."}
        else:
            skor_total += 1
            indikator_detail['fed'] = {"sekarang": fed_sekarang, "delta": selisih_fed, "icon": "🏦", "teks": "Suku bunga AS stabil di level rendah/menengah. Lingkungan makro yang sangat mendukung pergerakan emas."}

    # --- KEPUTUSAN ---
    if skor_total >= 2:
        keputusan, warna = "BUY", "success"
    elif skor_total <= -2:
        keputusan, warna = "SELL", "error"
    else:
        keputusan, warna = "HOLD", "warning"

    return skor_total, keputusan, warna, indikator_detail, harga_sekarang, harga_prediksi, selisih_prediksi, margin_rp