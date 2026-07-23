from ai_engine_lstm import hitung_prediksi_depan

def evaluasi_spk(hari_target, model, scaler, df):
    margin_rp = hari_target * 1500 
    
    harga_sekarang = df.iloc[-1]['Harga_Emas_IDR_per_Gram']
    harga_prediksi = hitung_prediksi_depan(hari_target, model, scaler, df)
    selisih_prediksi = harga_prediksi - harga_sekarang
    
    data_terakhir = df.iloc[-1]
    # Data t-30 historis
    data_30_hari_lalu = df.iloc[-31] 
    
    skor_total = 0
    indikator_detail = {} # Menggunakan kamus (dictionary) agar mudah ditata

    # --- 1. AI LSTM ---
    if selisih_prediksi > margin_rp:
        skor_total += 2
        indikator_detail['ai'] = {"sentimen": "Tren Positif", "icon": "🤖", "delta": selisih_prediksi, "teks": f"Harga diproyeksikan **naik Rp {int(selisih_prediksi):,}**, menembus batas aman margin spread sebesar Rp {int(margin_rp):,}."}
    elif selisih_prediksi < -margin_rp:
        skor_total -= 2
        indikator_detail['ai'] = {"sentimen": "Tren Negatif", "icon": "🤖", "delta": selisih_prediksi, "teks": f"Harga diproyeksikan **turun tajam (Rp {int(abs(selisih_prediksi)):,})** melebihi batas toleransi risiko."}
    else:
        indikator_detail['ai'] = {"sentimen": "Tren Netral", "icon": "🤖", "delta": selisih_prediksi, "teks": f"Proyeksi kenaikan/penurunan masih di bawah batas margin spread (Rp {int(margin_rp):,}). Disarankan menahan aset."}

    # --- 2. KURS USD ---
    kurs_sekarang = data_terakhir['Kurs_USD_IDR']
    kurs_30 = data_30_hari_lalu['Kurs_USD_IDR']  # Diperbaiki jadi _30
    selisih_kurs = kurs_sekarang - kurs_30       # Penyesuaian perhitungan
    if selisih_kurs > 0:
        skor_total += 1
        indikator_detail['kurs'] = {"sekarang": kurs_sekarang, "delta": selisih_kurs, "icon": "💵", "teks": "Rupiah melemah terhadap Dolar AS. Kondisi ini secara historis memicu dorongan naik pada harga emas lokal (konversi naik)."}
    else:
        skor_total -= 1
        indikator_detail['kurs'] = {"sekarang": kurs_sekarang, "delta": selisih_kurs, "icon": "💵", "teks": "Rupiah menguat terhadap Dolar AS. Kondisi ini menekan nilai konversi harga emas ke Rupiah."}

    # --- 3. MINYAK ---
    minyak_sekarang = data_terakhir['Harga_Minyak_IDR_per_Barel']
    minyak_30 = data_30_hari_lalu['Harga_Minyak_IDR_per_Barel']  # Diperbaiki jadi _30
    selisih_minyak = minyak_sekarang - minyak_30                 # Penyesuaian perhitungan
    if selisih_minyak > 0:
        skor_total += 1
        indikator_detail['minyak'] = {"sekarang": minyak_sekarang, "delta": selisih_minyak, "icon": "🛢️", "teks": "Harga energi melonjak, memicu risiko peningkatan inflasi. Daya tarik emas sebagai aset pelindung nilai (safe haven) meningkat."}
    else:
        skor_total -= 1
        indikator_detail['minyak'] = {"sekarang": minyak_sekarang, "delta": selisih_minyak, "icon": "🛢️", "teks": "Tren harga energi melandai, meredakan tekanan inflasi pasar dan ikut meredupkan pamor investasi komoditas."}

    # --- 4. FED RATE ---
    fed_sekarang = data_terakhir['Fed_Rate_Percent']
    fed_30 = data_30_hari_lalu['Fed_Rate_Percent']  # Diperbaiki jadi _30
    selisih_fed = fed_sekarang - fed_30             # Penyesuaian perhitungan
    if selisih_fed <= 0:
        skor_total += 1
        indikator_detail['fed'] = {"sekarang": fed_sekarang, "delta": selisih_fed, "icon": "🏦", "teks": "Suku bunga stabil/turun. Memegang aset emas menjadi jauh lebih menarik ketimbang menaruh uang di perbankan AS."}
    else:
        skor_total -= 1
        indikator_detail['fed'] = {"sekarang": fed_sekarang, "delta": selisih_fed, "icon": "🏦", "teks": "Suku bunga AS naik. Investor cenderung memindahkan dananya dari emas kembali ke mata uang Dolar atau obligasi."}

    # --- KEPUTUSAN ---
    if skor_total >= 2:
        keputusan, warna = "BUY", "success"
    elif skor_total <= -2:
        keputusan, warna = "SELL", "error"
    else:
        keputusan, warna = "HOLD", "warning"

    return skor_total, keputusan, warna, indikator_detail, harga_sekarang, harga_prediksi, selisih_prediksi, margin_rp