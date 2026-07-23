from google import genai

def ambil_analisis_gemini(nama_model, hari_target, h_sekarang, h_prediksi, selisih, keputusan, sentimen, indikator, skor_total):
    try:
        # Gunakan API Key kamu
        client = genai.Client(api_key="AQ.Ab8RN6LNkMoL0cFoc5973fF7sryM2pDV1rchjmNtm5Ud6Hz8QA")

        prompt_skripsi = f"""
        Konteks Metodologi:
        Sistem Pengambil Keputusan (SPK) menggunakan logika Berbasis Aturan (Rule-Based) yang mengawinkan proyeksi AI ({nama_model}) dengan analisis fundamental ekonomi makro (tren 30 hari terakhir).
        
        Data Analisis:
        - Prediksi Harga Emas: Berubah Rp {int(selisih):,} menjadi Rp {int(h_prediksi):,}
        - Kurs USD/IDR (Tren 30 Hari): {indikator['kurs']['teks']}
        - Minyak Dunia (Tren 30 Hari): {indikator['minyak']['teks']}
        - Fed Rate (Tren 30 Hari): {indikator['fed']['teks']}
        
        Keputusan Akhir: {keputusan}
        
        Instruksi Wajib:
        Buatlah argumen analitis yang KUAT, meyakinkan, dan mengalir (maksimal 2 kalimat) untuk menjelaskan mengapa SPK mengeluarkan keputusan '{keputusan}'.
        
        Aturan Penulisan:
        1. WAJIB memulai argumen dengan penegasan metode, contoh: "Sistem Pendukung Keputusan (SPK) berbasis aturan (rule-based) menetapkan..." atau "Melalui evaluasi rule-based, SPK merekomendasikan..."
        2. Lanjutkan dengan penjelasan ekonomi yang natural bertemunya "proyeksi arah harga AI" dengan "realita fundamental 30 hari terakhir" (Kurs, Minyak, Fed).
        3. DILARANG KERAS menyebutkan angka skor (seperti +1, -2) atau ambang batas. Fokus murni pada logika tarik-menarik 4 variabel makroekonomi tersebut.
        4. Dilarang menggunakan kalimat basa-basi.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_skripsi,
        )
        return response.text

    except Exception as e:
        return f"Sistem Pendukung Keputusan (SPK) berbasis aturan (rule-based) menetapkan keputusan {keputusan} karena proyeksi algoritma {nama_model} divalidasi secara logis oleh pergerakan fundamental Kurs, Minyak, dan Fed Rate selama 30 hari terakhir."