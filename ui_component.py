import streamlit as st
import datetime

# Import ketiga mesin kita
from ai_engine_lstm import muat_aset_ai as muat_aset_lstm
from ai_engine_gru import muat_aset_ai_gru
from ai_engine_xgb import muat_aset_ai_xgb
import output_lstm
import output_gru
import output_xgb

# Cache Loading agar tidak lemot
@st.cache_resource
def load_semua_model():
    m_lstm, s_lstm, df_lstm = muat_aset_lstm()
    m_gru, s_gru, _ = muat_aset_ai_gru()
    m_xgb, s_xgb, df_xgb = muat_aset_ai_xgb() 
    return m_lstm, s_lstm, df_lstm, m_gru, s_gru, m_xgb, s_xgb, df_xgb 

def render_ui_utama():
    model_lstm, scaler_lstm, df, model_gru, scaler_gru, model_xgb, scaler_xgb, df_xgb = load_semua_model()
    
    tanggal_terakhir_data = df.index[-1].date()
    hari_ini = datetime.date.today()

    # --- SUNTIKAN CUSTOM CSS ---
    st.markdown("""
        <style>
        /* Container utama dibiarkan lebar agar judul bisa leluasa */
        .block-container {
            max-width: 1200px !important; 
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Judul Utama Jumbo & 1 Baris */
        .judul-spk {
            text-align: center;
            margin-bottom: 8px;
            white-space: nowrap; 
            font-size: 2.8rem !important; 
            font-weight: 800 !important; 
            color: #111;
        }

        /* MEMPERTEBAL BORDER CONTAINER SECARA PAKSA */
        /* Menggunakan banyak selector agar CSS Streamlit benar-benar ketimpa */
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            border: 4px solid #1f2937 !important; /* 4px Tebel, Warna Abu-abu Sangat Gelap */
            border-radius: 12px !important;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Desain Global Tombol (Teks besar & bold) */
        div[data-testid="stButton"] button {
            transition: all 0.2s ease-in-out !important;
            border-radius: 8px !important;
            height: 48px !important;
            font-size: 17px !important; 
            font-weight: 700 !important; 
        }
        
        /* Tombol NON-AKTIF */
        div[data-testid="stButton"] button[kind="secondary"] {
            border: 1.5px solid #dcdcdc !important;
            background-color: #ffffff !important;
            color: #444 !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            border-color: #16a34a !important; 
            color: #16a34a !important;
            background-color: #f0fdf4 !important; 
            transform: translateY(-2px);
        }
        
        /* Tombol AKTIF & Eksekusi Utama (Tema Hijau) */
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #16a34a !important; 
            color: #ffffff !important;
            border: 1.5px solid #16a34a !important;
            box-shadow: 0 4px 10px rgba(22, 163, 74, 0.2) !important;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #15803d !important; 
            border-color: #15803d !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(22, 163, 74, 0.35) !important;
        }
        
        /* Sub-judul bersih tanpa link anchor */
        .sub-judul-bersih {
            color: #111; 
            font-size: 21px; 
            font-weight: bold; 
            margin-bottom: 12px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- HEADER UTAMA (FULL WIDTH) ---
    st.markdown("<h1 class='judul-spk'>🔮 Sistem Pendukung Keputusan (SPK) Investasi Emas</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555; font-size: 16px; font-weight: 500;'>Arsitektur Deep Learning & Machine Learning Berbasis Multimodal Data Indikator</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #eee; margin-top: 15px; margin-bottom: 30px;'>", unsafe_allow_html=True)

    if 'tanggal_target' not in st.session_state:
        st.session_state.tanggal_target = hari_ini
    if 'model_aktif' not in st.session_state:
        st.session_state.model_aktif = "LSTM"

    # --- MEMBUAT KOLOM PEMBATAS AGAR FORM TIDAK MELEBAR ---
    kosong_kiri, col_form, kosong_kanan = st.columns([1.5, 6, 1.5])

    with col_form:
        # ================= 1. FORM KONFIGURASI TANGGAL =================
        st.markdown("<div class='sub-judul-bersih'>1. Batas Tanggal Analisis Proyeksi</div>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("<p style='font-size: 16px; font-weight: 600; color: #333; margin-bottom: 12px;'>Pilihan Cepat (Proyeksi dari Hari Ini):</p>", unsafe_allow_html=True)
            
            t3 = hari_ini + datetime.timedelta(days=3)
            t7 = hari_ini + datetime.timedelta(days=7)
            t30 = hari_ini + datetime.timedelta(days=30)
            
            col_q1, col_q2, col_q3 = st.columns(3)
            
            if col_q1.button("3 Hari Kedepan", type="primary" if st.session_state.tanggal_target == t3 else "secondary", use_container_width=True):
                st.session_state.tanggal_target = t3
                st.rerun()
            if col_q2.button("1 Minggu Kedepan", type="primary" if st.session_state.tanggal_target == t7 else "secondary", use_container_width=True):
                st.session_state.tanggal_target = t7
                st.rerun()
            if col_q3.button("1 Bulan Kedepan", type="primary" if st.session_state.tanggal_target == t30 else "secondary", use_container_width=True):
                st.session_state.tanggal_target = t30
                st.rerun()

            st.write("") # Memberi sedikit jarak vertikal
            
            # Perbaikan tata letak kalender agar lebih rapi dan native
            _, col_kalender_tengah, _ = st.columns([1, 2, 1])
            with col_kalender_tengah:
                # Menggunakan label bawaan agar posisinya presisi dan tidak "melayang"
                tanggal_kalender_user = st.date_input(
                    "📅 Atau Tentukan Tanggal Manual:",
                    value=st.session_state.tanggal_target,
                    min_value=hari_ini,
                    max_value=hari_ini + datetime.timedelta(days=60)
                )
                st.session_state.tanggal_target = tanggal_kalender_user

        st.markdown("<br>", unsafe_allow_html=True)

        # ================= 2. FORM KONFIGURASI AI =================
        st.markdown("<div class='sub-judul-bersih'>2. Arsitektur Model Prediksi</div>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("<p style='font-size: 16px; font-weight: 600; color: #333; margin-bottom: 12px;'>Pilih Mesin Kecerdasan Buatan (AI):</p>", unsafe_allow_html=True)
            
            mod_col1, mod_col2, mod_col3 = st.columns(3)
            
            btn_lstm = mod_col1.button("🧠 Model LSTM", type="primary" if st.session_state.model_aktif == "LSTM" else "secondary", use_container_width=True)
            btn_xgb  = mod_col2.button("🚀 Model XGBOOST", type="primary" if st.session_state.model_aktif == "XGBOOST" else "secondary", use_container_width=True)
            btn_gru  = mod_col3.button("⚡ Model GRU", type="primary" if st.session_state.model_aktif == "GRU" else "secondary", use_container_width=True)

            if btn_lstm and st.session_state.model_aktif != "LSTM":
                st.session_state.model_aktif = "LSTM"
                st.rerun()
            if btn_xgb and st.session_state.model_aktif != "XGBOOST":
                st.session_state.model_aktif = "XGBOOST"
                st.rerun()
            if btn_gru and st.session_state.model_aktif != "GRU":
                st.session_state.model_aktif = "GRU"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ================= 3. VALIDASI & EKSEKUSI =================
        nama_hari = st.session_state.tanggal_target.strftime("%A")
        is_akhir_pekan = nama_hari in ["Saturday", "Sunday"]
        hari_aktif_ai = (st.session_state.tanggal_target - tanggal_terakhir_data).days

        tombol_proses = False
        with st.container():
            if is_akhir_pekan:
                hari_indo = "Sabtu" if nama_hari == "Saturday" else "Minggu"
                st.error(f"⚠️ **Pasar Tutup:** Tanggal target jatuh pada hari **{hari_indo}**. Bursa keuangan global libur pada akhir pekan.")
            elif hari_aktif_ai <= 0:
                st.warning("⚠️ **Evaluasi Invalid:** Tanggal target tidak boleh lebih mundur dari data historis terakhir di sistem.")
            else:
                _, col_btn, _ = st.columns([1, 2, 1])
                tombol_proses = col_btn.button("JALANKAN ANALISIS KEPUTUSAN AI", type="primary", use_container_width=True)

        st.markdown("<hr style='border: 1px dashed #ddd; margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True)

        # ================= OUTPUT RENDERER =================
        if tombol_proses:
            with st.spinner(f"Sedang memproses proyeksi menggunakan mesin {st.session_state.model_aktif}..."):
                if st.session_state.model_aktif == "LSTM":
                    output_lstm.tampilkan_hasil_lstm(hari_aktif_ai, model_lstm, scaler_lstm, df)
                elif st.session_state.model_aktif == "GRU":
                    output_gru.tampilkan_hasil_gru(hari_aktif_ai, model_gru, scaler_gru, df)
                elif st.session_state.model_aktif == "XGBOOST":
                    output_xgb.tampilkan_hasil_xgb(hari_aktif_ai, model_xgb, scaler_xgb, df_xgb)