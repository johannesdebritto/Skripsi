import streamlit as st
import datetime

# Import all AI engines
from ai_engine_lstm import muat_aset_ai as muat_aset_lstm
from ai_engine_gru import muat_aset_ai_gru
from ai_engine_xgb import muat_aset_ai_xgb
# ---> Additional imports for BI-LSTM and BI-GRU
from ai_engine_bilstm import muat_aset_ai_bilstm
from ai_engine_bigru import muat_aset_ai_bigru

import output_lstm
import output_gru
import output_xgb
# ---> Additional imports for BI-LSTM and BI-GRU output renderers
import output_bilstm
import output_bigru

# Resource caching to ensure optimal performance
@st.cache_resource
def load_semua_model():
    m_lstm, s_lstm, df_lstm = muat_aset_lstm()
    m_gru, s_gru, _ = muat_aset_ai_gru()
    m_xgb, s_xgb, df_xgb = muat_aset_ai_xgb() 
    
    # ---> Additional loading for BI-LSTM and BI-GRU
    # (Ensure ai_engine_bilstm.py and ai_engine_bigru.py return matching data structures)
    m_bilstm, s_bilstm, _ = muat_aset_ai_bilstm()
    m_bigru, s_bigru, _ = muat_aset_ai_bigru()
    
    return m_lstm, s_lstm, df_lstm, m_gru, s_gru, m_xgb, s_xgb, df_xgb, m_bilstm, s_bilstm, m_bigru, s_bigru 

def render_ui_utama():
    # Unpack all loaded models including new architectures
    (model_lstm, scaler_lstm, df, 
     model_gru, scaler_gru, 
     model_xgb, scaler_xgb, df_xgb,
     model_bilstm, scaler_bilstm,
     model_bigru, scaler_bigru) = load_semua_model()
    
    tanggal_terakhir_data = df.index[-1].date()
    hari_ini = datetime.date.today()

    # --- CUSTOM INJECTED CSS ---
    st.markdown("""
        <style>
        /* Main container width configuration */
        .block-container {
            max-width: 1200px !important; 
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Main Title Header Styling */
        .judul-spk {
            text-align: center;
            margin-bottom: 8px;
            white-space: nowrap; 
            font-size: 2.8rem !important; 
            font-weight: 800 !important; 
            color: #111;
        }

        /* FORCE ENHANCED CONTAINER BORDER */
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            border: 4px solid #1f2937 !important; 
            border-radius: 12px !important;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Global Button Styling */
        div[data-testid="stButton"] button {
            transition: all 0.2s ease-in-out !important;
            border-radius: 8px !important;
            height: 48px !important;
            font-size: 17px !important; 
            font-weight: 700 !important; 
        }
        
        /* INACTIVE BUTTONS */
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
        
        /* ACTIVE & PRIMARY EXECUTION BUTTONS (Green Theme) */
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
        
        /* Clean Subheadings without anchor links */
        .sub-judul-bersih {
            color: #111; 
            font-size: 21px; 
            font-weight: bold; 
            margin-bottom: 12px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- MAIN HEADER (FULL WIDTH) ---
    st.markdown("<h1 class='judul-spk'>Gold Investment Decision Support System </h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555; font-size: 16px; font-weight: 500;'>Deep Learning & Machine Learning Architecture Powered by Multimodal Indicator Data</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #eee; margin-top: 15px; margin-bottom: 30px;'>", unsafe_allow_html=True)

    if 'tanggal_target' not in st.session_state:
        st.session_state.tanggal_target = hari_ini
    if 'model_aktif' not in st.session_state:
        st.session_state.model_aktif = "LSTM"

    # --- CENTERED COLUMN LAYOUT FOR FORM ---
    kosong_kiri, col_form, kosong_kanan = st.columns([1.5, 6, 1.5])

    with col_form:
        # ================= 1. DATE CONFIGURATION FORM =================
        st.markdown("<div class='sub-judul-bersih'>1. Projection Analysis Target Date</div>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("<p style='font-size: 16px; font-weight: 600; color: #333; margin-bottom: 12px;'>Quick Select (Projections from Today):</p>", unsafe_allow_html=True)
            
            t3 = hari_ini + datetime.timedelta(days=3)
            t7 = hari_ini + datetime.timedelta(days=7)
            t30 = hari_ini + datetime.timedelta(days=30)
            
            col_q1, col_q2, col_q3 = st.columns(3)
            
            if col_q1.button("Next 3 Days", type="primary" if st.session_state.tanggal_target == t3 else "secondary", use_container_width=True):
                st.session_state.tanggal_target = t3
                st.rerun()
            if col_q2.button("Next 1 Week", type="primary" if st.session_state.tanggal_target == t7 else "secondary", use_container_width=True):
                st.session_state.tanggal_target = t7
                st.rerun()
            if col_q3.button("Next 1 Month", type="primary" if st.session_state.tanggal_target == t30 else "secondary", use_container_width=True):
                st.session_state.tanggal_target = t30
                st.rerun()

            st.write("") # Vertical spacing
            
            _, col_kalender_tengah, _ = st.columns([1, 2, 1])
            with col_kalender_tengah:
                tanggal_kalender_user = st.date_input(
                    "📅 Or Select Specific Target Date:",
                    value=st.session_state.tanggal_target,
                    min_value=hari_ini,
                    max_value=hari_ini + datetime.timedelta(days=60)
                )
                st.session_state.tanggal_target = tanggal_kalender_user

        st.markdown("<br>", unsafe_allow_html=True)

        # ================= 2. AI MODEL CONFIGURATION FORM =================
        st.markdown("<div class='sub-judul-bersih'>2. Predictive Model Architecture</div>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("<p style='font-size: 16px; font-weight: 600; color: #333; margin-bottom: 12px;'>Select Artificial Intelligence (AI) Engine:</p>", unsafe_allow_html=True)
            
            # ---> ROW 1: LSTM, XGBOOST, GRU (3 Columns)
            mod_col1, mod_col2, mod_col3 = st.columns(3)
            btn_lstm = mod_col1.button(" LSTM Model", type="primary" if st.session_state.model_aktif == "LSTM" else "secondary", use_container_width=True)
            btn_xgb  = mod_col2.button(" XGBOOST Model", type="primary" if st.session_state.model_aktif == "XGBOOST" else "secondary", use_container_width=True)
            btn_gru  = mod_col3.button(" GRU Model", type="primary" if st.session_state.model_aktif == "GRU" else "secondary", use_container_width=True)

            st.write("") # Spacing between button rows

            # ---> ROW 2: BI-LSTM, BI-GRU (2 Columns for balanced width)
            mod_col4, mod_col5 = st.columns(2)
            btn_bilstm = mod_col4.button(" BI-LSTM Model", type="primary" if st.session_state.model_aktif == "BI-LSTM" else "secondary", use_container_width=True)
            btn_bigru  = mod_col5.button(" BI-GRU Model", type="primary" if st.session_state.model_aktif == "BI-GRU" else "secondary", use_container_width=True)

            # Rerun Logic Row 1
            if btn_lstm and st.session_state.model_aktif != "LSTM":
                st.session_state.model_aktif = "LSTM"
                st.rerun()
            if btn_xgb and st.session_state.model_aktif != "XGBOOST":
                st.session_state.model_aktif = "XGBOOST"
                st.rerun()
            if btn_gru and st.session_state.model_aktif != "GRU":
                st.session_state.model_aktif = "GRU"
                st.rerun()
                
            # Rerun Logic Row 2
            if btn_bilstm and st.session_state.model_aktif != "BI-LSTM":
                st.session_state.model_aktif = "BI-LSTM"
                st.rerun()
            if btn_bigru and st.session_state.model_aktif != "BI-GRU":
                st.session_state.model_aktif = "BI-GRU"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ================= 3. VALIDATION & EXECUTION =================
        nama_hari = st.session_state.tanggal_target.strftime("%A")
        is_akhir_pekan = nama_hari in ["Saturday", "Sunday"]
        hari_aktif_ai = (st.session_state.tanggal_target - tanggal_terakhir_data).days

        tombol_proses = False
        with st.container():
            if is_akhir_pekan:
                st.error(f"⚠️ **Market Closed:** Selected target date falls on a **{nama_hari}**. Global financial markets are closed on weekends.")
            elif hari_aktif_ai <= 0:
                st.warning("⚠️ **Invalid Date Selection:** Target date cannot be earlier than or equal to the latest historical data point.")
            else:
                _, col_btn, _ = st.columns([1, 2, 1])
                tombol_proses = col_btn.button("RUN AI DECISION ANALYSIS", type="primary", use_container_width=True)

        st.markdown("<hr style='border: 1px dashed #ddd; margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True)

        # ================= OUTPUT RENDERER =================
        if tombol_proses:
            with st.spinner(f"Processing projection using the {st.session_state.model_aktif} engine..."):
                if st.session_state.model_aktif == "LSTM":
                    output_lstm.tampilkan_hasil_lstm(hari_aktif_ai, model_lstm, scaler_lstm, df)
                    
                elif st.session_state.model_aktif == "GRU":
                    output_gru.tampilkan_hasil_gru(hari_aktif_ai, model_gru, scaler_gru, df)
                    
                elif st.session_state.model_aktif == "XGBOOST":
                    output_xgb.tampilkan_hasil_xgb(hari_aktif_ai, model_xgb, scaler_xgb, df_xgb)
                    
                # ---> Additional Output Renderers
                elif st.session_state.model_aktif == "BI-LSTM":
                    output_bilstm.tampilkan_hasil_bilstm(hari_aktif_ai, model_bilstm, scaler_bilstm, df)
                    
                elif st.session_state.model_aktif == "BI-GRU":
                    output_bigru.tampilkan_hasil_bigru(hari_aktif_ai, model_bigru, scaler_bigru, df)