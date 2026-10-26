import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from dotenv import load_dotenv
import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Konfigurasi Halaman Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Sosiologi Digital",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk estetika tampilan
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
    }
    .narrative-container {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 12px;
        padding: 24px;
        line-height: 1.7;
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv()

# Cek dari Streamlit Secrets dulu, kalau tidak ada baru cek dari .env
try:
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------
# Sidebar: Pengaturan Data & API
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Konfigurasi Data")
    uploaded_file = st.file_uploader("Unggah File Survei (CSV):", type=["csv"])
    
    st.markdown("---")
    st.subheader("🔑 Status Gemini API")
    if api_key:
        st.sidebar.success("✅ Gemini API Terhubung")
    else:
        st.sidebar.error("❌ Gemini API Key Belum Dikonfigurasi")
    
    model_choice = st.selectbox(
        "Pilih Model Gemini:",
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
        index=0
    )
    st.caption("Model aktif generasi terbaru dengan latensi rendah.")

# ---------------------------------------------------------
# Membaca Data
# ---------------------------------------------------------
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"File berhasil diunggah: `{uploaded_file.name}`")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file CSV: {e}")
elif os.path.exists("data_survei.csv"):
    df = pd.read_csv("data_survei.csv")
    st.sidebar.info("Menggunakan file default: `data_survei.csv`")
else:
    st.error("File `data_survei.csv` tidak ditemukan dan belum ada file yang diunggah.")

# ---------------------------------------------------------
# Header Dashboard
# ---------------------------------------------------------
st.markdown('<div class="main-title">🏛️ Dashboard Analisis Sosiologi Digital</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Platform simulasi riset kuantitatif empiris yang dikombinasikan dengan narasi sosiologis kritis berbasis AI '
    'untuk membedah fenomena efisiensi kerja vs alienasi teknologi.'
    '</div>', 
    unsafe_allow_html=True
)

if df is not None:
    # Identifikasi kolom numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset membutuhkan minimal dua kolom numerik untuk analisis korelasi.")
        st.stop()

    # Default selection
    default_x_idx = 1 if "durasi_belajar_ai" in numeric_cols else 0
    default_y_idx = 2 if "skor_efisiensi_tugas" in numeric_cols else min(1, len(numeric_cols) - 1)

    # Tab Navigasi
    tab_data, tab_analisis = st.tabs(["📋 Data Survei", "📈 Uji Korelasi & Analisis Sosiologis"])

    with tab_data:
        st.subheader("Tinjauan Dataset")
        st.dataframe(df, use_container_width=True)
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.write("**Dimensi Dataset:**", f"{df.shape[0]} baris × {df.shape[1]} kolom")
        with col_stat2:
            st.write("**Kolom Numerik Terdeteksi:**", ", ".join(numeric_cols))

    with tab_analisis:
        st.subheader("1. Pemilihan Variabel Analisis")
        c1, c2 = st.columns(2)
        with c1:
            var_x = st.selectbox(
                "Pilih Variabel X (Independen):",
                numeric_cols,
                index=default_x_idx
            )
        with c2:
            var_y = st.selectbox(
                "Pilih Variabel Y (Dependen):",
                numeric_cols,
                index=default_y_idx
            )

        if var_x == var_y:
            st.warning("Peringatan: Variabel X dan Variabel Y sama. Disarankan memilih dua variabel berbeda.")

        # Kalkulasi Korelasi Pearson
        valid_df = df[[var_x, var_y]].dropna()
        r_val, p_val = pearsonr(valid_df[var_x], valid_df[var_y])

        st.subheader("2. Hasil Uji Statistik Korelasi Pearson")
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric(
                label="Koefisien Korelasi (r)", 
                value=f"{r_val:.4f}",
                delta="Linear Positif" if r_val > 0 else "Linear Negatif"
            )
        with m2:
            st.metric(
                label="P-Value", 
                value=f"{p_val:.4e}",
                help=f"Nilai eksak: {p_val:.8f}"
            )
        with m3:
            is_sig = p_val < 0.05
            st.metric(
                label="Signifikansi (α = 0.05)", 
                value="Signifikan" if is_sig else "Tidak Signifikan",
                delta="Terbukti" if is_sig else "H0 Diterima"
            )
        with m4:
            if abs(r_val) >= 0.8:
                kekuatan = "Sangat Kuat"
            elif abs(r_val) >= 0.6:
                kekuatan = "Kuat"
            elif abs(r_val) >= 0.4:
                kekuatan = "Sedang"
            else:
                kekuatan = "Lemah"
            st.metric(label="Kekuatan Hubungan", value=kekuatan)

        # Visualisasi Scatter Plot & Regresi
        st.subheader("3. Visualisasi Scatter Plot & Garis Regresi")
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.regplot(
            x=valid_df[var_x], 
            y=valid_df[var_y], 
            ax=ax,
            color="#2563EB",
            scatter_kws={"s": 80, "alpha": 0.85, "edgecolor": "black"},
            line_kws={"color": "#DC2626", "linewidth": 2, "label": f"Garis Tren (r = {r_val:.2f})"}
        )
        ax.set_title(f"Hubungan antara '{var_x}' dan '{var_y}'", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel(f"{var_x}", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{var_y}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", loc="upper left")
        st.pyplot(fig)

        # -----------------------------------------------------
        # Gemini AI Section
        # -----------------------------------------------------
        st.markdown("---")
        st.subheader("4. Interpretasi Sosiologis Berbasis Gemini AI")
        st.caption("AI bertindak sebagai Ahli Sosiologi Digital yang membongkar implikasi efisiensi kerja vs alienasi teknologi.")

        col_btn, col_hint = st.columns([1, 3])
        with col_btn:
            generate_button = st.button("✨ Generasi Analisis Sosiologis", type="primary", use_container_width=True)

        # Key untuk menyimpan hasil analisis di session state
        analysis_state_key = f"analysis_{var_x}_{var_y}"

        if generate_button:
            if not api_key:
                st.error("API Key Gemini tidak ditemukan! Konfigurasikan GEMINI_API_KEY di Streamlit Secrets atau file .env.")
            else:
                with st.spinner("Menghubungi Gemini AI untuk menyusun analisis sosiologis kritis..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        system_instruction = (
                            "Kamu adalah Ahli Sosiologi Digital. Tugasmu adalah menginterpretasikan hasil "
                            "uji statistik korelasi menjadi narasi analisis sosiologis yang tajam, kritis, "
                            "dan mengaitkannya dengan fenomena efisiensi kerja atau alienasi teknologi."
                        )

                        prompt = f"""Berikut adalah hasil analisis statistik dari survei {len(valid_df)} responden mahasiswa:
- Variabel Independen (X): {var_x} (Nilai: {valid_df[var_x].tolist()})
- Variabel Dependen (Y): {var_y} (Nilai: {valid_df[var_y].tolist()})
- Koefisien Korelasi Pearson (r): {r_val:.4f}
- Nilai signifikansi (p-value): {p_val:.6f}

Tolong interpretasikan temuan statistik di atas menjadi narasi analisis sosiologis komprehensif dengan struktur:
1. Interpretasi Empiris Statistik & Signifikansi.
2. Analisis Kritis Sosiologi: Apakah fenomena ini menunjukkan efisiensi kerja emansipatoris, ataukah justru memperdalam alienasi teknologi, komodifikasi waktu, dan reduksi nalar kritis mahasiswa?
3. Refleksi Sosial & Implikasi Jangka Panjang bagi dunia pendidikan tinggi di era AI.
"""

                        response = client.models.generate_content(
                            model=model_choice,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=0.7,
                            )
                        )
                        if response and response.text:
                            st.session_state[analysis_state_key] = {
                                "model": model_choice,
                                "text": response.text
                            }
                            st.success("Analisis sosiologis berhasil digenerasi!")
                        else:
                            st.error("Tidak ada respon yang dikembalikan oleh model Gemini.")
                    except Exception as err:
                        st.error(f"Terjadi kesalahan saat memanggil Gemini API: {err}")

        # Tampilkan narasi tersimpan jika ada
        if analysis_state_key in st.session_state:
            result_data = st.session_state[analysis_state_key]
            st.info(f"Model Analis: **{result_data['model']}**")
            with st.container():
                st.markdown('<div class="narrative-container">', unsafe_allow_html=True)
                st.markdown(result_data["text"])
                st.markdown('</div>', unsafe_allow_html=True)
