import os
import sys
import pandas as pd
from scipy.stats import pearsonr
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Memastikan output terminal mendukung encoding UTF-8 di Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    print("=" * 70)
    print("      SIMULASI RISET SOSIOLOGI: AI & EFISIENSI TUGAS MAHASISWA")
    print("=" * 70)

    # 1. Load API Key dari .env
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[Error] Variabel lingkungan GEMINI_API_KEY tidak ditemukan di file .env")
        return

    # 2. Baca file data_survei.csv menggunakan pandas
    csv_filename = "data_survei.csv"
    if not os.path.exists(csv_filename):
        print(f"[Error] File '{csv_filename}' tidak ditemukan.")
        return

    df = pd.read_csv(csv_filename)
    print("\n[1] DATA SURVEI RESPONDEN:")
    print("-" * 70)
    print(df.to_string(index=False))
    print("-" * 70)

    # 3. Hitung Korelasi Pearson (r dan p-value) menggunakan scipy.stats
    r_val, p_val = pearsonr(df["durasi_belajar_ai"], df["skor_efisiensi_tugas"])

    print("\n[2] HASIL UJI KORELASI PEARSON (scipy.stats):")
    print("-" * 70)
    print(f"Koefisien Korelasi (r) : {r_val:.4f}")
    print(f"P-value                : {p_val:.6e} ({p_val:.6f})")
    kategori_korelasi = "Sangat Kuat Positif" if r_val > 0.8 else "Sedang / Lemah"
    status_sig = "Signifikan secara statistik (p < 0.05)" if p_val < 0.05 else "Tidak signifikan (p >= 0.05)"
    print(f"Kekuatan Hubungan      : {kategori_korelasi}")
    print(f"Signifikansi           : {status_sig}")
    print("-" * 70)

    # 4. Inisialisasi Google GenAI Client
    client = genai.Client(api_key=api_key)

    # System Instruction sesuai permintaan pengguna
    system_instruction = (
        "Kamu adalah Ahli Sosiologi Digital. Tugasmu adalah menginterpretasikan hasil "
        "uji statistik korelasi menjadi narasi analisis sosiologis yang tajam, kritis, "
        "dan mengaitkannya dengan fenomena efisiensi kerja atau alienasi teknologi."
    )

    prompt = f"""Berikut adalah hasil statistik dari survei 10 responden mahasiswa mengenai interaksi mereka dengan AI:
- Variabel X: Durasi Belajar AI (jam/minggu) -> {df['durasi_belajar_ai'].tolist()}
- Variabel Y: Skor Efisiensi Pengerjaan Tugas (skala 1-10) -> {df['skor_efisiensi_tugas'].tolist()}
- Koefisien Korelasi Pearson (r): {r_val:.4f}
- Nilai signifikansi (p-value): {p_val:.6f}

Tolong buatkan narasi analisis sosiologis yang terstruktur:
1. Pembacaan empiris terhadap angka korelasi dan signifikansinya.
2. Analisis kritis: Apakah lonjakan efisiensi ini benar-benar membebaskan mahasiswa (efisiensi kerja sejati) ataukah menciptakan bentuk baru alienasi teknologi, dehumanisasi proses belajar, dan ketergantungan algoritmik?
3. Kesimpulan reflektif bagi sosiologi pendidikan tinggi di era kecerdasan artifisial.
"""

    print("\n[3] MENGIRIM DATA KE GEMINI API UNTUK INTERPRETASI SOSIOLOGIS...")
    
    # Model target: mencoba model yang diminta (gemini-2.0-flash / gemini-1.5-flash),
    # jika sudah dideprecasi oleh Google, otomatis fallback ke Flash aktif (gemini-3.5-flash-lite / gemini-3.8-flash)
    candidate_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
    response_text = None
    used_model = None

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                )
            )
            if response and response.text:
                response_text = response.text
                used_model = model_name
                break
        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg or "not found" in err_msg.lower() or "no longer available" in err_msg.lower():
                print(f"[Info] Model '{model_name}' telah dideprecasi oleh penyedia API. Beralih ke model aktif...")
            elif "503" in err_msg or "demand" in err_msg.lower():
                print(f"[Info] Model '{model_name}' sedang overload (503). Mencoba alternatif...")
            else:
                print(f"[Warning] Gagal dengan '{model_name}': {err_msg[:120]}. Mencoba model lain...")

    # 5. Cetak narasi sosiologis ke terminal dengan rapi
    if response_text:
        print(f"\n[Info] Berhasil dianalisis oleh Model: {used_model}\n")
        print("=" * 70)
        print("          NARASI ANALISIS SOSIOLOGI DIGITAL (GEMINI AI)")
        print("=" * 70)
        print(response_text)
        print("=" * 70)
    else:
        print("\n[Error] Gagal mendapatkan respon dari seluruh kandidat model Gemini.")

if __name__ == "__main__":
    main()
