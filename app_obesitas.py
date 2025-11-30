

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# === 1️⃣ Load Dataset ===
file_path = r"D:\Semester 5\obesity_terjemahan_dan_rekomendasi perbaikan.csv"
df = pd.read_csv(file_path)

# === 2️⃣ Kolom numerik & kategorikal ===
num_cols = [
    'Usia', 'Tinggi_Badan', 'Berat_Badan',
    'Frekuensi_Konsumsi_SayurBuah', 'Frekuensi_Aktivitas_Fisik',
    'Waktu_Penggunaan_Gawai', 'Asupan_Air'
]

cat_cols = [
    'Jenis_Kelamin', 'Riwayat_Keluarga_Obesitas',
    'Konsumsi_Makanan_Tinggi_Kalori', 'Frekuensi_Nyemil',
    'Kebiasaan_Merokok', 'Pantau_Asupan_Kalori',
    'Konsumsi_Alkohol', 'Transportasi_Utama'
]

# === 3️⃣ Bersihkan data numerik ===
for col in num_cols:
    df[col] = (
        df[col].astype(str)
        .str.replace(r"[^0-9.,]", "", regex=True)
        .str.replace(",", ".", regex=True)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# === 4️⃣ Normalisasi & One-hot encoding ===
scaler = MinMaxScaler()
scaler.fit(df[num_cols])

num_scaled = scaler.transform(df[num_cols])
cat_encoded = pd.get_dummies(df[cat_cols])
X = np.hstack((num_scaled, cat_encoded.values))


# ============================================================
# 🔧 Fungsi Filter Rekomendasi (Rule-Based)
# ============================================================
def filter_rekomendasi(input_user, rekom_text: str) -> str:

    if not isinstance(rekom_text, str) or rekom_text.strip() == "":
        return rekom_text

    kalimat = [k.strip() for k in rekom_text.split('.') if k.strip()]
    kalimat_baru = []

    for k in kalimat:
        low = k.lower()

        if input_user["Kebiasaan_Merokok"] == "no" and "merokok" in low:
            continue

        if input_user["Konsumsi_Alkohol"] == "no" and "alkohol" in low:
            continue

        if input_user["Konsumsi_Makanan_Tinggi_Kalori"] == "no":
            if "tinggi kalori" in low or "kalori tinggi" in low:
                continue

        if input_user["Frekuensi_Nyemil"] in ["no", "Sometimes"]:
            if "nyemil" in low or "cemilan" in low:
                continue

        kalimat_baru.append(k)

    if not kalimat_baru:
        return "Pertahankan pola hidup sehat dengan mengatur pola makan dan aktivitas fisik."

    return '. '.join(kalimat_baru) + '.'


# ============================================================
# 5️⃣ Fungsi Precision@K
# ============================================================
def precision_at_k(input_vector, kategori_target, K=3):
    sim_all = 1 - np.mean(np.abs(X - input_vector), axis=1)
    idx_top_k = np.argsort(sim_all)[-K:][::-1]

    relevan = 0
    for idx in idx_top_k:
        if df.loc[idx, "Tingkat_Obesitas"] == kategori_target:
            relevan += 1

    return relevan / K


# ============================================================
# 6️⃣ Fungsi Rekomendasi
# ============================================================
def rekomendasi_obesitas(input_data):

    # --- BMI Check tetap dijalankan ---
    tinggi = input_data['Tinggi_Badan']
    berat = input_data['Berat_Badan']
    bmi = berat / (tinggi ** 2)

    if bmi < 18.5:
        return {
            "Tingkat_Obesitas": "Kurus",
            "Rekomendasi": "Berat badan Anda di bawah normal. Tambahkan asupan kalori sehat.",
            "Similarity": 0.0
        }

    elif 18.5 <= bmi <= 24.9:
        return {
            "Tingkat_Obesitas": "Tidak Termasuk Obesitas",
            "Rekomendasi": "Berat badan ideal. Pertahankan pola hidup sehat.",
            "Similarity": 0.0
        }

    # --- Proses similarity (tanpa threshold) ---
    num_input_df = pd.DataFrame([input_data])[num_cols]
    num_input_scaled = scaler.transform(num_input_df)

    cat_input = pd.DataFrame([input_data])[cat_cols]
    cat_input_encoded = pd.get_dummies(cat_input)
    cat_input_encoded = cat_input_encoded.reindex(columns=cat_encoded.columns, fill_value=0)

    input_vector = np.hstack((num_input_scaled, cat_input_encoded.values))
    sim = 1 - np.mean(np.abs(X - input_vector), axis=1)

    idx_best = np.argmax(sim)
    skor = sim[idx_best]

    hasil = df.loc[idx_best, ["Tingkat_Obesitas", "Rekomendasi"]]
    rekom_final = filter_rekomendasi(input_data, hasil["Rekomendasi"])

    return {
        "Tingkat_Obesitas": hasil["Tingkat_Obesitas"],
        "Rekomendasi": rekom_final,
        "Similarity": skor
    }


# ============================================================
# 7️⃣ Streamlit UI
# ============================================================
st.set_page_config(page_title="Sistem Rekomendasi Obesitas", layout="centered")
st.title("🩺 Sistem Rekomendasi Obesitas & Gaya Hidup Sehat")

st.subheader("Masukkan Data Anda")

col1, col2 = st.columns(2)
with col1:
    usia = st.number_input("Usia", 10, 100, 25)
    tinggi = st.number_input("Tinggi Badan (meter)", 1.0, 2.5, 1.65)
    berat = st.number_input("Berat Badan (kg)", 30, 200, 60)
    sayur = st.slider("Frekuensi Konsumsi Sayur/Buah", 0, 5, 2)
    aktivitas = st.slider("Frekuensi Aktivitas Fisik", 0, 5, 2)
    gawai = st.slider("Waktu Penggunaan Gawai (jam)", 0, 10, 3)
    air = st.slider("Asupan Air (liter)", 0, 5, 2)

with col2:
    jk = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    riwayat = st.selectbox("Riwayat Obesitas Keluarga", ["yes", "no"])
    kalori = st.selectbox("Konsumsi Makanan Tinggi Kalori", ["yes", "no"])
    nyemil = st.selectbox("Frekuensi Nyemil", ["Frequently", "Sometimes", "no"])
    merokok = st.selectbox("Kebiasaan Merokok", ["yes", "no"])
    pantau = st.selectbox("Pantau Asupan Kalori", ["yes", "no"])
    alkohol = st.selectbox("Konsumsi Alkohol", ["Frequently", "Sometimes", "no"])
    transport = st.selectbox("Transportasi", ["Automobile", "Public_Transportation", "Walking", "Bike"])


# ============================================================
# Tombol Cek
# ============================================================
if st.button("🔍 Cek Rekomendasi"):

    input_user = {
        'Usia': usia, 'Tinggi_Badan': tinggi, 'Berat_Badan': berat,
        'Frekuensi_Konsumsi_SayurBuah': sayur, 'Frekuensi_Aktivitas_Fisik': aktivitas,
        'Waktu_Penggunaan_Gawai': gawai, 'Asupan_Air': air,
        'Jenis_Kelamin': jk, 'Riwayat_Keluarga_Obesitas': riwayat,
        'Konsumsi_Makanan_Tinggi_Kalori': kalori, 'Frekuensi_Nyemil': nyemil,
        'Kebiasaan_Merokok': merokok, 'Pantau_Asupan_Kalori': pantau,
        'Konsumsi_Alkohol': alkohol, 'Transportasi_Utama': transport
    }

    hasil = rekomendasi_obesitas(input_user)

    st.success(f"**Tingkat Obesitas:** {hasil['Tingkat_Obesitas']}")
    st.info(f"**Rekomendasi:**\n{hasil['Rekomendasi']}")


   
