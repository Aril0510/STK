import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1️⃣ Load Dataset
# ============================================================
file_path = r"D:\Semester 5\obesity_terjemahan_dan_rekomendasi perbaikan.csv"
df = pd.read_csv(file_path)

# ============================================================
# 2️⃣ Kolom numerik & kategorikal
# ============================================================
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

# ============================================================
# 3️⃣ Bersihkan data numerik
# ============================================================
for col in num_cols:
    df[col] = (
        df[col].astype(str)
        .str.replace(r"[^0-9.,]", "", regex=True)
        .str.replace(",", ".", regex=True)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ============================================================
# 4️⃣ Normalisasi & One-hot encoding
# ============================================================
scaler = MinMaxScaler()
scaler.fit(df[num_cols])

num_scaled = scaler.transform(df[num_cols])
cat_encoded = pd.get_dummies(df[cat_cols])
X = np.hstack((num_scaled, cat_encoded.values))

# ============================================================
# 5️⃣ Filter Rekomendasi (Threshold Rule-Based)
# ============================================================
def filter_rekomendasi(input_user, rekom_text: str) -> str:

    if not rekom_text or not isinstance(rekom_text, str):
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

    return ". ".join(kalimat_baru) + "."

# ============================================================
# 6️⃣ Precision@K
# ============================================================
def precision_at_k(input_vector, kategori_target, K=3):
    sim_all = 1 - np.mean(np.abs(X - input_vector), axis=1)
    idx_top_k = np.argsort(sim_all)[-K:][::-1]

    relevan = sum(df.loc[idx, "Tingkat_Obesitas"] == kategori_target for idx in idx_top_k)
    return relevan / K

# ============================================================
# 7️⃣ Fungsi Utama: Rekomendasi Obesitas
# ============================================================
def rekomendasi_obesitas(input_data):

    # --- BMI Screening ---
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

    # --- Similarity (Mean Absolute Difference) ---
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
        "Similarity": skor,
        "Input_Vector": input_vector
    }

# ============================================================
# 8️⃣ TESTING MODEL — Output PASTI MUNCUL
# ============================================================
if __name__ == "__main__":

    # Contoh input
    input_user = {
        'Usia': 25,
        'Tinggi_Badan': 1.65,
        'Berat_Badan': 70,
        'Frekuensi_Konsumsi_SayurBuah': 2,
        'Frekuensi_Aktivitas_Fisik': 2,
        'Waktu_Penggunaan_Gawai': 4,
        'Asupan_Air': 2,
        'Jenis_Kelamin': "Male",
        'Riwayat_Keluarga_Obesitas': "no",
        'Konsumsi_Makanan_Tinggi_Kalori': "yes",
        'Frekuensi_Nyemil': "Frequently",
        'Kebiasaan_Merokok': "no",
        'Pantau_Asupan_Kalori': "no",
        'Konsumsi_Alkohol': "no",
        'Transportasi_Utama': "Walking"
    }

    print("=== HASIL REKOMENDASI ===")
    hasil = rekomendasi_obesitas(input_user)
    print(hasil)

    # Precision@K
    input_vector = hasil["Input_Vector"]
    p5 = precision_at_k(input_vector, hasil["Tingkat_Obesitas"], K=5)

    print("\nPrecision@5 =", p5)
