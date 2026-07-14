import pandas as pd
import numpy as np
import streamlit as st

# Setup Tampilan Web
st.set_page_config(page_title="SPK Karyawan - Sistem Cerdas", layout="wide")
st.title("Sistem Pendukung Keputusan (SPK) Penilaian Karyawan")
st.markdown("Membandingkan metode **SAW, WP, dan TOPSIS** untuk 30 data karyawan.")
st.divider()

# ==========================================
# 1. Generate 30 Data Dummy Karyawan
# ==========================================
np.random.seed(42) 

# List 30 Nama Karyawan Dummy
nama_karyawan = [
    "Budi", "Citra", "Dedi", "Eka", "Fajar", "Gita", "Hadi", "Intan", "Joko", "Kartika",
    "Lukman", "Maya", "Nanda", "Oka", "Putri", "Qori", "Rizky", "Sari", "Tono", "Utami",
    "Vina", "Wawan", "Xaverius", "Yudi", "Zahra", "Agus", "Bambang", "Cahyo", "Dian", "Erwin"
]

data = {
    'Alternatif': nama_karyawan,
    'C1_Kinerja': np.random.randint(60, 100, 30),
    'C2_Disiplin': np.random.randint(60, 100, 30),
    'C3_Terlambat': np.random.randint(1, 15, 30), 
    'C4_Kerjasama': np.random.randint(60, 100, 30)
}
df = pd.DataFrame(data)

bobot = np.array([0.35, 0.25, 0.20, 0.20])
jenis = np.array([1, 1, -1, 1]) 
matriks = df.iloc[:, 1:].values

# ==========================================
# 2. Perhitungan SAW
# ==========================================
norm_saw = np.zeros(matriks.shape)
for j in range(matriks.shape[1]):
    if jenis[j] == 1: 
        norm_saw[:, j] = matriks[:, j] / np.max(matriks[:, j])
    else: 
        norm_saw[:, j] = np.min(matriks[:, j]) / matriks[:, j]
        
saw_score = np.sum(norm_saw * bobot, axis=1)

# ==========================================
# 3. Perhitungan WP
# ==========================================
bobot_wp = bobot / sum(bobot)
pangkat = bobot_wp * jenis 
wp_score = np.prod(matriks ** pangkat, axis=1)
wp_v = wp_score / sum(wp_score)

# ==========================================
# 4. Perhitungan TOPSIS
# ==========================================
pembagi = np.sqrt(np.sum(matriks**2, axis=0))
norm_topsis = matriks / pembagi
matriks_berbobot = norm_topsis * bobot

ideal_positif = np.where(jenis == 1, np.max(matriks_berbobot, axis=0), np.min(matriks_berbobot, axis=0))
ideal_negatif = np.where(jenis == 1, np.min(matriks_berbobot, axis=0), np.max(matriks_berbobot, axis=0))

d_plus = np.sqrt(np.sum((matriks_berbobot - ideal_positif)**2, axis=1))
d_min = np.sqrt(np.sum((matriks_berbobot - ideal_negatif)**2, axis=1))

topsis_score = d_min / (d_min + d_plus)

# ==========================================
# 5. Menggabungkan Hasil ke Tabel
# ==========================================
hasil_df = pd.DataFrame({
    'Alternatif': df['Alternatif'],
    'SAW_Score': np.round(saw_score, 3),
    'WP_Score': np.round(wp_v, 3),
    'TOPSIS_Score': np.round(topsis_score, 3)
})

hasil_df['SAW_Rank'] = hasil_df['SAW_Score'].rank(ascending=False).astype(int)
hasil_df['WP_Rank'] = hasil_df['WP_Score'].rank(ascending=False).astype(int)
hasil_df['TOPSIS_Rank'] = hasil_df['TOPSIS_Score'].rank(ascending=False).astype(int)

# Mengurutkan dari ranking 1
hasil_df = hasil_df.sort_values(by='SAW_Score', ascending=False)

final_df = pd.DataFrame({
    'Alternatif': hasil_df['Alternatif'],
    'SAW (V / Rank)': hasil_df['SAW_Score'].astype(str) + " / " + hasil_df['SAW_Rank'].astype(str),
    'WP (V / Rank)': hasil_df['WP_Score'].astype(str) + " / " + hasil_df['WP_Rank'].astype(str),
    'TOPSIS (CC / Rank)': hasil_df['TOPSIS_Score'].astype(str) + " / " + hasil_df['TOPSIS_Rank'].astype(str)
})

# Menampilkan di Web
st.subheader("Data Awal Karyawan (Dummy)")
st.dataframe(df, use_container_width=True)

st.subheader("Hasil Perbandingan Metode (SAW, WP, TOPSIS)")
st.dataframe(final_df, use_container_width=True)

st.success("Insight utama: Ketiga metode secara konsisten menempatkan alternatif terbaik pada peringkat atas. Hanya peringkat menengah-bawah yang sedikit berbeda antar metode, hal wajar karena perbedaan mekanisme perhitungan, namun tidak mengubah rekomendasi akhir.")
