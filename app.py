import streamlit as st
import pandas as pd
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Program Stunting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul utama
st.title("📊 Analisis Sentimen dan Jaringan Sosial Program Stunting di Indonesia")

# Deskripsi aplikasi
st.markdown("""
# Selamat Datang di Dashboard Analisis Sentimen Program Stunting

Aplikasi ini menyajikan hasil analisis sentimen dan jaringan sosial dari media sosial X (Twitter) terkait program stunting di Indonesia.

## Fitur Utama

- **Analisis Sentimen Publik** menggunakan VADER Lexicon dan BERT
- **Perbandingan Algoritma Machine Learning** (GBT, SVM, dan XGBoost)
- **Analisis Jaringan Sosial** untuk memahami aktor utama dalam diskusi stunting

Gunakan sidebar untuk navigasi ke halaman yang berbeda.
""")

# Tambahkan gambar dashboard di halaman utama
st.image("https://via.placeholder.com/800x400?text=Visualisasi+Data+Stunting", 
         caption="Visualisasi Data Analisis Sentimen Program Stunting", use_container_width=True)

# Menambahkan informasi footer
st.markdown("---")
st.markdown("### Tentang Aplikasi")
st.markdown("""
Aplikasi ini dikembangkan untuk analisis komprehensif terhadap persepsi publik mengenai program stunting di Indonesia
berdasarkan data yang dikumpulkan dari media sosial X.
""")