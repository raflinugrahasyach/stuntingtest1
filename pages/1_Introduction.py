import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Introduction - Analisis Sentimen Program Stunting",
    page_icon="📊",
    layout="wide"
)

# Judul halaman
st.title("📋 Introduction")

# Konten utama
st.markdown("""
## Analisis Sentimen dan Jaringan Sosial pada Media Sosial X untuk Menilai Persepsi Publik terhadap Program Stunting di Indonesia
""")

# Tampilkan konten dalam tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Latar Belakang & Tujuan", 
    "Manfaat Penelitian", 
    "Batasan & Penelitian Terdahulu", 
    "Metodologi", 
    "Data Sampel"
])

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Latar Belakang
        
        Stunting merupakan masalah kesehatan yang berdampak pada pertumbuhan dan perkembangan anak-anak. 
        Program penanggulangan stunting di Indonesia semakin diperhatikan oleh pemerintah. 
        Penelitian ini bertujuan untuk mengidentifikasi persepsi publik mengenai program stunting melalui 
        analisis media sosial X, dengan menggunakan metode analisis sentimen dan jaringan sosial.
        """)
    
    with col2:
        st.image("https://via.placeholder.com/400x300?text=Ilustrasi+Stunting", caption="Ilustrasi Program Stunting di Indonesia", use_column_width=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### Tujuan Penelitian
    
    1. Mengidentifikasi dan mengklasifikasikan sentimen publik terhadap program stunting 2024 di media sosial X menggunakan VADER Lexicon dan BERT.
    
    2. Membandingkan performa algoritma machine learning GBT, SVM, dan XGBoost untuk menentukan model dengan akurasi terbaik dalam klasifikasi sentimen publik terkait program stunting.
    
    3. Menganalisis posisi dan peran aktor dalam jaringan sosial yang terbentuk dari percakapan mengenai program stunting di media sosial X.
    """)

with tab2:
    st.markdown("""
    ### Manfaat Penelitian
    
    1. **Evaluasi Program**: Memberikan wawasan tentang persepsi publik terhadap program penanggulangan stunting, yang dapat digunakan untuk evaluasi dan perbaikan kebijakan.
    
    2. **Identifikasi Aktor Kunci**: Mengidentifikasi aktor kunci dalam pembentukan opini publik, menggunakan analisis jaringan sosial untuk memetakan pengaruh dalam diskusi mengenai program stunting.
    
    3. **Rekomendasi Kebijakan**: Memberikan masukan kepada pemerintah terkait strategi optimalisasi program penanggulangan stunting yang lebih responsif terhadap kebutuhan masyarakat.
    
    4. **Kontribusi Ilmiah**: Memberikan kontribusi pada pengembangan ilmu data, khususnya dalam analisis sentimen dan jaringan sosial, serta membuka peluang untuk penelitian lebih lanjut di bidang komunikasi digital dan kesehatan masyarakat.
    """)
    
    # Visualisasi manfaat penelitian (ilustrasi)
    st.markdown("#### Visualisasi Manfaat Penelitian")
    
    manfaat_data = {
        'Kategori': ['Kebijakan Publik', 'Kesehatan Masyarakat', 'Pengembangan Ilmu', 'Komunikasi Publik'],
        'Nilai': [85, 90, 75, 80]
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Kategori', y='Nilai', data=pd.DataFrame(manfaat_data), ax=ax, palette='viridis')
    ax.set_title('Distribusi Manfaat Penelitian Berdasarkan Kategori')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Tingkat Manfaat (%)')
    ax.set_xlabel('Kategori Manfaat')
    
    # Menambahkan nilai di atas bar
    for i, v in enumerate(manfaat_data['Nilai']):
        ax.text(i, v + 2, str(v), ha='center')
    
    st.pyplot(fig)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Batasan Masalah
        
        Penelitian ini terbatas pada analisis data dari media sosial X terkait persepsi publik terhadap program stunting di tahun 2024. Fokus utama adalah pada:
        
        - Analisis sentimen menggunakan VADER Lexicon dan BERT
        - Evaluasi performa model machine learning dalam klasifikasi sentimen
        - Analisis data hanya dari platform X
        - Periode pengamatan tahun 2024
        """)
    
    with col2:
        st.markdown("""
        ### Penelitian Terdahulu
        
        Beberapa studi terkait analisis sentimen dan jaringan sosial telah dilakukan untuk menilai persepsi publik terhadap isu kesehatan. Namun, penelitian ini mengambil pendekatan yang lebih terkini dengan:
        
        - Penggunaan model machine learning terbaru
        - Integrasi analisis jaringan sosial
        - Penggunaan data real-time dari media sosial
        - Fokus spesifik pada program stunting di Indonesia
        """)
    
    # Timeline penelitian (ilustrasi)
    st.markdown("#### Timeline Penelitian Terdahulu")
    
    timeline_data = {
        'Tahun': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'Jumlah Penelitian': [3, 5, 7, 9, 12, 15, 18]
    }
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Tahun', y='Jumlah Penelitian', data=pd.DataFrame(timeline_data), marker='o', linewidth=2, ax=ax)
    ax.set_title('Perkembangan Penelitian Terkait Analisis Sentimen Program Stunting')
    ax.set_ylabel('Jumlah Penelitian')
    ax.set_xlabel('Tahun')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Menambahkan nilai di atas titik
    for i, v in enumerate(timeline_data['Jumlah Penelitian']):
        ax.text(timeline_data['Tahun'][i], v + 0.5, str(v), ha='center')
    
    st.pyplot(fig)

with tab4:
    st.markdown("""
    ### Metodologi Penelitian - Pengumpulan Data
    
    Data yang digunakan dalam penelitian ini berasal dari media sosial X, yang dipilih karena banyaknya percakapan yang terjadi seputar program stunting. Pengumpulan data dilakukan dengan cara scraping tweet terkait stunting menggunakan API dari media sosial X.
    
    #### Proses Pengumpulan Data:
    
    1. **Penentuan Kata Kunci**: Mengidentifikasi kata kunci yang relevan dengan program stunting
    2. **Pengumpulan Data**: Menggunakan API X untuk mengumpulkan tweet berdasarkan kata kunci
    3. **Penyaringan Data**: Menyaring data yang tidak relevan atau duplikat
    4. **Pra-pemrosesan**: Membersihkan data untuk analisis lebih lanjut
    """)
    
    # Visualisasi alur metodologi
    st.markdown("#### Alur Metodologi Penelitian")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        📥 **Pengumpulan Data**
        - Scraping X API
        - Kata kunci stunting
        - Filter tweet
        """)
    
    with col2:
        st.markdown("""
        🧹 **Pra-pemrosesan**
        - Pembersihan teks
        - Normalisasi
        - Tokenisasi
        """)
    
    with col3:
        st.markdown("""
        🔍 **Analisis Sentimen**
        - VADER Lexicon
        - BERT Model
        - Komparasi hasil
        """)
    
    with col4:
        st.markdown("""
        📊 **Visualisasi & Insight**
        - Word Cloud
        - Distribusi sentimen
        - Jaringan interaksi
        """)

with tab5:
    st.markdown("### Data Sampel")
    
    # Sample data (dummy data for illustration)
    data = {
        'username': ['user1', 'user2', 'user3', 'user4', 'user5'],
        'full_text': [
            'Program stunting di daerah saya sangat membantu keluarga dengan ekonomi rendah. Terima kasih pemerintah!',
            'Pendistribusian dana program stunting masih belum merata di daerah terpencil. Tolong perhatikan!',
            'Sosialisasi program stunting perlu ditingkatkan agar masyarakat lebih paham manfaatnya.',
            'Saya lihat program stunting di desa kami sudah berjalan dengan baik, anak-anak jadi lebih sehat.',
            'Kenapa program stunting tidak menjangkau daerah kami? Padahal banyak yang membutuhkan.'
        ],
        'BERT_Label': ['positive', 'negative', 'neutral', 'positive', 'negative'],
        'VADER_Label': ['positive', 'negative', 'neutral', 'positive', 'negative']
    }
    
    df_sample = pd.DataFrame(data)
    
    # Tampilkan data sampel
    st.dataframe(df_sample, use_container_width=True)
    
    # Visualisasi distribusi sentimen dari data sampel
    st.markdown("#### Distribusi Sentimen dari Data Sampel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bert_counts = df_sample['BERT_Label'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(bert_counts, labels=bert_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
        ax.set_title('Distribusi Sentimen BERT')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        st.pyplot(fig)
    
    with col2:
        vader_counts = df_sample['VADER_Label'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('magma'))
        ax.set_title('Distribusi Sentimen VADER')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        st.pyplot(fig)

# Tambahkan informasi mengenai pengembangan aplikasi streamlit
st.markdown("---")
st.markdown("""
### Tentang Halaman Introduction

Halaman ini memberikan gambaran umum tentang penelitian analisis sentimen terhadap program stunting di Indonesia.
Navigasikan ke halaman selanjutnya untuk melihat hasil analisis berdasarkan rumusan masalah.
""")